#!/usr/bin/env python3
"""G0-6: Per-recipe audit of 70L TP8 decode compute union (5.73 ms),
intra-recipe no-compute bubbles (2.06 ms), and HBM weight roofline.

Reads raw SDK trace JSONL from high-QoS (uclamp=1024) 70-layer TP8 runs and
attributes:
  1. Per-recipe total span, TPC union, MME union,TPC|MME compute union;
  2. Per-recipe intra-recipe no-TPC/MME bubble (and EDMA/PDMA activity inside it);
  3. Per-operator-family and per-recipe-pattern aggregation across all 144 recipes/step;
  4. Comparison against theoretical HBM weight bandwidth roofline (2.45 TB/s peak, 2.08 TB/s @ 85%).
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "gaudi-kernels-public"))

from tools.validation.executor.analyze_bridge_device_trace import (  # noqa: E402
    engine_family,
    from_sdk_jsonl,
    union,
)


def duration(intervals):
    return sum(b - a for a, b in union(intervals))


def overlap(intervals, a, b):
    return duration([(max(a, x), min(b, y)) for x, y in intervals if x < b and y > a])


def classify_op_name(name: str) -> str:
    rules = [
        ("o_proj_bf16", "self_attn/o_proj"),
        ("lm_head_bf16", "logits_processor/aten__linear"),
        ("dense_mlp_gate", "mlp/gate/precision_fix__mm"),
        ("dense_mlp", "model/0/mlp"),
        ("moe_gp_scale_tail", "gp_scale_tail"),
        ("moe_down_broadcast", "downact"),
        ("moe_compact_gate", "compact_gate"),
        ("moe_combine", "pf_combine"),
        ("router_post8", "router_post8"),
        ("router_topk", "topk"),
        ("router_other", "router"),
        ("qkv_fp8_bmm", "block_fp8"),
        ("qkv_neumaier_reduce", "neumaier"),
        ("qkv_post_rope_cache", "qkv_post"),
        ("attn_swa128", "swa128"),
        ("attn_full", "self_attn/attn"),
        ("rmsnorm_residual", "residual_rmsnorm"),
        ("rmsnorm_input", "norm_block128"),
        ("tp_local_sum", "pf_sum"),
        ("isfinite", "isfinite"),
    ]
    for label, needle in rules:
        if needle in name:
            return label
    return "other"


# Per-invocation weight read bytes on 1 rank of TP8 (batch=1, 4K context)
# Used to compute per-family HBM roofline at 2.45 TB/s (peak) and 2.08 TB/s (85% effective)
FAMILY_BYTES_PER_LAYER_RANK = {
    # o_proj: BF16 [6144, 16384/8=2048] -> 6144 * 2048 * 2 = 25,165,824 B (~25.17 MB)
    "o_proj_bf16": 6144 * 2048 * 2,
    # qkv_proj: FP8 Q(128*192/8=3072) + K(8*192/8=192) + V(8*128/8=128) = 3392 cols * 6144 * 1 B = 20,840,448 B
    "qkv_fp8_bmm": 3392 * 6144 * 1,
    # MoE top-8 active experts per MoE layer (mxfp4 ~4.25 bits/param across gate_up + down):
    # Total per layer per rank ~20.0 MB: split ~2/3 gate_up (13.33 MB), ~1/3 down (6.67 MB)
    "moe_gp_scale_tail": int(13.33e6),
    "moe_down_broadcast": int(6.67e6),
}


def audit_trace(jsonl_path: Path):
    doc = from_sdk_jsonl(jsonl_path)
    events = doc["traceEvents"]
    threads = {
        (x["pid"], x["tid"]): x["args"]["name"]
        for x in events
        if x["ph"] == "M" and x["name"] == "thread_name"
    }
    hw_events = [
        dict(x, family=engine_family(threads.get((x["pid"], x["tid"]), "")))
        for x in events
        if x["ph"] == "X" and engine_family(threads.get((x["pid"], x["tid"]), ""))
    ]
    windows = sorted(
        [x for x in events if x["ph"] == "X" and x["name"] == "runner.sample_tokens"],
        key=lambda x: x["ts"],
    )

    step_recipe_records = []
    fine_op_totals = defaultdict(lambda: defaultdict(list))

    for step_idx, w in enumerate(windows):
        a, b = w["ts"], w["ts"] + w["dur"]
        inside = [
            (max(x["ts"], a), min(x["ts"] + x["dur"], b), x)
            for x in hw_events
            if x["ts"] < b and x["ts"] + x["dur"] > a
        ]
        compute = sorted(
            [x for x in inside if x[2]["family"] in ("TPC", "MME")],
            key=lambda x: x[0],
        )
        if not compute:
            continue

        by_engine = {
            f: union((p, q) for p, q, x in inside if x["family"] == f)
            for f in ["TPC", "MME", "PDMA", "EDMA", "NIC"]
        }

        # Group consecutive TPC/MME events by recipeId into recipe runs (144 per step)
        runs = []
        for begin, end, x in compute:
            rid = x["args"].get("recipeId")
            if runs and runs[-1]["recipe_id"] == rid:
                runs[-1]["end"] = max(runs[-1]["end"], end)
                runs[-1]["events"].append((begin, end, x))
            else:
                runs.append(
                    {
                        "begin": begin,
                        "end": end,
                        "recipe_id": rid,
                        "events": [(begin, end, x)],
                    }
                )

        # Fine operator classification across entire step
        by_fine_op = defaultdict(list)
        by_fine_op_tpc = defaultdict(list)
        by_fine_op_mme = defaultdict(list)
        for p, q, x in compute:
            cls = classify_op_name(x["name"])
            by_fine_op[cls].append((p, q))
            if x["family"] == "TPC":
                by_fine_op_tpc[cls].append((p, q))
            elif x["family"] == "MME":
                by_fine_op_mme[cls].append((p, q))

        for cls in set(by_fine_op.keys()):
            fine_op_totals[cls]["union_us"].append(duration(by_fine_op[cls]))
            fine_op_totals[cls]["tpc_us"].append(duration(by_fine_op_tpc[cls]))
            fine_op_totals[cls]["mme_us"].append(duration(by_fine_op_mme[cls]))

        # Analyze each of the 144 recipe runs in this step
        per_step_runs = []
        for ordinal_idx, r in enumerate(runs):
            r_begin, r_end = r["begin"], r["end"]
            r_span = r_end - r_begin
            tpc_iv = [(p, q) for p, q, x in r["events"] if x["family"] == "TPC"]
            mme_iv = [(p, q) for p, q, x in r["events"] if x["family"] == "MME"]
            comp_iv = [(p, q) for p, q, _ in r["events"]]

            tpc_u = duration(tpc_iv)
            mme_u = duration(mme_iv)
            comp_u = duration(comp_iv)
            intra_gap = max(0.0, r_span - comp_u)

            # Check how much of the intra-recipe no-TPC/MME gap overlaps with EDMA or PDMA
            edma_in_span = overlap(by_engine["EDMA"], r_begin, r_end)
            pdma_in_span = overlap(by_engine["PDMA"], r_begin, r_end)

            op_classes = sorted(
                set(classify_op_name(x["name"]) for _, _, x in r["events"])
            )
            sample_names = sorted(set(x["name"] for _, _, x in r["events"]))[:4]
            signature = "+".join(op_classes)

            per_step_runs.append(
                {
                    "ordinal": ordinal_idx,
                    "recipe_id": r["recipe_id"],
                    "signature": signature,
                    "op_classes": op_classes,
                    "sample_names": sample_names,
                    "event_count": len(r["events"]),
                    "span_us": r_span,
                    "compute_union_us": comp_u,
                    "tpc_union_us": tpc_u,
                    "mme_union_us": mme_u,
                    "intra_recipe_gap_us": intra_gap,
                    "edma_in_span_us": edma_in_span,
                    "pdma_in_span_us": pdma_in_span,
                }
            )
        step_recipe_records.append(per_step_runs)

    # Aggregate across steps by recipe_id and by recipe signature
    by_recipe_id = defaultdict(list)
    for step_runs in step_recipe_records:
        # Within one step, sum metrics per recipe_id first (in case a recipe_id runs multiple times)
        step_by_rid = defaultdict(
            lambda: {
                "count": 0,
                "span_us": 0.0,
                "compute_union_us": 0.0,
                "tpc_union_us": 0.0,
                "mme_union_us": 0.0,
                "intra_recipe_gap_us": 0.0,
                "edma_in_span_us": 0.0,
                "pdma_in_span_us": 0.0,
                "signature": "",
                "sample_names": [],
            }
        )
        for item in step_runs:
            rid = item["recipe_id"]
            d = step_by_rid[rid]
            d["count"] += 1
            d["span_us"] += item["span_us"]
            d["compute_union_us"] += item["compute_union_us"]
            d["tpc_union_us"] += item["tpc_union_us"]
            d["mme_union_us"] += item["mme_union_us"]
            d["intra_recipe_gap_us"] += item["intra_recipe_gap_us"]
            d["edma_in_span_us"] += item["edma_in_span_us"]
            d["pdma_in_span_us"] += item["pdma_in_span_us"]
            d["signature"] = item["signature"]
            if not d["sample_names"]:
                d["sample_names"] = item["sample_names"]
        for rid, d in step_by_rid.items():
            by_recipe_id[rid].append(d)

    # Also aggregate by recipe signature per step
    by_signature = defaultdict(list)
    for step_runs in step_recipe_records:
        step_by_sig = defaultdict(
            lambda: {
                "runs_per_step": 0,
                "span_us": 0.0,
                "compute_union_us": 0.0,
                "tpc_union_us": 0.0,
                "mme_union_us": 0.0,
                "intra_recipe_gap_us": 0.0,
                "edma_in_span_us": 0.0,
                "pdma_in_span_us": 0.0,
                "sample_names": [],
            }
        )
        for item in step_runs:
            sig = item["signature"]
            d = step_by_sig[sig]
            d["runs_per_step"] += 1
            d["span_us"] += item["span_us"]
            d["compute_union_us"] += item["compute_union_us"]
            d["tpc_union_us"] += item["tpc_union_us"]
            d["mme_union_us"] += item["mme_union_us"]
            d["intra_recipe_gap_us"] += item["intra_recipe_gap_us"]
            d["edma_in_span_us"] += item["edma_in_span_us"]
            d["pdma_in_span_us"] += item["pdma_in_span_us"]
            if not d["sample_names"]:
                d["sample_names"] = item["sample_names"]
        for sig, d in step_by_sig.items():
            by_signature[sig].append(d)

    signature_summary = []
    for sig, records in by_signature.items():
        signature_summary.append(
            {
                "signature": sig,
                "runs_per_step_median": statistics.median(
                    r["runs_per_step"] for r in records
                ),
                "step_span_median_us": statistics.median(r["span_us"] for r in records),
                "step_compute_union_median_us": statistics.median(
                    r["compute_union_us"] for r in records
                ),
                "step_tpc_union_median_us": statistics.median(
                    r["tpc_union_us"] for r in records
                ),
                "step_mme_union_median_us": statistics.median(
                    r["mme_union_us"] for r in records
                ),
                "step_intra_recipe_gap_median_us": statistics.median(
                    r["intra_recipe_gap_us"] for r in records
                ),
                "step_edma_in_span_median_us": statistics.median(
                    r["edma_in_span_us"] for r in records
                ),
                "step_pdma_in_span_median_us": statistics.median(
                    r["pdma_in_span_us"] for r in records
                ),
                "sample_names": records[0]["sample_names"],
            }
        )
    signature_summary.sort(key=lambda x: x["step_span_median_us"], reverse=True)

    fine_op_summary = []
    for cls, metrics in fine_op_totals.items():
        u_med = statistics.median(metrics["union_us"])
        tpc_med = statistics.median(metrics["tpc_us"])
        mme_med = statistics.median(metrics["mme_us"])
        invocations = 70 if "o_proj" in cls or "qkv" in cls else (69 if "moe" in cls else 1)
        bytes_per_inv = FAMILY_BYTES_PER_LAYER_RANK.get(cls, 0)
        total_bytes = bytes_per_inv * invocations
        roofline_peak_us = (total_bytes / 2.45e12) * 1e6 if total_bytes > 0 else None
        roofline_85_us = (total_bytes / 2.0825e12) * 1e6 if total_bytes > 0 else None
        fine_op_summary.append(
            {
                "operator_class": cls,
                "step_union_median_us": u_med,
                "step_tpc_median_us": tpc_med,
                "step_mme_median_us": mme_med,
                "estimated_step_weight_bytes": total_bytes if total_bytes > 0 else None,
                "roofline_peak_2450gbps_us": roofline_peak_us,
                "roofline_eff_2082gbps_us": roofline_85_us,
                "excess_over_eff_roofline_us": (
                    (u_med - roofline_85_us) if roofline_85_us is not None else None
                ),
            }
        )
    fine_op_summary.sort(key=lambda x: x["step_union_median_us"], reverse=True)

    return {
        "trace_file": str(jsonl_path),
        "steps_analyzed": len(step_recipe_records),
        "recipe_signature_breakdown": signature_summary,
        "fine_operator_breakdown": fine_op_summary,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace",
        default=str(
            REPO_ROOT
            / "instance-assets/critical-cpu-path-20261011/remote-critical/results/production70-critical-qos-b/sdk-scope-1-rank0.jsonl"
        ),
    )
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT / "GaudiRT/evidence/gate0-20261023/g0-6-recipe-roofline-audit.json"
        ),
    )
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = audit_trace(Path(args.trace))
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
