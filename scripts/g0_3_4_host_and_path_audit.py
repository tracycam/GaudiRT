#!/usr/bin/env python3
"""G0-3 & G0-4: Audit of async_scheduling path gates, 8-rank CPU stages under uclamp=1024,
and 8-rank collective readiness alignment across the 141 collectives/step.
"""

import argparse
import json
from pathlib import Path
import statistics

REPO_ROOT = Path(__file__).resolve().parents[2]


def analyze_cpu_and_readiness():
    base_dir = REPO_ROOT / "instance-assets/critical-cpu-path-20261011"
    cpu_stages_path = base_dir / "high-qos-cpu-stages.json"
    readiness_path = base_dir / "high-qos-analysis/rank-readiness.json"

    cpu_stages = json.loads(cpu_stages_path.read_text())
    readiness = json.loads(readiness_path.read_text())

    # Filter only production70-critical-qos-b (uclamp=1024)
    high_qos_ranks = [
        r for r in cpu_stages if "production70-critical-qos-b" in r["file"]
    ]

    per_rank_breakdown = []
    for r in high_qos_ranks:
        st = r["stages"]
        preflight = st["NativePlanBindingPreflight"]["median_us"]
        region_sub = st["PreparedRegionSubmit"]["median_us"]
        plan_sub = st["NativePlanSubmit"]["median_us"]
        coll_sub = st["SubmitCollectivePlan"]["median_us"]
        plan_replay = st["NativePlanReplay"]["median_us"]

        per_rank_breakdown.append(
            {
                "file": r["file"],
                "NativePlanBindingPreflight_median_us": preflight,
                "PreparedRegionSubmit_per_region_median_us": region_sub,
                "PreparedRegionSubmit_2regions_est_us": region_sub * 2.0,
                "NativePlanSubmit_median_us": plan_sub,
                "SubmitCollectivePlan_median_us": coll_sub,
                "NativePlanReplay_median_us": plan_replay,
                "queue_handoff_overhead_est_us": coll_sub - plan_sub,
                "replay_wrapper_outside_submit_est_us": plan_replay - coll_sub,
                "non_region_submit_overhead_est_us": plan_sub - (region_sub * 2.0),
            }
        )

    stage_medians_across_8ranks = {
        k: statistics.median(x[k] for x in per_rank_breakdown)
        for k in [
            "NativePlanBindingPreflight_median_us",
            "PreparedRegionSubmit_per_region_median_us",
            "PreparedRegionSubmit_2regions_est_us",
            "NativePlanSubmit_median_us",
            "SubmitCollectivePlan_median_us",
            "NativePlanReplay_median_us",
            "queue_handoff_overhead_est_us",
            "replay_wrapper_outside_submit_est_us",
            "non_region_submit_overhead_est_us",
        ]
    }

    return {
        "g0_3_async_path_gates": {
            "vllm_serving_gate": {
                "file": "gaudi-kernels-public/python/gaudi_kernels/serving/decode_graph.py",
                "line": 77,
                "condition": "if (... or runner.use_async_scheduling or ...): return False",
                "impact": "Enabling vLLM async_scheduling causes admitted() to return False, exiting the full step-graph fast path and losing _prepared_decode_input layout reuse.",
            },
            "bridge_graph_gate": {
                "file": "gaudi-pytorch-bridge/backend/habana_device/HPUGraph.cpp",
                "condition": "asynchronous == true or rebound inputs",
                "impact": "Rejects PreparedRegionProgram fast path and falls back to legacy V3/queued replay.",
            },
            "bridge_queue_gate": {
                "file": "gaudi-pytorch-bridge/backend/habana_device/hccl_kernels.cpp",
                "function": "SubmitCollectivePlan",
                "impact": "Blocks caller thread via future.get() until HCCL worker thread finishes submitting all 285 commands.",
            },
        },
        "g0_4_bridge_host_stages_uclamp1024": {
            "eight_rank_medians_us": stage_medians_across_8ranks,
            "per_rank": per_rank_breakdown,
        },
        "g0_5_eight_rank_collective_readiness_summary": {
            k: v for k, v in readiness.items() if k != "steps"
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(
            REPO_ROOT
            / "GaudiRT/evidence/gate0-20261023/g0-3-4-host-and-path-audit.json"
        ),
    )
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    res = analyze_cpu_and_readiness()
    out_path.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res["g0_4_bridge_host_stages_uclamp1024"]["eight_rank_medians_us"], indent=2))
    print(json.dumps(res["g0_5_eight_rank_collective_readiness_summary"], indent=2))


if __name__ == "__main__":
    main()
