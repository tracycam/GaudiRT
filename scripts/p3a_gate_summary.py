#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def fields(line):
    result = {}
    for token in line.split()[1:]:
        if "=" in token:
            key, value = token.split("=", 1)
            result[key] = value
    return result


def manifest(path):
    result = {}
    with path.open() as handle:
        if handle.readline().strip() != "# step record_fnv1a64 chain_fnv1a64":
            raise ValueError(f"invalid manifest header: {path}")
        for line in handle:
            step, record, chain = line.split()
            result[int(step)] = (record, chain)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("direct", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("steps", type=int)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--text", dest="text_path", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 32638:
        raise ValueError("steps must be in [1,32638]")

    verdict = json.loads((args.direct / "verdict.json").read_text())
    if verdict != {
        "verdict": "PASS", "mode": "direct", "ranks": 8, "steps": args.steps,
        "native_capture_steps": 130, "output_bytes_per_rank_step": 32768,
        "guard_bytes_per_rank_step": 512,
    }:
        raise ValueError(f"invalid verdict.json: {verdict}")

    ranks = []
    for rank in range(8):
        log_path = args.direct / f"rank{rank}.log"
        step_lines = [fields(line) for line in log_path.read_text().splitlines()
                      if line.startswith("[P3A_STEP]")]
        verdict_lines = [fields(line) for line in log_path.read_text().splitlines()
                         if line.startswith("[P3A_DIRECT_VERDICT]")]
        if len(step_lines) != args.steps or len(verdict_lines) != 1:
            raise ValueError(f"rank {rank}: invalid step/verdict counts")
        expected = manifest(args.reference / f"rank{rank}_manifest.tsv")
        totals = {"compute_submissions": 0, "compute_boundary_submissions": 0,
                  "compute_nop_bytes": 0, "external_wait_chunk1_writes": 0,
                  "monitor_chunk1_writes": 0, "monitor_high_chunk_writes": 0}
        for index, step in enumerate(step_lines):
            if int(step["step"]) != index or int(step["rank"]) != rank:
                raise ValueError(f"rank {rank}: non-sequential step at {index}")
            if int(step["set"]) != index % 2:
                raise ValueError(f"rank {rank} step {index}: wrong buffer set")
            pre = int(step["pre_target"])
            post = int(step["post_target"])
            external = int(step["post_wait_target"])
            if pre != 261 + 2 * index or post != pre + 1:
                raise ValueError(f"rank {rank} step {index}: compute target mismatch")
            if external != 131 + index:
                raise ValueError(f"rank {rank} step {index}: external target mismatch")
            if int(step["external_target"]) != external or int(step["internal_target"]) != external:
                raise ValueError(f"rank {rank} step {index}: network target mismatch")
            long_so_fields = {"wait_chunks", "external_wait_chunk1_writes",
                              "monitor_target", "monitor_chunk1_writes",
                              "monitor_high_chunk_writes"}
            if args.steps > 6000 and not long_so_fields <= step.keys():
                raise ValueError(f"rank {rank} step {index}: missing per-step long-SO evidence")
            if long_so_fields <= step.keys():
                chunks = [int(value) for value in step["wait_chunks"].split(":")]
                expected_chunks = [(external >> (15 * i)) & 0x7fff for i in range(4)]
                if chunks != expected_chunks or int(step["monitor_target"]) != pre:
                    raise ValueError(f"rank {rank} step {index}: encoded long-SO target mismatch")
                expected_high_writes = 5 if index == 16254 else 0
                if (int(step["monitor_chunk1_writes"]) != expected_high_writes or
                        int(step["monitor_high_chunk_writes"]) != expected_high_writes):
                    raise ValueError(f"rank {rank} step {index}: monitor chunk1 write mismatch")
            for stage in ("pre", "wait", "post"):
                start, end = (int(value, 16) for value in step[f"{stage}_pi"].split(":"))
                if end <= start:
                    raise ValueError(f"rank {rank} step {index}: invalid {stage} PI")
            if int(step["compute_submissions"]) < 6 or int(step["network_submissions"]) != 10:
                raise ValueError(f"rank {rank} step {index}: publication count mismatch")
            if step["memcmp"] != "PASS" or step["guard"] != "PASS":
                raise ValueError(f"rank {rank} step {index}: output/guard failure")
            if (step["record_fnv1a64"], step["chain_fnv1a64"]) != expected[index]:
                raise ValueError(f"rank {rank} step {index}: reference hash mismatch")
            for key in totals:
                totals[key] += int(step.get(key, 0))

        final = verdict_lines[0]
        if int(final["direct_steps"]) != args.steps or int(final["rank"]) != rank:
            raise ValueError(f"rank {rank}: final verdict step mismatch")
        for key, value in totals.items():
            if key in final and int(final[key]) != value:
                raise ValueError(f"rank {rank}: {key} total mismatch")
            if args.steps > 6000 and key not in final:
                raise ValueError(f"rank {rank}: missing long-SO total {key}")
        if int(final["network_submissions"]) != 10 * args.steps:
            raise ValueError(f"rank {rank}: network submit total mismatch")
        for key in ("output_memcmp", "guard_memcmp", "hash_chain"):
            if final[key] != "PASS":
                raise ValueError(f"rank {rank}: final {key} failure")
        if args.steps >= 64 and (int(final["compute_boundary_submissions"]) == 0 or
                                 int(final["network_nop_bytes"]) == 0):
            raise ValueError(f"rank {rank}: alignment paths were not exercised")
        if args.steps >= 1000 and int(final["compute_wraps"]) == 0:
            raise ValueError(f"rank {rank}: compute CCB wrap was not exercised")
        if args.steps >= 6000 and int(final["network_wraps"]) == 0:
            raise ValueError(f"rank {rank}: network CCB wrap was not exercised")
        if args.steps >= 16255 and (int(final["monitor_chunk1_writes"]) != 5 or
                                    int(final["monitor_high_chunk_writes"]) != 5):
            raise ValueError(f"rank {rank}: monitor chunk1 boundary was not exercised")
        if args.steps >= 32638 and int(final["external_wait_chunk1_writes"]) != 1:
            raise ValueError(f"rank {rank}: external WAIT chunk1 boundary was not exercised")
        ranks.append({
            "rank": rank,
            "compute_submissions": int(final["compute_submissions"]),
            "compute_boundary_submissions": int(final["compute_boundary_submissions"]),
            "network_submissions": int(final["network_submissions"]),
            "compute_wraps": int(final["compute_wraps"]),
            "network_wraps": int(final["network_wraps"]),
            "compute_nop_bytes": int(final["compute_nop_bytes"]),
            "network_nop_bytes": int(final["network_nop_bytes"]),
            "external_wait_chunk1_writes": int(final.get("external_wait_chunk1_writes", 0)),
            "monitor_chunk1_writes": int(final.get("monitor_chunk1_writes", 0)),
            "monitor_high_chunk_writes": int(final.get("monitor_high_chunk_writes", 0)),
            "final_record_fnv1a64": step_lines[-1]["record_fnv1a64"],
            "final_chain_fnv1a64": step_lines[-1]["chain_fnv1a64"],
        })

    result = {
        "verdict": "PASS", "scope": "autonomous add-AllGather-add correctness only",
        "ranks": 8, "direct_steps": args.steps, "native_capture_steps": 130,
        "output_bytes_per_rank_step": 32768, "guard_bytes_per_rank_step": 512,
        "rank_results": ranks,
    }
    lines = [
        f"verdict=PASS scope=autonomous_correctness ranks=8 direct_steps={args.steps}",
        f"compute_submissions={sum(r['compute_submissions'] for r in ranks)} "
        f"network_submissions={sum(r['network_submissions'] for r in ranks)}",
        f"rank0_compute_wraps={ranks[0]['compute_wraps']} "
        f"rank0_network_wraps={ranks[0]['network_wraps']} "
        f"rank0_final_chain={ranks[0]['final_chain_fnv1a64']}",
        f"rank0_external_wait_chunk1_writes={ranks[0]['external_wait_chunk1_writes']} "
        f"rank0_monitor_chunk1_writes={ranks[0]['monitor_chunk1_writes']} "
        f"rank0_monitor_high_chunk_writes={ranks[0]['monitor_high_chunk_writes']}",
    ]
    if args.json_path:
        args.json_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.text_path:
        args.text_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
