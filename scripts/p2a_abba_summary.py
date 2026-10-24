#!/usr/bin/env python3
import json
import re
import statistics
import sys
from pathlib import Path

ARMS = {
    "A1": "native",
    "B1": "direct",
    "B2": "direct",
    "A2": "native",
}
METRICS = ("submit_us", "wall_us", "completion_to_completion_us")


def stats(values):
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": min(values),
        "p50": statistics.median(values),
        "p90": ordered[(len(ordered) * 90) // 100],
        "max": max(values),
        "mean": statistics.mean(values),
    }


def reject_timed_diagnostics(path):
    markers = ("[P2A_WAIT]", "[P2A_WAIT_PROGRESS]", "[P2A_SYNC_STATE]",
               "[P2A_DYNAMIC]")
    for line in path.read_text().splitlines():
        if not any(marker in line for marker in markers):
            continue
        match = re.search(r"\bindex=(\d+)\b", line)
        if match and 130 <= int(match.group(1)) < 260:
            raise ValueError(f"timed diagnostic found in {path}: {line}")


def read_arm(root, label, expected_mode):
    per_rank = {}
    for rank in range(8):
        reject_timed_diagnostics(root / label / f"rank{rank}.log")
        path = root / label / f"rank{rank}_perf.json"
        with path.open() as handle:
            data = json.load(handle)
        if data["mode"] != expected_mode or len(data["samples"]) != 130:
            raise ValueError(f"invalid arm {label} rank {rank}")
        samples = data["samples"]
        for index, sample in enumerate(samples):
            if sample["index"] != 130 + index:
                raise ValueError(f"unexpected sample index in {path}")
            if sample["submit_us"] <= 0 or sample["wall_us"] <= 0:
                raise ValueError(f"invalid host timing in {path}")
            if sample["completion_ns"] <= 0:
                raise ValueError(f"invalid completion timestamp in {path}")
            if index and sample["completion_ns"] < samples[index - 1]["completion_ns"]:
                raise ValueError(f"non-monotonic completion timestamp in {path}")
        per_rank[rank] = samples
    return per_rank


def flatten(per_rank, metric):
    values = []
    for samples in per_rank.values():
        if metric == "completion_to_completion_us":
            values.extend(
                (current["completion_ns"] - previous["completion_ns"]) / 1000.0
                for previous, current in zip(samples, samples[1:])
            )
        else:
            values.extend(sample[metric] for sample in samples)
    return values


def arm_metrics(per_rank):
    return {metric: stats(flatten(per_rank, metric)) for metric in METRICS}


def rank_metrics(samples, metric):
    if metric == "completion_to_completion_us":
        values = [
            (current["completion_ns"] - previous["completion_ns"]) / 1000.0
            for previous, current in zip(samples, samples[1:])
        ]
    else:
        values = [sample[metric] for sample in samples]
    return stats(values)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: p2a_abba_summary.py ABBA_DIR")
    root = Path(sys.argv[1])
    per_arm = {label: read_arm(root, label, mode) for label, mode in ARMS.items()}
    metrics = {label: arm_metrics(per_arm[label]) for label in ARMS}

    metrics["A"] = {
        metric: stats(
            flatten(per_arm["A1"], metric) + flatten(per_arm["A2"], metric)
        )
        for metric in METRICS
    }
    metrics["B"] = {
        metric: stats(
            flatten(per_arm["B1"], metric) + flatten(per_arm["B2"], metric)
        )
        for metric in METRICS
    }

    comparisons = {}
    for left, right in (("A1", "B1"), ("B2", "A2"), ("A", "B")):
        comparisons[f"{left}_vs_{right}"] = {}
        for metric in METRICS:
            a = metrics[left][metric]["p50"]
            b = metrics[right][metric]["p50"]
            comparisons[f"{left}_vs_{right}"][metric] = {
                "baseline_p50_us": a,
                "candidate_p50_us": b,
                "delta_us": b - a,
                "delta_percent": (b / a - 1.0) * 100.0,
            }

    per_rank_summary = {}
    for label in ARMS:
        per_rank_summary[label] = {}
        for rank in range(8):
            per_rank_summary[label][rank] = {
                metric: rank_metrics(per_arm[label][rank], metric)
                for metric in METRICS
            }

    result = {
        "protocol": {
            "schedule": ["A1=native", "B1=direct", "B2=direct", "A2=native"],
            "process_isolation": True,
            "ld_preload": False,
            "ranks": 8,
            "samples_per_rank_per_arm": 130,
            "count_per_rank": 1024,
            "dtype": "FP32",
            "wait_policy": "nonblocking poll without sleep or success-path logging",
            "timed_loop_diagnostics": "none; template, address, SM/CI and decode checks are excluded",
            "wall_semantics": "host submit and busy-poll wait to common network CG; not isolated device time",
            "span_semantics": "loop cadence including inter-call H2D/D2H and full output verification; not pure device latency",
        },
        "metrics": metrics,
        "comparisons": comparisons,
        "per_rank": per_rank_summary,
    }
    with (root / "abba_summary.json").open("w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    for metric in METRICS:
        row = comparisons[f"A_vs_B"][metric]
        print(
            f"{metric}: native={row['baseline_p50_us']:.3f} us "
            f"direct={row['candidate_p50_us']:.3f} us "
            f"delta={row['delta_us']:+.3f} us ({row['delta_percent']:+.2f}%)"
        )
    print(f"evidence={root / 'abba_summary.json'}")


if __name__ == "__main__":
    main()
