#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
from pathlib import Path

NETWORK_STREAMS = {
    "network_garbage_collector_and_reduction2",
    "scaleup_send1",
    "scaleup_send2",
    "scaleup_receive1",
    "scaleup_receive2",
    "scaleout_send1",
    "scaleout_send2",
    "scaleout_receive1",
    "scaleout_receive2",
    "network_garbage_collector_and_reduction5",
}


def phases(trace):
    markers = list(re.finditer(r"^\[P0_PHASE\] >>> (\S+) <<<$", trace, re.M))
    result = {}
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(trace)
        result.setdefault(marker.group(1), []).append(trace[marker.end():end])
    return result


def one_phase(phase_map, name):
    blocks = phase_map.get(name, [])
    if len(blocks) != 1:
        raise ValueError(f"expected one {name} phase, found {len(blocks)}")
    return blocks[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("stderr", type=Path)
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--text", dest="text_path", type=Path)
    args = parser.parse_args()

    trace = args.trace.read_text()
    stderr = args.stderr.read_text()
    phase_map = phases(trace)
    verdicts = re.findall(
        r"\[P3A_SYNC_VERDICT\] rank=(\d+) native_steps=3 "
        r"closed_loop=PASS output_bytes=32768 guard_bytes=512", stderr)
    if sorted(map(int, verdicts)) != list(range(8)):
        raise ValueError(f"invalid rank verdicts: {verdicts}")

    iterations = []
    for index, label in enumerate(("COLD", "WARM", "MUTATED"), 1):
        pre = one_phase(phase_map, f"P3A_{label}_PRE")
        collective = one_phase(phase_map, f"P3A_{label}_ALLGATHER")
        post = one_phase(phase_map, f"P3A_{label}_POST")

        submissions = re.findall(r"^\[P0_SUBMIT\] phase=\S+ stream=(\S+)", collective, re.M)
        if len(submissions) != 10 or set(submissions) != NETWORK_STREAMS:
            raise ValueError(f"{label}: invalid network submissions: {submissions}")
        if any("stream=compute0" in line for line in collective.splitlines()
               if line.startswith("[P0_SUBMIT]")):
            raise ValueError(f"{label}: unexpected compute0 submission during AllGather")

        compute_targets = list(map(int, re.findall(
            r"\[P0_CG_SET_EXPECTED\][^\n]*cg=compute_completion_queue0 val=(\d+)", pre)))
        if not compute_targets:
            raise ValueError(f"{label}: missing pre-compute target")
        compute_target = compute_targets[-1]
        if compute_target != 2 * index:
            raise ValueError(f"{label}: pre-compute target {compute_target} != {2 * index}")

        monitor_targets = list(map(int, re.findall(
            r"DCORE3_MON_ARM\[\d+\]: watch SOB_group\(1472\.\.1479, "
            r"mask=0x0\) >= (\d+)", collective)))
        if len(monitor_targets) != 5 or set(monitor_targets) != {compute_target}:
            raise ValueError(f"{label}: invalid monitor targets: {monitor_targets}")

        external_targets = list(map(int, re.findall(
            r"cg=network_completion_queue_external_00 val=(\d+)", collective)))
        internal_targets = list(map(int, re.findall(
            r"cg=network_completion_queue_internal_00 val=(\d+)", collective)))
        if external_targets != [index] or internal_targets != [index]:
            raise ValueError(
                f"{label}: network targets {external_targets}/{internal_targets} != {index}")

        waits = re.findall(
            r"WAIT_FOR_EXT_SIGNAL: sm_id=3 sob_id=1440 target0=(\d+)", post)
        if waits != [str(index)]:
            raise ValueError(f"{label}: invalid post-compute waits: {waits}")
        post_targets = list(map(int, re.findall(
            r"\[P0_CG_SET_EXPECTED\][^\n]*cg=compute_completion_queue0 val=(\d+)", post)))
        if not post_targets or post_targets[-1] != 2 * index + 1:
            raise ValueError(f"{label}: invalid post-compute targets: {post_targets}")

        iterations.append({
            "label": label,
            "network_stream_submissions": len(submissions),
            "pre_compute_long_sob": "DCORE3_SOB[1472]",
            "pre_compute_target": compute_target,
            "network_monitor_targets": monitor_targets,
            "network_external_long_sob": "DCORE3_SOB[1440]",
            "network_external_target": index,
            "network_internal_target": index,
            "post_compute_wait_targets": list(map(int, waits)),
            "post_compute_target": post_targets[-1],
        })

    result = {
        "verdict": "PASS",
        "scope": "official native add-AllGather-add synchronization capture; not autonomous replay",
        "ranks": 8,
        "native_steps_per_rank": 3,
        "output_bytes_per_rank_step": 32768,
        "guard_bytes_per_rank_step": 512,
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "stderr_sha256": hashlib.sha256(args.stderr.read_bytes()).hexdigest(),
        "iterations": iterations,
    }
    text_lines = [
        "verdict=PASS (native synchronization capture only)",
        "ranks=8 native_steps_per_rank=3 output_bytes=32768 guard_bytes=512",
    ]
    for item in iterations:
        text_lines.append(
            f"{item['label']}: network_streams=10 "
            f"compute_DCORE3_SOB1472_target={item['pre_compute_target']} "
            f"network_monitors={item['network_monitor_targets']} "
            f"post_wait_DCORE3_SOB1440={item['post_compute_wait_targets']}")

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_path:
        args.json_path.write_text(encoded)
    if args.text_path:
        args.text_path.write_text("\n".join(text_lines) + "\n")
    print("\n".join(text_lines))


if __name__ == "__main__":
    main()
