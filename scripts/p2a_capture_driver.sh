#!/usr/bin/env bash
set -euo pipefail

ROOT="${GAUDIRT_ROOT:-$HOME/GaudiRT}"
OUT="${P2A_CAPTURE_OUT:-$ROOT/evidence/p2a-20261024/repro-$(date -u +%Y%m%dT%H%M%SZ)}"
BIN="$ROOT/build/p2a_autonomous_allgather_replay_probe"

mkdir -p "$OUT"
cmake -S "$ROOT" -B "$ROOT/build"
cmake --build "$ROOT/build" --target p2a_autonomous_allgather_replay_probe -j8
hl-smi > "$OUT/hl-smi.txt"
timeout 300s env \
    GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
    HABANA_LOGS="$OUT/habana-logs" \
    "$BIN" --mode capture --log-dir "$OUT/capture" \
    > "$OUT/capture_stdout.log" 2> "$OUT/capture_stderr.log"

{
    date -Ins
    hostname
    uname -a
    sha256sum \
        /usr/lib/habanalabs/libSynapse.so \
        /usr/lib/habanalabs/libscal.so \
        /usr/lib/habanalabs/libhcl.so \
        "$BIN" \
        "$ROOT/probes/p2a_autonomous_allgather_replay_probe.cpp" \
        "$ROOT/probes/autonomous_allgather_runtime.hpp" \
        "$ROOT/include/gaudirt/scal/ccb_writer.hpp"
} > "$OUT/environment.txt"
