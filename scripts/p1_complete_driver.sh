#!/usr/bin/env bash
set -euo pipefail

ROOT="${GAUDIRT_ROOT:-$HOME/GaudiRT}"
OUT="${P1_COMPLETE_OUT:-$ROOT/evidence/p1-complete-20261024/repro-$(date -u +%Y%m%dT%H%M%SZ)}"
BIN="$ROOT/build/p1_autonomous_recipe_replay_probe"

mkdir -p "$OUT"
cmake -S "$ROOT" -B "$ROOT/build"
cmake --build "$ROOT/build" --target p1_autonomous_recipe_replay_probe -j8
hl-smi > "$OUT/hl-smi.txt"
timeout 180s env \
    GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
    HABANA_LOGS="$OUT/habana-logs" \
    P0_LOG_FILE="$OUT/hook_trace.log" \
    "$BIN" > "$OUT/run_stdout.log" 2> "$OUT/run_stderr.log"

{
    date -Ins
    hostname
    uname -a
    sha256sum \
        /usr/lib/habanalabs/libSynapse.so \
        /usr/lib/habanalabs/libscal.so \
        /usr/lib/habanalabs/libhcl.so \
        "$BIN" \
        "$ROOT/probes/p0_scal_hook.cpp" \
        "$ROOT/probes/p1_autonomous_recipe_replay_probe.cpp" \
        "$ROOT/include/gaudirt/scal/ccb_writer.hpp"
} > "$OUT/environment.txt"
