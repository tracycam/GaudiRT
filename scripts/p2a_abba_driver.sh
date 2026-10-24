#!/usr/bin/env bash
set -euo pipefail

ROOT="${GAUDIRT_ROOT:-$HOME/GaudiRT}"
OUT="${P2A_ABBA_OUT:-$ROOT/evidence/p2a-20261024/abba-$(date -u +%Y%m%dT%H%M%SZ)}"
BIN="$ROOT/build/p2a_autonomous_allgather_replay_probe"
SUMMARY="$ROOT/scripts/p2a_abba_summary.py"

mkdir -p "$OUT"
test -f "$SUMMARY"
cmake -S "$ROOT" -B "$ROOT/build"
cmake --build "$ROOT/build" --target p2a_autonomous_allgather_replay_probe -j8
test -x "$BIN"

run_arm() {
    local label="$1"
    local mode="$2"
    mkdir -p "$OUT/$label"
    hl-smi > "$OUT/$label-hl-smi.txt"
    echo "[P2A_ABBA] arm=$label mode=$mode"
    timeout 300s env \
        GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
        HABANA_LOGS="$OUT/${label}-habana" \
        "$BIN" --mode "$mode" --log-dir "$OUT/$label" \
        > "$OUT/$label.stdout" 2> "$OUT/$label.stderr"
}

run_arm A1 native
run_arm B1 direct
run_arm B2 direct
run_arm A2 native

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

python3 "$SUMMARY" "$OUT" | tee "$OUT/summary.txt"
