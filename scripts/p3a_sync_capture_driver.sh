#!/usr/bin/env bash
set -euo pipefail

ROOT="${GAUDIRT_ROOT:-$HOME/GaudiRT}"
OUT="${P3A_SYNC_CAPTURE_OUT:-$ROOT/evidence/p3a-20261024/sync-capture/repro-$(date -u +%Y%m%dT%H%M%SZ)}"
BIN="$ROOT/build/p0_scal_capture_decode_probe"

mkdir -p "$OUT"
cmake -S "$ROOT" -B "$ROOT/build"
cmake --build "$ROOT/build" --target p0_scal_capture_decode_probe -j8
hl-smi > "$OUT/hl-smi.txt"

set +e
timeout -k 10s 240s env \
    GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
    HABANA_LOGS="$OUT/habana-logs" \
    P0_LOG_FILE="$OUT/rank7_scal_trace.log" \
    "$BIN" --mode p3a-sync \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
rc=$?
set -e

if [[ "$rc" -eq 0 ]]; then
    python3 "$ROOT/scripts/p3a_sync_capture_summary.py" \
        "$OUT/rank7_scal_trace.log" "$OUT/stderr.log" \
        --json "$OUT/summary.json" --text "$OUT/summary.txt" \
        > "$OUT/summary_stdout.log"
fi

{
    date -Ins
    hostname
    uname -a
    printf 'probe_exit_code=%d\n' "$rc"
    printf 'capture_mode=linked_scal_hook_correctness_only\n'
    sha256sum \
        /usr/lib/habanalabs/libSynapse.so \
        /usr/lib/habanalabs/libscal.so \
        /usr/lib/habanalabs/libhcl.so \
        "$BIN" \
        "$ROOT/probes/p0_scal_capture_decode_probe.cpp" \
        "$ROOT/probes/p0_scal_hook.cpp" \
        "$ROOT/include/gaudirt/scal/scal_packet_decoder.hpp"
} > "$OUT/environment.txt"
exit "$rc"
