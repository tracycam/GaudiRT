#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 reference|direct STEPS DATA_DIR OUT_DIR" >&2
    exit 2
fi

MODE="$1"
STEPS="$2"
DATA_DIR="$3"
OUT="$4"
ROOT="${GAUDIRT_ROOT:-$HOME/GaudiRT}"
BIN="$ROOT/build/p3a_compute_allgather_compute_probe"

if [[ "$MODE" != reference && "$MODE" != direct ]]; then
    echo "mode must be reference or direct" >&2
    exit 2
fi
mkdir -p "$DATA_DIR" "$OUT"
cmake -S "$ROOT" -B "$ROOT/build"
cmake --build "$ROOT/build" --target p3a_compute_allgather_compute_probe -j8
hl-smi > "$OUT/hl-smi.txt"

set +e
timeout -k 10s 900s env \
    GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so \
    HABANA_LOGS="$OUT/habana-logs" \
    "$BIN" --mode "$MODE" --steps "$STEPS" \
    --data-dir "$DATA_DIR" --log-dir "$OUT" \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
rc=$?
set -e

{
    date -Ins
    hostname
    uname -a
    printf 'probe_exit_code=%d\n' "$rc"
    printf 'mode=%s steps=%s correctness_only=1\n' "$MODE" "$STEPS"
    sha256sum \
        /usr/lib/habanalabs/libSynapse.so \
        /usr/lib/habanalabs/libscal.so \
        /usr/lib/habanalabs/libhcl.so \
        "$BIN" \
        "$ROOT/probes/p3a_compute_allgather_compute_probe.cpp" \
        "$ROOT/probes/autonomous_allgather_runtime.hpp" \
        "$ROOT/probes/p0_scal_hook.cpp" \
        "$ROOT/include/gaudirt/scal/ccb_writer.hpp"
} > "$OUT/environment.txt"
exit "$rc"
