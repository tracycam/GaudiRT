#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 reference|direct|artifact|native|perf STEPS DATA_DIR OUT_DIR" >&2
    exit 2
fi

MODE="$1"
STEPS="$2"
DATA_DIR="$3"
OUT="$4"
ROOT="${GAUDIRT_ROOT:-$HOME/GaudiRT}"
NO_HOOK="${P3A_NO_HOOK:-0}"
BUILD_DIR="${P3A_BUILD_DIR:-$ROOT/build}"
TIMEOUT_S="${P3A_TIMEOUT_S:-900}"

if [[ "$MODE" != reference && "$MODE" != direct && "$MODE" != artifact &&
      "$MODE" != native && "$MODE" != perf ]]; then
    echo "mode must be reference, direct, artifact, native, or perf" >&2
    exit 2
fi
if [[ "$NO_HOOK" == 1 && "$MODE" == artifact ]]; then
    echo "artifact generation requires the hooked build" >&2
    exit 2
fi
if [[ "$NO_HOOK" != 1 && ( "$MODE" == native || "$MODE" == perf ) ]]; then
    echo "native/perf requires P3A_NO_HOOK=1" >&2
    exit 2
fi
if [[ "$NO_HOOK" == 1 ]]; then
    BIN="$BUILD_DIR/p3a_nohook_compute_allgather_compute_probe"
    TARGET="p3a_nohook_compute_allgather_compute_probe"
else
    BIN="$BUILD_DIR/p3a_compute_allgather_compute_probe"
    TARGET="p3a_compute_allgather_compute_probe"
fi
mkdir -p "$DATA_DIR" "$OUT"
cmake -S "$ROOT" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --target "$TARGET" -j8
hl-smi > "$OUT/hl-smi.txt"

set +e
timeout -k 10s "${TIMEOUT_S}s" env \
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
    printf 'mode=%s steps=%s no_hook=%s\n' "$MODE" "$STEPS" "$NO_HOOK"
    printf 'binary=%s\nbuild_dir=%s\ntimeout_s=%s\n' "$BIN" "$BUILD_DIR" "$TIMEOUT_S"
    printf 'LD_PRELOAD=%s\n' "${LD_PRELOAD-}"
    printf 'P3A_REGION_TIMINGS=%s\n' "${P3A_REGION_TIMINGS-}"
    ldd "$BIN"
    sha256sum \
        /usr/lib/habanalabs/libSynapse.so \
        /usr/lib/habanalabs/libscal.so \
        /usr/lib/habanalabs/libhcl.so \
        "$BIN" \
        "$ROOT/probes/p3a_compute_allgather_compute_probe.cpp" \
        "$ROOT/probes/autonomous_allgather_runtime.hpp" \
        "$ROOT/probes/p0_scal_hook.cpp" \
        "$ROOT/include/gaudirt/scal/ccb_writer.hpp" \
        "$ROOT/include/gaudirt/scal/long_so.hpp"
} > "$OUT/environment.txt"
exit "$rc"
