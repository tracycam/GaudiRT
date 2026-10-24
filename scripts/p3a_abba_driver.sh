#!/usr/bin/env bash
set -euo pipefail

ROOT="${GAUDIRT_ROOT:-$HOME/GaudiRT}"
OUT="${P3A_ABBA_OUT:-$ROOT/evidence/p3a-20261024/abba-$(date -u +%Y%m%dT%H%M%SZ)}"
DRIVER="$ROOT/scripts/p3a_compute_allgather_compute_driver.sh"
DATA="$OUT/reference"
BIN="$ROOT/build/p3a_nohook_compute_allgather_compute_probe"
SUMMARY="$ROOT/scripts/p2a_abba_summary.py"

mkdir -p "$OUT" "$DATA"
bash "$DRIVER" reference 130 "$DATA" "$OUT/reference"
bash "$DRIVER" artifact 1 "$DATA" "$OUT/artifact"

run_arm() {
    local label="$1"
    local mode="$2"
    echo "[P3A_ABBA] arm=$label mode=$mode"
    P3A_NO_HOOK=1 bash "$DRIVER" "$mode" 130 "$DATA" "$OUT/$label"
}

run_arm A1 native
run_arm B1 perf
run_arm B2 perf
run_arm A2 native

{
    date -Ins
    hostname
    uname -a
    printf 'binary=%s\n' "$BIN"
    printf 'LD_PRELOAD=%s\n' "${LD_PRELOAD-}"
    ldd "$BIN"
    sha256sum \
        /usr/lib/habanalabs/libSynapse.so \
        /usr/lib/habanalabs/libscal.so \
        /usr/lib/habanalabs/libhcl.so \
        "$BIN" \
        "$ROOT/probes/p3a_compute_allgather_compute_probe.cpp" \
        "$ROOT/probes/autonomous_allgather_runtime.hpp" \
        "$ROOT/scripts/p3a_compute_allgather_compute_driver.sh" \
        "$ROOT/scripts/p2a_abba_summary.py"
} > "$OUT/environment.txt"

python3 "$SUMMARY" "$OUT" p3a | tee "$OUT/summary.txt"
