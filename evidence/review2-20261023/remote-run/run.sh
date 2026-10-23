#!/usr/bin/env bash
# Runs on ai@192.168.100.153. Module 7, exclusive flock, system SDK only, no LD_PRELOAD.
set -euo pipefail
D=/home/ai/gaudi-experiments/gaudirt-review2-20261023
cd "$D"
g++ -O2 -std=c++17 -I/usr/include/habanalabs prequeue_capacity_probe.cpp -L/usr/lib/habanalabs \
    -Wl,-rpath,/usr/lib/habanalabs -lSynapse -lpthread -o prequeue_capacity_probe
exec 9>/home/ai/gaudi-llm-experiments/.codex-kernel-locks/module-7.lock
flock -n 9
unset LD_PRELOAD BYPASS_PREPARE_LAUNCH_UTILS BYPASS_ANALYZE_TENSORS BYPASS_VALIDATE_SECTIONS
export LD_LIBRARY_PATH=/usr/lib/habanalabs
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
export HABANA_LOGS="$D/logs"
{
  date -u; sha256sum prequeue_capacity_probe.cpp prequeue_capacity_probe /usr/lib/habanalabs/libSynapse.so \
    /usr/lib/habanalabs/libscal.so; hl-smi -Q index,memory.used,utilization.aip -f csv
} > env.txt
timeout 600 taskset -c 100 ./prequeue_capacity_probe 7 > plain.log 2>&1 || echo "plain exit $?" >> plain.log
if [[ "${1:-}" == gdb ]]; then
  timeout 600 taskset -c 100 gdb -q -batch \
    -ex "set pagination off" -ex "handle SIGUSR1 stop print nopass" -ex "handle SIGPIPE nostop noprint pass" \
    -ex run -ex "thread apply all bt 40" -ex "kill" \
    --args ./prequeue_capacity_probe 7 trap > gdb.log 2>&1 || echo "gdb exit $?" >> gdb.log
fi
date -u > completed.txt
