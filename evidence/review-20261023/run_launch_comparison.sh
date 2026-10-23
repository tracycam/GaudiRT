#!/usr/bin/env bash
set -euo pipefail
cd /home/ai/gaudi-experiments/gaudirt-review-20261023
g++ -O2 -I/usr/include/habanalabs -I/usr/include/drm hardware_query.cpp -L/usr/lib/habanalabs -Wl,-rpath,/usr/lib/habanalabs -lhl-thunk -o hardware_query
g++ -O2 -std=c++17 -I/usr/include/habanalabs launch_probe.cpp -L/usr/lib/habanalabs -Wl,-rpath,/usr/lib/habanalabs -lSynapse -ldl -o launch_probe
exec 9>/home/ai/gaudi-llm-experiments/.codex-kernel-locks/module-0.lock
flock -n 9
export LD_LIBRARY_PATH=/usr/lib/habanalabs
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
export LOG_LEVEL_ALL=6
unset LD_PRELOAD BYPASS_PREPARE_LAUNCH_UTILS BYPASS_ANALYZE_TENSORS BYPASS_VALIDATE_SECTIONS
{
    date -u
    uname -a
    dpkg-query -W 'habanalabs*' 2>/dev/null || true
    sha256sum launch_probe.cpp launch_probe /usr/lib/habanalabs/libSynapse.so /usr/lib/habanalabs/libscal.so /usr/lib/habanalabs/libhcl.so /usr/lib/habanalabs/libtpc_kernels.so
    sha256sum /opt/habanalabs/engines_fw/sched_cmpt_media.bin
    sha256sum /home/ai/gaudi-experiments/first-principles-bridge-20261022/synapse_launch_breakdown.cpp /home/ai/gaudi-experiments/first-principles-bridge-20261022/synapse_launch_breakdown.so
    ldd launch_probe
} > environment.txt
./hardware_query > hardware.txt
for arm in baseline-a1 instrumented-b1 instrumented-b2 baseline-a2; do
    export HABANA_LOGS="$PWD/logs-$arm"
    if [[ $arm == instrumented* ]]; then
        export LD_PRELOAD=/home/ai/gaudi-experiments/first-principles-bridge-20261022/synapse_launch_breakdown.so
    else
        unset LD_PRELOAD
    fi
    timeout 60 taskset -c 2 ./launch_probe > "$arm.log" 2>&1
done
unset LD_PRELOAD
export HABANA_LOGS="$PWD/logs-rmw"
timeout 60 taskset -c 2 ./launch_probe rmw > rmw.log 2>&1
date -u > completed.txt
