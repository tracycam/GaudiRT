#!/usr/bin/env bash
set -euo pipefail
cd /home/ai/gaudi-experiments/gaudirt-review-20261023
g++ -O2 -std=c++17 -I/usr/include/habanalabs sram_mme_probe.cpp -L/usr/lib/habanalabs -Wl,-rpath,/usr/lib/habanalabs -lSynapse -o sram_mme_probe
exec 9>/home/ai/gaudi-llm-experiments/.codex-kernel-locks/module-0.lock
flock -n 9
unset LD_PRELOAD
export LD_LIBRARY_PATH=/usr/lib/habanalabs
export GC_KERNEL_PATH=/usr/lib/habanalabs/libtpc_kernels.so
export LOG_LEVEL_ALL=6
export HABANA_LOGS="$PWD/logs-sram-mme"
export ENABLE_EXPERIMENTAL_FLAGS=true
export DUMP_POST_GRAPHS="$PWD/sram-mme-post.json"
timeout 60 taskset -c 2 ./sram_mme_probe > sram-mme.log 2>&1
