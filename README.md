# GaudiRT

Zero-legacy-bridge C++20 runtime, Synapse C API graph stack, and PyTorch
`PrivateUse1` backend for LLM inference on Intel Gaudi2 (`hl-225`), licensed
under [Apache-2.0](LICENSE).

Companion custom TPC C kernel repository:
**[`tracycam/gaudi-kernels`](https://github.com/tracycam/gaudi-kernels)** (paired snapshot: `d23e585`).

---

## Why GaudiRT Exists

Gaudi2 hardware provides 24 TPC vector cores, 2 MME FP8/BF16 matrix engines,
48 MB SRAM, and ~2.45 TB/s HBM bandwidth per card, but standard PyTorch serving
paths suffer from host dispatch overhead, dynamic recipe recompilation, and
implicit bridge abstractions (`gaudi-pytorch-bridge` / `habana_frameworks` /
`PT_HPU_LAZY_MODE`).

`GaudiRT` cuts out the legacy PyTorch bridge completely on the candidate
execution path:

- **Direct C ABI only**: links solely against `libSynapse.so` + `libhcl.so`
  (plus standard upstream PyTorch C++ extension APIs with
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0`).
- **Runtime Fail-Fast guard**: [`models/mimo/runner.py`](models/mimo/runner.py)
  inspects `sys.modules`, environment variables, and `/proc/self/maps` at
  startup and aborts immediately if `habana_frameworks` or
  `gaudi-pytorch-bridge` is mapped in the candidate process.
- **Static HBM Arena & Persistent Recipe Registry**: deterministic workspace/KV
  allocation (`HBMArena`), SHA256-fingerprinted disk recipe caching
  (`synRecipeSerialize` / `synRecipeDeSerialize`), pre-resolved tensor launch
  tables (`PreparedLaunch`), and native HCL TP8 collective orchestration.

---

## Verified Measured Status & Explicit Scope Boundaries

All measurements below were collected on an 8× Gaudi2 (`hl-225`, Synapse/HCL
`1.24.1-482`) node running **70-layer `MiMo-V2-Flash` (309B MoE, hybrid SWA +
Full Attention, Block-FP8 + BF16) on TP8**:

| Workload / Milestone | Baseline (`vllm-gaudi` Production Path) | `GaudiRT` Zero-Bridge Candidate (`c568284` + `gaudi-kernels` `d23e585`) | Numerical / Determinism Gate |
| :--- | :--- | :--- | :--- |
| **Standalone 70L TP8 B1 Decode (`ctx=4096`, 4-arm ABBA)** | `11.4145 ms` / `11.2448 ms` median step | **`10.5119 ms` / `10.6084 ms`** (**`-7.91%` / `-5.66%`**) | 128-step greedy tokens **100% match**; final step `max_kl = 0.002016` |
| **Empty-KV 4096-token Prefill (`M=32` chunks) + 128-step Continuous Decode** | Offline CPU `FP64` full-prompt reference (`CPU FP32 vs FP64 max_kl = 0.043453`) | Cold/warm cache **bit-identical** (`1.805 s` warm load across 10 recipes); full-prompt candidate vs `FP64` **`max_kl = 0.034032`** (`21.7%` closer to `FP64` than pure CPU `FP32`) | `top1_match = 1.0`, Top-4 logit order exact match; 0 NaN/Inf |
| **Full `vLLM` `engine.step()` Serving ABBA (`4096` Prefill + `128` Decode, 4-arm)** | **TTFT p50**: `2570.86` / `2567.78 ms` (`2569.3 ms` mean)<br>**TPOT p50**: `11.522` / `11.384 ms` (`11.453 ms` mean) | **TPOT p50**: **`11.016` / `10.952 ms`** (**`10.984 ms`**, **`-4.10%` faster**)<br>**TTFT p50**: `8436.30` / `8435.89 ms` (`8436.1 ms`, **`3.28×` slower** — `M=32` ungrouped TPC MoE Prefill bottleneck; Milestone 4.4 active) | Cold/warm 20 recipes, logits, 129 tokens & 70L KV **bit-identical** to standalone runner; Candidate vs `FP64` `max_kl = 0.034032` vs Prod `0.056061` |

### Current Scope & What Is *Not* Claimed Yet

1. **Model-coupled graph builders**: Current layer/attention/MoE Synapse C++
   graph builders live under [`models/mimo/`](models/mimo/) and are specialized
   for `MiMo-V2-Flash` shapes and weight layouts; a model-agnostic graph IR has
   intentionally not been extracted prior to multi-model validation.
2. **Single-stream `vLLM` serving validated, TTFT optimization in progress**:
   Real `vLLM` scheduler, paged KV allocation, 8-worker IPC, streamed CPU->HBM
   weight loading, and `OutputProcessor` execution are validated end-to-end with
   zero legacy bridge and `TPOT` `-4.10%` faster than production, **but** `M=32`
   Prefill currently causes `3.28×` higher `TTFT` (`8436.1 ms` vs `2569.3 ms`).
   Large static prefill buckets (`M=256/512`) + grouped-expert MME MoE on shared
   resident weights are tracked as Milestone 4.4 in [`ROADMAP.md`](ROADMAP.md);
   variable tail chunks, multi-request concurrency, and general sampling remain
   subsequent milestones.
3. **Hardware arithmetic mitigations**: Gaudi2 TPC `v_exp_f32` exhibits severe
   step discontinuities at 750 exact FP32 threshold points (`~11.8%` relative
   error), and Synapse silently caps TPC library providers per process (`<=15`
   safe limit). Every discovered hardware, compiler, numerical, and lifecycle
   pitfall is documented with reproducers and code mitigations in
   [`docs/gaudi2-pitfalls.md`](docs/gaudi2-pitfalls.md).

---

## Repository Layout

```text
include/gaudirt/
  common/       RAII handles, SHA256, host policy, ABBA timing harness
  runtime/      DeviceRuntime, HBMArena, RecipeRegistry, PreparedLaunch, TPReduce
  scal/         SCAL & Gaudi2 QMAN ARC packet inspection helpers
src/            Core C++20 runtime library (libgaudirt.a)
models/mimo/    70-layer TP8 Synapse C++ graph builders, PrivateUse1 extension, Python runner
tests/          CPU unit tests & single-card / TP8 Gaudi2 device verification tests
docs/           Design notes, Gaudi2 pitfalls ledger, hardware/software autopsy, MXFP4 verdict
```

---

## Prerequisites & Build

### Prerequisites (see [`MANIFEST.md`](MANIFEST.md))

- Linux x86_64 with Intel Gaudi2 driver, firmware, Synapse (`libSynapse.so`),
  and HCL (`libhcl.so`) `1.24.x` installed under `/usr/include/habanalabs` and
  `/usr/lib/habanalabs`.
- CMake `>= 3.20`, GCC/Clang supporting C++20.
- Companion TPC kernel bundle built from
  [`tracycam/gaudi-kernels`](https://github.com/tracycam/gaudi-kernels) and
  supplied via `GC_KERNEL_PATH` (`<bundled_provider.so>:/usr/lib/habanalabs/libtpc_kernels.so`).

### CMake Build & Unit Tests

```bash
cmake -S . -B build   -DHABANA_INC_DIR=/usr/include/habanalabs   -DHABANA_LIB_DIR=/usr/lib/habanalabs
cmake --build build -j"$(nproc)"

# Pure runtime metadata / unit checks:
./build/prefill_metadata_test
./build/mimo_step_metadata_test
./build/host_policy_inherit_test
```

---

## Key Documentation

- [`ROADMAP.md`](ROADMAP.md) — 4-layer target architecture, milestone gates, and current phase status.
- [`docs/gaudi2-pitfalls.md`](docs/gaudi2-pitfalls.md) — Master ledger of Gaudi2 hardware ISA bugs (`v_exp_f32`), Synapse compiler limits, numerical traps, and runtime lifecycle rules.
- [`docs/mimo-numeric-correction.md`](docs/mimo-numeric-correction.md) — Root-cause analysis of 70-layer MoE trajectory drift and compensated QKV midpoint rounding.
- [`docs/gaudi2-hardware-vs-software-autopsy.md`](docs/gaudi2-hardware-vs-software-autopsy.md) — Hardware capability vs software stack bottleneck autopsy on Gaudi2.
- [`docs/mxfp4-verdict.md`](docs/mxfp4-verdict.md) — Quantitative roof-line and instruction-cost verdict on software-emulated MXFP4 on Gaudi2.
- [`MANIFEST.md`](MANIFEST.md) — Validated driver, firmware, Synapse, HCL, and PyTorch version contract.
- [`docs/PUBLICATION.md`](docs/PUBLICATION.md) — Open-source snapshot provenance and sanitization notes.
