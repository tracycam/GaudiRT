# Public source provenance

- **`v0.1.0-phase4.2` (`9bc993e`)**: Initial public snapshot exported from
  verified Phase 4.2 development commit `8fc3b3b` (`Use compensated QKV rounding
  for model attention`, 2026-10-31), paired with companion kernel repository
  [`tracycam/gaudi-kernels`](https://github.com/tracycam/gaudi-kernels) snapshot
  commit `d23e585` (`v0.2.0-phase4.2`).
- **`v0.1.1-phase4.3`**: Incremental sync of Phase 4.3 zero-legacy-bridge `vLLM`
  serving integration commits (`19164ee..c568284`: scheduled prefill chunks &
  validated KV page updates, CPU-only streamed `safetensors` weight loader,
  serving runner adapter, KV page reset across requests, and worker host-policy
  affinity inheritance) together with updated [`ROADMAP.md`](../ROADMAP.md) and
  [`docs/gaudi2-pitfalls.md`](gaudi2-pitfalls.md) (`P-RT-17`..`P-RT-20`,
  `P-BENCH-06`).

The public repository starts with fresh Git history under [Apache-2.0](../LICENSE).
Runtime, graph-builder, `PrivateUse1` backend, and device/unit test sources match
the verified development tree without arithmetic or behavioral changes.

## Omitted local artifacts

In accordance with repository hygiene rules (`AGENTS.md`), this release omits:

- Prior local development Git history containing private lab host addresses;
- Out-of-tree raw JSON receipts, profiler traces, and run logs (`instance-assets/`);
- Compiled shared libraries, cached Synapse `.recipe` binaries, and build trees;
- Model checkpoints and private evaluation prompts.

SHA256 digests referenced in [`ROADMAP.md`](../ROADMAP.md) and
[`docs/gaudi2-pitfalls.md`](gaudi2-pitfalls.md) identify offline verification
receipts archived outside the source repository.
