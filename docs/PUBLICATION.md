# Public source provenance

This initial public snapshot of `GaudiRT` is exported from verified Phase 4.2
development commit `8fc3b3b` (`Use compensated QKV rounding for model attention`,
2026-10-31), paired with companion kernel repository
[`tracycam/gaudi-kernels`](https://github.com/tracycam/gaudi-kernels) snapshot
commit `d23e585` (`Add compensated QKV midpoint rounding for model TPC bundle`).

The public repository starts with fresh Git history under [Apache-2.0](../LICENSE).
Runtime, graph-builder, `PrivateUse1` backend, and device/unit test sources are
exported verbatim from `8fc3b3b` without arithmetic or behavioral changes.

## Omitted local artifacts

In accordance with repository hygiene rules (`AGENTS.md`), this release omits:

- Prior local development Git history containing private lab host addresses;
- Out-of-tree raw JSON receipts, profiler traces, and run logs (`instance-assets/`);
- Compiled shared libraries, cached Synapse `.recipe` binaries, and build trees;
- Model checkpoints and private evaluation prompts.

SHA256 digests referenced in [`ROADMAP.md`](../ROADMAP.md) and
[`docs/gaudi2-pitfalls.md`](gaudi2-pitfalls.md) identify offline verification
receipts archived outside the source repository.
