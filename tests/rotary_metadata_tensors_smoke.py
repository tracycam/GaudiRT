"""Check tensor-only RoPE metadata against the validated metadata interface."""

import os
import sys
import hashlib

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import rotary_runtime_smoke as reference


def main():
    torch.random.default_generator.manual_seed(20261105)
    gaudirt.initialize(hbm_bytes=512 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for rows in (1, 3, 137, 513):
        metadata, positions_cpu = reference.metadata(rows)
        cache_cpu = torch.randn(reference.CONTEXT, 128, dtype=torch.bfloat16)
        q_cpu = torch.randn(rows, 56 * 128, dtype=torch.bfloat16)
        k_cpu = torch.randn(rows, 8 * 128, dtype=torch.bfloat16)
        q, k, cache = (value.to("gaudirt:0") for value in (q_cpu, k_cpu, cache_cpu))
        meta_positions = metadata.rotary_positions()
        meta_header = metadata.rotary_header()
        assert meta_positions.shape == (rows,) and meta_positions.dtype == torch.int32
        assert meta_header.shape == (1,) and metadata.context() == reference.CONTEXT
        old = torch.ops.gaudirt.rotary_embedding.default(q, k, cache, metadata, 128, 128, True)
        new = torch.ops.gaudirt._rotary_embedding_validated.default(
            q, k, cache, meta_positions, meta_header, metadata.context(), 128, 128, True
        )
        for actual, before, source in zip(new, old, (q_cpu, k_cpu)):
            assert torch.equal(actual.cpu().view(torch.int16), before.cpu().view(torch.int16))
            data, error, floor, absolute = reference.check(
                actual, source, cache_cpu, positions_cpu, 128, 128, True
            )
            payload.extend(data); errors.append(error); floors.append(floor); absolutes.append(absolute)
        counts = gaudirt.recipe_cache_stats()
        torch.ops.gaudirt._rotary_embedding_validated.default(
            q, k, cache, meta_positions, meta_header, metadata.context(), 128, 128, True
        )
        assert gaudirt.recipe_cache_stats() == counts
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[0] == 0 and counts[1] > 0, counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[0] > 0 and counts[1] == 0, counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("Validated tensor RoPE metadata matches existing interface and FP64;",
          "recipes", counts, "max row error", max(errors), "FP32 floor", max(floors),
          "max abs", max(absolutes), "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
