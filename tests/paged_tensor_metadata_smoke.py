"""Validate the tensor-only paged attention entry against the existing API."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import paged_attention_smoke as reference


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(7012)
    gaudirt.initialize(hbm_bytes=512 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for rows in (1, 7, 137, 513):
        metadata, locations, requests, positions, mapping = reference.metadata(rows)
        query_cpu, key_cpu, value_cpu = [torch.randn(rows, width, dtype=torch.bfloat16)
                                        for width in (reference.Q_WIDTH, reference.KV_WIDTH,
                                                      reference.KV_WIDTH)]
        base_k, base_v = [torch.randn(reference.SLOTS, reference.KV_HEADS, reference.DIM,
                                      dtype=torch.bfloat16) for _ in range(2)]
        old = reference.run_layer(query_cpu, key_cpu, value_cpu, base_k, base_v, metadata,
                                  locations, requests, positions, mapping, sliced=rows == 137)
        before = gaudirt.recipe_cache_stats()
        key_pool, value_pool = base_k.to("gaudirt:0"), base_v.to("gaudirt:0")
        result = torch.ops.gaudirt._paged_attn_validated.default(
            old[4], old[5], old[6], key_pool, value_pool,
            metadata.row_locations(), metadata.row_requests(), metadata.rotary_positions(),
            metadata.request_mapping(), metadata.rotary_header(), reference.DIM, reference.SCALE,
        )
        assert gaudirt.recipe_cache_stats() == before, "tensor metadata changed recipe identity"
        for actual, expected in ((result, old[11]), (key_pool, old[7]), (value_pool, old[8])):
            assert torch.equal(actual.cpu().view(torch.int16), expected.cpu().view(torch.int16))
        payload.extend(old[0]); errors.append(old[1]); floors.append(old[2]); absolutes.append(old[3])
        print("tensor paged metadata rows", rows, "stats", before, flush=True)
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 8], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [8, 0], counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("tensor paged metadata stats", counts, "max row error", max(errors), "FP32 floor", max(floors),
          "max abs", max(absolutes), "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
