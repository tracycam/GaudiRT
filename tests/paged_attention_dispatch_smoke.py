"""Exercise the C++ short/long decode choice with live header aliases."""

import gc
import math
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch

import gaudirt


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(1907)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    heads, kv_heads, dim, context = 56, 8, 128, 1024
    slots = context + 2
    mapping = torch.arange(1, context + 1, dtype=torch.int32).view(1, context)
    zeros = torch.zeros((slots, kv_heads, dim), dtype=torch.bfloat16)
    key_pool, value_pool = zeros.to("gaudirt:0"), zeros.to("gaudirt:0")
    query = torch.randn((1, heads * dim), dtype=torch.bfloat16).to("gaudirt:0")
    key = torch.randn((1, kv_heads * dim), dtype=torch.bfloat16).to("gaudirt:0")
    value = torch.randn_like(key.cpu()).to("gaudirt:0")
    # The first two inputs share the TPC recipe, the next two share the MME
    # recipe, and switching back must reuse the first one.
    residents = []
    for position in (18, 254, 255, 511, 18):
        metadata = torch.classes.gaudirt.PagedMetadata(
            torch.tensor([position + 1]), torch.tensor([0]), torch.tensor([position]), mapping, slots
        )
        locations, requests, positions = (metadata.row_locations(), metadata.row_requests(),
                                           metadata.rotary_positions())
        device_mapping, header = metadata.request_mapping(), metadata.rotary_header()
        del metadata
        gc.collect()
        output = torch.ops.gaudirt._paged_attn_validated.default(
            query, key, value, key_pool, value_pool, locations, requests, positions,
            device_mapping, header, dim, 1 / math.sqrt(dim),
        )
        assert torch.isfinite(output.cpu()).all()
        residents.append(gaudirt.recipe_cache_stats()[2])
        del output, locations, requests, positions, device_mapping, header
    assert residents == [2, 2, 3, 3, 3], residents
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats == [0, 3, 3], stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats == [3, 0, 3], stats
    print("paged decode recipe residents", residents, "cache", stats, flush=True)


if __name__ == "__main__":
    main()
