"""Unsupported attention shapes fail before compiling or mutating the KV pool."""

import math
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def reject(rows, context, positions):
    slots = context + 1
    mapping = torch.arange(1, slots, dtype=torch.int32).reshape(1, context)
    requests = torch.zeros(rows, dtype=torch.int64)
    positions = torch.tensor(positions, dtype=torch.int64)
    locations = mapping[requests, positions].to(torch.int64)
    metadata = torch.classes.gaudirt.PagedMetadata(locations, requests, positions, mapping, slots)
    query = torch.ones((rows, 56 * 128), dtype=torch.bfloat16, device="gaudirt:0")
    key = torch.ones((rows, 8 * 128), dtype=torch.bfloat16, device="gaudirt:0")
    value = torch.ones_like(key)
    key_pool = torch.zeros((slots, 8, 128), dtype=torch.bfloat16, device="gaudirt:0")
    value_pool = torch.zeros_like(key_pool)
    before = gaudirt.recipe_cache_stats()
    try:
        torch.ops.gaudirt.paged_attn.default(query, key, value, key_pool, value_pool,
                                             metadata, 128, 1 / math.sqrt(128))
    except RuntimeError as error:
        assert "tiled MME attention" in str(error), error
    else:
        raise AssertionError("unsupported attention geometry ran instead of failing")
    assert gaudirt.recipe_cache_stats() == before
    assert not torch.count_nonzero(key_pool.cpu()).item()
    assert not torch.count_nonzero(value_pool.cpu()).item()
    print("rejected rows", rows, "context", context, "without KV writes or recipe compilation", flush=True)


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    reject(3, 32, [0, 1, 2])
    reject(1, 4096, [4095])
    reject(1, 4608, [4607])


if __name__ == "__main__":
    main()
