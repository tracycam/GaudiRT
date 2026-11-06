"""Check 32:8 attention at both supported head widths and decode paths."""

import hashlib
import math
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import paged_attention_smoke as attention


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(4439)
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    digest = hashlib.sha256()
    errors, floors, absolutes, tiled_errors = [], [], [], []

    for dimension in (64, 128):
        attention.HEADS, attention.KV_HEADS, attention.DIM = 32, 8, dimension
        attention.Q_WIDTH, attention.KV_WIDTH = 32 * dimension, 8 * dimension
        attention.SCALE = 1 / math.sqrt(dimension)
        attention.SCALE_FP32 = torch.tensor(attention.SCALE, dtype=torch.float32).item()
        for rows, context, position in [
            (1, 32, None), (7, 32, None), (33, 32, None),
            (137, 32, None), (1025, 32, None),
            (1, 512, 254), (1, 512, 255), (1, 512, 511),
        ]:
            attention.REQUESTS, attention.CONTEXT = (64, context) if position is None else (1, context)
            attention.SLOTS = attention.REQUESTS * context + 1
            if position is None:
                prepared, locations, requests, positions, mapping = attention.metadata(rows)
            else:
                mapping = torch.arange(1, attention.SLOTS, dtype=torch.int32).view(1, context)
                requests = torch.tensor([0], dtype=torch.int64)
                positions = torch.tensor([position], dtype=torch.int64)
                locations = mapping[requests, positions].to(torch.int64)
                prepared = torch.classes.gaudirt.PagedMetadata(
                    locations, requests, positions, mapping, attention.SLOTS
                )
            query, key, value = [torch.randn(rows, width, dtype=torch.bfloat16)
                                 for width in (attention.Q_WIDTH, attention.KV_WIDTH, attention.KV_WIDTH)]
            for layer in range(2):
                pool_shape = (attention.SLOTS, attention.KV_HEADS, dimension)
                key_pool = torch.randn(pool_shape, dtype=torch.bfloat16)
                value_pool = torch.randn(pool_shape, dtype=torch.bfloat16)
                before = gaudirt.recipe_cache_stats()
                result = attention.run_layer(query, key, value, key_pool, value_pool, prepared,
                                             locations, requests, positions, mapping, sliced=rows == 137)
                if layer == 1:
                    assert gaudirt.recipe_cache_stats() == before, "same-geometry layer recompiled"
                digest.update(result[0])
                errors.append(result[1]); floors.append(result[2]); absolutes.append(result[3])
                tiled_errors.append(result[12])
                del result
            print("geometry", 32, 8, dimension, "rows", rows, "position", position,
                  "cache", gaudirt.recipe_cache_stats(), flush=True)
            del prepared

    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[0] == 0 and stats[1] > 0, stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[0] > 0 and stats[1] == 0, stats
    print("32:8 GQA audit", "cache", stats, "max_row_relative_l2", max(errors),
          "max_tiled_bf16_pv_error", max(tiled_errors), "max_cpu_fp32_floor_diagnostic", max(floors),
          "max_abs", max(absolutes),
          "output_and_kv_sha256", digest.hexdigest(), flush=True)
    gaudirt.empty_cache()
    gaudirt.shutdown()


if __name__ == "__main__":
    main()
