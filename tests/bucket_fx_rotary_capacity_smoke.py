"""Share rotary FX recipes across cache capacities and validated contexts."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F

import gaudirt
import gaudirt.compiler as compiler
from bucket_fx_rotary_smoke import check


CONTEXT = 4608


def projection_and_rotary(x, weight, positions, header, cache):
    q, k, v = F.linear(x, weight).split([128, 64, 64], dim=-1)
    q, k = torch.ops.gaudirt._rotary_embedding_validated.default(
        q, k, cache, positions, header, CONTEXT, 64, 64, True
    )
    return q, k, v


def metadata(context, positions):
    mapping = torch.arange(1, context + 1, dtype=torch.int32).reshape(1, -1)
    requests = torch.zeros_like(positions)
    locations = mapping[0, positions.long()].contiguous()
    return torch.classes.gaudirt.PagedMetadata(locations, requests, positions, mapping, context + 1)


def main():
    global CONTEXT
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(9043)
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    cache_cpu = torch.randn(32768, 64, dtype=torch.bfloat16)
    weight_cpu = (torch.randn(256, 256) * 0.05).to(torch.bfloat16)
    weight = weight_cpu.to("gaudirt:0")
    run = compiler.compile(projection_and_rotary, dynamic_arg_indices=(0, 2))
    payload = bytearray()
    for rows in (1, 17):
        x_cpu = torch.randn(rows, 256, dtype=torch.bfloat16)
        x = x_cpu.to("gaudirt:0")
        shared = None
        for context, end in ((4608, 4608), (32768, 4608), (32768, 32768)):
            CONTEXT = context
            positions_cpu = torch.arange(end - rows, end, dtype=torch.int32)
            prepared = metadata(context, positions_cpu)
            cache = cache_cpu[:context].contiguous().to("gaudirt:0")
            before = gaudirt.recipe_cache_stats()
            output = run(x, weight, prepared.rotary_positions(), prepared.rotary_header(), cache)
            after = gaudirt.recipe_cache_stats()
            data, error, floor, _ = check(output, x_cpu, weight_cpu, cache_cpu[:context], positions_cpu.long())
            if context == 4608:
                shared = data
            elif end == 4608:
                assert data == shared, "capacity changed rotary output"
                assert after == before, (before, after)
            else:
                assert after == before, (before, after)
            payload.extend(data)
            print("rows", rows, "context", context, "last_position", end - 1,
                  "cache", after, "relative_l2", error, "fp32_floor", floor, flush=True)
    expected = [0, 2] if os.environ["GAUDIRT_TEST_CACHE"] == "cold" else [2, 0]
    stats = gaudirt.recipe_cache_stats()
    assert stats[:2] == expected, stats
    short_cache = cache_cpu[:4608].contiguous().to("gaudirt:0")
    try:
        run(x, weight, prepared.rotary_positions(), prepared.rotary_header(), short_cache)
    except RuntimeError as error:
        assert "rotary" in str(error).lower(), str(error)
    else:
        raise AssertionError("rotary cache shorter than the validated context was accepted")
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as out:
            out.write(payload)
    print("rotary capacity cache", stats, "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
