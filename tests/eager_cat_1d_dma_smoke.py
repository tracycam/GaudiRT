"""One-dimensional concatenation stays on device without shape-specific recipes."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    for dtype in (torch.bool, torch.int32, torch.int64, torch.float32, torch.bfloat16):
        base = torch.arange(515, dtype=torch.int32).to(dtype)
        device = base.to("gaudirt")
        for views in ((slice(1, 257, 2), slice(0, 513, 3)),
                      (slice(1, 4), slice(4, 515)),
                      (slice(0, 256), slice(0, 256))):
            inputs = tuple(device[index] for index in views)
            expected = torch.cat(tuple(base[index] for index in views))
            before = gaudirt.recipe_cache_stats()[1]
            actual = torch.cat(inputs)
            assert gaudirt.recipe_cache_stats()[1] == before, "cat compiled a recipe"
            del inputs
            assert torch.equal(actual.cpu(), expected), (dtype, views)
    left = torch.tensor([1, -2, 3], dtype=torch.int32)
    right = torch.tensor([0.5, -1.25], dtype=torch.float32)
    promoted = torch.cat((left.to("gaudirt"), right.to("gaudirt")))
    assert promoted.dtype == torch.float32 and torch.equal(promoted.cpu(), torch.cat((left, right)))

    streams = [gaudirt.Stream(), gaudirt.Stream()]
    results = []
    expected = torch.arange(1024, dtype=torch.int64)
    for stream in streams:
        with gaudirt.stream(stream):
            left = expected[:513].to("gaudirt", non_blocking=True)
            right = expected[513:].to("gaudirt", non_blocking=True)
            results.append(torch.cat((left, right)))
            del left, right
    for stream, result in zip(streams, results):
        stream.synchronize()
        with gaudirt.stream(stream):
            assert torch.equal(result.cpu(), expected)
    print("1D cat: contiguous, strided, unaligned and aliased inputs; no recipe compilation", flush=True)


if __name__ == "__main__":
    main()
