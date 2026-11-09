"""Check int64 argmax output and edge cases after fusing its cast."""

import ctypes
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=4 * 1024**2)
    torch.random.default_generator.manual_seed(841)
    counter = ctypes.CDLL(None).gaudirt_test_launch_count
    counter.restype = ctypes.c_uint64
    for dtype in (torch.bfloat16, torch.float32, torch.int32):
        values = torch.randint(-5, 6, (4, 129)).to(dtype)
        if dtype != torch.int32:
            values[0, 1], values[0, 3] = float("nan"), float("nan")
            values[1, :] = -float("inf")
            values[2, 5], values[2, 7] = float("inf"), float("inf")
        device = values.to("gaudirt:0")
        before = gaudirt.recipe_cache_stats()
        launches = counter()
        result = torch.argmax(device, dim=-1)
        assert counter() - launches == 1, "argmax must launch one recipe"
        after = gaudirt.recipe_cache_stats()
        assert after[1] - before[1] <= 1, (before, after)
        assert result.dtype == torch.int64 and torch.equal(result.cpu(), torch.argmax(values, dim=-1))
        for dim in (None, 0, 1):
            for keep in (False, True):
                actual = torch.argmax(device.t(), dim=dim, keepdim=keep).cpu()
                expected = torch.argmax(values.t(), dim=dim, keepdim=keep)
                assert torch.equal(actual, expected), (dtype, dim, keep, actual, expected)
    for width in (1, 3, 127, 1025):
        values = torch.zeros((2, width), dtype=torch.float32)
        values[:, min(1, width - 1)] = values[:, -1] = 2
        assert torch.equal(torch.argmax(values.to("gaudirt:0"), dim=1).cpu(), torch.argmax(values, dim=1))
    values = torch.tensor([[-(2**63), 2**62, 2**62 + 1], [-1, -(2**63), -1]], dtype=torch.int64)
    for dim in (None, 0, 1):
        assert torch.equal(torch.argmax(values.to("gaudirt:0"), dim=dim).cpu(), torch.argmax(values, dim=dim))
    if os.environ.get("GAUDIRT_EXPECT_WARM") == "1":
        assert gaudirt.recipe_cache_stats()[1] == 0, gaudirt.recipe_cache_stats()
    print("argmax single-recipe int64 output and special cases passed", flush=True)


if __name__ == "__main__":
    main()
