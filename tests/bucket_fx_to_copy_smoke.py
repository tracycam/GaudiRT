"""Explicit FX casts and copies retain PyTorch's allocation semantics."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch

import gaudirt
import gaudirt.compiler as compiler


def to_bf16(x):
    return torch.ops.aten._to_copy.default(x, dtype=torch.bfloat16)


def to_f32(x):
    return torch.ops.aten._to_copy.default(x, dtype=torch.float32)


def method_bf16(x):
    return x.to(dtype=torch.bfloat16)


def method_f32(x):
    return x.to(dtype=torch.float32)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(1083)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    bf16, f32, method_b, method_f = [compiler.compile(fn) for fn in
                                     (to_bf16, to_f32, method_bf16, method_f32)]
    digest = hashlib.sha256()
    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 3, 17, 137, 513, 4096):
            cpu = torch.randn(rows, 15).to(dtype)
            if rows == 17:
                storage = torch.empty((rows, 30), dtype=dtype, device="gaudirt:0")
                device = storage[:, ::2]
                device.copy_(cpu)
            else:
                device = cpu.to("gaudirt:0")
            for target, explicit, method in ((torch.bfloat16, bf16, method_b),
                                              (torch.float32, f32, method_f)):
                copied = explicit(device)
                assert copied.dtype == target and copied.data_ptr() != device.data_ptr()
                expected = cpu.to(target)
                actual = copied.cpu()
                assert torch.isfinite(actual).all() and torch.equal(actual, expected)
                before = gaudirt.recipe_cache_stats()
                converted = method(device)
                assert gaudirt.recipe_cache_stats() == before
                assert torch.equal(converted.cpu(), expected)
                assert (converted.data_ptr() == device.data_ptr()) == (dtype == target)
                if dtype == target and rows == 17:
                    copied[0, 0].copy_(torch.tensor(5.0, dtype=target))
                    assert torch.equal(device.cpu(), cpu), "_to_copy output aliases its input"
                digest.update(actual.contiguous().view(torch.uint8).numpy().tobytes())
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[:2] == [0, 24], stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[:2] == [24, 0], stats
    print("FX _to_copy", "stats", stats, "output_sha256", digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
