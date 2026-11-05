"""Broadcast static feature vectors inside dynamic BF16/FP32 FX programs."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
from fp64_reference import direct_bf16

import gaudirt
import gaudirt.compiler as compiler


def graph(x, bias):
    return x + bias, bias - x, x * bias, x / bias, bias / x


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(6917)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    captures = 0
    original = compiler.gaudirt_backend

    def capture(module, inputs):
        nonlocal captures
        captures += 1
        return original(module, inputs)

    compiler.gaudirt_backend = capture
    compiled = compiler.compile(graph)
    digest = hashlib.sha256()
    worst = 0.0
    for dtype in (torch.bfloat16, torch.float32):
        for rank in (1, 2):
            for rows in (1, 3, 17, 137, 513, 4096):
                x = (torch.rand(rows, 65) + 0.75).to(dtype)
                bias = (torch.rand(65) + 0.75).to(dtype)
                if rank == 2:
                    bias = bias.view(1, -1)
                results = compiled(x.to("gaudirt:0"), bias.to("gaudirt:0"))
                truth = graph(x.double(), bias.double())
                floor = graph(x.float(), bias.float())
                for actual, ref64, ref32 in zip(results, truth, floor):
                    if dtype == torch.bfloat16:
                        ref64, ref32 = direct_bf16(ref64), ref32.to(torch.bfloat16).double()
                    else:
                        ref64, ref32 = ref64.float().double(), ref32.double()
                    actual = actual.cpu().double()
                    assert torch.isfinite(actual).all() and torch.isfinite(ref64).all() and torch.isfinite(ref32).all()
                    denominator = torch.linalg.vector_norm(ref64, dim=-1).clamp_min(1e-30)
                    error = (torch.linalg.vector_norm(actual - ref64, dim=-1) / denominator).max().item()
                    floor_error = (torch.linalg.vector_norm(ref32 - ref64, dim=-1) / denominator).max().item()
                    assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2
                    if dtype == torch.bfloat16:
                        assert torch.equal(actual, ref64)
                    worst = max(worst, error)
                    digest.update(actual.to(dtype).contiguous().view(torch.uint8).numpy().tobytes())
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[:2] == [0, 24], stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[:2] == [24, 0], stats
    assert captures == 4, captures
    print("FX feature broadcast", "stats", stats, "max_row_relative_l2", worst,
          "output_sha256", digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
