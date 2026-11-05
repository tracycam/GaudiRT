"""Dynamic outer-batch GEMM against independent CPU FP64."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
from fp64_reference import direct_bf16

import gaudirt
import gaudirt.compiler as compiler


def bmm(x, y):
    return torch.bmm(x, y)


def matmul(x, y):
    return torch.matmul(x, y)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(3219)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    captures = 0
    original = compiler.gaudirt_backend

    def capture(graph_module, inputs):
        nonlocal captures
        captures += 1
        return original(graph_module, inputs)

    compiler.gaudirt_backend = capture
    first = compiler.compile(bmm, dynamic_arg_indices=(0, 1))
    second = compiler.compile(matmul, dynamic_arg_indices=(0, 1))
    digest = hashlib.sha256()
    worst = 0.0
    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 3, 17, 137, 513, 4096):
            x = torch.randn(rows, 3, 7).to(dtype)
            y = torch.randn(rows, 7, 5).to(dtype)
            if rows == 17:
                backing = torch.empty((rows, 3, 14), dtype=dtype, device="gaudirt:0")
                dx = backing[:, :, ::2]
                dx.copy_(x)
                assert not dx.is_contiguous()
            else:
                dx = x.to("gaudirt:0")
            dy = y.to("gaudirt:0")
            result = first(dx, dy)
            assert result.shape == (rows, 3, 5) and result.is_contiguous()
            assert result.untyped_storage().nbytes() >= (1 << (rows - 1).bit_length()) * 15 * result.element_size()
            actual = result.cpu().double()
            truth = torch.bmm(x.double(), y.double())
            floor = torch.bmm(x.float(), y.float())
            if dtype == torch.bfloat16:
                truth = direct_bf16(truth)
                floor = floor.to(torch.bfloat16).double()
            else:
                truth = truth.float().double()
                floor = floor.double()
            assert torch.isfinite(actual).all() and torch.isfinite(truth).all() and torch.isfinite(floor).all()
            denom = torch.linalg.vector_norm(truth.reshape(rows, -1), dim=-1).clamp_min(1e-30)
            error = (torch.linalg.vector_norm((actual - truth).reshape(rows, -1), dim=-1) / denom).max().item()
            floor_error = (torch.linalg.vector_norm((floor - truth).reshape(rows, -1), dim=-1) / denom).max().item()
            assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2, (dtype, rows, error, floor_error)
            worst = max(worst, error)
            before = gaudirt.recipe_cache_stats()
            again = second(dx, dy)
            assert gaudirt.recipe_cache_stats() == before, "equivalent bmm/matmul recompiled"
            assert torch.equal(result.cpu(), again.cpu())
            digest.update(result.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        x = torch.randn(1, 3, 7).to(dtype)
        y = torch.randn(1, 7, 5).to(dtype).to("gaudirt:0")
        normal = x.to("gaudirt:0")
        backing = torch.empty((2, 3, 7), dtype=dtype, device="gaudirt:0")
        unusual = backing[::2]
        unusual.copy_(x)
        assert unusual.is_contiguous() and unusual.stride(0) != normal.stride(0)
        before_captures = captures
        expected = first(normal, y).cpu()
        actual = first(unusual, y).cpu()
        assert torch.equal(actual, expected)
        assert captures == before_captures, "singleton-stride input triggered a new capture"
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[:2] == [0, 12], stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[:2] == [12, 0], stats
    assert captures == 4, captures
    for rows in (0, 4097):
        x = torch.empty((rows, 3, 7), dtype=torch.float32, device="gaudirt:0")
        y = torch.empty((rows, 7, 5), dtype=torch.float32, device="gaudirt:0")
        try:
            first(x, y)
        except RuntimeError as error:
            assert "invalid dynamic input" in str(error), str(error)
        else:
            raise AssertionError(f"accepted invalid batch size {rows}")
    assert gaudirt.recipe_cache_stats() == stats
    print("batch GEMM FX", "stats", stats, "max_row_relative_l2", worst,
          "output_sha256", digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
