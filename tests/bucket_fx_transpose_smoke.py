"""Token-preserving transpose views and batch GEMM against CPU FP64."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
from fp64_reference import direct_bf16

import gaudirt
import gaudirt.compiler as compiler


def view_method(x):
    return x.transpose(1, 2)


def view_function(x):
    return torch.transpose(x, -2, -1)


def view_aten(x):
    return torch.ops.aten.transpose.int(x, 1, 2)


def view_roundtrip(x):
    return x.transpose(1, 2).transpose(1, 2)


def transpose_reshape(x):
    return x.transpose(1, 2).reshape(-1, 21)


def bmm_method(x, y):
    return torch.bmm(x.transpose(1, 2), y)


def bmm_function(x, y):
    return torch.bmm(torch.transpose(x, -2, -1), y)


def bmm_with_view(x, y):
    view = torch.ops.aten.transpose.int(x, 1, 2)
    return torch.bmm(view, y), view, x


def bmm_right(x, y):
    return torch.bmm(x, y.transpose(1, 2))


def bmm_both(x, y):
    return torch.bmm(x.transpose(1, 2), y.transpose(1, 2))


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(7529)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    views = [compiler.compile(fn) for fn in (view_method, view_function, view_aten)]
    roundtrip, reshape_copy = [compiler.compile(fn) for fn in (view_roundtrip, transpose_reshape)]
    left, function, with_view, right, both = [
        compiler.compile(fn, dynamic_arg_indices=(0, 1))
        for fn in (bmm_method, bmm_function, bmm_with_view, bmm_right, bmm_both)
    ]
    digest = hashlib.sha256()
    worst = 0.0
    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 3, 17, 137, 513, 4096):
            x_cpu = torch.randn(rows, 7, 3).to(dtype)
            if rows == 17:
                storage = torch.empty((rows, 7, 6), dtype=dtype, device="gaudirt:0")
                x = storage[:, :, ::2]
                x.copy_(x_cpu)
            else:
                x = x_cpu.to("gaudirt:0")
            y_cpu = torch.randn(rows, 7, 5).to(dtype)
            y = y_cpu.to("gaudirt:0")
            for view in (fn(x) for fn in views):
                assert view.data_ptr() == x.data_ptr()
                assert view.stride() == x.transpose(1, 2).stride()
                assert torch.equal(view.cpu(), x_cpu.transpose(1, 2))
            restored = roundtrip(x)
            assert restored.data_ptr() == x.data_ptr() and restored.stride() == x.stride()
            copied = reshape_copy(x)
            assert copied.untyped_storage()._cdata != x.untyped_storage()._cdata
            assert torch.equal(copied.cpu(), x_cpu.transpose(1, 2).reshape(-1, 21))
            if rows == 17:
                view[0, 1, 0].copy_(torch.tensor(1.25, dtype=dtype))
                x_cpu[0, 0, 1] = 1.25
                assert torch.equal(x.cpu(), x_cpu)
            result = left(x, y)
            before = gaudirt.recipe_cache_stats()
            equivalent = function(x, y)
            same, alias, original = with_view(x, y)
            assert gaudirt.recipe_cache_stats() == before, "equivalent transpose graph recompiled"
            assert alias.data_ptr() == x.data_ptr() and alias.stride() == x.transpose(1, 2).stride()
            assert original.data_ptr() == x.data_ptr()
            assert torch.equal(result.cpu(), equivalent.cpu()) and torch.equal(result.cpu(), same.cpu())
            p = x_cpu.transpose(1, 2).contiguous().to("gaudirt:0")
            z = y_cpu.transpose(1, 2).contiguous().to("gaudirt:0")
            assert torch.equal(result.cpu(), right(p, z).cpu())
            assert torch.equal(result.cpu(), both(x, z).cpu())
            truth = torch.bmm(x_cpu.double().transpose(1, 2), y_cpu.double())
            floor = torch.bmm(x_cpu.float().transpose(1, 2), y_cpu.float())
            if dtype == torch.bfloat16:
                truth = direct_bf16(truth)
                floor = floor.to(torch.bfloat16).double()
            else:
                truth, floor = truth.float().double(), floor.double()
            actual = result.cpu().double()
            assert torch.isfinite(actual).all() and torch.isfinite(truth).all() and torch.isfinite(floor).all()
            denom = torch.linalg.vector_norm(truth.reshape(rows, -1), dim=-1).clamp_min(1e-30)
            error = (torch.linalg.vector_norm((actual - truth).reshape(rows, -1), dim=-1) / denom).max().item()
            floor_error = (torch.linalg.vector_norm((floor - truth).reshape(rows, -1), dim=-1) / denom).max().item()
            assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2
            worst = max(worst, error)
            digest.update(result.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[:2] == [0, 48], stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[:2] == [48, 0], stats
    print("FX transpose and bmm", "stats", stats, "max_row_relative_l2", worst,
          "output_sha256", digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
