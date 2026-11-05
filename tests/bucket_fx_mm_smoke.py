"""Dynamic 2-D GEMM with a standard contiguous [K, N] right operand."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
from fp64_reference import direct_bf16

import gaudirt
import gaudirt.compiler as compiler


def mm(x, weight):
    return torch.mm(x, weight)


def matmul(x, weight):
    return torch.matmul(x, weight)


def addmm(x, weight, bias):
    return torch.addmm(bias, x, weight)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(3349)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    captures = []
    original = compiler.gaudirt_backend

    def capture(graph, inputs):
        captures.append(1)
        return original(graph, inputs)

    compiler.gaudirt_backend = capture
    first, equivalent, biased = [compiler.compile(fn) for fn in (mm, matmul, addmm)]
    payload = bytearray()
    worst = worst_floor = 0.0
    for dtype in (torch.bfloat16, torch.float32):
        weight_cpu = (torch.randn(17, 23) * 0.2).to(dtype)
        bias_cpu = (torch.randn(23) * 0.2).to(dtype)
        weight, bias = weight_cpu.to("gaudirt:0"), bias_cpu.to("gaudirt:0")
        for rows in (1, 3, 17, 137, 513, 4096):
            x_cpu = torch.randn(rows, 17).to(dtype)
            if rows == 17:
                storage = torch.empty((rows, 34), dtype=dtype, device="gaudirt:0")
                x = storage[:, ::2]
                x.copy_(x_cpu)
            else:
                x = x_cpu.to("gaudirt:0")
            plain = first(x, weight)
            before = gaudirt.recipe_cache_stats()
            same = equivalent(x, weight)
            assert gaudirt.recipe_cache_stats() == before, "equivalent GEMM recompiled"
            assert torch.equal(plain.cpu(), same.cpu())
            result = biased(x, weight, bias)
            for actual, offset in ((plain, None), (result, bias_cpu)):
                truth = x_cpu.double() @ weight_cpu.double()
                floor = x_cpu.float() @ weight_cpu.float()
                if offset is not None:
                    truth = truth + offset.double()
                    floor = floor + offset.float()
                if dtype == torch.bfloat16:
                    truth = direct_bf16(truth)
                    floor = floor.to(torch.bfloat16).double()
                else:
                    truth, floor = truth.float().double(), floor.double()
                value = actual.cpu().double()
                assert torch.isfinite(value).all() and torch.isfinite(truth).all() and torch.isfinite(floor).all()
                denominator = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-30)
                error = (torch.linalg.vector_norm(value - truth, dim=-1) / denominator).max().item()
                floor_error = (torch.linalg.vector_norm(floor - truth, dim=-1) / denominator).max().item()
                assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2, (rows, error, floor_error)
                worst, worst_floor = max(worst, error), max(worst_floor, floor_error)
                payload.extend(actual.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 24], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [24, 0], counts
    assert len(captures) == 6, captures
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("FX standard matrix multiply", "stats", counts, "max_row_relative_l2", worst,
          "max_fp32_floor", worst_floor, "output_sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
