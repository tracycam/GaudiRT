"""Scalar elementwise FX operations against PyTorch and direct CPU FP64."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch

import gaudirt
import gaudirt.compiler as compiler
from fp64_reference import direct_bf16


def graph(x):
    return (x + 0.003907, 3.14159 + x, x - 0.1, 3.14159 - x,
            x * 0.1234567, 3.14159 * x, x / 3.14159, 3.14159 / x, x / 0.1)


def graph_aten(x):
    return (torch.ops.aten.add.Scalar(x, 0.003907), 3.14159 + x,
            torch.ops.aten.sub.Scalar(x, 0.1), 3.14159 - x,
            torch.ops.aten.mul.Scalar(x, 0.1234567), 3.14159 * x,
            torch.ops.aten.div.Scalar(x, 3.14159), 3.14159 / x,
            torch.ops.aten.div.Scalar(x, 0.1))


def invalid_scalar(x):
    return x + float("inf")


def reference(x):
    dtype = x.dtype
    a, b_add, c = (torch.tensor(value, dtype=dtype).double() for value in (0.003907, 3.14159, 0.1))
    b, c_div, d = (torch.tensor(value, dtype=torch.float32).double()
                   for value in (3.14159, 0.1, 0.1234567))
    x = x.double()
    return (x + a, b_add + x, x - c, b_add - x, x * d, b * x, x / b, b / x, x / c_div)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(5847)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    captures = []
    original = compiler.gaudirt_backend

    def capture(module, inputs):
        captures.append(1)
        return original(module, inputs)

    compiler.gaudirt_backend = capture
    compiled = compiler.compile(graph)
    aten = compiler.compile(graph_aten)
    payload = bytearray()
    worst = worst_floor = 0.0
    division_rounding_mismatches = 0
    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 3, 17, 137, 513, 4096):
            cpu = (torch.rand(rows, 65) + 0.75).to(dtype)
            cpu[0, 0] = 1.0
            if rows == 17:
                storage = torch.empty((rows, 130), dtype=dtype, device="gaudirt:0")
                device = storage[:, ::2]
                device.copy_(cpu)
            else:
                device = cpu.to("gaudirt:0")
            outputs = compiled(device)
            before = gaudirt.recipe_cache_stats()
            equivalent = aten(device)
            assert gaudirt.recipe_cache_stats() == before, "equivalent scalar graph recompiled"
            assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(outputs, equivalent))
            for index, (actual, expected, pytorch) in enumerate(zip(outputs, reference(cpu), graph(cpu))):
                truth = direct_bf16(expected) if dtype == torch.bfloat16 else expected.float().double()
                actual = actual.cpu().double()
                assert torch.isfinite(actual).all() and torch.isfinite(truth).all()
                denominator = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-30)
                error = (torch.linalg.vector_norm(actual - truth, dim=-1) / denominator).max().item()
                floor = (torch.linalg.vector_norm(pytorch.double() - truth, dim=-1) / denominator).max().item()
                assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (error, floor)
                if dtype == torch.bfloat16 and index not in (7, 8):
                    assert torch.equal(actual, pytorch.double()), (
                        rows, index, int(torch.count_nonzero(actual != pytorch.double())),
                        (actual - pytorch.double()).abs().max().item(), error, floor,
                    )
                elif dtype == torch.bfloat16:
                    division_rounding_mismatches += int(torch.count_nonzero(actual != pytorch.double()))
                worst = max(worst, error)
                worst_floor = max(worst_floor, floor)
                payload.extend(actual.to(dtype).contiguous().view(torch.uint8).numpy().tobytes())
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[:2] == [0, 12], stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[:2] == [12, 0], stats
    assert len(captures) == 4, captures
    try:
        compiler.compile(invalid_scalar)(torch.empty((1, 65), dtype=torch.float32, device="gaudirt:0"))
    except RuntimeError as error:
        assert "scalar must be finite" in str(error)
    else:
        raise AssertionError("nonfinite scalar was accepted")
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("FX scalar arithmetic", "stats", stats, "max_row_relative_l2", worst,
          "max_fp32_floor", worst_floor, "division_rounding_mismatches", division_rounding_mismatches,
          "output_sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
