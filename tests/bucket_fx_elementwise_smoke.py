"""Connected dynamic BF16/FP32 sub, multiply and division programs."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
from fp64_reference import direct_bf16

import gaudirt
import gaudirt.compiler as compiler


def graph(x, y):
    difference = x - y
    product = difference * x
    quotient = product / y
    return difference, product, quotient


def same_graph(x, y):
    return graph(x, y)


def fused_graph(x, y):
    return ((x - y) * x) / y


def reference(x, y, fp64):
    input_dtype = x.dtype
    dtype = torch.float64 if fp64 else torch.float32
    x, y = x.to(dtype), y.to(dtype)
    if x.dtype == torch.float64:
        rounded = direct_bf16 if input_dtype == torch.bfloat16 else lambda value: value.float().double()
    else:
        rounded = (lambda value: value.to(torch.bfloat16).float()) if input_dtype == torch.bfloat16 else lambda value: value
    difference = rounded(x - y)
    product = rounded(difference * x)
    quotient = rounded(product / y)
    return tuple(value.double() for value in (difference, product, quotient))


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(1148)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    captures = 0
    original = compiler.gaudirt_backend

    def capture(graph_module, inputs):
        nonlocal captures
        captures += 1
        return original(graph_module, inputs)

    compiler.gaudirt_backend = capture
    first = compiler.compile(graph, dynamic_arg_indices=(0, 1))
    second = compiler.compile(same_graph, dynamic_arg_indices=(0, 1))
    fused = compiler.compile(fused_graph, dynamic_arg_indices=(0, 1))
    digest = hashlib.sha256()
    worst = 0.0
    for input_dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 3, 17, 137, 513, 4096):
            x = torch.randn(rows, 65).to(input_dtype)
            y = (torch.rand(rows, 65) + 0.75).to(input_dtype)
            dx, dy = x.to("gaudirt:0"), y.to("gaudirt:0")
            outputs = first(dx, dy)
            truth, floor = reference(x, y, True), reference(x, y, False)
            for name, result, expected, baseline in zip(("sub", "mul", "div"), outputs, truth, floor):
                actual = result.cpu()
                assert torch.isfinite(actual).all() and torch.isfinite(expected).all() and torch.isfinite(baseline).all()
                denominator = torch.linalg.vector_norm(expected, dim=-1).clamp_min(1e-30)
                error = (torch.linalg.vector_norm(actual.double() - expected, dim=-1) / denominator).max().item()
                floor_error = (torch.linalg.vector_norm(baseline - expected, dim=-1) / denominator).max().item()
                assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2, (name, input_dtype, rows, error, floor_error, (actual.double() - expected).abs().max().item())
                if input_dtype == torch.bfloat16:
                    assert torch.equal(actual.double(), expected), (name, rows, "BF16 boundary differs")
                worst = max(worst, error)
                digest.update(actual.contiguous().view(torch.uint8).numpy().tobytes())
            before = gaudirt.recipe_cache_stats()
            again = second(dx, dy)
            assert gaudirt.recipe_cache_stats() == before, "same elementwise graph recompiled"
            assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(outputs, again))
        x = torch.randn(137, 65).to(input_dtype)
        y = (torch.rand(137, 65) + 0.75).to(input_dtype)
        actual = fused(x.to("gaudirt:0"), y.to("gaudirt:0")).cpu()
        expected, baseline = reference(x, y, True)[-1], reference(x, y, False)[-1]
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all() and torch.isfinite(baseline).all()
        denominator = torch.linalg.vector_norm(expected, dim=-1).clamp_min(1e-30)
        error = (torch.linalg.vector_norm(actual.double() - expected, dim=-1) / denominator).max().item()
        floor_error = (torch.linalg.vector_norm(baseline - expected, dim=-1) / denominator).max().item()
        assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2, (input_dtype, error, floor_error)
        if input_dtype == torch.bfloat16:
            assert torch.equal(actual.double(), expected), ("fused", "BF16 boundary differs")
        worst = max(worst, error)
        digest.update(actual.contiguous().view(torch.uint8).numpy().tobytes())
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[:2] == [0, 14], stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[:2] == [14, 0], stats
    assert captures == 6, captures
    print("elementwise FX", "stats", stats, "max_row_relative_l2", worst,
          "output_sha256", digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
