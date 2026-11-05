"""Two-layer FX programs with persistent intermediate and reordered outputs."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
from fp64_reference import direct_bf16
import gaudirt
import gaudirt.compiler as compiler

torch.set_num_threads(4)
torch.random.default_generator.manual_seed(3371)


def layer(x, weight, residual):
    projected = torch.nn.functional.linear(x, weight)
    return projected + residual, projected


def reference(x, weight, residual, fp64):
    dtype = torch.float64 if fp64 else torch.float32
    rounded = direct_bf16 if fp64 else lambda value: value.to(torch.bfloat16).float()
    x, weight, residual = (value.to(dtype) for value in (x, weight, residual))
    projected = rounded(x @ weight.t())
    return rounded(projected + residual).double(), projected.double()


def check(outputs, x, weight, residual):
    ref64 = reference(x, weight, residual, True)
    ref32 = reference(x, weight, residual, False)
    results = []
    for output, truth, baseline in zip(outputs, ref64, ref32):
        assert output.shape == x.shape and output.is_contiguous()
        bucket = 1 << (x.size(0) - 1).bit_length()
        assert output.untyped_storage().nbytes() >= bucket * x.size(1) * 2
        assert output.untyped_storage().nbytes() % 256 == 0
        actual = output.cpu().double()
        assert torch.isfinite(actual).all() and torch.isfinite(truth).all() and torch.isfinite(baseline).all()
        denominator = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-30)
        error = (torch.linalg.vector_norm(actual - truth, dim=-1) / denominator).max().item()
        floor = (torch.linalg.vector_norm(baseline - truth, dim=-1) / denominator).max().item()
        absolute = (actual - truth).abs().max().item()
        assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (x.size(0), error, floor, absolute)
        results.append((actual.to(torch.bfloat16).view(torch.int16).numpy().tobytes(), error, floor, absolute))
    return results


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    original = compiler.gaudirt_backend
    captures = []

    def capture(graph_module, example_inputs):
        captures.append(1)
        return original(graph_module, example_inputs)

    compiler.gaudirt_backend = capture
    compiled = compiler.compile(layer, dynamic_arg_indices=(0, 2))
    weights_cpu = [(torch.randn(64, 64) * 0.125).to(torch.bfloat16) for _ in range(2)]
    weights = [weight.to("gaudirt:0") for weight in weights_cpu]
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for rows in (1, 3, 17, 65, 137, 513, 2048, 4096):
        x_cpu, r1_cpu, r2_cpu = [torch.randn(rows, 64, dtype=torch.bfloat16) for _ in range(3)]
        x, r1, r2 = [value.to("gaudirt:0") for value in (x_cpu, r1_cpu, r2_cpu)]
        first = compiled(x, weights[0], r1)
        first_results = check(first, x_cpu, weights_cpu[0], r1_cpu)
        before_second = gaudirt.recipe_cache_stats()
        second = compiled(first[0], weights[1], r2)
        assert gaudirt.recipe_cache_stats() == before_second, "isomorphic layer recompiled"
        second_results = check(second, first[0].cpu(), weights_cpu[1], r2_cpu)
        for result in first_results + second_results:
            payload.extend(result[0]); errors.append(result[1]); floors.append(result[2]); absolutes.append(result[3])
        print("rows", rows, "stats", before_second,
              "relative_l2", max(result[1] for result in first_results + second_results), flush=True)
        del first, second, x, r1, r2, result
    assert len(captures) == 1, captures
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 8], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [8, 0], counts
    print("multi-output FX stats:", counts, "max_row_relative_l2:", max(errors),
          "max_cpu_fp32_floor:", max(floors), "max_abs:", max(absolutes),
          "output sha256:", hashlib.sha256(payload).hexdigest(), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    del weights
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("multi-output FX two-layer device audit passed", flush=True)


if __name__ == "__main__":
    main()
