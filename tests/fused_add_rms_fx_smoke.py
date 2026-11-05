"""Fused residual BF16 boundary in eager and connected two-layer FX paths."""

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
torch.random.default_generator.manual_seed(713)
EPSILON = 1e-6
EPSILON_FP32 = torch.tensor(EPSILON, dtype=torch.float32).item()


def layer(x, residual, gamma, weight):
    normalized, updated = torch.ops.gaudirt.fused_add_rms_norm.default(x, residual, gamma, EPSILON)
    return torch.nn.functional.linear(normalized, weight), updated


def reference(x, residual, gamma, weight, fp64):
    dtype = torch.float64 if fp64 else torch.float32
    rounded = direct_bf16 if fp64 else lambda value: value.to(torch.bfloat16).float()
    x, residual, gamma = (value.to(dtype) for value in (x, residual, gamma))
    updated = rounded(x + residual)
    normalized = rounded(updated * torch.rsqrt((updated * updated).mean(-1, keepdim=True) + EPSILON_FP32) * gamma)
    result = normalized if weight is None else rounded(normalized @ weight.to(dtype).t())
    return result.double(), updated.double()


def check(outputs, x, residual, gamma, weight=None):
    ref64 = reference(x, residual, gamma, weight, True)
    ref32 = reference(x, residual, gamma, weight, False)
    checked = []
    for index, (output, truth, baseline) in enumerate(zip(outputs, ref64, ref32)):
        assert output.shape == x.shape and output.is_contiguous()
        bucket = 1 << (x.size(0) - 1).bit_length()
        assert output.untyped_storage().nbytes() >= bucket * x.size(1) * 2
        assert output.untyped_storage().nbytes() % 256 == 0
        actual = output.cpu().double()
        assert torch.isfinite(actual).all() and torch.isfinite(truth).all() and torch.isfinite(baseline).all()
        if index == 1:
            assert torch.equal(actual, truth), "BF16 residual boundary differs"
        denominator = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-30)
        error = (torch.linalg.vector_norm(actual - truth, dim=-1) / denominator).max().item()
        floor = (torch.linalg.vector_norm(baseline - truth, dim=-1) / denominator).max().item()
        absolute = (actual - truth).abs().max().item()
        assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (x.size(0), error, floor, absolute)
        checked.append((actual.to(torch.bfloat16).view(torch.int16).numpy().tobytes(), error, floor, absolute))
    return checked


def main():
    gaudirt.initialize(hbm_bytes=512 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    original = compiler.gaudirt_backend
    captures = []

    def capture(graph_module, example_inputs):
        captures.append(1)
        return original(graph_module, example_inputs)

    compiler.gaudirt_backend = capture
    compiled = compiler.compile(layer, dynamic_arg_indices=(0, 1))
    gammas_cpu = [(torch.randn(64) * 0.1 + 1).to(torch.bfloat16) for _ in range(2)]
    weights_cpu = [(torch.randn(64, 64) * 0.125).to(torch.bfloat16) for _ in range(2)]
    gammas = [gamma.to("gaudirt:0") for gamma in gammas_cpu]
    weights = [weight.to("gaudirt:0") for weight in weights_cpu]
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for rows in (1, 3, 17, 65, 137, 513, 2048, 4096):
        x_cpu, residual_cpu = [torch.randn(rows, 64, dtype=torch.bfloat16) for _ in range(2)]
        x, residual = x_cpu.to("gaudirt:0"), residual_cpu.to("gaudirt:0")
        first = compiled(x, residual, gammas[0], weights[0])
        first_results = check(first, x_cpu, residual_cpu, gammas_cpu[0], weights_cpu[0])
        before_second = gaudirt.recipe_cache_stats()
        second = compiled(first[0], first[1], gammas[1], weights[1])
        assert gaudirt.recipe_cache_stats() == before_second, "isomorphic fused layer recompiled"
        second_results = check(second, first[0].cpu(), first[1].cpu(), gammas_cpu[1], weights_cpu[1])
        for result in first_results + second_results:
            payload.extend(result[0]); errors.append(result[1]); floors.append(result[2]); absolutes.append(result[3])
        print("FX rows", rows, "stats", before_second, flush=True)
        del first, second, x, residual, result
    assert len(captures) == 1, captures
    del gammas, weights

    gamma_cpu = (torch.randn(7168) * 0.1 + 1).to(torch.bfloat16)
    gamma = gamma_cpu.to("gaudirt:0")
    for rows in (1, 7, 33, 137, 1025):
        x_cpu, residual_cpu = [torch.randn(rows, 7168, dtype=torch.bfloat16) for _ in range(2)]
        x, residual = x_cpu.to("gaudirt:0"), residual_cpu.to("gaudirt:0")
        first = torch.ops.gaudirt.fused_add_rms_norm.default(x, residual, gamma, EPSILON)
        first_results = check(first, x_cpu, residual_cpu, gamma_cpu)
        before_second = gaudirt.recipe_cache_stats()
        second = torch.ops.gaudirt.fused_add_rms_norm.default(first[0], first[1], gamma, EPSILON)
        assert gaudirt.recipe_cache_stats() == before_second, "same-geometry eager fused norm recompiled"
        second_results = check(second, first[0].cpu(), first[1].cpu(), gamma_cpu)
        for result in first_results + second_results:
            payload.extend(result[0]); errors.append(result[1]); floors.append(result[2]); absolutes.append(result[3])
        print("target width 7168 rows", rows, "stats", before_second,
              "relative_l2", max(result[1] for result in first_results + second_results), flush=True)
        if rows == 7:
            try:
                torch.ops.gaudirt.fused_add_rms_norm.default(x, residual, gamma, 0.0)
            except RuntimeError as error:
                assert "epsilon" in str(error)
            else:
                raise AssertionError("invalid epsilon was accepted")
        del first, second, x, residual, result
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 13], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [13, 0], counts
    print("fused RMSNorm stats:", counts, "max_row_relative_l2:", max(errors),
          "max_cpu_fp32_floor:", max(floors), "max_abs:", max(absolutes),
          "output sha256:", hashlib.sha256(payload).hexdigest(), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    del gamma
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("fused RMSNorm eager and FX device audit passed", flush=True)


if __name__ == "__main__":
    main()
