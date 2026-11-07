"""Check dynamic buckets against a reference rounded at each stored op boundary.

This is a compiler and storage-boundary smoke test, not a full decoder-layer
numerical gate: the latter requires pure FP32 or FP64 inside the entire layer.
"""

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
torch.random.default_generator.manual_seed(903)
EPSILON = 1e-6
EPSILON_FP32 = torch.tensor(EPSILON, dtype=torch.float32).item()


def layer(x, gamma, weight, bias, residual):
    normalized = torch.ops.gaudirt.rms_norm.default(x, gamma, EPSILON)
    return torch.nn.functional.linear(normalized, weight, bias) + residual


def same_layer(x, gamma, weight, bias, residual):
    normalized = torch.ops.gaudirt.rms_norm.default(x, gamma, EPSILON)
    return torch.nn.functional.linear(normalized, weight, bias) + residual


def different_epsilon(x, gamma, weight, bias, residual):
    normalized = torch.ops.gaudirt.rms_norm.default(x, gamma, 0.5)
    return torch.nn.functional.linear(normalized, weight, bias) + residual


def check_norm(x, gamma, gamma_cpu):
    actual = torch.ops.gaudirt.rms_norm.default(x, gamma, EPSILON).cpu().double()
    source = x.cpu()
    x64, g64 = source.double(), gamma_cpu.double()
    truth = direct_bf16(x64 * torch.rsqrt((x64 * x64).mean(-1, keepdim=True) + EPSILON_FP32) * g64)
    x32, g32 = source.float(), gamma_cpu.float()
    floor = (x32 * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + EPSILON_FP32) * g32).to(torch.bfloat16).double()
    assert all(torch.isfinite(value).all() for value in (actual, truth, floor))
    denom = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-30)
    error = (torch.linalg.vector_norm(actual - truth, dim=-1) / denom).max().item()
    floor_error = (torch.linalg.vector_norm(floor - truth, dim=-1) / denom).max().item()
    assert error <= max(3e-3, 2 * floor_error) and error <= 3e-3, (error, floor_error)


def check_stored_boundaries(output, x, gamma, weight, bias, residual, epsilon=EPSILON_FP32):
    actual = output.cpu().double()
    x64, g64 = x.double(), gamma.double()
    norm64 = direct_bf16(x64 * torch.rsqrt((x64 * x64).mean(-1, keepdim=True) + epsilon) * g64)
    linear64 = direct_bf16(norm64 @ weight.double().t() + bias.double())
    reference64 = direct_bf16(linear64 + residual.double())
    x32, g32 = x.float(), gamma.float()
    norm32 = (x32 * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + epsilon) * g32).to(torch.bfloat16)
    linear32 = (norm32.float() @ weight.float().t() + bias.float()).to(torch.bfloat16)
    reference32 = (linear32.float() + residual.float()).to(torch.bfloat16).double()
    assert torch.isfinite(actual).all() and torch.isfinite(reference64).all() and torch.isfinite(reference32).all()
    denom = torch.linalg.vector_norm(reference64, dim=-1).clamp_min(1e-30)
    error = (torch.linalg.vector_norm(actual - reference64, dim=-1) / denom).max().item()
    floor = (torch.linalg.vector_norm(reference32 - reference64, dim=-1) / denom).max().item()
    absolute = (actual - reference64).abs().max().item()
    if error > max(3e-3, 2 * floor) or error > 1e-2:
        gpu_x, gpu_gamma, gpu_weight, gpu_bias = [
            value.to("gaudirt:0") for value in (x, gamma, weight, bias)
        ]
        norm_device = torch.ops.gaudirt.rms_norm.default(gpu_x, gpu_gamma, epsilon).cpu()
        linear_device = gaudirt._C.bucket_linear(norm_device.to("gaudirt:0"), gpu_weight, gpu_bias).cpu()
        split_output = (linear_device.float() + residual.float()).to(torch.bfloat16).double()
        bad = (norm_device.double() != norm64).nonzero()[:6]
        print("diagnostic m:", x.size(0), "norm_bit_differences:",
              torch.count_nonzero(norm_device.double() != norm64).item(),
              "combined_vs_separate_bit_differences:", torch.count_nonzero(actual != split_output).item(),
              "norm_max_abs:", (norm_device.double() - norm64).abs().max().item(),
              "samples:", [(idx.tolist(), norm_device[tuple(idx)].item(), norm64[tuple(idx)].item(),
                            norm32[tuple(idx)].item(),
                            (x64 * torch.rsqrt((x64 * x64).mean(-1, keepdim=True) + epsilon) * g64)
                            [tuple(idx)].item(),
                            (x32 * torch.rsqrt((x32 * x32).mean(-1, keepdim=True) + epsilon) * g32)
                            [tuple(idx)].item()) for idx in bad], flush=True)
    assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (error, floor, absolute)
    return actual.to(torch.bfloat16).view(torch.int16).numpy().tobytes(), error, floor, absolute


def main():
    gaudirt.initialize(hbm_bytes=128 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    original = compiler.gaudirt_backend
    captures = []

    def capture(graph_module, example_inputs):
        captures.append(1)
        return original(graph_module, example_inputs)

    compiler.gaudirt_backend = capture
    compiled = compiler.compile(layer, dynamic_arg_indices=(0, 4))
    weights_cpu = [torch.randn(64, 64, dtype=torch.bfloat16) for _ in range(2)]
    gamma_cpu = [torch.randn(64, dtype=torch.bfloat16) for _ in range(2)]
    bias_cpu = [torch.randn(64, dtype=torch.bfloat16) for _ in range(2)]
    weights = [value.to("gaudirt:0") for value in weights_cpu]
    gammas = [value.to("gaudirt:0") for value in gamma_cpu]
    biases = [value.to("gaudirt:0") for value in bias_cpu]
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for m in (1, 3, 17, 65, 137, 513, 2048, 4096):
        x_cpu, r1_cpu, r2_cpu = [torch.randn(m, 64, dtype=torch.bfloat16) for _ in range(3)]
        x, r1, r2 = [value.to("gaudirt:0") for value in (x_cpu, r1_cpu, r2_cpu)]
        first = compiled(x, gammas[0], weights[0], biases[0], r1)
        before_second = gaudirt.recipe_cache_stats()
        assert first.shape == (m, 64) and first.untyped_storage().nbytes() % 256 == 0
        data, error, floor, absolute = check_stored_boundaries(first, x_cpu, gamma_cpu[0], weights_cpu[0], bias_cpu[0], r1_cpu)
        payload.extend(data); errors.append(error); floors.append(floor); absolutes.append(absolute)
        if m == 17:
            probe_x, probe_r, probe_output = x, r1, first.cpu()
        if m == 65:
            midpoint_x = x
        second = compiled(first, gammas[1], weights[1], biases[1], r2)
        assert gaudirt.recipe_cache_stats() == before_second, "isomorphic layer recompiled"
        data, error, floor, absolute = check_stored_boundaries(second, first.cpu(), gamma_cpu[1], weights_cpu[1], bias_cpu[1], r2_cpu)
        payload.extend(data); errors.append(error); floors.append(floor); absolutes.append(absolute)
    assert len(captures) == 1, captures
    before = gaudirt.recipe_cache_stats()
    other = compiler.compile(same_layer, dynamic_arg_indices=(0, 4))(probe_x, gammas[0], weights[0], biases[0], probe_r)
    assert gaudirt.recipe_cache_stats() == before and len(captures) == 2
    assert torch.equal(other.cpu().view(torch.int16), probe_output.view(torch.int16))
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert before[:2] == [0, 8], before
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert before[:2] == [8, 0], before
    print("norm program cache stats:", before, "max_row_relative_l2:", max(errors),
          "max_cpu_fp32_per_op_stored_boundary_error:", max(floors), "max_abs:", max(absolutes),
          "output sha256:", hashlib.sha256(payload).hexdigest(), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    variant = compiler.compile(different_epsilon, dynamic_arg_indices=(0, 4))(
        probe_x, gammas[0], weights[0], biases[0], probe_r
    )
    after_variant = gaudirt.recipe_cache_stats()
    assert after_variant[2] == before[2] + 1 and len(captures) == 3
    assert not torch.equal(variant.cpu().view(torch.int16), probe_output.view(torch.int16))
    check_stored_boundaries(variant, probe_x.cpu(), gamma_cpu[0], weights_cpu[0], bias_cpu[0], probe_r.cpu(), epsilon=0.5)
    for epsilon in (0.0, -1.0, float("inf"), float("nan")):
        try:
            torch.ops.gaudirt.rms_norm.default(probe_x, gammas[0], epsilon)
        except RuntimeError as error:
            assert "epsilon" in str(error)
        else:
            raise AssertionError("invalid RMSNorm epsilon accepted")
    check_norm(probe_x, gammas[0], gamma_cpu[0])
    check_norm(midpoint_x, gammas[0], gamma_cpu[0])
    for invalid_m in (0, 4097):
        invalid = torch.empty((invalid_m, 64), dtype=torch.bfloat16, device="gaudirt:0")
        try:
            torch.ops.gaudirt.rms_norm.default(invalid, gammas[0], EPSILON)
        except RuntimeError as error:
            assert "token count" in str(error)
        else:
            raise AssertionError("invalid token count accepted")
    del x, r1, r2, first, second, probe_x, probe_r, midpoint_x, other, variant, invalid
    del weights, gammas, biases
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("connected FX RMSNorm programs passed", flush=True)


if __name__ == "__main__":
    main()
