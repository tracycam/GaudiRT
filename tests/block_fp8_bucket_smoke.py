"""Exercise prepared block-FP8 weights in dynamic compiled FX programs."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import gaudirt.compiler as compiler
from gaudi_kernels.fp8_codec import reencode_e4m3fn_half


def with_bias(x, weight, scale, bias):
    return torch.ops.gaudirt.block_fp8_linear.default(x, weight, scale, bias)


def without_bias(x, weight, scale):
    return torch.ops.gaudirt.block_fp8_linear.default(x, weight, scale)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(2026)
    n, k, groups = 256, 256, 2
    original = torch.randn(n, k).clamp(-4, 4).to(torch.float8_e4m3fn)
    original_scales = (torch.rand(n // 128, groups) + 0.125).float()
    encoded, scales, _ = reencode_e4m3fn_half(original, original_scales)
    packed = encoded.view(torch.uint8).reshape(n, groups, 128).permute(1, 0, 2).contiguous()
    packed = packed.view(torch.float8_e4m3fn)
    bias = torch.randn(n, dtype=torch.bfloat16)
    reference_weight = (original.double().reshape(n // 128, 128, groups, 128) *
                        original_scales.double().reshape(n // 128, 1, groups, 1)).reshape(n, k)

    gaudirt.initialize(hbm_bytes=512 * 1024 * 1024, pinned_bytes=8 * 1024 * 1024)
    w_device, s_device, b_device = (value.to("gaudirt:0") for value in (packed, scales, bias))
    compiled_bias = compiler.compile(with_bias, dynamic_arg_indices=(0,))
    compiled_no_bias = compiler.compile(without_bias, dynamic_arg_indices=(0,))
    outputs, metrics = bytearray(), []

    def check(input_cpu, result, expected_bias):
        grouped = input_cpu.float().reshape(-1, groups, 128)
        act_scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
        original_q = (grouped / act_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        act64 = (original_q.double() * act_scale.double()).reshape(-1, k)
        reference = act64 @ reference_weight.T
        raw_reference = input_cpu.double().reshape(-1, k) @ reference_weight.T
        if expected_bias is not None:
            reference += expected_bias.double()
            raw_reference += expected_bias.double()
        actual_bf16 = result.cpu()
        actual = actual_bf16.double().reshape(-1, n)
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all() and torch.isfinite(raw_reference).all()
        def row_error(target):
            denominator = torch.linalg.vector_norm(target, dim=-1)
            assert (denominator > 0).all()
            return (torch.linalg.vector_norm(actual - target, dim=-1) / denominator).max().item()
        error, raw_error = row_error(reference), row_error(raw_reference)
        print("FP8 bucket", tuple(input_cpu.shape), "fp64_ocp_w8a8_l2", error,
              "fp64_unquantized_activation_l2", raw_error, flush=True)
        if error > 0.02 and (path := os.environ.get("GAUDIRT_FP8_DIAGNOSTIC")):
            torch.save({"input": input_cpu, "weight": original, "weight_scale": original_scales,
                        "bias": expected_bias, "output": actual_bf16}, path)
        assert error <= 0.02, (error, raw_error)
        metrics.append((error, raw_error))
        outputs.extend(actual_bf16.view(torch.int16).numpy().tobytes())

    for m in (1, 3, 17, 137, 513, 4096):
        x_cpu = torch.randn(m, k).to(torch.bfloat16)
        x = x_cpu.to("gaudirt:0")
        result = compiled_bias(x, w_device, s_device, b_device)
        assert result.shape == (m, n)
        check(x_cpu, result, bias)
    x3_cpu = torch.randn(3, 2, k).to(torch.bfloat16)
    x3 = x3_cpu.to("gaudirt:0")
    output3 = compiled_no_bias(x3, w_device, s_device)
    assert output3.shape == (3, 2, n)
    check(x3_cpu, output3, None)
    stats = gaudirt.recipe_cache_stats()
    mode = os.environ.get("GAUDIRT_TEST_CACHE")
    if mode == "warm":
        assert stats[1] == 0, stats
    print("FP8 buckets", "cache", stats,
          "fp64_ocp_w8a8_max_l2", max(value[0] for value in metrics),
          "fp64_unquantized_activation_max_l2", max(value[1] for value in metrics),
          "sha256", hashlib.sha256(outputs).hexdigest(), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(outputs)
    del result, output3, x, x3, w_device, s_device, b_device
    gaudirt.empty_cache()
    gaudirt.shutdown()
    assert "habana_frameworks" not in sys.modules


if __name__ == "__main__":
    main()
