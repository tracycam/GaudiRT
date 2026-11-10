"""Check dynamic-token/channel FP8 FX buckets against independent CPU FP64."""

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
    return torch.ops.gaudirt.fp8_linear.default(x, weight, scale, bias)


def without_bias(x, weight, scale):
    return torch.ops.gaudirt.fp8_linear.default(x, weight, scale)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(3026)
    n, k = 256, 256
    original = torch.randn(n, k).clamp(-4, 4).to(torch.float8_e4m3fn)
    original_scale = (torch.rand(n) + 0.125).float()
    encoded, scales, _ = reencode_e4m3fn_half(original, original_scale)
    reference_weight = original.double() * original_scale.double().unsqueeze(1)
    bias = torch.randn(n, dtype=torch.bfloat16)
    gaudirt.initialize(hbm_bytes=512 * 1024 * 1024, pinned_bytes=8 * 1024 * 1024)
    w_device, s_device, b_device = (value.to("gaudirt:0") for value in (encoded, scales, bias))
    compiled_bias = compiler.compile(with_bias, dynamic_arg_indices=(0,))
    compiled_no_bias = compiler.compile(without_bias, dynamic_arg_indices=(0,))
    outputs, metrics = bytearray(), []

    def check(input_cpu, result, expected_bias):
        flat = input_cpu.float().reshape(-1, k)
        act_scale = flat.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
        original_q = (flat / act_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        reference = (original_q.double() * act_scale.double()) @ reference_weight.T
        unquantized = input_cpu.double().reshape(-1, k) @ reference_weight.T
        if expected_bias is not None:
            reference += expected_bias.double()
            unquantized += expected_bias.double()
        result_bf16 = result.cpu()
        actual = result_bf16.double().reshape(-1, n)
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all() and torch.isfinite(unquantized).all()

        def row_error(target):
            denominator = torch.linalg.vector_norm(target, dim=-1)
            assert (denominator > 0).all()
            return (torch.linalg.vector_norm(actual - target, dim=-1) / denominator).max().item()

        error, diagnostic = row_error(reference), row_error(unquantized)
        print("FP8 channel bucket", tuple(input_cpu.shape), "fp64_ocp_w8a8_l2", error,
              "fp64_unquantized_activation_l2", diagnostic, flush=True)
        assert error <= 0.02, (error, diagnostic)
        metrics.append((error, diagnostic))
        outputs.extend(result_bf16.view(torch.int16).numpy().tobytes())

    for m in (1, 3, 17, 137, 513, 4096):
        x_cpu = torch.randn(m, k).to(torch.bfloat16)
        if m == 3:
            x_cpu[0].zero_()
        result = compiled_bias(x_cpu.to("gaudirt:0"), w_device, s_device, b_device)
        assert result.shape == (m, n)
        check(x_cpu, result, bias)
    x3_cpu = torch.randn(3, 2, k).to(torch.bfloat16)
    output3 = compiled_no_bias(x3_cpu.to("gaudirt:0"), w_device, s_device)
    assert output3.shape == (3, 2, n)
    check(x3_cpu, output3, None)
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[1] == 0, stats
    print("FP8 channel buckets", "cache", stats,
          "fp64_ocp_w8a8_max_l2", max(value[0] for value in metrics),
          "fp64_unquantized_activation_max_l2", max(value[1] for value in metrics),
          "sha256", hashlib.sha256(outputs).hexdigest(), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(outputs)
    del result, output3, w_device, s_device, b_device
    gaudirt.empty_cache()
    gaudirt.shutdown()
    assert "habana_frameworks" not in sys.modules
    assert "libhabana_pytorch" not in open("/proc/self/maps").read()


if __name__ == "__main__":
    main()
