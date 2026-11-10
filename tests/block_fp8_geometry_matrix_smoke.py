"""Check both bias modes and all requested row buckets for one FP8 geometry."""

import argparse
import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
from gaudi_kernels.fp8_codec import reencode_e4m3fn_half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("output")
    args = parser.parse_args()
    n, k = args.n, args.k
    if n <= 0 or k <= 0 or n % 128 or k % 128:
        raise ValueError("N and K must be positive multiples of 128")
    torch.set_num_threads(8)
    torch.random.default_generator.manual_seed(61492)
    original = torch.randn(n, k).clamp_(-4, 4).to(torch.float8_e4m3fn)
    original_scale = torch.rand(n // 128, k // 128, dtype=torch.float32) * 0.1 + 0.01
    bias = torch.randn(n, dtype=torch.bfloat16)
    encoded, adjusted, _ = reencode_e4m3fn_half(original, original_scale)
    packed = encoded.view(torch.uint8).reshape(n, k // 128, 128).permute(1, 0, 2).contiguous()
    packed = packed.view(torch.float8_e4m3fn)
    del encoded
    gaudirt.initialize(hbm_bytes=3 * 1024**3, pinned_bytes=8 * 1024**2)
    w_device, s_device, b_device = (value.to("gaudirt:0") for value in (packed, adjusted, bias))
    digest = hashlib.sha256()
    with open(args.output, "wb") as target:
        for m in (1, 16, 128, 1024, 4096):
            x = torch.randn(m, k).to(torch.bfloat16)
            device_x = x.to("gaudirt:0")
            no_bias = torch.ops.gaudirt.block_fp8_linear.default(device_x, w_device, s_device).cpu()
            with_bias = torch.ops.gaudirt.block_fp8_linear.default(device_x, w_device, s_device, b_device).cpu()
            assert no_bias.shape == with_bias.shape == (m, n)
            group = x.float().reshape(m, k // 128, 128)
            act_scale = group.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
            q = (group / act_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
            qx = (q.double() * act_scale.double()).reshape(m, k)
            raw_x = x.double()
            errors = torch.zeros((2, m), dtype=torch.float64)
            energy = torch.zeros_like(errors)
            unquant_errors = torch.zeros_like(errors)
            unquant_energy = torch.zeros_like(errors)
            for start in range(0, n, 512):
                width = min(512, n - start)
                decoded = (original[start:start + width].double().reshape(width // 128, 128, k // 128, 128) *
                           original_scale[start // 128:(start + width) // 128].double().reshape(
                               width // 128, 1, k // 128, 1)).reshape(width, k)
                reference, unquantized = qx @ decoded.t(), raw_x @ decoded.t()
                for index, result in enumerate((no_bias, with_bias)):
                    addition = bias[start:start + width].double() if index else 0
                    expected, raw_expected = reference + addition, unquantized + addition
                    actual = result[:, start:start + width].double()
                    assert torch.isfinite(actual).all() and torch.isfinite(expected).all() and torch.isfinite(raw_expected).all()
                    errors[index] += (actual - expected).square().sum(-1)
                    energy[index] += expected.square().sum(-1)
                    unquant_errors[index] += (actual - raw_expected).square().sum(-1)
                    unquant_energy[index] += raw_expected.square().sum(-1)
            assert (energy > 0).all() and (unquant_energy > 0).all()
            for index, result in enumerate((no_bias, with_bias)):
                main_error = torch.sqrt(errors[index] / energy[index]).max().item()
                diagnostic = torch.sqrt(unquant_errors[index] / unquant_energy[index]).max().item()
                assert main_error <= 0.02, (m, index, main_error, diagnostic)
                payload = result.view(torch.int16).numpy().tobytes()
                target.write(payload)
                digest.update(payload)
                print(f"M={m} N={n} K={k} bias={index} fp64_ocp_w8a8_l2={main_error:.9g} "
                      f"fp64_unquantized_activation_l2={diagnostic:.9g}", flush=True)
            del device_x, no_bias, with_bias
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[1] == 0, stats
    print("FP8_GEOMETRY_MATRIX", "cache", stats, "sha256", digest.hexdigest(), flush=True)
    del w_device, s_device, b_device
    gaudirt.empty_cache()
    gaudirt.shutdown()
    assert "habana_frameworks" not in sys.modules


if __name__ == "__main__":
    main()
