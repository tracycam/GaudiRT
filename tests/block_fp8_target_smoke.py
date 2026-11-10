"""Check one full-sized block-FP8 linear against two independent FP64 references."""

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
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("k", type=int)
    parser.add_argument("bias", choices=("0", "1"))
    parser.add_argument("output")
    args = parser.parse_args()
    m, n, k = args.m, args.n, args.k
    if not 0 < m <= 4096 or n <= 0 or k <= 0 or n % 128 or k % 128:
        raise ValueError("expected M=1..4096 and positive aligned N/K")
    torch.set_num_threads(8)
    torch.random.default_generator.manual_seed(39012)
    weight = torch.randn(n, k).clamp_(-4, 4).to(torch.float8_e4m3fn)
    weight_scales = torch.rand(n // 128, k // 128, dtype=torch.float32) * 0.1 + 0.01
    x = torch.randn(m, k).to(torch.bfloat16)
    bias = torch.randn(n, dtype=torch.bfloat16) if args.bias == "1" else None
    encoded, adjusted, _ = reencode_e4m3fn_half(weight, weight_scales)
    packed = encoded.view(torch.uint8).reshape(n, k // 128, 128).permute(1, 0, 2).contiguous()
    packed = packed.view(torch.float8_e4m3fn)
    del encoded
    gaudirt.initialize(hbm_bytes=2 * 1024**3, pinned_bytes=8 * 1024**2)
    x_device, w_device, s_device = (value.to("gaudirt:0") for value in (x, packed, adjusted))
    bias_device = None if bias is None else bias.to("gaudirt:0")
    output = torch.ops.gaudirt.block_fp8_linear.default(x_device, w_device, s_device, bias_device)
    result = output.cpu()
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[1] == 0, stats

    grouped = x.float().reshape(m, k // 128, 128)
    scale = grouped.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448.0
    quant = (grouped / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    qx = (quant.double() * scale.double()).reshape(m, k)
    raw_x = x.double()
    errors = torch.zeros(m, dtype=torch.float64)
    ref_energy = torch.zeros_like(errors)
    raw_errors = torch.zeros_like(errors)
    raw_energy = torch.zeros_like(errors)
    for start in range(0, n, 512):
        width = min(512, n - start)
        decoded = (weight[start:start + width].double().reshape(width // 128, 128, k // 128, 128) *
                   weight_scales[start // 128:(start + width) // 128].double().reshape(
                       width // 128, 1, k // 128, 1)).reshape(width, k)
        reference = qx @ decoded.T
        raw_reference = raw_x @ decoded.T
        if bias is not None:
            reference += bias[start:start + width].double()
            raw_reference += bias[start:start + width].double()
        actual = result[:, start:start + width].double()
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all() and torch.isfinite(raw_reference).all()
        errors += (actual - reference).square().sum(-1)
        ref_energy += reference.square().sum(-1)
        raw_errors += (actual - raw_reference).square().sum(-1)
        raw_energy += raw_reference.square().sum(-1)
    assert (ref_energy > 0).all() and (raw_energy > 0).all()
    error = torch.sqrt(errors / ref_energy).max().item()
    diagnostic = torch.sqrt(raw_errors / raw_energy).max().item()
    payload = result.view(torch.int16).numpy().tobytes()
    with open(args.output, "wb") as target:
        target.write(payload)
    print(f"M={m} N={n} K={k} bias={args.bias} fp64_ocp_w8a8_l2={error:.9g} "
          f"fp64_unquantized_activation_l2={diagnostic:.9g} cache={stats} "
          f"sha256={hashlib.sha256(payload).hexdigest()}", flush=True)
    assert error <= 0.02, error
    del output, x_device, w_device, s_device, bias_device
    gaudirt.empty_cache()
    gaudirt.shutdown()
    assert "habana_frameworks" not in sys.modules


if __name__ == "__main__":
    main()
