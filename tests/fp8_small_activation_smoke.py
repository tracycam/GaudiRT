"""Check block and token FP8 for zero and small nonzero activations."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
from gaudi_kernels.fp8_codec import reencode_e4m3fn_half


def main():
    torch.random.default_generator.manual_seed(915)
    n = k = 256
    original = torch.randn(n, k).clamp(-4, 4).to(torch.float8_e4m3fn)
    block_scale = (torch.rand(n // 128, k // 128) + 0.5).float()
    channel_scale = (torch.rand(n) + 0.5).float()
    block, adjusted_block, _ = reencode_e4m3fn_half(original, block_scale)
    channel, adjusted_channel, _ = reencode_e4m3fn_half(original, channel_scale)
    packed = block.view(torch.uint8).reshape(n, k // 128, 128).permute(1, 0, 2).contiguous()
    packed = packed.view(torch.float8_e4m3fn)
    amplitudes = [1.0, 1e-8, 1e-10, 1e-12, 1e-15, 0.0]
    x = (torch.randn(len(amplitudes), k) * torch.tensor(amplitudes).reshape(-1, 1)).to(torch.bfloat16)
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    device_x = x.to("gaudirt")
    outputs = (
        torch.ops.gaudirt.block_fp8_linear.default(
            device_x, packed.to("gaudirt"), adjusted_block.to("gaudirt")).cpu(),
        torch.ops.gaudirt.fp8_linear.default(
            device_x, channel.to("gaudirt"), adjusted_channel.to("gaudirt")).cpu(),
    )
    weights = (
        (original.double().reshape(n // 128, 128, k // 128, 128) *
         block_scale.double().reshape(n // 128, 1, k // 128, 1)).reshape(n, k),
        original.double() * channel_scale.double().unsqueeze(1),
    )
    data = bytearray()
    for kind, output, weight in zip(("block", "token"), outputs, weights):
        group = x.float().reshape(len(amplitudes), -1, 128 if kind == "block" else k)
        scale = group.abs().amax(-1, keepdim=True) / 448.0
        quantized = (group / torch.where(scale == 0, 1.0, scale)).clamp(-448, 448).to(torch.float8_e4m3fn)
        reference = (quantized.double() * scale.double()).reshape(-1, k) @ weight.t()
        unquantized = x.double() @ weight.t()
        actual = output.double()
        assert all(torch.isfinite(value).all() for value in (actual, reference, unquantized))

        def distances(target):
            norm = torch.linalg.vector_norm(target, dim=-1)
            diff = torch.linalg.vector_norm(actual - target, dim=-1)
            return torch.where(norm == 0, diff, diff / norm)

        error = distances(reference)
        diagnostic = distances(unquantized)
        print(kind, "fp64_ocp_w8a8_l2", error.tolist(),
              "fp64_unquantized_activation_l2", diagnostic.tolist(), flush=True)
        assert (error <= 0.02).all(), (kind, error)
        assert torch.equal(output[-1], torch.zeros_like(output[-1]))
        data.extend(output.view(torch.uint8).numpy().tobytes())
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[1] == 0, stats
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as result:
            result.write(data)
    print("small activations", "cache", stats, "sha256", hashlib.sha256(data).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
