"""Regression for a BF16 RMSNorm midpoint found in a connected FX graph."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
from fp64_reference import direct_bf16
import gaudirt


def sample(bits):
    return torch.frombuffer(bytearray.fromhex(bits), dtype=torch.bfloat16).clone()


def main():
    x = sample("f2bf093dbb3f46be3a3f7b3f7a3e25bedb3f193ece3ef43ff73ec5bfdf3d8fbe"
               "42bfa73e5dbf553fa1bf0f3f33bf91bfb3bfef3fa6bfa73c893fb6be9b3f55bd"
               "d4bda0bfe53fe13e7b3fa83f0bbfe03f0f40753d0b3f3b3f05bdb03fe53ecb3f"
               "533fb03fb03f50be443f903e03bedfbd803fb83f7fbf14c0bfbf51beb83e7cbd")
    residual = sample("d1bf93bf14c0ee3e7d3d233f793f903f24bf47bf963ee43e103eccbf913fbe3e"
                      "3cbf08bf903dac3feebfef3bfd3e97bf1b3f243f7e3f97bf3bbf93bfa43ffe3f"
                      "fb3e3abe0000833fd4bec0be8cbea3bf84bf523f00004e3d55bf4b3e133f063f"
                      "253f053f72be4ebe353ef2bf2fbf69bfb6bebb3e133f673f3cbc3f3f18bf943d")
    gamma = sample("6d3f7f3f6d3f663f853f943f923f6d3f5a3f983f853f6c3f813f8b3f653f883f"
                   "813f813f823f6b3f7f3f643f653f723f933f703f913f7a3f8a3f543f683f743f"
                   "873f713f763f793f6d3f503f803f703f883f7d3f8e3f913f723f5f3f6e3f853f"
                   "763f823f893f763f653f803f883f7b3f963f6b3f903f723fa33f8a3f963f6f3f")
    assert x.numel() == residual.numel() == gamma.numel() == 64
    boundary = direct_bf16(x.double() + residual.double())
    eps = torch.tensor(1e-6, dtype=torch.float32).item()
    truth = direct_bf16(boundary * torch.rsqrt((boundary * boundary).mean() + eps) * gamma.double())
    assert truth[57] == 1.1796875
    gaudirt.initialize(hbm_bytes=256 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    weight = gamma.to("gaudirt:0")
    for rows in (1, 137):
        left = x.repeat(rows, 1).to("gaudirt:0")
        right = residual.repeat(rows, 1).to("gaudirt:0")
        norm, updated = torch.ops.gaudirt.fused_add_rms_norm.default(left, right, weight, 1e-6)
        plain = torch.ops.gaudirt.rms_norm.default(updated, weight, 1e-6)
        assert torch.equal(updated.cpu().double(), boundary.expand(rows, -1))
        assert torch.equal(norm.cpu().double(), truth.expand(rows, -1))
        assert torch.equal(plain.cpu().double(), truth.expand(rows, -1))
        del left, right, norm, updated, plain
    del weight
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("fused and plain RMSNorm BF16 midpoint regression passed", flush=True)


if __name__ == "__main__":
    main()
