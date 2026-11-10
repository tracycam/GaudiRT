"""Check the raw FP8 checkpoint transfers required by merged-weight loading."""

import sys

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=64 * 1024 * 1024)
    raw = torch.arange(96, dtype=torch.uint8).reshape(6, 16)
    weight = raw.view(torch.float8_e4m3fn)
    device = torch.empty(weight.shape, dtype=weight.dtype, device="gaudirt:0")
    device.copy_(weight)
    assert torch.equal(device.cpu().view(torch.uint8), raw)

    replacement = torch.arange(40, dtype=torch.uint8).reshape(5, 8).view(torch.float8_e4m3fn)
    device.narrow(0, 1, 5).narrow(1, 4, 8).copy_(replacement)
    expected = raw.clone()
    expected[1:6, 4:12] = replacement.view(torch.uint8)
    assert torch.equal(device.cpu().view(torch.uint8), expected)

    source_scales = torch.tensor([[0.5, 0.25], [0.125, 0.0625]], dtype=torch.bfloat16)
    scales = torch.empty(source_scales.shape, dtype=torch.float32, device="gaudirt:0")
    scales.copy_(source_scales)
    assert torch.equal(scales.cpu(), source_scales.float())
    del device, scales
    gaudirt.synchronize()
    gaudirt.shutdown()
    assert "habana_frameworks" not in sys.modules
    print("FP8 raw storage, strided loading, and BF16-to-FP32 scale copy passed")


if __name__ == "__main__":
    main()
