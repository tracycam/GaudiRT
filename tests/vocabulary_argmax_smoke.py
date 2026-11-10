"""Single-recipe vocabulary argmax preserves all PyTorch floating edge cases."""

import ctypes
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=4 * 1024**2)
    launch = ctypes.CDLL(None).gaudirt_test_launch_count
    launch.restype = ctypes.c_uint64
    torch.manual_seed(962)
    for dtype in (torch.bfloat16, torch.float32):
        for width in (1, 3, 127, 128, 129, 4099, 65537, 176131):
            values = torch.randn((8, width), dtype=torch.float32).to(dtype)
            values[0].fill_(-float("inf"))
            values[1].fill_(float("inf"))
            values[2].fill_(0)
            values[2, 0] = -0.0
            if width > 1:
                values[3, 0] = float("inf")
                values[3, width // 2] = float("nan")
                values[4, width - 1] = float("nan")
                values[6].fill_(-100)
                values[6, width // 7] = values[6, width - 1] = 100
                values[7, width // 4] = values[7, width - 1] = float("nan")
            device = values.to("gaudirt:0")
            before = launch()
            actual = torch.argmax(device, dim=-1)
            assert launch() - before == 1
            expected = torch.argmax(values, dim=-1)
            assert torch.equal(actual.cpu(), expected), (dtype, width, actual.cpu(), expected)
    print("BF16/FP32 vocabulary argmax: one scan, ties and nonfinite values passed", flush=True)


if __name__ == "__main__":
    main()
