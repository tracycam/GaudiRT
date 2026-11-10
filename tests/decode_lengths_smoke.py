"""Decode length updates preserve old batches without scalar DMA."""

import ctypes
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=4 * 1024**2)
    counter = ctypes.CDLL(None)
    for name in ("gaudirt_test_launch_count", "gaudirt_test_h2d_copy_count", "gaudirt_test_d2h_copy_count"):
        getattr(counter, name).restype = ctypes.c_uint64
    for count in (1, 3, 16):
        seq = torch.arange(count, dtype=torch.int64) * (2**33) + 17
        orig = torch.arange(count, dtype=torch.int32) + 31
        device_seq, device_orig = seq.to("gaudirt"), orig.to("gaudirt")
        torch.ops.gaudirt._advance_decode_lengths(device_seq, device_orig)
        before = tuple(getattr(counter, name)() for name in (
            "gaudirt_test_launch_count", "gaudirt_test_h2d_copy_count", "gaudirt_test_d2h_copy_count"))
        next_seq, next_orig = torch.ops.gaudirt._advance_decode_lengths(device_seq, device_orig)
        after = tuple(getattr(counter, name)() for name in (
            "gaudirt_test_launch_count", "gaudirt_test_h2d_copy_count", "gaudirt_test_d2h_copy_count"))
        assert tuple(b - a for a, b in zip(before, after)) == (1, 0, 0), (before, after)
        assert torch.equal(next_seq.cpu(), seq + 1) and torch.equal(next_orig.cpu(), orig + 1)
        assert torch.equal(device_seq.cpu(), seq) and torch.equal(device_orig.cpu(), orig)
    print("decode lengths: one launch, no H2D/D2H, old batches preserved", flush=True)


if __name__ == "__main__":
    main()
