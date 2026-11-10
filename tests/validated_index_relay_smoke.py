"""The scheduler-owned index path is asynchronous and keeps public checks."""

import ctypes
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    probe = ctypes.CDLL(None)
    launches, d2h = probe.gaudirt_test_launch_count, probe.gaudirt_test_d2h_copy_count
    launches.restype = d2h.restype = ctypes.c_uint64
    for dtype in (torch.int32, torch.int64):
        indices_cpu = torch.tensor([0, 11, 256], dtype=dtype)
        indices = indices_cpu.to("gaudirt:0")
        expected = torch.arange(257, dtype=torch.int64)
        values_cpu = torch.tensor([73, -8, 42], dtype=torch.int64)
        values = values_cpu.to("gaudirt:0")
        target = expected.to("gaudirt:0")
        # Compile and warm the exact graphs before counting steady launches.
        torch.ops.gaudirt._index_select_validated(target, 0, indices)
        torch.ops.gaudirt._index_put_validated_(target, indices, values)
        gaudirt.synchronize()
        before = launches(), d2h()
        result = torch.ops.gaudirt._index_select_validated(target, 0, indices)
        assert (launches() - before[0], d2h() - before[1]) == (1, 0)
        expected[indices_cpu.long()] = values_cpu
        assert torch.equal(result.cpu(), expected[indices_cpu.long()])
        before = launches(), d2h()
        torch.ops.gaudirt._index_put_validated_(target, indices, values)
        assert (launches() - before[0], d2h() - before[1]) == (1, 0)
        assert torch.equal(target.cpu(), expected)
        for bad in (-1, 257):
            invalid = torch.tensor([bad], dtype=dtype).to("gaudirt:0")
            try:
                torch.index_select(target, 0, invalid)
            except RuntimeError as error:
                assert "index out of bounds" in str(error), str(error)
            else:
                raise AssertionError("public index_select accepted an invalid index")
    print("validated relay: one launch, no D2H; public index bounds checked", flush=True)


if __name__ == "__main__":
    main()
