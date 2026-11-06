"""Verify that disjoint split outputs share one actual Synapse split node."""

import ctypes
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    counter = ctypes.CDLL(None).gaudirt_test_split_node_count
    counter.restype = ctypes.c_uint64
    torch.random.default_generator.manual_seed(824)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=4 * 1024**2)
    program = gaudirt._C.BucketProgram(
        [1], [[4, 0, 0, 4, 16], [4, 0, 4, 6, 16], [4, 0, 10, 6, 16], [15, 1, 2, 3]], [4]
    )
    for rows, expected_nodes in ((3, 1), (3, 0), (17, 1)):
        value = torch.randn(rows, 16, dtype=torch.bfloat16)
        before = counter()
        actual = program.run([value.to("gaudirt:0")])[0].cpu()
        assert torch.equal(actual, value), (rows, actual, value)
        assert counter() - before == expected_nodes, (rows, counter() - before)

    overlapping = gaudirt._C.BucketProgram(
        [1], [[4, 0, 0, 8, 16], [4, 0, 4, 8, 16], [15, 1, 2]], [3]
    )
    value = torch.randn(3, 16, dtype=torch.bfloat16)
    before = counter()
    actual = overlapping.run([value.to("gaudirt:0")])[0].cpu()
    assert torch.equal(actual, torch.cat((value[:, :8], value[:, 4:12]), dim=-1))
    assert counter() - before == 2, counter() - before
    print("disjoint QKV uses one split; overlapping slices preserve their values", flush=True)


if __name__ == "__main__":
    main()
