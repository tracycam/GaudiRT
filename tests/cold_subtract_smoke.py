"""Cold device subtraction for scheduler integer metadata and strided tensors."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    payload = bytearray()
    for dtype in (torch.int32, torch.int64, torch.float32, torch.bfloat16):
        if dtype == torch.int64:
            cpu = torch.tensor([0, 1, -1, 2**40, -(2**40), 2**63 - 1, -(2**63)], dtype=dtype)
            scalars = (1, -2, -(2**63))
        else:
            cpu = torch.tensor([0, 1, -1, 3, -3, 126, -127], dtype=dtype)
            scalars = (1, -2)
        for scalar in scalars:
            result = (cpu.to("gaudirt:0") - scalar).cpu()
            expected = cpu - scalar
            assert torch.equal(result, expected), (dtype, scalar, result, expected)
            bits = result.view(torch.int16) if dtype == torch.bfloat16 else result
            payload.extend(bits.numpy().tobytes())
        grid = torch.arange(42, dtype=torch.int32).reshape(3, 14).to(dtype)
        offset = torch.tensor([[1], [3], [7]], dtype=dtype)
        device_grid = grid.to("gaudirt:0")
        device_offset = offset.to("gaudirt:0")
        result = device_grid[:, ::2] - device_offset
        expected = grid[:, ::2] - offset
        assert torch.equal(result.cpu(), expected), dtype
        output = torch.empty((3, 7), dtype=dtype, device="gaudirt:0")
        torch.sub(device_grid[:, ::2], device_offset, out=output)
        assert torch.equal(output.cpu(), expected), dtype
    try:
        torch.sub(torch.ones(3, dtype=torch.int64, device="gaudirt:0"), 1, alpha=2)
    except RuntimeError as error:
        assert "alpha=1" in str(error), str(error)
    else:
        raise AssertionError("unsupported subtraction alpha was accepted")
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[0] == 0 and counts[1] > 0, counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[0] > 0 and counts[1] == 0, counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("cold subtraction stats", counts, "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
