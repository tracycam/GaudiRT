"""Cold device addition for scheduler metadata and strided tensors."""

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
        data = [0, 1, -1, 3, -3, 126, -127]
        if dtype == torch.int64:
            data = [0, 1, -1, 2**40, -(2**40), 2**63 - 1, -(2**63)]
        cpu = torch.tensor(data, dtype=dtype)
        for scalar in (1, -2):
            result = (cpu.to("gaudirt:0") + scalar).cpu()
            assert torch.equal(result, cpu + scalar), (dtype, scalar, result)
            bits = result.view(torch.int16) if dtype == torch.bfloat16 else result
            payload.extend(bits.numpy().tobytes())
        grid = torch.arange(42, dtype=torch.int32).reshape(3, 14).to(dtype)
        offset = torch.tensor([[1], [3], [7]], dtype=dtype)
        device_grid, device_offset = grid.to("gaudirt:0"), offset.to("gaudirt:0")
        expected = grid[:, ::2] + offset
        assert torch.equal((device_grid[:, ::2] + device_offset).cpu(), expected), dtype
        output = torch.empty((3, 7), dtype=dtype, device="gaudirt:0")
        torch.add(device_grid[:, ::2], device_offset, out=output)
        assert torch.equal(output.cpu(), expected), dtype
    try:
        torch.add(torch.ones(3, dtype=torch.int64, device="gaudirt:0"), 1, alpha=2)
    except RuntimeError as error:
        assert "alpha=1" in str(error), str(error)
    else:
        raise AssertionError("unsupported addition alpha was accepted")
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[0] == 0 and counts[1] > 0, counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[0] > 0 and counts[1] == 0, counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("cold addition stats", counts, "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
