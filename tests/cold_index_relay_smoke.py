"""Device request-indexed INT64 relay with fail-fast bounds checking."""

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
    for count in (1, 7, 33, 137):
        for index_dtype in (torch.int32, torch.int64):
            cpu_base = torch.arange(257, dtype=torch.int64) - 99
            cpu_indices = (torch.arange(count, dtype=index_dtype) * 17 + 1) % 257
            cpu_values = torch.arange(count, dtype=torch.int64) * 41 - 2**40
            storage = torch.empty(514, dtype=torch.int64, device="gaudirt:0")
            device_base = storage[::2]
            device_base.copy_(cpu_base)
            index_storage = torch.empty(count * 2, dtype=index_dtype, device="gaudirt:0")
            indices = index_storage[::2]
            indices.copy_(cpu_indices)
            value_storage = torch.empty(count * 2, dtype=torch.int64, device="gaudirt:0")
            values = value_storage[::2]
            values.copy_(cpu_values)
            device_base[indices] = values
            expected = cpu_base.clone()
            expected[cpu_indices.long()] = cpu_values
            actual = device_base.cpu()
            assert torch.equal(actual, expected), (count, index_dtype, "write")
            gathered = device_base[indices].cpu()
            assert torch.equal(gathered, cpu_values), (count, index_dtype, "read")
            payload.extend(actual.numpy().tobytes())
            payload.extend(gathered.numpy().tobytes())

    cpu_base = torch.arange(32, dtype=torch.int64)
    indices_cpu = torch.tensor([1, 3, 5, 7, 9, 11, 13], dtype=torch.int64)
    expected = cpu_base.clone()
    expected[indices_cpu] = cpu_base[:7].clone()
    device_base, indices = cpu_base.to("gaudirt:0"), indices_cpu.to("gaudirt:0")
    device_base[indices] = device_base[:7]
    assert torch.equal(device_base.cpu(), expected), "overlapping source"
    scalar = torch.tensor(11, dtype=torch.int64).to("gaudirt:0")
    device_base[indices] = scalar
    expected[indices_cpu] = 11
    assert torch.equal(device_base.cpu(), expected), "device scalar broadcast"
    device_base[indices] = -7
    expected[indices_cpu] = -7
    assert torch.equal(device_base.cpu(), expected), "host scalar broadcast"
    for invalid in (-1, 32, 2**32):
        before = device_base.cpu()
        ids = torch.tensor([invalid], dtype=torch.int64).to("gaudirt:0")
        value = torch.tensor([100], dtype=torch.int64).to("gaudirt:0")
        try:
            device_base[ids] = value
        except RuntimeError as error:
            assert "index out of bounds" in str(error), str(error)
        else:
            raise AssertionError("invalid scatter index was accepted")
        assert torch.equal(device_base.cpu(), before), "invalid write changed the destination"
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[0] == 0 and counts[1] > 0, counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[0] > 0 and counts[1] == 0, counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("device relay stats", counts, "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
