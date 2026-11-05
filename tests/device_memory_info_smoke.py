"""Device lifecycle and the allocatable pool's memory accounting."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    assert not gaudirt.is_initialized()
    capacity = 256 * 1024 * 1024
    gaudirt.initialize(hbm_bytes=capacity, pinned_bytes=4 * 1024 * 1024)
    assert gaudirt.is_initialized()
    free, total = gaudirt.mem_get_info()
    physical_free, physical_total = gaudirt.physical_mem_get_info()
    assert (free, total) == (capacity, capacity)
    assert 0 <= physical_free <= physical_total and physical_total >= capacity
    value = torch.empty((257,), dtype=torch.uint8, device="gaudirt:0")
    assert gaudirt.memory_allocated() == 512
    assert gaudirt.mem_get_info() == (capacity - 512, capacity)
    assert gaudirt.max_memory_allocated() >= 512
    try:
        gaudirt.mem_get_info("gaudirt:1")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid device was accepted")
    del value
    assert gaudirt.mem_get_info() == (capacity, capacity)
    gaudirt.shutdown()
    assert not gaudirt.is_initialized()
    print("device lifecycle and pool memory accounting passed", flush=True)


if __name__ == "__main__":
    main()
