"""Fill, arange and checked device indexing reuse their stream events."""

import ctypes
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    counter = ctypes.CDLL(None)
    created = counter.gaudirt_test_event_create_count
    destroyed = counter.gaudirt_test_event_destroy_count
    created.restype = destroyed.restype = ctypes.c_uint64
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    streams = [gaudirt.Stream(), gaudirt.Stream()]
    cpu_source = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    cpu_index = torch.tensor([0, 3, 7, 12], dtype=torch.int64)
    source, index = cpu_source.to("gaudirt:0"), cpu_index.to("gaudirt:0")
    filled = [torch.empty((13, 19), dtype=torch.int32, device="gaudirt:0") for _ in streams]
    ranges = [torch.empty((64,), dtype=torch.int32, device="gaudirt:0") for _ in streams]
    gaudirt.synchronize()

    def exercise(number):
        filled[number].fill_(number + 7)
        torch.arange(0, 64, 1, out=ranges[number])
        selected = torch.index_select(source, 0, index)
        assert torch.equal(selected.cpu(), cpu_source[cpu_index])

    for number, stream in enumerate(streams):
        with gaudirt.stream(stream):
            exercise(number)
            exercise(number)
    gaudirt.synchronize()
    before_create, before_destroy = created(), destroyed()
    for iteration in range(8):
        number = iteration % 2
        with gaudirt.stream(streams[number]):
            exercise(number)
    gaudirt.synchronize()
    assert created() == before_create, (before_create, created())
    assert destroyed() == before_destroy, (before_destroy, destroyed())
    for number in range(2):
        assert torch.equal(filled[number].cpu(), torch.full((13, 19), number + 7, dtype=torch.int32))
        assert torch.equal(ranges[number].cpu(), torch.arange(64, dtype=torch.int32))
    print("cold fill/arange/index checks: 8 steps, event creations 0 destructions 0", flush=True)


if __name__ == "__main__":
    main()
