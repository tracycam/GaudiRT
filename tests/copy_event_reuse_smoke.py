"""Pinned and pageable copies reuse cross-stream events after warmup."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import threading

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
    cases = []
    for index in range(2):
        cpu = torch.arange(13 * 19, dtype=torch.float32).reshape(13, 19) + index * 1000
        pinned_in = torch.empty_like(cpu, pin_memory=True)
        pinned_in.copy_(cpu)
        pinned_out = torch.empty_like(cpu, pin_memory=True)
        host_out = torch.empty_like(cpu)
        device = torch.empty((13, 38), dtype=torch.float32, device="gaudirt:0")[:, ::2]
        contiguous = torch.empty_like(cpu, device="gaudirt:0")
        cases.append((cpu, pinned_in, pinned_out, host_out, device, contiguous))

    def exercise(index):
        cpu, pinned_in, pinned_out, host_out, device, contiguous = cases[index]
        contiguous.copy_(pinned_in, non_blocking=True)
        pinned_out.copy_(contiguous, non_blocking=True)
        contiguous.copy_(cpu, non_blocking=True)
        host_out.copy_(contiguous, non_blocking=False)
        device.copy_(pinned_in, non_blocking=True)
        pinned_out.copy_(device, non_blocking=True)
        device.copy_(cpu, non_blocking=True)
        host_out.copy_(device, non_blocking=False)

    for index, stream in enumerate(streams):
        with gaudirt.stream(stream):
            exercise(index)
    gaudirt.synchronize()
    before_create, before_destroy = created(), destroyed()
    for iteration in range(32):
        index = iteration % 2
        with gaudirt.stream(streams[index]):
            exercise(index)
    gaudirt.synchronize()
    assert created() == before_create, (before_create, created())
    assert destroyed() == before_destroy, (before_destroy, destroyed())
    for cpu, _, pinned_out, host_out, _, _ in cases:
        assert torch.equal(cpu, pinned_out)
        assert torch.equal(cpu, host_out)

    def concurrent_round(steps):
        gate = threading.Barrier(2)

        def worker(index):
            with gaudirt.stream(streams[0]):
                gate.wait(timeout=30)
                for _ in range(steps):
                    exercise(index)
                    assert torch.equal(cases[index][0], cases[index][3])

        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(worker, range(2)))

    concurrent_round(2)
    gaudirt.synchronize()
    before_create, before_destroy = created(), destroyed()
    concurrent_round(32)
    gaudirt.synchronize()
    assert created() == before_create, (before_create, created())
    assert destroyed() == before_destroy, (before_destroy, destroyed())
    print("copy steps 32 and two-thread same-stream steps 64, pinned/pageable contiguous/strided H2D/D2H,",
          "event creations 0 destructions 0", flush=True)


if __name__ == "__main__":
    main()
