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
    for index in range(4):
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

    for index in range(len(cases)):
        with gaudirt.stream(streams[index % len(streams)]):
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

    def concurrent_round(steps, mixed_streams):
        gate = threading.Barrier(len(cases))

        def worker(index):
            with gaudirt.stream(streams[index % len(streams) if mixed_streams else 0]):
                gate.wait(timeout=30)
                for _ in range(steps):
                    exercise(index)
                    assert torch.equal(cases[index][0], cases[index][3])

        with ThreadPoolExecutor(max_workers=len(cases)) as executor:
            list(executor.map(worker, range(len(cases))))

    for mixed_streams in (False, True):
        concurrent_round(2, mixed_streams)
    gaudirt.synchronize()
    before_create, before_destroy = created(), destroyed()
    for mixed_streams in (False, True):
        concurrent_round(32, mixed_streams)
    gaudirt.synchronize()
    assert created() == before_create, (before_create, created())
    assert destroyed() == before_destroy, (before_destroy, destroyed())
    for cpu, _, pinned_out, host_out, _, _ in cases:
        assert torch.equal(cpu, pinned_out)
        assert torch.equal(cpu, host_out)
    # A strided copy larger than the staging limit must wait before reusing
    # the temporary for each subsequent chunk, without leaving a final record.
    large_cpu = torch.arange(33 * 65536, dtype=torch.float32).reshape(33, 65536)
    large_storage = torch.empty((33, 65537), dtype=torch.float32, device="gaudirt:0")
    large_device = large_storage[:, :65536]
    large_device.copy_(large_cpu, non_blocking=False)
    assert torch.equal(large_device.cpu(), large_cpu)
    print("four-thread same-stream and mixed-stream copies: 256 steps; pinned/pageable, contiguous/strided;",
          "event creations 0 destructions 0; multichunk strided copy passed", flush=True)


if __name__ == "__main__":
    main()
