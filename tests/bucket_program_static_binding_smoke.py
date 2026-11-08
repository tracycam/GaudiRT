"""Verify weight prebinding, changing weights, alias staging, and lifetimes."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
import gc
import os
import sys
import threading
import weakref

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    torch.set_num_threads(4)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    node = gaudirt._C.FxNode.linear(0, 1, 2, True)
    program = gaudirt._C.BucketProgram([1, 0, 0], [node], [3])
    probe = ctypes.CDLL(None)
    probe.gaudirt_test_records.restype = ctypes.c_uint64
    probe.gaudirt_test_event_create_count.restype = ctypes.c_uint64
    probe.gaudirt_test_event_destroy_count.restype = ctypes.c_uint64
    rng = torch.Generator().manual_seed(1803)
    x_cpu = torch.randn(3, 64, generator=rng)
    weight_cpu = torch.randn(64, 64, generator=rng)
    bias_cpu = torch.randn(64, generator=rng)
    x, weight, bias = (tensor.to("gaudirt:0") for tensor in (x_cpu, weight_cpu, bias_cpu))
    other_cpu = torch.randn(64, 64, generator=rng)
    other = other_cpu.to("gaudirt:0")
    stream = gaudirt.Stream()
    gaudirt.synchronize()
    with gaudirt.stream(stream):
        probe.gaudirt_test_reset_records()
        first = program.run([x, weight, bias])[0]
        first_records = probe.gaudirt_test_records()
        probe.gaudirt_test_reset_records()
        second = program.run([x, weight, bias])[0]
        warm_records = probe.gaudirt_test_records()
        other_result = program.run([x, other, bias])[0]
        alias_result = program.run([weight[:3], weight, bias])[0]
    gaudirt.synchronize()
    assert first_records == 3 and warm_records == 1, (first_records, warm_records)
    for actual, expected in ((first, x_cpu @ weight_cpu.T + bias_cpu),
                             (second, x_cpu @ weight_cpu.T + bias_cpu),
                             (other_result, x_cpu @ other_cpu.T + bias_cpu),
                             (alias_result, weight_cpu[:3] @ weight_cpu.T + bias_cpu)):
        assert torch.allclose(actual.cpu(), expected, atol=2e-4, rtol=2e-4)
    weight_cpu.mul_(0.5)
    weight.copy_(weight_cpu)
    gaudirt.synchronize()
    with gaudirt.stream(stream):
        changed = program.run([x, weight, bias])[0]
    gaudirt.synchronize()
    assert torch.allclose(changed.cpu(), x_cpu @ weight_cpu.T + bias_cpu, atol=2e-4, rtol=2e-4)
    with gaudirt.stream(stream):
        for _ in range(4):
            transient = program.run([x, weight, bias])[0]
            del transient
        gaudirt.synchronize()
        before_events = (probe.gaudirt_test_event_create_count(), probe.gaudirt_test_event_destroy_count())
        for _ in range(8):
            transient = program.run([x, weight, bias])[0]
            del transient
        after_events = (probe.gaudirt_test_event_create_count(), probe.gaudirt_test_event_destroy_count())
    assert before_events == after_events, (before_events, after_events)
    second_stream = gaudirt.Stream()
    barrier = threading.Barrier(2)

    def concurrent(active_stream):
        with gaudirt.stream(active_stream):
            barrier.wait(timeout=30)
            return program.run([x, weight, bias])[0]

    with ThreadPoolExecutor(max_workers=2) as executor:
        parallel = list(executor.map(concurrent, (stream, second_stream)))
    gaudirt.synchronize()
    for item in parallel:
        assert torch.allclose(item.cpu(), x_cpu @ weight_cpu.T + bias_cpu, atol=2e-4, rtol=2e-4)
    assert gaudirt.recipe_cache_stats()[:2] == [0, 1]
    weight_ref = weakref.ref(weight)
    del first, second, other_result, alias_result, changed, actual, item, parallel
    del x, weight, bias, other, stream, second_stream
    gc.collect()
    assert weight_ref() is None, "prebinding extended the weight's lifetime"
    gaudirt.empty_cache()
    gaudirt.shutdown()  # The compiled program can outlive the device context.
    print("static bindings cold/warm registrations", first_records, warm_records,
          "steady event creates/destroys 0/0; replacement, alias, concurrency, shutdown passed", flush=True)


if __name__ == "__main__":
    main()
