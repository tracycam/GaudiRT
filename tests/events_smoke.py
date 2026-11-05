"""Cross-stream Synapse event dependencies and shutdown ownership."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    first, second = gaudirt.Stream(), gaudirt.Stream()
    ready, complete = gaudirt.Event(), gaudirt.Event()
    assert ready.query()
    cpu = torch.arange(262144, dtype=torch.int32)
    with gaudirt.stream(first):
        source = cpu.to("gaudirt:0", non_blocking=True)
        ready.record()
    with gaudirt.stream(second):
        second.wait_event(ready)
        result = torch.empty_like(source)
        result.copy_(source, non_blocking=True)
        source.record_stream(second)
        del source
        complete.record()
    with gaudirt.stream(first):
        replacements = [torch.empty_like(result).fill_(i) for i in range(4)]
    complete.synchronize()
    assert complete.query()
    with gaudirt.stream(second):
        assert torch.equal(result.cpu(), cpu)

    with gaudirt.stream(first):
        source = cpu.to("gaudirt:0", non_blocking=True)
    with gaudirt.stream(second):
        second.wait_stream(first)
        result.copy_(source, non_blocking=True)
        source.record_stream(second)
    second.synchronize()
    with gaudirt.stream(second):
        assert torch.equal(result.cpu(), cpu)

    start, end = gaudirt.Event(enable_timing=True), gaudirt.Event(enable_timing=True)
    with gaudirt.stream(first):
        start.record()
        source.copy_(cpu, non_blocking=True)
        end.record()
    end.synchronize()
    assert start.elapsed_time(end) >= 0
    first.synchronize()
    del source, result, replacements, start, end, complete
    gaudirt.empty_cache()
    gaudirt.shutdown()
    try:
        ready.query()
    except RuntimeError as error:
        assert "shut down runtime" in str(error)
    else:
        raise AssertionError("event outlived its runtime without an error")
    print("cross-stream waits, event timing, and event shutdown passed", flush=True)


if __name__ == "__main__":
    main()
