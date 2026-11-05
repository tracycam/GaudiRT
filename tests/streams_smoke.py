"""Real compute streams, stream-local dispatch, and allocator lifetime."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    gaudirt.set_device(0)
    default = gaudirt.current_stream()
    first, second = gaudirt.Stream(), gaudirt.Stream()
    assert len({default.stream_id, first.stream_id, second.stream_id}) == 3
    assert len({default.native_handle, first.native_handle, second.native_handle}) == 3
    source_cpu = torch.arange(65536, dtype=torch.int32)
    with gaudirt.stream(first):
        assert gaudirt.current_stream().stream_id == first.stream_id
        source = source_cpu.to("gaudirt:0", non_blocking=True)
        copied = torch.empty_like(source)
        copied.copy_(source, non_blocking=True)
    assert gaudirt.current_stream().stream_id == default.stream_id
    first.synchronize()
    with gaudirt.StreamContext(second):
        output = torch.empty_like(copied)
        output.copy_(copied, non_blocking=True)
        copied.record_stream(second)
        del copied
    with gaudirt.stream(first):
        temporary = [torch.empty_like(source).fill_(i) for i in range(6)]
    second.synchronize()
    with gaudirt.stream(second):
        actual = output.cpu()
    assert torch.equal(actual, source_cpu)
    first.synchronize()
    del output, source, temporary
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("distinct streams, current-stream dispatch, and allocator reuse passed", flush=True)


if __name__ == "__main__":
    main()
