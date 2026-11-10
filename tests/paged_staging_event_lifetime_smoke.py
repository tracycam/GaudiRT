"""Growing metadata staging slots reuses events prepared for each stream."""

import ctypes
import gc
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    probe = ctypes.CDLL(None)
    created, destroyed = probe.gaudirt_test_event_create_count, probe.gaudirt_test_event_destroy_count
    created.restype = destroyed.restype = ctypes.c_uint64
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    streams = [gaudirt.Stream(), gaudirt.Stream()]
    mapping = torch.arange(64, dtype=torch.int32).reshape(1, 64).to("gaudirt")
    locations, requests, tokens = [torch.tensor([value], dtype=torch.int64).to("gaudirt")
                                   for value in (7, 0, 17)]
    positions = torch.tensor([7], dtype=torch.int64)

    def metadata():
        return torch.classes.gaudirt.PagedMetadata._from_scheduler_fused(
            locations, requests, positions, mapping, [7], [1], 2048, tokens, 32000, 4, 2, 128)

    # Initialize the shared graph workspace and its events before counting.
    with gaudirt.stream(streams[0]):
        warm = metadata()
    del warm
    gc.collect()
    gaudirt.synchronize()
    before = created(), destroyed()
    pending = []
    for index in range(16):
        with gaudirt.stream(streams[index % 2]):
            pending.append(metadata())
    assert (created(), destroyed()) == before, (before, created(), destroyed())
    for item in pending:
        assert item.row_locations().cpu().tolist() == [7]
        assert item.rotary_positions().cpu().tolist() == [7]
    del item
    # Release in reverse stream order and immediately issue more plans. A
    # shared completion event must cover every prior slot on that stream.
    while pending:
        pending.pop()
    gc.collect()
    for index in range(16):
        with gaudirt.stream(streams[index % 2]):
            pending.append(metadata())
    gaudirt.synchronize()
    assert (created(), destroyed()) == before, (before, created(), destroyed())
    for item in pending:
        assert item.row_locations().cpu().tolist() == [7]
    del item, pending, mapping, locations, requests, positions, tokens, streams
    gc.collect()
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("32 in-flight metadata plans on two streams: correct, zero event creation or destruction", flush=True)


if __name__ == "__main__":
    main()
