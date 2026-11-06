"""Metadata staging reuses the events owned by each compute stream."""

import ctypes
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    stream_events = ctypes.CDLL(None).gaudirt_test_stream_event_create_count
    stream_events.restype = ctypes.c_uint64
    all_events = ctypes.CDLL(None).gaudirt_test_event_create_count
    all_events.restype = ctypes.c_uint64
    destroyed_events = ctypes.CDLL(None).gaudirt_test_event_destroy_count
    destroyed_events.restype = ctypes.c_uint64
    initial = stream_events()
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    streams = [gaudirt.Stream(), gaudirt.Stream()]
    assert stream_events() - initial >= 24, "event interposer did not observe stream setup"
    mapping = torch.arange(1, 129, dtype=torch.int32).reshape(2, 64)
    device_mapping = mapping.to("gaudirt:0")
    cases = []
    for request, prefix in ((0, 5), (1, 12)):
        positions = torch.arange(prefix, prefix + 3, dtype=torch.int64)
        requests = torch.full((3,), request, dtype=torch.int64)
        locations = mapping[requests, positions].long()
        batch_requests = torch.tensor([request], dtype=torch.int64)
        token_ids = torch.tensor([11, 12, 13], dtype=torch.int64)
        device = [tensor.to("gaudirt:0") for tensor in (locations, batch_requests, positions, token_ids)]
        cases.append((locations, requests, positions, prefix, device))

    def construct(index, mode):
        locations, requests, positions, prefix, device = cases[index]
        if mode == "scheduler":
            return torch.classes.gaudirt.PagedMetadata._from_scheduler(
                device[0], device[1], device[2], device_mapping, [prefix], [3], 129, device[3], 32000
            )
        if mode == "checked":
            return torch.classes.gaudirt.PagedMetadata.from_device_with_tokens(
                device[0], device[1], device[2], device_mapping, [prefix], [3], 129, device[3], 32000
            )
        return torch.classes.gaudirt.PagedMetadata(locations, requests, positions, mapping, 129)

    for index, stream in enumerate(streams):
        with gaudirt.StreamContext(stream):
            for mode in ("cpu", "scheduler", "checked"):
                construct(index, mode)
    gaudirt.synchronize()
    before_stream, before_all, before_destroyed = stream_events(), all_events(), destroyed_events()
    pending = []
    for iteration in range(24):
        index = iteration % 2
        with gaudirt.StreamContext(streams[index]):
            for mode in ("cpu", "scheduler", "checked"):
                pending.append((index, construct(index, mode)))
    gaudirt.synchronize()
    assert stream_events() == before_stream, (before_stream, stream_events())
    assert all_events() == before_all, (before_all, all_events())
    assert destroyed_events() == before_destroyed, (before_destroyed, destroyed_events())
    for index, metadata in pending:
        locations, requests, positions, _, _ = cases[index]
        assert torch.equal(metadata.row_locations().cpu().long(), locations)
        assert torch.equal(metadata.row_requests().cpu().long(), requests)
        assert torch.equal(metadata.rotary_positions().cpu().long(), positions)
    print("metadata constructions", len(pending), "event creations 0 event destructions 0", flush=True)


if __name__ == "__main__":
    main()
