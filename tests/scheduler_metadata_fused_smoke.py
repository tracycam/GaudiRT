"""Compare fused scheduler metadata and its single host-to-device copy."""

import ctypes
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    copies = ctypes.CDLL(None).gaudirt_test_h2d_copy_count
    copies.restype = ctypes.c_uint64
    waits = ctypes.CDLL(None).gaudirt_test_h2d_stream_wait_count
    waits.restype = ctypes.c_uint64
    creates = ctypes.CDLL(None).gaudirt_test_event_create_count
    creates.restype = ctypes.c_uint64
    for rows in (1, 3, 17):
        context, slots = 64, 2048
        prefixes = [3 + i for i in range(rows)]
        mapping = torch.zeros((rows, context), dtype=torch.int32)
        for row, prefix in enumerate(prefixes):
            mapping[row, :prefix + 1] = torch.arange(row * context, row * context + prefix + 1)
        locations = torch.tensor([row * context + prefix for row, prefix in enumerate(prefixes)], dtype=torch.int64)
        requests = torch.arange(rows, dtype=torch.int64)
        positions = torch.tensor(prefixes, dtype=torch.int64)
        tokens = torch.arange(rows, dtype=torch.int64) + 17
        loc_dev, req_dev, pos_dev, map_dev, tok_dev = [
            value.to("gaudirt:0") for value in (locations, requests, positions, mapping, tokens)
        ]
        lengths = [1] * rows
        old = torch.classes.gaudirt.PagedMetadata._from_scheduler(
            loc_dev, req_dev, pos_dev, map_dev, prefixes, lengths, slots, tok_dev, 32000
        )
        old_plan = old.attention_plan(4, 2, 128)
        warm = torch.classes.gaudirt.PagedMetadata._from_scheduler_fused(
            loc_dev, req_dev, positions, map_dev, prefixes, lengths, slots, tok_dev, 32000, 4, 2, 128
        )
        gaudirt.synchronize()
        before = copies()
        fused = torch.classes.gaudirt.PagedMetadata._from_scheduler_fused(
            loc_dev, req_dev, positions, map_dev, prefixes, lengths, slots, tok_dev, 32000, 4, 2, 128
        )
        assert copies() - before == 1, "fused metadata must use one H2D copy"
        for name in ("row_locations", "row_requests", "rotary_positions", "rotary_header",
                     "last_token_indices", "token_ids"):
            assert torch.equal(getattr(old, name)().cpu(), getattr(fused, name)().cpu()), (rows, name)
        assert torch.equal(fused.model_positions().cpu(), positions)
        tasks = 1 << ((rows * 8 - 1).bit_length())
        assert torch.equal(old_plan[:tasks * 64].cpu(), fused.attention_plan(4, 2, 128)[:tasks * 64].cpu())
        del old, warm, fused, old_plan
    # Retained metadata must remain independent even when multiple streams
    # submit work before either stream finishes.
    streams = [gaudirt.Stream(), gaudirt.Stream()]
    positions = torch.tensor([7], dtype=torch.int64)
    mapping = torch.arange(64, dtype=torch.int32).reshape(1, 64).to("gaudirt:0")
    locations, requests, tokens = [torch.tensor([value], dtype=torch.int64).to("gaudirt:0")
                                   for value in (7, 0, 17)]
    pending = []
    for stream in streams:
        with gaudirt.StreamContext(stream):
            pending.append(torch.classes.gaudirt.PagedMetadata._from_scheduler_fused(
                locations, requests, positions, mapping, [7], [1], 2048, tokens, 32000, 4, 2, 128))
    assert pending[0].attention_plan(4, 2, 128).data_ptr() != pending[1].attention_plan(4, 2, 128).data_ptr()
    gaudirt.synchronize()
    for metadata in pending:
        assert torch.equal(metadata.model_positions().cpu(), positions)
    del metadata, pending
    gaudirt.synchronize()

    def once():
        metadata = torch.classes.gaudirt.PagedMetadata._from_scheduler_fused(
            locations, requests, positions, mapping, [7], [1], 2048, tokens, 32000, 4, 2, 128)
        del metadata
        gaudirt.synchronize()

    for _ in range(5):
        once()
    before_copies, before_waits, before_creates = copies(), waits(), creates()
    for _ in range(20):
        once()
    assert copies() - before_copies == 20
    assert waits() == before_waits, (before_waits, waits())
    assert creates() == before_creates, (before_creates, creates())
    print("fused scheduler metadata matches existing path with one H2D", flush=True)


if __name__ == "__main__":
    main()
