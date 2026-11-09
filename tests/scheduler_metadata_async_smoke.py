"""Check scheduler-owned metadata matches checked input without a host wait."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch

import gaudirt
import paged_device_metadata_smoke as paged


def make_metadata(rows, sliced=False):
    locations, requests, positions, mapping, prefixes, lengths, _ = paged.data(rows)
    inputs, keep = [], []
    for source in (locations, requests, positions, mapping):
        value, storage = paged.to_device(source, sliced and source.ndim == 1)
        inputs.append(value)
        keep.append(storage)
    tokens = (torch.arange(rows, dtype=torch.int64) * 73) % 32000
    token_device, token_storage = paged.to_device(tokens, sliced)
    args = (*inputs, prefixes, lengths, paged.SLOTS, token_device, 32000)
    return args, (inputs, keep, token_device, token_storage), tokens


def main():
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=4 * 1024**2)
    digest = hashlib.sha256()
    for rows in (1, 7, 33, 137, 1025):
        args, keep, tokens = make_metadata(rows, sliced=rows == 137)
        checked = torch.classes.gaudirt.PagedMetadata.from_device_with_tokens(*args)
        scheduler = torch.classes.gaudirt.PagedMetadata._from_scheduler(*args)
        parts = scheduler.forward_tensors(4, 2, 128)
        for part, name in zip(parts[:6], ("token_ids", "last_token_indices", "rotary_positions",
                                          "rotary_header", "row_locations", "row_requests")):
            assert torch.equal(part.cpu(), getattr(scheduler, name)().cpu()), (rows, name)
        assert parts[6].data_ptr() == scheduler.attention_plan(4, 2, 128).data_ptr()
        assert parts[7] == scheduler.context()
        for name in ("row_locations", "row_requests", "rotary_positions", "last_token_indices", "token_ids"):
            a, b = getattr(checked, name)().cpu(), getattr(scheduler, name)().cpu()
            assert torch.equal(a, b), (rows, name)
            digest.update(b.contiguous().view(torch.uint8).numpy().tobytes())
        token_view = scheduler.token_ids()
        bucket = 1 << (rows - 1).bit_length()
        padded = token_view.as_strided((bucket,), (1,), token_view.storage_offset()).cpu()
        assert torch.equal(padded[:rows], tokens.int()) and torch.count_nonzero(padded[rows:]) == 0
        del checked, scheduler, args, keep, token_view, parts, part

    # Warm the recipe, then put enough device copies ahead of it that a host
    # synchronization inside metadata preparation would make the event complete.
    args, keep, _ = make_metadata(7)
    metadata = torch.classes.gaudirt.PagedMetadata._from_scheduler(*args)
    gaudirt.synchronize()
    source = torch.empty((32 * 1024**2,), dtype=torch.uint8, device="gaudirt:0")
    target = torch.empty_like(source)
    for _ in range(512):
        target.copy_(source, non_blocking=True)
    queued = gaudirt.Event()
    queued.record()
    assert not queued.query(), "device work completed before the asynchronous check"
    next_metadata = torch.classes.gaudirt.PagedMetadata._from_scheduler(*args)
    assert not queued.query(), "scheduler metadata waited for earlier device work"
    queued.synchronize()
    assert torch.equal(next_metadata.token_ids().cpu(), metadata.token_ids().cpu())
    print("scheduler metadata matches checked mixed-varlen inputs and does not wait; sha256",
          digest.hexdigest(), flush=True)
    del queued, source, target, metadata, next_metadata, args, keep
    gaudirt.shutdown()


if __name__ == "__main__":
    main()
