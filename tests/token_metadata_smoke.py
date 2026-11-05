"""Validate token IDs in the same device metadata gate as the page table."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import paged_device_metadata_smoke as paged


def prepare(rows, source, sliced=False):
    locations, requests, positions, mapping, prefixes, lengths, _ = paged.data(rows)
    values = [value.to("gaudirt:0") for value in (locations, requests, positions, mapping)]
    tokens, storage = paged.to_device(source, sliced)
    metadata = torch.classes.gaudirt.PagedMetadata.from_device_with_tokens(
        *values, prefixes, lengths, paged.SLOTS, tokens, 32000
    )
    return metadata, (values, tokens, storage)


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    payload = bytearray()
    for rows in (1, 3, 7, 137, 1025, 4096):
        for dtype in (torch.int32, torch.int64):
            source = (torch.arange(rows, dtype=dtype) * 73) % 32000
            metadata, keep = prepare(rows, source, sliced=rows == 137)
            tokens = metadata.token_ids()
            bucket = 1 << (rows - 1).bit_length()
            assert metadata.vocabulary() == 32000 and metadata.rows() == rows
            assert tokens.shape == (rows,) and tokens.dtype == torch.int32
            assert tokens.untyped_storage().nbytes() >= bucket * 4
            full = tokens.as_strided((bucket,), (1,), 0).cpu()
            assert torch.equal(full[:rows], source.to(torch.int32))
            assert torch.count_nonzero(full[rows:]).item() == 0
            payload.extend(full.numpy().tobytes())
            del metadata, keep, tokens
        print("validated token metadata rows", rows, "stats", gaudirt.recipe_cache_stats(), flush=True)
    source = torch.arange(7, dtype=torch.int64)
    for invalid in (-1, 32000, 1 << 32):
        bad = source.clone(); bad[0] = invalid
        try:
            prepare(7, bad)
        except RuntimeError as error:
            assert "token ID outside vocabulary" in str(error), str(error)
        else:
            raise AssertionError(f"out-of-vocabulary token {invalid} accepted")
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 12], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [12, 0], counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("validated token metadata stats", counts, "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
