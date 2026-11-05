"""Validate host-side request metadata before its asynchronous device upload."""

import gc
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def metadata(rows):
    mapping = torch.arange(192, dtype=torch.int32).reshape(3, 64)
    requests = torch.arange(rows, dtype=torch.int64) % 3
    positions = torch.arange(rows, dtype=torch.int64) // 3 + 2
    locations = mapping[requests, positions].to(torch.int64)
    return locations, requests, positions, mapping


def rejected(values, message):
    try:
        torch.classes.gaudirt.PagedMetadata(*values, 192)
    except RuntimeError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("invalid paged metadata was accepted")


def main():
    gaudirt.initialize(hbm_bytes=32 * 1024 * 1024, pinned_bytes=1024 * 1024)
    for rows in (1, 3, 17, 137):
        values = metadata(rows)
        prepared = torch.classes.gaudirt.PagedMetadata(*values, 192)
        assert prepared.rows() == rows and prepared.slots() == 192
        del prepared
    locations, requests, positions, mapping = metadata(7)
    duplicate = locations.clone(); duplicate[1] = duplicate[0]
    dup_requests = requests.clone(); dup_requests[1] = requests[0]
    dup_positions = positions.clone(); dup_positions[1] = positions[0]
    rejected((duplicate, dup_requests, dup_positions, mapping), "duplicate output KV slot")
    invalid = locations.clone(); invalid[0] = 192
    rejected((invalid, requests, positions, mapping), "out of range")
    missing = mapping.clone(); missing[0, 0] = -1
    rejected((locations, requests, positions, missing), "missing KV slot")
    alias = mapping.clone(); alias[1, 0] = locations[0]
    rejected((locations, requests, positions, alias), "aliases another visible prefix")
    mismatch = mapping.clone(); mismatch[requests[0], positions[0]] = 191
    rejected((locations, requests, positions, mismatch), "does not match")
    empty = torch.empty(0, dtype=torch.int64)
    rejected((empty, empty, empty, mapping), "dimensions outside supported range")
    too_many = torch.zeros(4097, dtype=torch.int64)
    rejected((too_many, too_many, too_many, mapping), "dimensions outside supported range")
    gc.collect()
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("paged metadata entry validation passed", flush=True)


if __name__ == "__main__":
    main()
