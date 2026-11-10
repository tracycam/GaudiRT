"""Cached staging is released at shutdown without invalidating live metadata."""

import gc
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def make_metadata():
    mapping = torch.arange(64, dtype=torch.int32).reshape(1, 64).to("gaudirt")
    locations, requests, tokens = [torch.tensor([value], dtype=torch.int64).to("gaudirt")
                                   for value in (7, 0, 17)]
    return torch.classes.gaudirt.PagedMetadata._from_scheduler_fused(
        locations, requests, torch.tensor([7], dtype=torch.int64), mapping, [7], [1], 2048,
        tokens, 32000, 4, 2, 128)


def main():
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    metadata = make_metadata()
    try:
        gaudirt.shutdown()
    except RuntimeError as error:
        assert "attention metadata" in str(error), error
    else:
        raise AssertionError("shutdown accepted live attention metadata")
    assert metadata.model_positions().cpu().tolist() == [7]
    del metadata
    gc.collect()
    gaudirt.shutdown()
    print("staging preserves live metadata and releases its pools on shutdown", flush=True)


if __name__ == "__main__":
    main()
