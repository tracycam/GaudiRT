"""Scheduler metadata recipe is independent of the global KV/request capacities."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    slots, context, requests = (int(os.environ[name]) for name in
                                ("GAUDIRT_TEST_SLOTS", "GAUDIRT_TEST_CONTEXT", "GAUDIRT_TEST_REQUESTS"))
    mode = os.environ["GAUDIRT_TEST_CACHE"]
    assert slots > 6 and context >= 6 and requests > 0 and mode in ("cold", "warm")
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    mapping = torch.zeros((requests, context), dtype=torch.int32)
    mapping[0, :6] = torch.arange(1, 7, dtype=torch.int32)
    locations = torch.tensor([4, 5, 6], dtype=torch.int64)
    request = torch.tensor([0], dtype=torch.int64)
    positions = torch.tensor([3, 4, 5], dtype=torch.int64)
    tokens = torch.tensor([21, 22, 23], dtype=torch.int64)
    mapping_device = mapping.to("gaudirt:0")
    device = [value.to("gaudirt:0") for value in (locations, request, positions, tokens)]
    metadata = torch.classes.gaudirt.PagedMetadata._from_scheduler(
        device[0], device[1], device[2], mapping_device, [3], [3], slots, device[3], 32000
    )
    stats = gaudirt.recipe_cache_stats()
    assert stats[:2] == ([0, 1] if mode == "cold" else [1, 0]), stats
    actual = [metadata.row_locations().cpu(), metadata.row_requests().cpu(),
              metadata.rotary_positions().cpu(), metadata.token_ids().cpu(),
              metadata.last_token_indices().cpu()]
    expected = [locations.int(), torch.zeros(3, dtype=torch.int32), positions.int(),
                tokens.int(), torch.tensor([2], dtype=torch.int32)]
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert metadata.slots() == slots and metadata.context() == context
    assert metadata.request_mapping().data_ptr() == mapping_device.data_ptr()
    high_locations = torch.arange(slots - 3, slots, dtype=torch.int64)
    mapping_device[0, 3:6].copy_(high_locations.int())
    high_metadata = torch.classes.gaudirt.PagedMetadata._from_scheduler(
        high_locations.to("gaudirt:0"), device[1], device[2], mapping_device,
        [3], [3], slots, device[3], 32000
    )
    assert torch.equal(high_metadata.row_locations().cpu(), high_locations.int())
    assert gaudirt.recipe_cache_stats() == stats
    payload = b"".join(value.numpy().tobytes() for value in actual)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("scheduler metadata", "slots", slots, "context", context, "requests", requests,
          "stats", stats, "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
