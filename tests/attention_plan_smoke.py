"""Exercise one step's device tile-index plan without invoking attention."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    requests = int(os.environ["GAUDIRT_TEST_REQUESTS"])
    context = int(os.environ["GAUDIRT_TEST_CONTEXT"])
    slots = int(os.environ["GAUDIRT_TEST_SLOTS"])
    gaudirt.initialize(hbm_bytes=512 * 1024 * 1024, pinned_bytes=8 * 1024 * 1024)
    positions = torch.tensor([3, 4, 5, *range(7, 17)], dtype=torch.int32)
    row_requests = torch.tensor([0] * 3 + [1] * 10, dtype=torch.int32)
    mapping = ((torch.arange(requests, dtype=torch.int64)[:, None] * 256 +
                torch.arange(context, dtype=torch.int64)[None, :]) % slots).to(torch.int32)
    locations = mapping[row_requests.long(), positions.long()].contiguous()
    metadata = torch.classes.gaudirt.PagedMetadata(locations, row_requests, positions, mapping, slots)
    plan = metadata.attention_plan(4, 2, 64)
    assert metadata.attention_plan(4, 2, 64).data_ptr() == plan.data_ptr()
    try:
        metadata.attention_plan(8, 2, 64)
        raise AssertionError("geometry change did not fail")
    except RuntimeError as error:
        assert "different head geometry" in str(error)
    gaudirt.synchronize()
    indices = plan.cpu()[:16 * 64].reshape(2, 2, 2, 2, 64)
    for block, (request, visible) in enumerate(((0, 6), (1, 17))):
        for kind in range(2):
            for head in range(2):
                expected = torch.full((128,), (kind * 2 + head) * slots, dtype=torch.int32)
                expected[:visible] += mapping[request, :visible]
                expected[64:] = 0  # Entirely masked tasks use safe index zero.
                actual = indices[block, kind, head].reshape(-1)
                assert torch.equal(actual, expected), (block, kind, head, actual[:24].tolist(), expected[:24].tolist())
    stats = gaudirt.recipe_cache_stats()
    mode = os.environ["GAUDIRT_TEST_CACHE"]
    assert stats[:2] == ([0, 1] if mode == "cold" else [1, 0]), stats
    print("attention plan indices and cached geometry matched", mode, stats, flush=True)
    assert "habana_frameworks" not in sys.modules
    with open("/proc/self/maps", encoding="utf-8") as maps:
        assert not any("habana_pytorch" in line or "libhabana_pytorch" in line for line in maps)


if __name__ == "__main__":
    main()
