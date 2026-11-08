"""Exercise page-sized slot allocation and deduplication on the device."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    payload = bytearray()
    for page_size in (16, 64):
        prefixes, lengths = [0, 7, page_size, 2 * page_size - 1], [19, 14, 20, 66]
        previous = [-1, 81 * page_size + 6, 82 * page_size + page_size - 1,
                    83 * page_size + page_size - 2]
        pages = [13, 7, 23, 41, 12, 33, 55, 61, 15, 18, 19, 20, 21, 22, 24, 25]
        expected, offset = [], 0
        for prefix, length, last in zip(prefixes, lengths, previous):
            existing = min(length, (-prefix) % page_size)
            expected.extend(last + 1 + i for i in range(existing))
            remaining = length - existing
            for i in range(remaining):
                expected.append(pages[offset + i // page_size] * page_size + i % page_size)
            offset += (remaining + page_size - 1) // page_size
        last_device = torch.tensor(previous, dtype=torch.int64).to("gaudirt:0")
        pages_device = torch.tensor(pages, dtype=torch.int64).to("gaudirt:0")
        slots, used = gaudirt._C._allocate_paged_slots(last_device, pages_device,
                                                        prefixes, lengths, page_size)
        assert used == offset and slots.cpu().tolist() == expected, (page_size, used, offset)
        released = [13 * page_size + 7, 7 * page_size, 13 * page_size,
                    23 * page_size + page_size - 1, 7 * page_size + 3]
        released_device = torch.tensor(released, dtype=torch.int64).to("gaudirt:0")
        unique = gaudirt._C._release_paged_slots(released_device, page_size, 512)
        assert unique.cpu().tolist() == [13, 7, 23], (page_size, unique.cpu())
        payload.extend(slots.cpu().view(torch.uint8).numpy().tobytes())
        payload.extend(unique.cpu().view(torch.uint8).numpy().tobytes())
        for invalid in (0, 513 * page_size, 1 << 40):
            try:
                gaudirt._C._release_paged_slots(torch.tensor([invalid]).to("gaudirt:0"), page_size, 512)
            except RuntimeError as error:
                assert "outside the KV pool" in str(error), str(error)
            else:
                raise AssertionError((page_size, invalid))
        print("device pages", page_size, "new", used, "released", unique.cpu().tolist(), flush=True)
        del last_device, pages_device, slots, released_device, unique
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[:2] == [0, 6], stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[:2] == [6, 0], stats
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("page slots stats", stats, "sha256", hashlib.sha256(payload).hexdigest(), flush=True)
    gaudirt.empty_cache()
    gaudirt.shutdown()


if __name__ == "__main__":
    main()
