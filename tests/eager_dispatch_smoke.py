"""Exercise the registered cold operators on actual device tensors."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    source = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    index = torch.tensor([3, 0, 2], dtype=torch.int64)
    device = source.to("gaudirt:0")
    device_index = index.to("gaudirt:0")
    mask = device > 10
    zeros = torch.zeros_like(device)
    cases = (
        (mask, source > 10),
        (torch.where(mask, device, zeros), torch.where(source > 10, source, torch.zeros_like(source))),
        (device + 3, source + 3),
        (torch.cat((device[:2], device[2:]), dim=0), source),
        (torch.index_select(device, 0, device_index), torch.index_select(source, 0, index)),
        (torch.argmax(device, dim=1), torch.argmax(source, dim=1)),
        (device.to(torch.bfloat16), source.to(torch.bfloat16)),
    )
    for actual, expected in cases:
        assert torch.equal(actual.cpu(), expected)
    del actual, expected, cases, mask, zeros, device, device_index
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("cold comparison, where, arithmetic, cat, indexing, argmax and cast passed", flush=True)


if __name__ == "__main__":
    main()
