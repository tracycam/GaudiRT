"""Check that every layer's fixed KV descriptor stays inside one allocation."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=192 * 1024**2, pinned_bytes=4 * 1024**2)
    layers, slots, heads, dim = 3, 65, 2, 64
    pool = gaudirt._C._allocate_paged_kv(layers, slots, heads, dim)
    assert tuple(pool.shape) == (layers, 2, heads, slots, dim)
    layer_bytes = 2 * heads * slots * dim * 2
    descriptor_bytes = 2 * heads * 131072 * dim * 2
    assert pool.untyped_storage().nbytes() == (layers - 1) * layer_bytes + descriptor_bytes
    assert torch.count_nonzero(pool.cpu()).item() == 0

    key = pool[1, 0].permute(1, 0, 2)
    value = pool[1, 1].permute(1, 0, 2)
    assert not key.is_contiguous()
    assert key.data_ptr() == pool.data_ptr() + layer_bytes
    assert value.data_ptr() == key.data_ptr() + heads * slots * dim * 2
    source = torch.arange(slots * heads * dim, dtype=torch.float32).reshape(slots, heads, dim).to(torch.bfloat16)
    key.copy_(source)
    assert torch.equal(key.cpu(), source)
    assert torch.count_nonzero(value.cpu()).item() == 0
    assert torch.count_nonzero(pool[0].cpu()).item() == 0
    assert torch.count_nonzero(pool[2].cpu()).item() == 0

    for args in ((0, 65, 1, 128), (1, 131073, 1, 128), (1, 65, 1, 64)):
        try:
            gaudirt._C._allocate_paged_kv(*args)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid KV geometry accepted: {args}")
    del key, value, pool
    gaudirt.synchronize()
    assert "habana_frameworks" not in sys.modules
    print("head-major KV allocation and descriptor bounds passed", flush=True)


if __name__ == "__main__":
    main()
