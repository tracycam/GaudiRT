"""Fixed-window attention with causal prefixes, full KV audit, and warm caching."""

import atexit
import hashlib
import math
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"


def assert_no_bridge():
    assert "habana_frameworks" not in sys.modules
    with open("/proc/self/maps", encoding="utf-8") as maps:
        assert not any("habana_pytorch" in line or "libhabana_pytorch" in line for line in maps)


assert_no_bridge()
atexit.register(assert_no_bridge)

import torch
import gaudirt
from fp64_reference import direct_bf16
from paged_attention_reference import isolated_errors, tiled_bf16_pv


def main():
    torch.set_num_threads(4)
    heads = int(os.environ.get("GAUDIRT_TEST_HEADS", "4"))
    assert heads in (4, 56)
    kv_heads, dim = (2, 128) if heads == 4 else (8, 128)
    slots = int(os.environ.get("GAUDIRT_TEST_SLOTS", "2048"))
    context = int(os.environ.get("GAUDIRT_TEST_CONTEXT", "1024"))
    assert slots > 1024 and context >= 1024
    gaudirt.initialize(hbm_bytes=1024 * 1024 * 1024, pinned_bytes=8 * 1024 * 1024)
    mapping = (torch.arange(context, dtype=torch.int32) % (slots - 1) + 1).reshape(1, -1)
    head_map = torch.arange(heads) // (heads // kv_heads)
    scale = torch.tensor(1 / math.sqrt(dim), dtype=torch.float32).item()
    payload = bytearray()
    for rows in (7, 137):
        prefix = 512
        rng = torch.Generator().manual_seed(3000 + rows)
        query = torch.randn(rows, heads * dim, generator=rng).to(torch.bfloat16)
        key = torch.randn(rows, kv_heads * dim, generator=rng).to(torch.bfloat16)
        value = torch.randn(rows, kv_heads * dim, generator=rng).to(torch.bfloat16)
        key_base = torch.randn(slots, kv_heads, dim, generator=torch.Generator().manual_seed(520)).to(torch.bfloat16)
        value_base = torch.randn(slots, kv_heads, dim, generator=torch.Generator().manual_seed(521)).to(torch.bfloat16)
        positions = torch.arange(prefix, prefix + rows, dtype=torch.int32)
        requests = torch.zeros(rows, dtype=torch.int32)
        locations = mapping[0, positions.long()].contiguous()
        metadata = torch.classes.gaudirt.PagedMetadata(locations, requests, positions, mapping, slots)
        combined = gaudirt._C._allocate_paged_kv(1, slots, kv_heads, dim)
        key_pool, value_pool = combined[0, 0].transpose(0, 1), combined[0, 1].transpose(0, 1)
        combined[0, 0].copy_(key_base.transpose(0, 1).contiguous())
        combined[0, 1].copy_(value_base.transpose(0, 1).contiguous())
        key_storage = torch.empty((rows, kv_heads * dim + 1), dtype=torch.bfloat16, device="gaudirt:0")
        key_view = key_storage[:, 1:]
        key_view.copy_(key)
        result = torch.ops.gaudirt.paged_attn.default(query.to("gaudirt:0"), key_view,
            value.to("gaudirt:0"), key_pool, value_pool, metadata, dim, scale)
        gaudirt.synchronize()
        expected_key, expected_value = key_base.clone(), value_base.clone()
        expected_key[locations.long()] = key.reshape(rows, kv_heads, dim)
        expected_value[locations.long()] = value.reshape(rows, kv_heads, dim)
        actual_key = combined[0, 0].cpu().transpose(0, 1)
        actual_value = combined[0, 1].cpu().transpose(0, 1)
        assert torch.equal(actual_key.view(torch.int16), expected_key.view(torch.int16))
        assert torch.equal(actual_value.view(torch.int16), expected_value.view(torch.int16))
        ref64 = []
        for row in range(rows):
            indices = mapping[0, :positions[row].item() + 1].long()
            ks, vs = expected_key[indices][:, head_map, :], expected_value[indices][:, head_map, :]
            q = query[row].reshape(heads, dim).double()
            scores = torch.einsum("hd,thd->ht", q, ks.double()) * scale
            ref64.append(torch.einsum("ht,thd->hd", scores.softmax(-1), vs.double()))
        ref64 = direct_bf16(torch.stack(ref64).reshape(rows, -1))
        ref_pv = tiled_bf16_pv(query, expected_key, expected_value, requests, positions, mapping,
                               heads, kv_heads, dim, scale, 512)
        actual = result.cpu().double()
        errors = isolated_errors(actual, ref64, ref_pv)
        payload.extend(result.cpu().view(torch.int16).numpy().tobytes())
        print("tiled extend", rows, "prefix", prefix, "max relative l2", errors,
              "untouched KV exact", flush=True)
        del result, key_pool, value_pool, combined, metadata, key_view, key_storage
    counts = gaudirt.recipe_cache_stats()
    expected = [0, 11] if os.environ["GAUDIRT_TEST_CACHE"] == "cold" else [11, 0]
    assert counts[:2] == expected, counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("cache", counts, "output sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
