"""The last chunked row and a single query share the same online KV prefix."""

import atexit
import hashlib
import math
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"


def isolation():
    assert "habana_frameworks" not in sys.modules
    with open("/proc/self/maps", encoding="utf-8") as maps:
        assert not any("habana_pytorch" in line or "gaudi-pytorch-bridge" in line for line in maps)


isolation()
atexit.register(isolation)

import torch
import gaudirt
from fp64_reference import direct_bf16
from paged_attention_reference import isolated_errors, tiled_bf16_pv


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(7319)
    heads = int(os.environ.get("GAUDIRT_TEST_HEADS", "4"))
    kv_heads = int(os.environ.get("GAUDIRT_TEST_KV_HEADS", "2"))
    dim = int(os.environ.get("GAUDIRT_TEST_HEAD_DIM", "128"))
    prefix = int(os.environ.get("GAUDIRT_TEST_PREFIX", "1024"))
    rows = 137
    total = prefix + rows
    slots = 1 << (total + 1).bit_length()
    assert heads % kv_heads == 0 and dim in (64, 128) and kv_heads * dim % 128 == 0
    gaudirt.initialize(hbm_bytes=2 * 1024**3, pinned_bytes=8 * 1024**2)
    mapping = torch.arange(1, slots, dtype=torch.int32).reshape(1, slots - 1)
    requests = torch.zeros(rows, dtype=torch.int32)
    positions = torch.arange(prefix, total, dtype=torch.int32)
    locations = mapping[0, positions.long()].contiguous()
    query = torch.randn(rows, heads * dim, dtype=torch.bfloat16)
    key = torch.randn(rows, kv_heads * dim, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    base_keys = torch.zeros(slots, kv_heads, dim, dtype=torch.bfloat16)
    base_values = torch.zeros_like(base_keys)
    base_keys[1:prefix + 1] = torch.randn(prefix, kv_heads, dim, dtype=torch.bfloat16)
    base_values[1:prefix + 1] = torch.randn_like(base_keys[1:prefix + 1])
    many_metadata = torch.classes.gaudirt.PagedMetadata(locations, requests, positions, mapping, slots)
    many_pool = gaudirt._C._allocate_paged_kv(1, slots, kv_heads, dim)
    many_pool[0, 0].copy_(base_keys.transpose(0, 1).contiguous())
    many_pool[0, 1].copy_(base_values.transpose(0, 1).contiguous())
    q, k, v = (tensor.to("gaudirt:0") for tensor in (query, key, value))
    scale = torch.tensor(1 / math.sqrt(dim), dtype=torch.float32).item()
    many = torch.ops.gaudirt.paged_attn.default(q, k, v, many_pool[0, 0].transpose(0, 1),
        many_pool[0, 1].transpose(0, 1), many_metadata, dim, scale).cpu()
    expected_keys, expected_values = base_keys.clone(), base_values.clone()
    expected_keys[locations.long()] = key.reshape(rows, kv_heads, dim)
    expected_values[locations.long()] = value.reshape(rows, kv_heads, dim)
    actual_keys = many_pool[0, 0].cpu().transpose(0, 1)
    actual_values = many_pool[0, 1].cpu().transpose(0, 1)
    assert torch.equal(actual_keys, expected_keys) and torch.equal(actual_values, expected_values)

    previous_keys, previous_values = expected_keys.clone(), expected_values.clone()
    previous_keys[locations[-1]] = base_keys[locations[-1]]
    previous_values[locations[-1]] = base_values[locations[-1]]
    one_pool = gaudirt._C._allocate_paged_kv(1, slots, kv_heads, dim)
    one_pool[0, 0].copy_(previous_keys.transpose(0, 1).contiguous())
    one_pool[0, 1].copy_(previous_values.transpose(0, 1).contiguous())
    one_metadata = torch.classes.gaudirt.PagedMetadata(
        locations[-1:], requests[-1:], positions[-1:], mapping, slots)
    one = torch.ops.gaudirt.paged_attn.default(q[-1:], k[-1:], v[-1:],
        one_pool[0, 0].transpose(0, 1), one_pool[0, 1].transpose(0, 1),
        one_metadata, dim, scale).cpu()
    assert torch.equal(one_pool[0, 0].cpu().transpose(0, 1), expected_keys)
    assert torch.equal(one_pool[0, 1].cpu().transpose(0, 1), expected_values)
    maximum = (one.double() - many[-1:].double()).abs().max().item()
    assert torch.equal(one.view(torch.int16), many[-1:].view(torch.int16)), maximum

    head_map = torch.arange(heads) // (heads // kv_heads)
    keys = expected_keys[1:total + 1][:, head_map].double()
    values = expected_values[1:total + 1][:, head_map].double()
    scores = torch.einsum("hd,thd->ht", query[-1].reshape(heads, dim).double(), keys) * scale
    truth = direct_bf16(torch.einsum("ht,thd->hd", scores.softmax(-1), values).reshape(1, -1))
    tiled = tiled_bf16_pv(query[-1:], expected_keys, expected_values, requests[-1:], positions[-1:],
                          mapping, heads, kv_heads, dim, scale, 512)
    errors = isolated_errors(one.double(), truth, tiled)
    stats = gaudirt.recipe_cache_stats()
    mode = os.environ["GAUDIRT_TEST_CACHE"]
    assert (mode == "cold" and stats[1] > 0) or (mode == "warm" and stats[0] > 0 and stats[1] == 0), stats
    digest = hashlib.sha256()
    for tensor in (one, actual_keys.contiguous(), actual_values.contiguous()):
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "w", encoding="utf-8") as output:
            output.write(digest.hexdigest() + "\n")
    print("same-prefix rows", rows, "prefix", prefix, "max_abs", maximum,
          "relative_l2", errors, "KV exact", "cache", stats, "sha256", digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
