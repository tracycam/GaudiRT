"""Audit one KV update across mixed fused and tiled query blocks."""

import atexit
import hashlib
import math
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"


def no_bridge():
    assert "habana_frameworks" not in sys.modules
    with open("/proc/self/maps", encoding="utf-8") as maps:
        assert not any("habana_pytorch" in line for line in maps)


no_bridge()
atexit.register(no_bridge)

import gaudirt
import torch


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(41753)
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    heads, kv_heads, dim = 4, 2, 64
    context, requests_count = 1024, 3
    slots = context * requests_count + 1
    prefixes, lengths = (0, 512, 800), (19, 137, 1)
    mapping = torch.arange(1, slots, dtype=torch.int32).reshape(requests_count, context)
    requests = torch.cat([torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(lengths)])
    positions = torch.cat([torch.arange(p, p + n, dtype=torch.int32) for p, n in zip(prefixes, lengths)])
    locations = mapping[requests.long(), positions.long()].contiguous()
    rows = len(locations)
    metadata = torch.classes.gaudirt.PagedMetadata(locations, requests, positions, mapping, slots)
    q, k, v = [torch.randn(rows, width, dtype=torch.bfloat16) for width in
               (heads * dim, kv_heads * dim, kv_heads * dim)]
    original_k, original_v = [torch.randn(slots, kv_heads, dim, dtype=torch.bfloat16) for _ in range(2)]
    combined = gaudirt._C._allocate_paged_kv(1, slots, kv_heads, dim)
    combined[0, 0].copy_(original_k.transpose(0, 1).contiguous())
    combined[0, 1].copy_(original_v.transpose(0, 1).contiguous())
    key_pool, value_pool = combined[0, 0].transpose(0, 1), combined[0, 1].transpose(0, 1)
    key_storage = torch.empty((rows, kv_heads * dim + 1), dtype=torch.bfloat16, device="gaudirt:0")
    key_view = key_storage[:, 1:]
    key_view.copy_(k)
    result = torch.ops.gaudirt.paged_attn.default(q.to("gaudirt:0"), key_view, v.to("gaudirt:0"),
                                                   key_pool, value_pool, metadata, dim, 1 / math.sqrt(dim))
    actual_k, actual_v = combined[0, 0].cpu().transpose(0, 1), combined[0, 1].cpu().transpose(0, 1)
    expected_k, expected_v = original_k.clone(), original_v.clone()
    expected_k[locations.long()] = k.reshape(rows, kv_heads, dim)
    expected_v[locations.long()] = v.reshape(rows, kv_heads, dim)
    assert torch.equal(actual_k, expected_k) and torch.equal(actual_v, expected_v)
    head_map = torch.arange(heads) // (heads // kv_heads)
    actual = result.cpu().double()
    errors = []
    for row in range(rows):
        indices = mapping[requests[row], :positions[row] + 1].long()
        keys = expected_k[indices][:, head_map].double()
        values = expected_v[indices][:, head_map].double()
        scores = torch.einsum("hd,thd->ht", q[row].reshape(heads, dim).double(), keys) / math.sqrt(dim)
        truth = torch.einsum("ht,thd->hd", scores.softmax(-1), values).reshape(-1)
        assert torch.isfinite(truth).all() and torch.isfinite(actual[row]).all()
        errors.append((torch.linalg.vector_norm(actual[row] - truth) /
                       torch.linalg.vector_norm(truth).clamp_min(1e-30)).item())
    assert max(errors) <= 1e-2, max(errors)
    payload = b"".join(t.contiguous().view(torch.int16).numpy().tobytes()
                       for t in (result.cpu(), actual_k, actual_v))
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[1] == 0, stats
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("mixed fused/tiled KV audit", "max_relative_l2", max(errors), "cache", stats,
          "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
