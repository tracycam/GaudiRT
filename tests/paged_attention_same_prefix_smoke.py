"""Compare the last M=137 attention row with M=1 on the same KV prefix."""

import math
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch

import gaudirt
import paged_attention_smoke as reference
from paged_attention_reference import device_kv_tile, isolated_errors, tiled_bf16_pv


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(28194)
    assert (reference.HEADS, reference.KV_HEADS, reference.DIM) == (56, 8, 128)
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    prefix, rows, context = {
        "0": (17, 137, 160), "1": (511, 137, 1024), "2": (3959, 137, 4224)
    }[os.environ.get("GAUDIRT_TEST_LONG_PREFIX", "0")]
    slots = context + 1
    mapping = torch.arange(1, slots, dtype=torch.int32).reshape(1, context)
    requests = torch.zeros(rows, dtype=torch.int64)
    positions = torch.arange(prefix, prefix + rows, dtype=torch.int64)
    locations = mapping[requests, positions].long()
    query, key, value = [torch.randn(rows, width, dtype=torch.bfloat16)
                         for width in (7168, 1024, 1024)]
    base_key = torch.zeros(slots, 8, 128, dtype=torch.bfloat16)
    base_value = torch.zeros_like(base_key)
    base_key[1:prefix + 1] = torch.randn(prefix, 8, 128, dtype=torch.bfloat16)
    base_value[1:prefix + 1] = torch.randn(prefix, 8, 128, dtype=torch.bfloat16)
    meta_many = torch.classes.gaudirt.PagedMetadata(locations, requests, positions, mapping, slots)
    key_pool, value_pool = base_key.to("gaudirt:0"), base_value.to("gaudirt:0")
    many = torch.ops.gaudirt.paged_attn.default(
        query.to("gaudirt:0"), key.to("gaudirt:0"), value.to("gaudirt:0"),
        key_pool, value_pool, meta_many, 128, 1 / math.sqrt(128)
    )
    key_after, value_after = key_pool.cpu(), value_pool.cpu()
    assert torch.equal(key_after[locations], key.reshape(rows, 8, 128))
    assert torch.equal(value_after[locations], value.reshape(rows, 8, 128))
    untouched = torch.ones(slots, dtype=torch.bool)
    untouched[locations] = False
    assert torch.equal(key_after[untouched], base_key[untouched])
    assert torch.equal(value_after[untouched], base_value[untouched])

    key_before_last, value_before_last = key_after.clone(), value_after.clone()
    key_before_last[locations[-1]] = base_key[locations[-1]]
    value_before_last[locations[-1]] = base_value[locations[-1]]
    key_single, value_single = key_before_last.to("gaudirt:0"), value_before_last.to("gaudirt:0")
    meta_single = torch.classes.gaudirt.PagedMetadata(
        locations[-1:], requests[-1:], positions[-1:], mapping, slots
    )
    single = torch.ops.gaudirt.paged_attn.default(
        query[-1:].to("gaudirt:0"), key[-1:].to("gaudirt:0"), value[-1:].to("gaudirt:0"),
        key_single, value_single, meta_single, 128, 1 / math.sqrt(128)
    )
    assert torch.equal(key_single.cpu().view(torch.int16), key_after.view(torch.int16))
    assert torch.equal(value_single.cpu().view(torch.int16), value_after.view(torch.int16))
    many_last, one = many[-1:].cpu().double(), single.cpu().double()
    truth = reference.expected(query[-1:], key_after, value_after, requests[-1:],
                               positions[-1:], mapping, True)
    floor = reference.expected(query[-1:], key_after, value_after, requests[-1:],
                               positions[-1:], mapping, False)
    tiled_args = (query[-1:], key_after, value_after, requests[-1:], positions[-1:], mapping,
                  reference.HEADS, reference.KV_HEADS, reference.DIM, reference.SCALE_FP32)
    many_tiled = tiled_bf16_pv(*tiled_args, device_kv_tile(rows))
    one_tiled = tiled_bf16_pv(*tiled_args, device_kv_tile(1))
    assert torch.isfinite(floor).all()
    many_error, many_tile_error = isolated_errors(many_last, truth, many_tiled)
    one_error, one_tile_error = isolated_errors(one, truth, one_tiled)
    denom = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-30)
    floor_error = (torch.linalg.vector_norm(floor - truth, dim=-1) / denom).max().item()
    delta = (torch.linalg.vector_norm(many_last - one, dim=-1) / denom).max().item()
    threshold = max(3e-3, 2 * floor_error)
    assert delta <= threshold and delta <= 1e-2
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts == [0, 4, 4], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts == [4, 0, 4], counts
    print("same-prefix attention", "many_error", many_error, "single_error", one_error,
          "many_tiled_error", many_tile_error, "single_tiled_error", one_tile_error,
          "cpu_fp32_floor", floor_error, "row_difference", delta,
          "max_abs_between", (many_last - one).abs().max().item(),
          "threshold", threshold, "cache", counts, flush=True)


if __name__ == "__main__":
    main()
