"""Audit 128 target-head decode steps against frozen CPU references."""

import hashlib
import math
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch

import gaudirt
from fp64_reference import direct_bf16
from paged_attention_reference import device_kv_tile, isolated_errors, tiled_bf16_pv


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(8721)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    heads, kv_heads, head_dim, steps = 56, 8, 128, 128
    prefix = 512 if os.environ.get("GAUDIRT_TEST_LONG_PREFIX") == "1" else 0
    context, slots = prefix + steps, prefix + steps + (2 if prefix else 1)
    scale = 1 / math.sqrt(head_dim)
    scale32 = torch.tensor(scale, dtype=torch.float32).item()
    mapping = torch.arange(1, context + 1, dtype=torch.int32).view(1, context)
    queries = (torch.randn(steps, heads * head_dim) * 0.125).to(torch.bfloat16)
    keys = (torch.randn(steps, kv_heads * head_dim) * 0.125).to(torch.bfloat16)
    values = (torch.randn(steps, kv_heads * head_dim) * 0.125).to(torch.bfloat16)
    expected_key = torch.zeros(slots, kv_heads, head_dim, dtype=torch.bfloat16)
    expected_value = torch.zeros_like(expected_key)
    if prefix:
        expected_key[1:prefix + 1] = (torch.randn(prefix, kv_heads, head_dim) * 0.125).to(torch.bfloat16)
        expected_value[1:prefix + 1] = (torch.randn(prefix, kv_heads, head_dim) * 0.125).to(torch.bfloat16)
    key_pool, value_pool = expected_key.to("gaudirt:0"), expected_value.to("gaudirt:0")
    head_map = torch.arange(heads) // (heads // kv_heads)
    audit = os.environ.get("GAUDIRT_STATE_AUDIT") == "1"
    payload = bytearray()
    worst, worst_floor, worst_absolute, worst_tiled = 0.0, 0.0, 0.0, 0.0
    kv_tile = None
    for step in range(steps):
        location = torch.tensor([prefix + step + 1], dtype=torch.int64)
        requests = torch.tensor([0], dtype=torch.int64)
        position = torch.tensor([prefix + step], dtype=torch.int64)
        metadata = torch.classes.gaudirt.PagedMetadata(location, requests, position, mapping, slots)
        output = torch.ops.gaudirt.paged_attn.default(
            queries[step:step + 1].to("gaudirt:0"), keys[step:step + 1].to("gaudirt:0"),
            values[step:step + 1].to("gaudirt:0"), key_pool, value_pool, metadata, head_dim, scale,
        )
        expected_key[prefix + step + 1] = keys[step].view(kv_heads, head_dim)
        expected_value[prefix + step + 1] = values[step].view(kv_heads, head_dim)
        if audit:
            assert torch.equal(key_pool.cpu().view(torch.int16), expected_key.view(torch.int16))
            assert torch.equal(value_pool.cpu().view(torch.int16), expected_value.view(torch.int16))
        actual_bf16 = output.cpu()
        actual = actual_bf16.double().view(heads, head_dim)

        def reference(dtype):
            query = queries[step].view(heads, head_dim).to(dtype)
            key = expected_key[1:prefix + step + 2, head_map].to(dtype)
            value = expected_value[1:prefix + step + 2, head_map].to(dtype)
            scores = torch.einsum("hd,thd->ht", query, key) * scale32
            return torch.einsum("ht,thd->hd", torch.softmax(scores, dim=-1), value)

        truth = direct_bf16(reference(torch.float64))
        floor = reference(torch.float32).to(torch.bfloat16).double()
        if kv_tile is None:
            kv_tile = device_kv_tile(1)
        tiled = tiled_bf16_pv(queries[step:step + 1], expected_key, expected_value,
                              requests, position, mapping, heads, kv_heads, head_dim, scale32, kv_tile)
        assert torch.isfinite(floor).all()
        error, tiled_error = isolated_errors(actual.reshape(1, -1), truth.reshape(1, -1), tiled)
        denom = torch.linalg.vector_norm(truth.reshape(-1)).clamp_min(1e-30)
        floor_error = (torch.linalg.vector_norm((floor - truth).reshape(-1)) / denom).item()
        absolute = (actual - truth).abs().max().item()
        worst, worst_floor, worst_absolute = max(worst, error), max(worst_floor, floor_error), max(worst_absolute, absolute)
        worst_tiled = max(worst_tiled, tiled_error)
        payload.extend(actual_bf16.contiguous().view(torch.uint8).numpy().tobytes())
    actual_key, actual_value = key_pool.cpu(), value_pool.cpu()
    assert torch.equal(actual_key.view(torch.int16), expected_key.view(torch.int16))
    assert torch.equal(actual_value.view(torch.int16), expected_value.view(torch.int16))
    for tensor in (actual_key, actual_value):
        payload.extend(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output_file:
            output_file.write(payload)
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[0] == 0 and stats[1] > 0, stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[0] > 0 and stats[1] == 0, stats
    print("128-step paged GQA", "prefix", prefix, "audit", audit, "stats", stats,
          "max_row_relative_l2", worst, "max_tiled_bf16_pv_error", worst_tiled,
          "max_fp32_floor_diagnostic", worst_floor, "max_abs", worst_absolute,
          "output_and_kv_sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
