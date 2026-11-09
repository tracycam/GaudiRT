"""Device integration of fused decode and short causal prefill with KV auditing."""

import atexit
import hashlib
import math
import os
import sys
import time

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
from paged_attention_reference import isolated_errors


def main():
    torch.set_num_threads(4)
    heads = int(os.environ.get("GAUDIRT_TEST_HEADS", "4"))
    kv_heads = int(os.environ.get("GAUDIRT_TEST_KV_HEADS", "2" if heads == 4 else "8"))
    dim = int(os.environ.get("GAUDIRT_TEST_HEAD_DIM", "128"))
    slots = int(os.environ.get("GAUDIRT_TEST_SLOTS", "512"))
    context = int(os.environ.get("GAUDIRT_TEST_CONTEXT", "256"))
    decode_prefix = int(os.environ.get("GAUDIRT_TEST_PREFIX", "143"))
    assert heads > 0 and kv_heads > 0 and heads % kv_heads == 0 and dim in (64, 128)
    assert kv_heads * dim % 128 == 0 and context >= max(192, decode_prefix + 1)
    assert slots > max(150, decode_prefix + 1)
    gaudirt.initialize(hbm_bytes=1024 * 1024 * 1024, pinned_bytes=8 * 1024 * 1024)
    mapping = (torch.arange(context, dtype=torch.int32) % (slots - 1) + 1).reshape(1, -1)
    head_map = torch.arange(heads) // (heads // kv_heads)
    scale = torch.tensor(1 / math.sqrt(dim), dtype=torch.float32).item()
    payload = bytearray()
    cases = (("decode", 1, decode_prefix), ("prefill", 137, 0))
    if selected := os.environ.get("GAUDIRT_TEST_CASE"):
        cases = tuple(case for case in cases if case[0] == selected)
        assert cases
    for case, rows, prefix in cases:
        rng = torch.Generator().manual_seed(900 + rows)
        query = torch.randn(rows, heads * dim, generator=rng).to(torch.bfloat16)
        key = torch.randn(rows, kv_heads * dim, generator=rng).to(torch.bfloat16)
        value = torch.randn(rows, kv_heads * dim, generator=rng).to(torch.bfloat16)
        key_base = torch.randn(slots, kv_heads, dim, generator=torch.Generator().manual_seed(512)).to(torch.bfloat16)
        value_base = torch.randn(slots, kv_heads, dim, generator=torch.Generator().manual_seed(513)).to(torch.bfloat16)
        positions = torch.arange(prefix, prefix + rows, dtype=torch.int32)
        requests = torch.zeros(rows, dtype=torch.int32)
        locations = mapping[0, positions.long()].contiguous()
        metadata = torch.classes.gaudirt.PagedMetadata(locations, requests, positions, mapping, slots)
        combined = gaudirt._C._allocate_paged_kv(1, slots, kv_heads, dim)
        key_pool, value_pool = combined[0, 0].transpose(0, 1), combined[0, 1].transpose(0, 1)
        combined[0, 0].copy_(key_base.transpose(0, 1).contiguous())
        combined[0, 1].copy_(value_base.transpose(0, 1).contiguous())
        device_query, device_key, device_value = (tensor.to("gaudirt:0") for tensor in (query, key, value))
        def launch():
            return torch.ops.gaudirt.paged_attn.default(device_query, device_key, device_value,
                key_pool, value_pool, metadata, dim, scale)
        result = launch()
        gaudirt.synchronize()
        if case == "decode" and os.environ.get("GAUDIRT_TEST_REQUIRE_FUSED") == "1":
            recipe_keys = gaudirt._C._recipe_cache_keys()
            assert any(key.startswith("paged_fused_") for key in recipe_keys), recipe_keys
            assert not any(key.startswith("paged_tile_") for key in recipe_keys), recipe_keys
        expected_key, expected_value = key_base.clone(), value_base.clone()
        expected_key[locations.long()] = key.reshape(rows, kv_heads, dim)
        expected_value[locations.long()] = value.reshape(rows, kv_heads, dim)
        actual_key = combined[0, 0].cpu().transpose(0, 1)
        actual_value = combined[0, 1].cpu().transpose(0, 1)
        assert torch.equal(actual_key.view(torch.int16), expected_key.view(torch.int16))
        assert torch.equal(actual_value.view(torch.int16), expected_value.view(torch.int16))
        ref64, ref_pv = [], []
        for row in range(rows):
            indices = mapping[0, :positions[row].item() + 1].long()
            ks = expected_key[indices][:, head_map, :]
            vs = expected_value[indices][:, head_map, :]
            q = query[row].reshape(heads, dim)
            scores64 = torch.einsum("hd,thd->ht", q.double(), ks.double()) * scale
            ref64.append(torch.einsum("ht,thd->hd", scores64.softmax(-1), vs.double()))
            scores32 = torch.einsum("hd,thd->ht", q.float(), ks.float()) * scale
            probabilities = scores32.softmax(-1).to(torch.bfloat16).float()
            ref_pv.append(torch.einsum("ht,thd->hd", probabilities, vs.float()))
        ref64 = direct_bf16(torch.stack(ref64).reshape(rows, -1))
        ref_pv = torch.stack(ref_pv).reshape(rows, -1).to(torch.bfloat16).double()
        actual = result.cpu().double()
        err64, err_pv = isolated_errors(actual, ref64, ref_pv)
        payload.extend(result.cpu().view(torch.int16).numpy().tobytes())
        payload.extend(actual_key.contiguous().view(torch.int16).numpy().tobytes())
        payload.extend(actual_value.contiguous().view(torch.int16).numpy().tobytes())
        print(case, "rows", rows, "max relative l2", err64, "BF16-PV", err_pv,
              "untouched KV exact", flush=True)
        if repetitions := int(os.environ.get("GAUDIRT_TEST_TIMING_REPETITIONS", "0")):
            assert repetitions > 0
            start, end = gaudirt.Event(enable_timing=True), gaudirt.Event(enable_timing=True)
            gaudirt.synchronize()
            start.record()
            host_start = time.perf_counter()
            for _ in range(repetitions):
                timed = launch()
            host_ms = 1000 * (time.perf_counter() - host_start) / repetitions
            end.record()
            end.synchronize()
            print("device_stream_ms", start.elapsed_time(end) / repetitions,
                  "host_submit_ms", host_ms, "repetitions", repetitions, flush=True)
    stats = gaudirt.recipe_cache_stats()
    count = 2 * len(cases)
    expected = [0, count] if os.environ["GAUDIRT_TEST_CACHE"] == "cold" else [count, 0]
    assert stats[:2] == expected, stats
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("cache", stats, "payload sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
