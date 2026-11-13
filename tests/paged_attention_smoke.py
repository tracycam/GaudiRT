"""Two-layer paged attention, dynamic buckets, and full KV pool audit."""

import hashlib
import math
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
from fp64_reference import direct_bf16
from paged_attention_reference import device_kv_tile, isolated_errors, tiled_bf16_pv
import gaudirt

torch.set_num_threads(4)
torch.random.default_generator.manual_seed(3127)
GEOMETRY = os.environ.get("GAUDIRT_TEST_GEOMETRY", "small")
assert GEOMETRY in ("small", "nanbeige")
HEADS, KV_HEADS, DIM = (4, 2, 64) if GEOMETRY == "small" else (56, 8, 128)
Q_WIDTH, KV_WIDTH = HEADS * DIM, KV_HEADS * DIM
REQUESTS, CONTEXT = (64, 128) if GEOMETRY == "small" else (64, 32)
SLOTS = REQUESTS * CONTEXT + 1
SCALE = 1.0 / math.sqrt(DIM)
SCALE_FP32 = torch.tensor(SCALE, dtype=torch.float32).item()
ROWS = (1, 3, 17, 137, 513, 1025, 4096) if GEOMETRY == "small" else (1, 7, 33, 137, 1025)


def metadata(rows):
    mapping = torch.arange(1, SLOTS, dtype=torch.int32).reshape(REQUESTS, CONTEXT)
    requests = torch.arange(rows, dtype=torch.int64) % REQUESTS
    positions = torch.arange(rows, dtype=torch.int64) // REQUESTS + 3
    locations = mapping[requests, positions].to(torch.int64)
    prepared = torch.classes.gaudirt.PagedMetadata(locations, requests, positions, mapping, SLOTS)
    return prepared, locations, requests, positions, mapping


def expected_unrounded(query, key_pool, value_pool, requests, positions, mapping, dtype):
    query = query.reshape(-1, HEADS, DIM).to(dtype)
    key_pool, value_pool = key_pool.to(dtype), value_pool.to(dtype)
    head_map = torch.arange(HEADS) // (HEADS // KV_HEADS)
    rows = []
    for row in range(query.size(0)):
        loc = mapping[requests[row], : positions[row] + 1].long()
        keys = key_pool[loc][:, head_map, :]
        values = value_pool[loc][:, head_map, :]
        scores = torch.einsum("hd,thd->ht", query[row], keys) * SCALE_FP32
        rows.append(torch.einsum("ht,thd->hd", torch.softmax(scores, dim=-1), values))
    return torch.stack(rows).reshape(query.size(0), Q_WIDTH)


def expected(query, key_pool, value_pool, requests, positions, mapping, fp64):
    dtype = torch.float64 if fp64 else torch.float32
    output = expected_unrounded(query, key_pool, value_pool, requests, positions, mapping, dtype)
    return direct_bf16(output) if fp64 else output.to(torch.bfloat16).double()


def source_tensor(cpu, sliced):
    if not sliced:
        return cpu.to("gaudirt:0"), None
    storage = torch.empty((cpu.size(0), cpu.size(1) + 1), dtype=cpu.dtype, device="gaudirt:0")
    view = storage[:, 1:]
    view.copy_(cpu)
    return view, storage


def run_layer(query_cpu, key_cpu, value_cpu, key_base, value_base, prepared,
              locations, requests, positions, mapping, sliced):
    query = query_cpu.to("gaudirt:0")
    key, key_storage = source_tensor(key_cpu, sliced)
    value, value_storage = source_tensor(value_cpu, sliced)
    combined = gaudirt._C._allocate_paged_kv(1, SLOTS, KV_HEADS, DIM)
    key_pool = combined[0, 0].transpose(0, 1)
    value_pool = combined[0, 1].transpose(0, 1)
    key_pool.copy_(key_base)
    value_pool.copy_(value_base)
    output = torch.ops.gaudirt.paged_attn.default(query, key, value, key_pool, value_pool,
                                                   prepared, DIM, SCALE)
    assert output.shape == query_cpu.shape and output.is_contiguous()
    assert output.untyped_storage().nbytes() >= (1 << (query_cpu.size(0) - 1).bit_length()) * Q_WIDTH * 2
    assert output.untyped_storage().nbytes() % 256 == 0
    expected_key, expected_value = key_base.clone(), value_base.clone()
    expected_key[locations] = key_cpu.reshape(-1, KV_HEADS, DIM)
    expected_value[locations] = value_cpu.reshape(-1, KV_HEADS, DIM)
    actual_key, actual_value = key_pool.cpu(), value_pool.cpu()
    assert torch.equal(actual_key.view(torch.int16), expected_key.view(torch.int16))
    assert torch.equal(actual_value.view(torch.int16), expected_value.view(torch.int16))
    actual = output.cpu().double()
    ref64 = expected(query_cpu, expected_key, expected_value, requests, positions, mapping, True)
    ref32 = expected(query_cpu, expected_key, expected_value, requests, positions, mapping, False)
    tiled = tiled_bf16_pv(query_cpu, expected_key, expected_value, requests, positions, mapping,
                          HEADS, KV_HEADS, DIM, SCALE_FP32, device_kv_tile(query_cpu.size(0)))
    assert torch.isfinite(ref32).all()
    denominator = torch.linalg.vector_norm(ref64, dim=-1).clamp_min(1e-30)
    error, tiled_error = isolated_errors(actual, ref64, tiled)
    floor = (torch.linalg.vector_norm(ref32 - ref64, dim=-1) / denominator).max().item()
    absolute = (actual - ref64).abs().max().item()
    payload = b"".join(value.view(torch.int16).numpy().tobytes()
                       for value in (output.cpu(), actual_key, actual_value))
    return (payload, error, floor, absolute, query, key, value, key_pool, value_pool,
            key_storage, value_storage, output, tiled_error)


def main():
    gaudirt.initialize(hbm_bytes=2048 * 1024 * 1024, pinned_bytes=16 * 1024 * 1024)
    payload, errors, floors, absolutes, tiled_errors = bytearray(), [], [], [], []
    for rows in ROWS:
        prepared, locations, requests, positions, mapping = metadata(rows)
        query_cpu, key_cpu, value_cpu = [
            torch.randn(rows, width, dtype=torch.bfloat16)
            for width in (Q_WIDTH, KV_WIDTH, KV_WIDTH)
        ]
        bases = [(torch.randn(SLOTS, KV_HEADS, DIM, dtype=torch.bfloat16),
                  torch.randn(SLOTS, KV_HEADS, DIM, dtype=torch.bfloat16)) for _ in range(2)]
        first = run_layer(query_cpu, key_cpu, value_cpu, *bases[0], prepared,
                          locations, requests, positions, mapping, sliced=rows in (17, 137))
        before_second = gaudirt.recipe_cache_stats()
        second = run_layer(query_cpu, key_cpu, value_cpu, *bases[1], prepared,
                           locations, requests, positions, mapping, sliced=rows in (17, 137))
        assert gaudirt.recipe_cache_stats() == before_second, "same-geometry layer recompiled"
        print("rows", rows, "recipe stats", before_second, "relative_l2", max(first[1], second[1]),
              "tiled_bf16_pv_error", max(first[12], second[12]),
              "cpu_fp32_floor_diagnostic", max(first[2], second[2]), flush=True)
        if rows == 3:
            before_pool = first[7].cpu()
            try:
                torch.ops.gaudirt.paged_attn.default(first[4], first[5], first[6], first[7], first[7],
                                                      prepared, DIM, SCALE)
            except RuntimeError as error:
                assert "disjoint" in str(error) or "combined storage" in str(error), error
            else:
                raise AssertionError("overlapping KV pools were accepted")
            assert torch.equal(first[7].cpu().view(torch.int16), before_pool.view(torch.int16))
            try:
                torch.ops.gaudirt.paged_attn.default(first[4], first[5], first[6], first[7], first[8],
                                                      prepared, 96, SCALE)
            except RuntimeError as error:
                assert "head dimension" in str(error)
            else:
                raise AssertionError("unsupported head dimension was accepted")
        for result in (first, second):
            payload.extend(result[0]); errors.append(result[1]); floors.append(result[2]); absolutes.append(result[3])
            tiled_errors.append(result[12])
        del result
        if rows == 137:
            one = torch.classes.gaudirt.PagedMetadata(locations[-1:].contiguous(), requests[-1:].contiguous(),
                                                      positions[-1:].contiguous(), mapping, SLOTS)
            replay = torch.ops.gaudirt.paged_attn.default(first[4][-1:], first[5][-1:], first[6][-1:],
                                                          first[7], first[8], one, DIM, SCALE)
            assert torch.equal(replay.cpu().view(torch.int16), first[11].cpu()[-1:].view(torch.int16))
            del one, replay
        del first, second, prepared
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 2 * len(ROWS)], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [2 * len(ROWS), 0], counts
    print("paged attention cache stats:", counts, "max_row_relative_l2:", max(errors),
          "max_tiled_bf16_pv_error:", max(tiled_errors),
          "max_cpu_fp32_floor_diagnostic:", max(floors), "max_abs:", max(absolutes),
          "output and KV sha256:", hashlib.sha256(payload).hexdigest(), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("paged attention two-layer device and KV audit passed", flush=True)


if __name__ == "__main__":
    main()
