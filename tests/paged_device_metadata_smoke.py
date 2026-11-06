"""Device-resident SGLang-shaped page maps with validated mixed-varlen metadata."""

import hashlib
import math
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import paged_attention_smoke as attention

torch.set_num_threads(4)
torch.random.default_generator.manual_seed(8017)
GEOMETRY = os.environ.get("GAUDIRT_TEST_GEOMETRY", "small")
assert GEOMETRY in ("small", "nanbeige")
REQUESTS, CONTEXT = (8, 1024) if GEOMETRY == "small" else (4, 512)
SLOTS = REQUESTS * CONTEXT + 1
attention.REQUESTS, attention.CONTEXT, attention.SLOTS = REQUESTS, CONTEXT, SLOTS
if GEOMETRY == "nanbeige":
    attention.HEADS, attention.KV_HEADS, attention.DIM = 56, 8, 128
    attention.Q_WIDTH = attention.HEADS * attention.DIM
    attention.KV_WIDTH = attention.KV_HEADS * attention.DIM
    attention.SCALE = 1.0 / math.sqrt(attention.DIM)
    attention.SCALE_FP32 = torch.tensor(attention.SCALE, dtype=torch.float32).item()
ROWS = (1, 7, 33, 137, 1025, 4096) if GEOMETRY == "small" else (1, 7, 33, 137, 1025)
REQUEST_ORDER = [7, 2, 5, 1, 6, 0, 4, 3] if GEOMETRY == "small" else [3, 1, 2, 0]


def data(rows):
    mapping = torch.arange(1, SLOTS, dtype=torch.int32).reshape(REQUESTS, CONTEXT)
    batch = min(rows, REQUESTS)
    requests = torch.tensor(REQUEST_ORDER[:batch], dtype=torch.int64)
    lengths = [rows // batch + int(i < rows % batch) for i in range(batch)]
    prefixes = [2 + i for i in range(batch)]
    row_requests = torch.repeat_interleave(requests, torch.tensor(lengths))
    positions = torch.cat([torch.arange(prefix, prefix + length, dtype=torch.int64)
                           for prefix, length in zip(prefixes, lengths)])
    locations = mapping[row_requests, positions].to(torch.int64)
    return locations, requests, positions, mapping, prefixes, lengths, row_requests


def to_device(source, sliced=False):
    if not sliced:
        return source.to("gaudirt:0"), None
    storage = torch.empty(source.numel() * 2, dtype=source.dtype, device="gaudirt:0")
    view = storage[::2]
    view.copy_(source)
    return view, storage


def prepare(locations, requests, positions, mapping, prefixes, lengths, sliced=False):
    device_locs, loc_storage = to_device(locations, sliced)
    device_requests, req_storage = to_device(requests, sliced)
    device_positions, pos_storage = to_device(positions, sliced)
    device_mapping = mapping.to("gaudirt:0")
    prepared = torch.classes.gaudirt.PagedMetadata.from_device(
        device_locs, device_requests, device_positions, device_mapping, prefixes, lengths, SLOTS
    )
    return prepared, (device_locs, device_requests, device_positions, device_mapping,
                      loc_storage, req_storage, pos_storage)


def rejected(values, code):
    try:
        prepared, keep = prepare(*values)
    except RuntimeError as error:
        assert f"code {code}" in str(error), str(error)
    else:
        del prepared, keep
        raise AssertionError("invalid device metadata was accepted")


def main():
    gaudirt.initialize(hbm_bytes=(256 if GEOMETRY == "small" else 512) * 1024 * 1024,
                       pinned_bytes=4 * 1024 * 1024)
    payload, errors, floors, absolutes, tiled_errors = bytearray(), [], [], [], []
    for rows in ROWS:
        locations, requests, positions, mapping, prefixes, lengths, row_requests = data(rows)
        if rows in (33, 1025):
            positions = positions.to(torch.int32)
        prepared, keep = prepare(locations, requests, positions, mapping, prefixes, lengths, sliced=rows == 137)
        assert prepared.rows() == rows and prepared.slots() == SLOTS
        indices = gaudirt.validated_last_token_indices(prepared)
        assert indices.dtype == torch.int32 and getattr(indices, "_gaudirt_validated_rows") == rows
        expected_indices = torch.cumsum(torch.tensor(lengths, dtype=torch.int32), dim=0) - 1
        assert torch.equal(indices.cpu().long(), expected_indices)
        payload.extend(indices.cpu().numpy().tobytes())
        if rows < 4096:
            query_cpu, key_cpu, value_cpu = [torch.randn(rows, width, dtype=torch.bfloat16)
                                            for width in (attention.Q_WIDTH, attention.KV_WIDTH, attention.KV_WIDTH)]
            bases = [(torch.randn(SLOTS, attention.KV_HEADS, attention.DIM, dtype=torch.bfloat16),
                      torch.randn(SLOTS, attention.KV_HEADS, attention.DIM, dtype=torch.bfloat16)) for _ in range(2)]
            first = attention.run_layer(query_cpu, key_cpu, value_cpu, *bases[0], prepared,
                                        locations, row_requests, positions, mapping, sliced=rows == 137)
            before_second = gaudirt.recipe_cache_stats()
            second = attention.run_layer(query_cpu, key_cpu, value_cpu, *bases[1], prepared,
                                         locations, row_requests, positions, mapping, sliced=rows == 137)
            assert gaudirt.recipe_cache_stats() == before_second, "same-geometry layer recompiled"
            for result in (first, second):
                payload.extend(result[0]); errors.append(result[1]); floors.append(result[2]); absolutes.append(result[3])
                tiled_errors.append(result[12])
            del first, second, result
        print("device metadata rows", rows, "recipe stats", gaudirt.recipe_cache_stats(), flush=True)
        del prepared, keep, indices

    locations, requests, positions, mapping, prefixes, lengths, row_requests = data(7)
    invalid = locations.clone(); invalid[0] += 1 << 32
    rejected((invalid, requests, positions, mapping, prefixes, lengths), 1)
    invalid = requests.clone(); invalid[0] += 1 << 32
    rejected((locations, invalid, positions, mapping, prefixes, lengths), 2)
    invalid = positions.clone(); invalid[0] += 1 << 32
    rejected((locations, requests, invalid, mapping, prefixes, lengths), 1)
    invalid = locations.clone(); invalid[0] += 1
    rejected((invalid, requests, positions, mapping, prefixes, lengths), 3)
    bad_mapping = mapping.clone(); bad_mapping[requests[0], 0] = -1
    rejected((locations, requests, positions, bad_mapping, prefixes, lengths), 6)
    bad_mapping = mapping.clone(); bad_mapping[requests[1], 0] = locations[0]
    rejected((locations, requests, positions, bad_mapping, prefixes, lengths), 7)
    duplicate_requests = torch.tensor([0, 0], dtype=torch.int64)
    duplicate_positions = torch.tensor([2, 2], dtype=torch.int64)
    duplicate_locations = mapping[duplicate_requests, duplicate_positions].to(torch.int64)
    rejected((duplicate_locations, duplicate_requests, duplicate_positions, mapping, [2, 2], [1, 1]), 4)
    try:
        prepared, keep = prepare(locations, requests, positions, mapping, prefixes, [lengths[0] + 1] + lengths[1:])
    except RuntimeError as error:
        assert "sequence lengths" in str(error)
    else:
        del prepared, keep
        raise AssertionError("inconsistent sequence lengths were accepted")
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[0] == 0 and counts[1] > 0, counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[0] > 0 and counts[1] == 0, counts
    print("device metadata stats:", counts, "max_row_relative_l2:", max(errors),
          "max_tiled_bf16_pv_error:", max(tiled_errors),
          "max_cpu_fp32_floor_diagnostic:", max(floors), "max_abs:", max(absolutes),
          "output and KV sha256:", hashlib.sha256(payload).hexdigest(), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("device metadata and two-layer paged attention audit passed", flush=True)


if __name__ == "__main__":
    main()
