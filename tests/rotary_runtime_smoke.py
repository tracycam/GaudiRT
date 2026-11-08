"""Unpadded rotary runtime inputs, shared recipes, and independent references."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
from fp64_reference import direct_bf16
import gaudirt

torch.set_num_threads(4)
torch.random.default_generator.manual_seed(1403)
REQUESTS, CONTEXT = 64, 64
SLOTS = REQUESTS * CONTEXT + 1


def metadata(rows):
    mapping = torch.arange(1, SLOTS, dtype=torch.int32).reshape(REQUESTS, CONTEXT)
    requests = torch.arange(rows, dtype=torch.int64) % REQUESTS
    positions = torch.arange(rows, dtype=torch.int64) // REQUESTS + 3
    locations = mapping[requests, positions].to(torch.int64)
    return torch.classes.gaudirt.PagedMetadata(locations, requests, positions, mapping, SLOTS), positions


def reference(input, cache, positions, head_dim, rotary_dim, neox, fp64):
    dtype = torch.float64 if fp64 else torch.float32
    rounded = direct_bf16 if fp64 else lambda value: value.to(torch.bfloat16).float()
    x = input.to(dtype).reshape(input.size(0), -1, head_dim)
    table = cache.to(dtype)[positions]
    cosine, sine = table.chunk(2, dim=-1)
    cosine, sine = cosine.unsqueeze(1), sine.unsqueeze(1)
    rotated, remainder = x[..., :rotary_dim], x[..., rotary_dim:]
    a, b = rotated.chunk(2, dim=-1) if neox else (rotated[..., ::2], rotated[..., 1::2])
    first = rounded(a * cosine - b * sine)
    second = rounded(b * cosine + a * sine)
    rotated = torch.cat((first, second), -1) if neox else torch.stack((first, second), -1).flatten(-2)
    return torch.cat((rotated, remainder), -1).reshape(input.shape).double()


def device_source(cpu, sliced):
    if not sliced:
        return cpu.to("gaudirt:0"), None
    storage = torch.empty((cpu.size(0), cpu.size(1) + 1), dtype=cpu.dtype, device="gaudirt:0")
    view = storage[:, 1:]
    view.copy_(cpu)
    return view, storage


def check(actual_device, cpu, cache, positions, head_dim, rotary_dim, neox):
    actual = actual_device.cpu().double()
    ref64 = reference(cpu, cache, positions, head_dim, rotary_dim, neox, True)
    ref32 = reference(cpu, cache, positions, head_dim, rotary_dim, neox, False)
    assert torch.isfinite(actual).all() and torch.isfinite(ref64).all() and torch.isfinite(ref32).all()
    denom = torch.linalg.vector_norm(ref64, dim=-1).clamp_min(1e-30)
    error = (torch.linalg.vector_norm(actual - ref64, dim=-1) / denom).max().item()
    floor = (torch.linalg.vector_norm(ref32 - ref64, dim=-1) / denom).max().item()
    absolute = (actual - ref64).abs().max().item()
    assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (error, floor, absolute)
    return actual.to(torch.bfloat16).view(torch.int16).numpy().tobytes(), error, floor, absolute


def layer(q_cpu, k_cpu, cache_cpu, cache_device, prepared, positions, head_dim, rotary_dim, neox, sliced):
    q, q_storage = device_source(q_cpu, sliced)
    k, k_storage = device_source(k_cpu, sliced)
    q_out, k_out = torch.ops.gaudirt.rotary_embedding.default(
        q, k, cache_device, prepared, head_dim, rotary_dim, neox
    )
    bucket = 1 << (q_cpu.size(0) - 1).bit_length()
    for actual, source in ((q_out, q_cpu), (k_out, k_cpu)):
        assert actual.shape == source.shape and actual.is_contiguous()
        assert actual.untyped_storage().nbytes() >= bucket * source.size(1) * 2
        assert actual.untyped_storage().nbytes() % 256 == 0
    result = [check(actual, source, cache_cpu, positions, head_dim, rotary_dim, neox)
              for actual, source in ((q_out, q_cpu), (k_out, k_cpu))]
    del q, k, q_out, k_out, q_storage, k_storage
    return result


def main():
    gaudirt.initialize(hbm_bytes=512 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    payload, errors, floors, absolutes = bytearray(), [], [], []
    cases = [(56, 8, 128, 128, neox, rows)
             for neox in (True, False) for rows in (1, 7, 33, 137, 1025)]
    cases += [(4, 2, 96, 62, neox, 7) for neox in (True, False)]
    for q_heads, k_heads, head_dim, rotary_dim, neox, rows in cases:
        prepared, positions = metadata(rows)
        cache_cpu = torch.randn(CONTEXT, rotary_dim, dtype=torch.bfloat16)
        cache_device = cache_cpu.to("gaudirt:0")
        q_cpu = torch.randn(rows, q_heads * head_dim, dtype=torch.bfloat16)
        k_cpu = torch.randn(rows, k_heads * head_dim, dtype=torch.bfloat16)
        first = layer(q_cpu, k_cpu, cache_cpu, cache_device, prepared, positions,
                      head_dim, rotary_dim, neox, sliced=rows == 137)
        counts = gaudirt.recipe_cache_stats()
        second = layer(q_cpu, k_cpu, cache_cpu, cache_device, prepared, positions,
                       head_dim, rotary_dim, neox, sliced=rows == 137)
        assert gaudirt.recipe_cache_stats() == counts, "same-geometry layer recompiled"
        for q_or_k in first + second:
            payload.extend(q_or_k[0]); errors.append(q_or_k[1]); floors.append(q_or_k[2]); absolutes.append(q_or_k[3])
        print("rows", rows, "heads", q_heads, k_heads, "rotary", rotary_dim, "neox", neox,
              "stats", counts, "relative_l2", max(first[0][1], first[1][1]), flush=True)
        if rows == 7 and q_heads == 4 and neox:
            query = q_cpu.to("gaudirt:0")
            key = k_cpu.to("gaudirt:0")
            for invalid_cache, invalid_rotary, message in (
                (cache_device[:32], rotary_dim, "cache"),
                (cache_device, rotary_dim - 1, "rotary dimension"),
            ):
                try:
                    torch.ops.gaudirt.rotary_embedding.default(query, key, invalid_cache, prepared,
                                                               head_dim, invalid_rotary, neox)
                except RuntimeError as error:
                    assert message in str(error), str(error)
                else:
                    raise AssertionError("invalid rotary inputs were accepted")
            del query, key, invalid_cache
        del prepared, cache_device, q_or_k
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, len(cases)], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [len(cases), 0], counts
    print("rotary runtime stats:", counts, "max_row_relative_l2:", max(errors),
          "max_cpu_fp32_floor:", max(floors), "max_abs:", max(absolutes),
          "output sha256:", hashlib.sha256(payload).hexdigest(), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("rotary runtime two-layer device audit passed", flush=True)


if __name__ == "__main__":
    main()
