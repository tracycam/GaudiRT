"""One FX graph for QKV projection, split and validated RoPE."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F
import gaudirt
import gaudirt.compiler as compiler
from fp64_reference import direct_bf16
from rotary_runtime_smoke import reference as rotary_reference

INPUT_WIDTH = 256


def metadata(rows):
    requests, context = 64, 128
    slots = requests * context + 1
    mapping = torch.arange(1, slots, dtype=torch.int32).reshape(requests, context)
    req = torch.arange(rows, dtype=torch.int64) % requests
    positions = torch.arange(rows, dtype=torch.int64) // requests + 1
    locations = mapping[req, positions].to(torch.int64)
    value = torch.classes.gaudirt.PagedMetadata(locations, req, positions, mapping, slots)
    return value, positions


def graph(x, weight, positions, header, cache):
    q, k, v = F.linear(x, weight).split([128, 64, 64], dim=-1)
    q, k = torch.ops.gaudirt._rotary_embedding_validated.default(
        q, k, cache, positions, header, 128, 64, 64, True
    )
    return q, k, v


def second_layer(x, weight, positions, header, cache):
    return graph(x, weight, positions, header, cache)


def check(outputs, x, weight, cache, positions):
    projection64 = direct_bf16(x.double() @ weight.double().t())
    projection32 = (x.float() @ weight.float().t()).to(torch.bfloat16).double()
    outputs64 = list(projection64.split([128, 64, 64], dim=-1))
    outputs32 = list(projection32.split([128, 64, 64], dim=-1))
    for i in (0, 1):
        outputs64[i] = rotary_reference(outputs64[i], cache, positions, 64, 64, True, True)
        outputs32[i] = rotary_reference(outputs32[i], cache, positions, 64, 64, True, False)
    errors, floors, absolutes, payload = [], [], [], bytearray()
    for actual, ref64, ref32 in zip(outputs, outputs64, outputs32):
        actual = actual.cpu().double()
        assert torch.isfinite(actual).all() and torch.isfinite(ref64).all() and torch.isfinite(ref32).all()
        denom = torch.linalg.vector_norm(ref64, dim=-1).clamp_min(1e-30)
        error = (torch.linalg.vector_norm(actual - ref64, dim=-1) / denom).max().item()
        floor = (torch.linalg.vector_norm(ref32 - ref64, dim=-1) / denom).max().item()
        absolute = (actual - ref64).abs().max().item()
        assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (error, floor, absolute)
        payload.extend(actual.to(torch.bfloat16).view(torch.int16).numpy().tobytes())
        errors.append(error); floors.append(floor); absolutes.append(absolute)
    return payload, max(errors), max(floors), max(absolutes)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(7219)
    gaudirt.initialize(hbm_bytes=512 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    weights_cpu = [(torch.randn(256, INPUT_WIDTH) * 0.05).to(torch.bfloat16) for _ in range(2)]
    weights = [value.to("gaudirt:0") for value in weights_cpu]
    cache_cpu = torch.randn(128, 64, dtype=torch.bfloat16)
    cache = gaudirt._reserve_rotary_cache(cache_cpu)
    first_graph = compiler.compile(graph, dynamic_arg_indices=(0, 2))
    second_graph = compiler.compile(second_layer, dynamic_arg_indices=(0, 2))
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for rows in (1, 3, 17, 137, 513, 2048, 4096):
        prepared, positions_cpu = metadata(rows)
        positions, header = prepared.rotary_positions(), prepared.rotary_header()
        x_cpu = torch.randn(rows, INPUT_WIDTH, dtype=torch.bfloat16)
        x = x_cpu.to("gaudirt:0")
        first = first_graph(x, weights[0], positions, header, cache)
        before = gaudirt.recipe_cache_stats()
        second = second_graph(x, weights[1], positions, header, cache)
        assert gaudirt.recipe_cache_stats() == before, "same-geometry layer recompiled"
        for result, weight in ((first, weights_cpu[0]), (second, weights_cpu[1])):
            data, error, floor, absolute = check(result, x_cpu, weight, cache_cpu, positions_cpu)
            payload.extend(data); errors.append(error); floors.append(floor); absolutes.append(absolute)
        print("FX linear/split/RoPE rows", rows, "recipe stats", before, flush=True)
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 7], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [7, 0], counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("FX rotary input width", INPUT_WIDTH, "recipes", counts, "max row error", max(errors), "FP32 floor", max(floors),
          "max abs", max(absolutes), "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
