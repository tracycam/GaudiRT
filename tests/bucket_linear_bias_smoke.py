"""Validate biased FX linear over all buckets against independent FP64."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
torch.manual_seed(875)
import gaudirt
import gaudirt.compiler as compiler


def linear(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)


gaudirt.initialize(hbm_bytes=128 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
backend = compiler.gaudirt_backend
captures = []


def capture(graph_module, example_inputs):
    captures.append(1)
    return backend(graph_module, example_inputs)


compiler.gaudirt_backend = capture
digest = hashlib.sha256()
output_bytes = bytearray()
errors, floors = [], []
for dtype in (torch.bfloat16, torch.float32):
    weight_cpu = torch.randn(64, 64, dtype=dtype)
    bias_cpu = torch.randn(64, dtype=dtype)
    weight = weight_cpu.to("gaudirt:0")
    bias = bias_cpu.to("gaudirt:0")
    compiled = compiler.compile(linear)
    for m in (1, 2, 3, 5, 9, 17, 33, 65, 137, 257, 513, 2048, 4096):
        input_cpu = torch.randn(m, 64, dtype=dtype)
        x = input_cpu.to("gaudirt:0")
        result = compiled(x, weight, bias).cpu()
        ref64 = input_cpu.double() @ weight_cpu.double().t() + bias_cpu.double()
        ref32 = (input_cpu.float() @ weight_cpu.float().t() + bias_cpu.float()).double()
        actual = result.double()
        assert torch.isfinite(actual).all() and torch.isfinite(ref64).all() and torch.isfinite(ref32).all()
        denom = torch.linalg.vector_norm(ref64, dim=-1).clamp_min(1e-30)
        error = (torch.linalg.vector_norm(actual - ref64, dim=-1) / denom).max().item()
        floor = (torch.linalg.vector_norm(ref32 - ref64, dim=-1) / denom).max().item()
        assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (dtype, m, error, floor)
        payload = result.view(torch.int16 if dtype == torch.bfloat16 else torch.int32).numpy().tobytes()
        digest.update(payload)
        output_bytes.extend(payload)
        errors.append(error)
        floors.append(floor)
    # A bias aliasing the input and a strided bias both require distinct
    # persistent sections, without changing the cached graph.
    alias_cpu = torch.randn(1, 64, dtype=dtype)
    alias = alias_cpu.to("gaudirt:0")
    bias_storage_cpu = torch.randn(128, dtype=dtype)
    bias_storage = bias_storage_cpu.to("gaudirt:0")
    for device_bias, cpu_bias in ((alias[0], alias_cpu[0]),
                                  (bias_storage[::2], bias_storage_cpu[::2])):
        before = gaudirt.recipe_cache_stats()
        actual = gaudirt._C.bucket_linear(alias, weight, device_bias).cpu()
        assert gaudirt.recipe_cache_stats() == before
        ref64 = alias_cpu.double() @ weight_cpu.double().t() + cpu_bias.double()
        ref32 = (alias_cpu.float() @ weight_cpu.float().t() + cpu_bias.float()).double()
        assert torch.isfinite(actual).all() and torch.isfinite(ref64).all() and torch.isfinite(ref32).all()
        denom = torch.linalg.vector_norm(ref64, dim=-1).clamp_min(1e-30)
        error = (torch.linalg.vector_norm(actual.double() - ref64, dim=-1) / denom).max().item()
        floor = (torch.linalg.vector_norm(ref32 - ref64, dim=-1) / denom).max().item()
        assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (dtype, error, floor)
        payload = actual.view(torch.int16 if dtype == torch.bfloat16 else torch.int32).numpy().tobytes()
        digest.update(payload)
        output_bytes.extend(payload)
        errors.append(error)
        floors.append(floor)
    del alias, bias_storage, device_bias
    del x, weight, bias
counts = gaudirt.recipe_cache_stats()
assert len(captures) == 2, captures
print("biased FX captures:", len(captures), "cache stats:", counts, flush=True)
print("biased output sha256:", digest.hexdigest(), "max_row_relative_l2:", max(errors),
      "max_cpu_fp32_floor:", max(floors), flush=True)
if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
    with open(path, "wb") as output:
        output.write(output_bytes)
if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
    assert counts[:2] == [0, 26], counts
elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
    assert counts[:2] == [26, 0], counts
gaudirt.empty_cache()
gaudirt.shutdown()
print("biased FX linear passed", flush=True)
