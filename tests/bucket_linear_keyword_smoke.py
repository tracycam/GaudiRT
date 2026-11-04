"""Keyword and positional linear bias must use the same device recipe."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
torch.manual_seed(876)
import gaudirt
from gaudirt.compiler import compile


def positional(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)


def keyword(x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias=bias)


gaudirt.initialize(hbm_bytes=64 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
cpu_x = torch.randn(17, 64, dtype=torch.bfloat16)
cpu_weight = torch.randn(64, 64, dtype=torch.bfloat16)
cpu_bias = torch.randn(64, dtype=torch.bfloat16)
x, weight, bias = (tensor.to("gaudirt:0") for tensor in (cpu_x, cpu_weight, cpu_bias))
first = compile(positional)(x, weight, bias)
stats = gaudirt.recipe_cache_stats()
second = compile(keyword)(x, weight, bias)
assert gaudirt.recipe_cache_stats() == stats == [0, 1, 1], stats
assert torch.equal(first.cpu().view(torch.int16), second.cpu().view(torch.int16))
reference = cpu_x.double() @ cpu_weight.double().t() + cpu_bias.double()
output = first.cpu().double()
assert torch.isfinite(reference).all() and torch.isfinite(output).all()
denom = torch.linalg.vector_norm(reference, dim=-1).clamp_min(1e-30)
error = (torch.linalg.vector_norm(output - reference, dim=-1) / denom).max().item()
assert error <= 3e-3, error
del x, weight, bias, first, second
gaudirt.empty_cache()
gaudirt.shutdown()
print("keyword and positional linear bias share one FP64-validated recipe", flush=True)
