"""Normalize bias-free functional linear and transposed GEMM to one recipe."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
torch.manual_seed(874)
import gaudirt
from gaudirt.compiler import compile


def linear(x, weight):
    return torch.nn.functional.linear(x, weight)


def matrix_product(x, weight):
    return torch.mm(x, weight.t())


def matmul_method(x, weight):
    return torch.matmul(x, weight.transpose(0, 1))


def matmul_operator(x, weight):
    return x @ torch.transpose(weight, -2, -1)


gaudirt.initialize(hbm_bytes=64 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
cpu_x = torch.randn(137, 64, dtype=torch.bfloat16)
cpu_weight = torch.randn(64, 64, dtype=torch.bfloat16)
x = cpu_x.to("gaudirt:0")
weight = cpu_weight.to("gaudirt:0")
first = compile(linear)(x, weight)
stats = gaudirt.recipe_cache_stats()
second = compile(matrix_product)(x, weight)
third = compile(matmul_method)(x, weight)
fourth = compile(matmul_operator)(x, weight)
assert gaudirt.recipe_cache_stats() == stats == [0, 1, 1], stats
assert torch.equal(first.cpu().view(torch.int16), second.cpu().view(torch.int16))
assert torch.equal(first.cpu().view(torch.int16), third.cpu().view(torch.int16))
assert torch.equal(first.cpu().view(torch.int16), fourth.cpu().view(torch.int16))
reference = cpu_x.double() @ cpu_weight.double().t()
result = first.cpu().double()
assert torch.isfinite(reference).all() and torch.isfinite(result).all()
denom = torch.linalg.vector_norm(reference, dim=-1).clamp_min(1e-30)
error = (torch.linalg.vector_norm(result - reference, dim=-1) / denom).max().item()
assert error <= 3e-3, error
del x, weight, first, second, third, fourth
gaudirt.empty_cache()
gaudirt.shutdown()
print("bias-free linear, mm and matmul transpose forms reused one recipe", flush=True)
