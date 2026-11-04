"""Exercise the FP32 GEMM lowering with a non-power-of-two token count."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
torch.manual_seed(873)
import gaudirt
from gaudirt.compiler import compile


def linear(x, weight):
    return torch.mm(x, weight.t())


gaudirt.initialize(hbm_bytes=64 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
input_cpu = torch.randn(17, 64)
weight_cpu = torch.randn(64, 64)
input_device = input_cpu.to("gaudirt:0")
weight_device = weight_cpu.to("gaudirt:0")
compiled = compile(linear)
output = compiled(input_device, weight_device).cpu().double()
reference = input_cpu.double() @ weight_cpu.double().t()
floor = (input_cpu @ weight_cpu.t()).double()
assert torch.isfinite(output).all() and torch.isfinite(reference).all() and torch.isfinite(floor).all()
denom = torch.linalg.vector_norm(reference, dim=-1).clamp_min(1e-30)
error = (torch.linalg.vector_norm(output - reference, dim=-1) / denom).max().item()
floor_error = (torch.linalg.vector_norm(floor - reference, dim=-1) / denom).max().item()
assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2, (error, floor_error)
print("FP32 bucket GEMM max_row_relative_l2:", error, "CPU_FP32 floor:", floor_error, flush=True)
del input_device, weight_device
gaudirt.empty_cache()
gaudirt.shutdown()
