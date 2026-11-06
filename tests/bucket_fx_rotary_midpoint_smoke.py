"""Check adversarial QKV midpoint inputs against the FP64 error contract."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F

import gaudirt
import gaudirt.compiler as compiler
from bucket_fx_rotary_smoke import metadata
from fp64_reference import direct_bf16


def graph(x, weight, positions, header, cache):
    qkv = F.linear(x, weight)
    q, k, v = qkv.split([7168, 1024, 1024], dim=-1)
    q, k = torch.ops.gaudirt._rotary_embedding_validated.default(
        q, k, cache, positions, header, 128, 128, 128, True
    )
    return qkv, q, k, v


def main():
    torch.set_num_threads(4)
    gaudirt.initialize(hbm_bytes=1024**3, pinned_bytes=8 * 1024**2)
    weight_cpu = torch.zeros((9216, 7168), dtype=torch.bfloat16)
    row = 0
    for sign in (1.0, -1.0):
        for multiple in (1, 3):
            for tail in (-1, 0, 1):
                weight_cpu[row, 0] = sign
                weight_cpu[row, 1] = sign * multiple * 2**-8
                weight_cpu[row, 2] = sign * tail * 2**-32
                row += 1
    reference = torch.zeros(9216, dtype=torch.float64)
    reference[:12] = direct_bf16(weight_cpu[:12].double().sum(-1))
    reference32 = weight_cpu.float().sum(-1).to(torch.bfloat16).double()
    denominator = torch.linalg.vector_norm(reference).clamp_min(1e-30)
    floor = (torch.linalg.vector_norm(reference32 - reference) / denominator).item()
    weight = weight_cpu.to("gaudirt:0")
    cache = torch.zeros((128, 128), dtype=torch.bfloat16).to("gaudirt:0")
    compiled = compiler.compile(graph, dynamic_arg_indices=(0, 2))
    payload = bytearray()
    for rows in (1, 3):
        prepared, _ = metadata(rows)
        x = torch.ones((rows, 7168), dtype=torch.bfloat16).to("gaudirt:0")
        actual = compiled(x, weight, prepared.rotary_positions(), prepared.rotary_header(), cache)[0].cpu()
        assert torch.isfinite(actual).all()
        error = (torch.linalg.vector_norm(actual.double() - reference, dim=-1) / denominator).max().item()
        assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (error, floor)
        payload.extend(actual.contiguous().view(torch.uint8).numpy().tobytes())
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 2], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [2, 0], counts
    print("rotary QKV FP64 midpoint error gate", "cache", counts, "FP32 floor", floor,
          "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
