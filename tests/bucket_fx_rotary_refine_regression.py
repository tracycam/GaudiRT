"""Check that a multi-row rotary projection actually refines a BF16 disagreement."""

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


def projection(x, weight):
    return F.linear(x, weight)


def projection_and_rotary(x, weight, positions, header, cache):
    q, k, v = F.linear(x, weight).split([128, 64, 4096], dim=-1)
    q, k = torch.ops.gaudirt._rotary_embedding_validated.default(
        q, k, cache, positions, header, 128, 64, 64, True
    )
    return q, k, v


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(4137)
    gaudirt.initialize(hbm_bytes=2 * 1024**3, pinned_bytes=8 * 1024**2)
    x_cpu = torch.randn(137, 7168).to(torch.bfloat16)
    weight_cpu = (torch.randn(4288, 7168) * 0.05).to(torch.bfloat16)
    x, weight = x_cpu.to("gaudirt:0"), weight_cpu.to("gaudirt:0")
    prepared, _ = metadata(137)
    positions, header = prepared.rotary_positions(), prepared.rotary_header()
    cache = torch.randn(128, 64).to(torch.bfloat16).to("gaudirt:0")
    plain = compiler.compile(projection, dynamic_arg_indices=(0,))(x, weight)
    refined = compiler.compile(projection_and_rotary, dynamic_arg_indices=(0, 2))(
        x, weight, positions, header, cache
    )[2]
    actual = refined.cpu()[-1].double()
    baseline = plain.cpu()[-1, 192:].double()
    reference = direct_bf16(x_cpu[-1:].double() @ weight_cpu[192:].double().t())[0]
    assert torch.isfinite(actual).all() and torch.isfinite(baseline).all() and torch.isfinite(reference).all()
    changed = actual != baseline
    assert changed.any(), "test did not exercise multi-row refinement"
    assert torch.equal(actual[changed], reference[changed]), "refined outputs differ from direct FP64"
    print("multi-row rotary refinement", "changed", changed.sum().item(),
          "reference_mismatches", (actual != reference).sum().item(), flush=True)


if __name__ == "__main__":
    main()
