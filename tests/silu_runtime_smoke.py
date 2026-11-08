"""Check the fused SiLU product against pure FP32 and FP64 references."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
from fp64_reference import direct_bf16


def main():
    torch.set_num_threads(4)
    rng = torch.Generator().manual_seed(9347)
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    payload = bytearray()
    cases = ((1, 7), (128, 1), (129, 7), (22016, 1), (22016, 7), (22016, 137))
    errors = []
    for width, rows in cases:
        source = (torch.randn((rows, 2 * width), generator=rng) * 3).bfloat16()
        actual = torch.ops.gaudirt.silu_and_mul.default(source.to("gaudirt:0")).cpu().double()
        gate64, up64 = source.double().chunk(2, dim=-1)
        truth = direct_bf16((gate64 / (1 + torch.exp(-gate64))) * up64).double()
        gate32, up32 = source.float().chunk(2, dim=-1)
        floor = ((gate32 / (1 + torch.exp(-gate32))) * up32).bfloat16().double()
        assert all(torch.isfinite(value).all() for value in (actual, truth, floor))
        denominator = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-30)
        error = (torch.linalg.vector_norm(actual - truth, dim=-1) / denominator).max().item()
        floor_error = (torch.linalg.vector_norm(floor - truth, dim=-1) / denominator).max().item()
        assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2, (width, rows, error, floor_error)
        errors.append(error)
        payload.extend(actual.bfloat16().view(torch.uint8).numpy().tobytes())
        print("silu width", width, "rows", rows, "relative_l2", error, "fp32_floor", floor_error, flush=True)
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[:2] == [0, len(cases)], stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[:2] == [len(cases), 0], stats
    print("silu stats", stats, "max_error", max(errors),
          "sha256", hashlib.sha256(payload).hexdigest(), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    gaudirt.empty_cache()
    gaudirt.shutdown()


if __name__ == "__main__":
    main()
