"""The FX multi-output norm node agrees bitwise with the independent eager launch."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import gaudirt.compiler as compiler


def graph(x, residual, weight):
    return torch.ops.gaudirt.fused_add_rms_norm.default(x, residual, weight, 1e-6)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(6291)
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=4 * 1024**2)
    compiled = compiler.compile(graph, dynamic_arg_indices=(0, 1))
    weight = torch.randn(7168, dtype=torch.bfloat16).to("gaudirt:0")
    payload = bytearray()
    for rows in (1, 7, 137):
        x, residual = [torch.randn(rows, 7168, dtype=torch.bfloat16).to("gaudirt:0") for _ in range(2)]
        actual = compiled(x, residual, weight)
        eager = graph(x, residual, weight)
        for result, reference in zip(actual, eager):
            result, reference = result.cpu(), reference.cpu()
            assert torch.equal(result.view(torch.int16), reference.view(torch.int16)), rows
            payload.extend(result.view(torch.uint8).numpy().tobytes())
        stats = gaudirt.recipe_cache_stats()
        repeated = compiled(x, residual, weight)
        assert gaudirt.recipe_cache_stats() == stats
        assert all(torch.equal(a.cpu().view(torch.int16), b.cpu().view(torch.int16))
                   for a, b in zip(actual, repeated))
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("FX fused norm multi-output matches eager; recipe stats", gaudirt.recipe_cache_stats(),
          "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
