"""Dynamic feature slicing and concatenation without Python token padding."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch

import gaudirt
import gaudirt.compiler as compiler


def graph(x, y):
    left = x[:, 1:-1]
    middle = torch.ops.aten.slice.Tensor(y, 1, 1, 4, 1)
    combined = torch.cat((left, middle, x[:, -2:]), dim=-1)
    return combined[:, 2:-1], combined


def same_graph(x, y):
    return graph(x, y)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(4728)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    captures = 0
    original = compiler.gaudirt_backend

    def capture(graph_module, inputs):
        nonlocal captures
        captures += 1
        return original(graph_module, inputs)

    compiler.gaudirt_backend = capture
    first = compiler.compile(graph, dynamic_arg_indices=(0, 1))
    second = compiler.compile(same_graph, dynamic_arg_indices=(0, 1))
    digest = hashlib.sha256()
    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 3, 17, 137, 513, 4096):
            x, y = torch.randn(rows, 17).to(dtype), torch.randn(rows, 5).to(dtype)
            dx, dy = x.to("gaudirt:0"), y.to("gaudirt:0")
            outputs = first(dx, dy)
            expected = graph(x, y)
            for actual, reference in zip(outputs, expected):
                actual = actual.cpu()
                assert actual.shape == reference.shape and torch.equal(actual, reference)
                assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
                digest.update(actual.contiguous().view(torch.uint8).numpy().tobytes())
            before = gaudirt.recipe_cache_stats()
            again = second(dx, dy)
            assert gaudirt.recipe_cache_stats() == before, "same feature graph recompiled"
            assert all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(outputs, again))
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[:2] == [0, 12], stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[:2] == [12, 0], stats
    assert captures == 4, captures
    print("feature cat FX", "stats", stats, "output_sha256", digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
