"""Two-layer FX linear and QKV feature splits over unpadded token counts."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
from fp64_reference import direct_bf16
import gaudirt
import gaudirt.compiler as compiler

torch.set_num_threads(4)
torch.random.default_generator.manual_seed(4107)
GEOMETRY = os.environ.get("GAUDIRT_TEST_GEOMETRY", "small")
assert GEOMETRY in ("small", "nanbeige")
HIDDEN = 64 if GEOMETRY == "small" else 7168
SIZES = [64, 32, 32] if GEOMETRY == "small" else [7168, 1024, 1024]
ROWS = (1, 3, 17, 65, 137, 513, 2048, 4096) if GEOMETRY == "small" else (1, 7, 137)
PROBE_ROWS = 17 if GEOMETRY == "small" else 7


def layer(x, weight):
    qkv = torch.nn.functional.linear(x, weight)
    return qkv.split(SIZES, dim=-1)


def functional_layer(x, weight):
    qkv = torch.nn.functional.linear(x, weight)
    return torch.split(qkv, SIZES, dim=-1)


def aten_layer(x, weight):
    qkv = torch.nn.functional.linear(x, weight)
    return torch.ops.aten.split_with_sizes.default(qkv, SIZES, -1)


def check(outputs, x, weight):
    truth = direct_bf16(x.double() @ weight.double().t()).split(SIZES, dim=-1)
    baseline = (x.float() @ weight.float().t()).to(torch.bfloat16).float().double().split(SIZES, dim=-1)
    checked = []
    for output, expected, floor_value in zip(outputs, truth, baseline):
        assert output.shape == expected.shape and output.is_contiguous()
        bucket = 1 << (x.size(0) - 1).bit_length()
        assert output.untyped_storage().nbytes() >= bucket * output.size(1) * 2
        assert output.untyped_storage().nbytes() % 256 == 0
        actual = output.cpu().double()
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all() and torch.isfinite(floor_value).all()
        denominator = torch.linalg.vector_norm(expected, dim=-1).clamp_min(1e-30)
        error = (torch.linalg.vector_norm(actual - expected, dim=-1) / denominator).max().item()
        floor = (torch.linalg.vector_norm(floor_value - expected, dim=-1) / denominator).max().item()
        absolute = (actual - expected).abs().max().item()
        assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (x.size(0), error, floor, absolute)
        checked.append((actual.to(torch.bfloat16).view(torch.int16).numpy().tobytes(), error, floor, absolute))
    return checked


def main():
    gaudirt.initialize(hbm_bytes=(256 if GEOMETRY == "small" else 1024) * 1024 * 1024,
                       pinned_bytes=4 * 1024 * 1024)
    original = compiler.gaudirt_backend
    captures = []

    def capture(graph_module, example_inputs):
        captures.append(1)
        return original(graph_module, example_inputs)

    compiler.gaudirt_backend = capture
    compiled = compiler.compile(layer)
    weights_cpu = [(torch.randn(sum(SIZES), HIDDEN) * 0.125).to(torch.bfloat16) for _ in range(2)]
    weights = [weight.to("gaudirt:0") for weight in weights_cpu]
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for rows in ROWS:
        x_cpu = torch.randn(rows, HIDDEN, dtype=torch.bfloat16)
        x = x_cpu.to("gaudirt:0")
        first = compiled(x, weights[0])
        first_results = check(first, x_cpu, weights_cpu[0])
        if rows == PROBE_ROWS:
            probe_x = x
            probe_output = tuple(output.cpu() for output in first)
        before_second = gaudirt.recipe_cache_stats()
        second = compiled(first[0], weights[1])
        assert gaudirt.recipe_cache_stats() == before_second, "isomorphic layer recompiled"
        second_results = check(second, first[0].cpu(), weights_cpu[1])
        for result in first_results + second_results:
            payload.extend(result[0]); errors.append(result[1]); floors.append(result[2]); absolutes.append(result[3])
        print("rows", rows, "stats", before_second,
              "relative_l2", max(result[1] for result in first_results + second_results), flush=True)
        del first, second, x, result
    assert len(captures) == 1, captures
    counts = gaudirt.recipe_cache_stats()
    for equivalent in (functional_layer, aten_layer):
        same = compiler.compile(equivalent)(probe_x, weights[0])
        assert gaudirt.recipe_cache_stats() == counts, "equivalent split graph recompiled"
        assert all(torch.equal(a.cpu().view(torch.int16), b.view(torch.int16))
                   for a, b in zip(same, probe_output))
    assert len(captures) == 3, captures
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, len(ROWS)], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [len(ROWS), 0], counts
    print("QKV split FX stats:", counts, "max_row_relative_l2:", max(errors),
          "max_cpu_fp32_floor:", max(floors), "max_abs:", max(absolutes),
          "output sha256:", hashlib.sha256(payload).hexdigest(), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    invalid = gaudirt._C.BucketProgram([1], [[4, 0, 0, 4, 9]], [1])
    invalid_input = torch.empty((3, 8), dtype=torch.bfloat16, device="gaudirt:0")
    try:
        invalid.run([invalid_input])
    except RuntimeError as error:
        assert "feature split geometry" in str(error)
    else:
        raise AssertionError("split sizes that do not cover the feature width were accepted")
    del invalid, invalid_input
    del weights, same, probe_x
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("QKV split FX two-layer device audit passed", flush=True)


if __name__ == "__main__":
    main()
