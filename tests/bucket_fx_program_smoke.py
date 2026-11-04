"""Connected FX linear+residual programs on two independent layer inputs."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
torch.manual_seed(877)
from fp64_reference import direct_bf16
import gaudirt
import gaudirt.compiler as compiler


def layer(x, weight, bias, residual):
    return torch.nn.functional.linear(x, weight, bias) + residual


def same_layer(x, weight, bias, residual):
    return torch.nn.functional.linear(x, weight, bias) + residual


def check(actual, x, weight, bias, residual):
    result = actual.cpu().double()
    fp64_linear = x.double() @ weight.double().t() + bias.double()
    ref64 = direct_bf16(direct_bf16(fp64_linear) + residual.double())
    fp32_linear = x.float() @ weight.float().t() + bias.float()
    ref32 = (fp32_linear.to(torch.bfloat16).float() + residual.float()).to(torch.bfloat16).double()
    assert torch.isfinite(result).all() and torch.isfinite(ref64).all() and torch.isfinite(ref32).all()
    denom = torch.linalg.vector_norm(ref64, dim=-1).clamp_min(1e-30)
    error = (torch.linalg.vector_norm(result - ref64, dim=-1) / denom).max().item()
    floor = (torch.linalg.vector_norm(ref32 - ref64, dim=-1) / denom).max().item()
    if not (error <= max(3e-3, 2 * floor) and error <= 1e-2):
        device_linear = gaudirt._C.bucket_linear(x.to("gaudirt:0"), weight.to("gaudirt:0"),
                                                bias.to("gaudirt:0")).cpu()
        expected_add = (device_linear.float() + residual.float()).to(torch.bfloat16)
        actual_bits = result.to(torch.bfloat16).view(torch.int16)
        unrounded = direct_bf16(fp64_linear + residual.double())
        bad = (actual_bits != expected_add.view(torch.int16)).nonzero()[:6]
        print("diagnostic m:", x.size(0), "linear_bf16_bit_differences:",
              torch.count_nonzero(device_linear.double() != direct_bf16(fp64_linear)).item(),
              "add_bit_differences:", torch.count_nonzero(actual_bits != expected_add.view(torch.int16)).item(),
              "unrounded_fusion_differences:", torch.count_nonzero(result != unrounded).item(),
              "samples:", [(idx.tolist(), result[tuple(idx)].item(), expected_add[tuple(idx)].item(),
                            unrounded[tuple(idx)].item()) for idx in bad],
              flush=True)
    assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (error, floor)
    payload = result.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    return payload, error, floor, (result - ref64).abs().max().item()


def run():
    midpoint = torch.tensor([1 + 2**-8 + 2**-32], dtype=torch.float64)
    assert direct_bf16(midpoint).item() == 1 + 2**-7
    gaudirt.initialize(hbm_bytes=128 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    backend = compiler.gaudirt_backend
    captures = []

    def capture(graph_module, example_inputs):
        captures.append(1)
        return backend(graph_module, example_inputs)

    compiler.gaudirt_backend = capture
    compiled = compiler.compile(layer, dynamic_arg_indices=(0, 3))
    w1_cpu, w2_cpu = (torch.randn(64, 64, dtype=torch.bfloat16) for _ in range(2))
    b1_cpu, b2_cpu = (torch.randn(64, dtype=torch.bfloat16) for _ in range(2))
    w1, w2, b1, b2 = (value.to("gaudirt:0") for value in (w1_cpu, w2_cpu, b1_cpu, b2_cpu))
    digest = hashlib.sha256()
    data = bytearray()
    errors, floors, absolute_errors = [], [], []
    for m in (1, 3, 17, 65, 137, 513, 2048, 4096):
        x_cpu, r1_cpu, r2_cpu = (torch.randn(m, 64, dtype=torch.bfloat16) for _ in range(3))
        x, r1, r2 = (value.to("gaudirt:0") for value in (x_cpu, r1_cpu, r2_cpu))
        first = compiled(x, w1, b1, r1)
        after_first = gaudirt.recipe_cache_stats()
        assert first.shape == (m, 64) and first.is_contiguous()
        assert first.untyped_storage().nbytes() >= (1 << (m - 1).bit_length()) * 64 * 2
        payload, error, floor, absolute = check(first, x_cpu, w1_cpu, b1_cpu, r1_cpu)
        digest.update(payload); data.extend(payload)
        errors.append(error); floors.append(floor); absolute_errors.append(absolute)
        if m == 17:
            probe_x, probe_r = x, r1
            probe_expected = first.cpu()
        second = compiled(first, w2, b2, r2)
        assert gaudirt.recipe_cache_stats() == after_first, "second layer recompiled the program"
        payload, error, floor, absolute = check(second, first.cpu(), w2_cpu, b2_cpu, r2_cpu)
        digest.update(payload); data.extend(payload)
        errors.append(error); floors.append(floor); absolute_errors.append(absolute)
    assert len(captures) == 1, ("unexpected layer capture count", len(captures))
    before = gaudirt.recipe_cache_stats()
    other = compiler.compile(same_layer, dynamic_arg_indices=(0, 3))(probe_x, w1, b1, probe_r)
    assert gaudirt.recipe_cache_stats() == before, "isomorphic FX graph recompiled a recipe"
    assert torch.equal(other.cpu().view(torch.int16), probe_expected.view(torch.int16))
    assert len(captures) == 2, captures
    counts = gaudirt.recipe_cache_stats()
    print("program captures: 1 per graph; cache stats:", counts, flush=True)
    print("program output sha256:", digest.hexdigest(), "max_row_relative_l2:", max(errors),
          "max_cpu_fp32_floor:", max(floors), "max_abs:", max(absolute_errors), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(data)
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 8], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [8, 0], counts
    del x, r1, r2, first, second, w1, w2, b1, b2, probe_x, probe_r, other
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("connected FX linear+residual programs passed", flush=True)


if __name__ == "__main__":
    run()
