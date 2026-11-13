"""Two connected BF16 RMSNorm/GateUp/SiLU/Down/residual subgraphs."""

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
torch.random.default_generator.manual_seed(1223)
EPSILON = 1e-6
EPSILON_FP32 = torch.tensor(EPSILON, dtype=torch.float32).item()


def layer(x, gamma, gate_weight, gate_bias, down_weight, down_bias, residual):
    normalized = torch.ops.gaudirt.rms_norm.default(x, gamma, EPSILON)
    gate_up = torch.nn.functional.linear(normalized, gate_weight, gate_bias)
    activated = torch.ops.gaudirt.silu_and_mul.default(gate_up)
    return torch.nn.functional.linear(activated, down_weight, down_bias) + residual


def same_layer(x, gamma, gate_weight, gate_bias, down_weight, down_bias, residual):
    normalized = torch.ops.gaudirt.rms_norm.default(x, gamma, EPSILON)
    gate_up = torch.nn.functional.linear(normalized, gate_weight, gate_bias)
    activated = torch.ops.gaudirt.silu_and_mul.default(gate_up)
    return torch.nn.functional.linear(activated, down_weight, down_bias) + residual


def reference(x, gamma, gate_weight, gate_bias, down_weight, down_bias, residual, fp64):
    dtype = torch.float64 if fp64 else torch.float32
    rounded = direct_bf16 if fp64 else lambda value: value.to(torch.bfloat16).float()
    x, gamma, gate_weight, gate_bias, down_weight, down_bias, residual = [
        value.to(dtype) for value in (x, gamma, gate_weight, gate_bias, down_weight, down_bias, residual)
    ]
    normalized = rounded(x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPSILON_FP32) * gamma)
    gate_up = rounded(normalized @ gate_weight.t() + gate_bias)
    gate, up = gate_up.chunk(2, dim=-1)
    multiplied = rounded((gate / (1.0 + torch.exp(-gate))) * up)
    down = rounded(multiplied @ down_weight.t() + down_bias)
    return rounded(down + residual).double()


def check(output, inputs):
    actual = output.cpu().double()
    ref64 = reference(*inputs, fp64=True)
    ref32 = reference(*inputs, fp64=False)
    assert torch.isfinite(actual).all() and torch.isfinite(ref64).all() and torch.isfinite(ref32).all()
    denom = torch.linalg.vector_norm(ref64, dim=-1).clamp_min(1e-30)
    error = (torch.linalg.vector_norm(actual - ref64, dim=-1) / denom).max().item()
    floor = (torch.linalg.vector_norm(ref32 - ref64, dim=-1) / denom).max().item()
    absolute = (actual - ref64).abs().max().item()
    assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (inputs[0].size(0), error, floor, absolute)
    return actual.to(torch.bfloat16).view(torch.int16).numpy().tobytes(), error, floor, absolute


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    original = compiler.gaudirt_backend
    captures = []

    def capture(graph_module, example_inputs):
        captures.append(1)
        return original(graph_module, example_inputs)

    compiler.gaudirt_backend = capture
    compiled = compiler.compile(layer, dynamic_arg_indices=(0, 6))
    parameters_cpu = []
    for _ in range(2):
        gamma = (torch.randn(64) * 0.1 + 1).to(torch.bfloat16)
        gate_weight = (torch.randn(256, 64) * 0.1).to(torch.bfloat16)
        gate_bias = (torch.randn(256) * 0.1).to(torch.bfloat16)
        down_weight = (torch.randn(64, 128) * 0.1).to(torch.bfloat16)
        down_bias = (torch.randn(64) * 0.1).to(torch.bfloat16)
        parameters_cpu.append((gamma, gate_weight, gate_bias, down_weight, down_bias))
    parameters = [tuple(value.to("gaudirt:0") for value in group) for group in parameters_cpu]
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for m in (1, 3, 17, 65, 137, 513, 2048, 4096):
        x_cpu, r1_cpu, r2_cpu = [torch.randn(m, 64, dtype=torch.bfloat16) for _ in range(3)]
        x, r1, r2 = [value.to("gaudirt:0") for value in (x_cpu, r1_cpu, r2_cpu)]
        first = compiled(x, *parameters[0], r1)
        before_second = gaudirt.recipe_cache_stats()
        assert first.shape == (m, 64) and first.untyped_storage().nbytes() % 256 == 0
        data, error, floor, absolute = check(first, (x_cpu, *parameters_cpu[0], r1_cpu))
        payload.extend(data); errors.append(error); floors.append(floor); absolutes.append(absolute)
        if m == 17:
            probe_x, probe_r, probe_output = x, r1, first.cpu()
        second = compiled(first, *parameters[1], r2)
        assert gaudirt.recipe_cache_stats() == before_second, "isomorphic layer recompiled"
        data, error, floor, absolute = check(second, (first.cpu(), *parameters_cpu[1], r2_cpu))
        payload.extend(data); errors.append(error); floors.append(floor); absolutes.append(absolute)
    assert len(captures) == 1, captures
    before = gaudirt.recipe_cache_stats()
    other = compiler.compile(same_layer, dynamic_arg_indices=(0, 6))(probe_x, *parameters[0], probe_r)
    assert gaudirt.recipe_cache_stats() == before and len(captures) == 2
    assert torch.equal(other.cpu().view(torch.int16), probe_output.view(torch.int16))
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert before[:2] == [0, 8], before
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert before[:2] == [8, 0], before
    print("MLP program cache stats:", before, "max_row_relative_l2:", max(errors),
          "max_cpu_fp32_floor:", max(floors), "max_abs:", max(absolutes),
          "output sha256:", hashlib.sha256(payload).hexdigest(), flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    packed_cpu = torch.randn(3, 258, dtype=torch.bfloat16)
    packed = packed_cpu.to("gaudirt:0")
    eager = torch.ops.gaudirt.silu_and_mul.default(packed)
    assert eager.shape == (3, 129) and eager.untyped_storage().nbytes() % 256 == 0
    gate, up = packed_cpu.double().chunk(2, dim=-1)
    truth = direct_bf16((gate / (1 + torch.exp(-gate))) * up)
    actual = eager.cpu().double()
    assert torch.isfinite(actual).all() and torch.isfinite(truth).all()
    eager_error = (torch.linalg.vector_norm(actual - truth, dim=-1) /
                   torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-30)).max().item()
    assert eager_error <= 3e-3, eager_error
    invalid = torch.empty((3, 255), dtype=torch.bfloat16, device="gaudirt:0")
    try:
        torch.ops.gaudirt.silu_and_mul.default(invalid)
    except RuntimeError as error:
        assert "gated activation geometry" in str(error), str(error)
    else:
        raise AssertionError("odd gate/up width was accepted")
    for invalid_m in (0, gaudirt.MAX_BUCKET_ROWS + 1):
        invalid_rows = torch.empty((invalid_m, 256), dtype=torch.bfloat16, device="gaudirt:0")
        try:
            torch.ops.gaudirt.silu_and_mul.default(invalid_rows)
        except RuntimeError as error:
            assert "token count" in str(error)
        else:
            raise AssertionError("invalid token count was accepted")
    del x, r1, r2, first, second, probe_x, probe_r, other, invalid, invalid_rows, packed, eager, parameters
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("connected FX Dense MLP programs passed", flush=True)


if __name__ == "__main__":
    main()
