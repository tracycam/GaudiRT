"""Validated embedding and two connected synthetic Dense layers across token buckets."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F

import gaudirt
import gaudirt.compiler as compiler
import token_metadata_smoke as metadata_reference
from fp64_reference import direct_bf16


EPSILON = 1e-6
EPSILON_FP32 = torch.tensor(EPSILON, dtype=torch.float32).item()


def embedding(ids, table):
    return F.embedding(ids.long(), table)


def layer(x, residual, gamma1, gamma2, qkv_weight, out_weight, up_weight, down_weight):
    normalized, residual = torch.ops.gaudirt.fused_add_rms_norm.default(x, residual, gamma1, EPSILON)
    q, k, v = F.linear(normalized, qkv_weight).split([64, 32, 32], dim=-1)
    # Exercise all Q/K/V outputs without claiming this tensor mix is attention.
    mixed = torch.cat((q[:, :32] + k + v, q[:, 32:]), dim=-1)
    projected = F.linear(mixed, out_weight)
    normalized, residual = torch.ops.gaudirt.fused_add_rms_norm.default(projected, residual, gamma2, EPSILON)
    activated = torch.ops.gaudirt.silu_and_mul.default(F.linear(normalized, up_weight))
    return F.linear(activated, down_weight), residual


def same_layer(x, residual, gamma1, gamma2, qkv_weight, out_weight, up_weight, down_weight):
    return layer(x, residual, gamma1, gamma2, qkv_weight, out_weight, up_weight, down_weight)


def reference(x, residual, params, fp64):
    dtype = torch.float64 if fp64 else torch.float32
    rounded = direct_bf16 if fp64 else lambda value: value.to(torch.bfloat16).float()
    x, residual = x.to(dtype), residual.to(dtype)
    gamma1, gamma2, qkv, out, up, down = (value.to(dtype) for value in params)
    residual = rounded(x + residual)
    normalized = rounded(residual * torch.rsqrt((residual * residual).mean(-1, keepdim=True) + EPSILON_FP32)
                         * gamma1)
    q, k, v = rounded(normalized @ qkv.t()).split([64, 32, 32], dim=-1)
    mixed = torch.cat((rounded(rounded(q[:, :32] + k) + v), q[:, 32:]), dim=-1)
    projected = rounded(mixed @ out.t())
    residual = rounded(projected + residual)
    normalized = rounded(residual * torch.rsqrt((residual * residual).mean(-1, keepdim=True) + EPSILON_FP32)
                         * gamma2)
    gate, value = rounded(normalized @ up.t()).chunk(2, dim=-1)
    activated = rounded(rounded(gate / (1 + torch.exp(-gate))) * value)
    return rounded(activated @ down.t()).double(), residual.double()


def check(outputs, x, residual, params):
    truth, floor = reference(x, residual, params, True), reference(x, residual, params, False)
    payload = bytearray()
    worst = floor_max = absolute = 0.0
    for device, expected, baseline in zip(outputs, truth, floor):
        actual = device.cpu().double()
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all() and torch.isfinite(baseline).all()
        denom = torch.linalg.vector_norm(expected, dim=-1).clamp_min(1e-30)
        error = (torch.linalg.vector_norm(actual - expected, dim=-1) / denom).max().item()
        floor_error = (torch.linalg.vector_norm(baseline - expected, dim=-1) / denom).max().item()
        assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2, (error, floor_error)
        worst, floor_max = max(worst, error), max(floor_max, floor_error)
        absolute = max(absolute, (actual - expected).abs().max().item())
        payload.extend(device.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return payload, worst, floor_max, absolute


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(2906)
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    table_cpu = torch.randn(257, 64).to(torch.bfloat16)
    table = table_cpu.to("gaudirt:0")
    parameters_cpu = []
    for _ in range(2):
        gamma1, gamma2 = [(1 + torch.randn(64) * 0.1).to(torch.bfloat16) for _ in range(2)]
        weights = [(torch.randn(*shape) * 0.08).to(torch.bfloat16)
                   for shape in ((128, 64), (64, 64), (256, 64), (64, 128))]
        parameters_cpu.append((gamma1, gamma2, *weights))
    parameters = [tuple(value.to("gaudirt:0") for value in values) for values in parameters_cpu]
    captures = []
    original_backend = compiler.gaudirt_backend

    def capture(graph, inputs):
        captures.append(1)
        return original_backend(graph, inputs)

    compiler.gaudirt_backend = capture
    embed = compiler.compile(embedding)
    compiled = compiler.compile(layer, dynamic_arg_indices=(0, 1))
    payload = bytearray()
    errors, floors, absolutes = [], [], []
    for rows in (1, 3, 17, 65, 137, 513, 2048, 4096):
        ids_cpu = torch.arange(rows, dtype=torch.int64) % table_cpu.size(0)
        metadata, keep = metadata_reference.prepare(rows, ids_cpu, vocabulary=table_cpu.size(0))
        ids = gaudirt.validated_token_ids(metadata)
        residual_cpu = torch.randn(rows, 64).to(torch.bfloat16)
        residual = residual_cpu.to("gaudirt:0")
        x = embed(ids, table)
        assert torch.equal(x.cpu(), table_cpu[ids_cpu])
        first = compiled(x, residual, *parameters[0])
        record = check(first, table_cpu[ids_cpu], residual_cpu, parameters_cpu[0])
        before_second = gaudirt.recipe_cache_stats()
        second = compiled(first[0], first[1], *parameters[1])
        assert gaudirt.recipe_cache_stats() == before_second, "same-geometry second layer recompiled"
        second_record = check(second, first[0].cpu(), first[1].cpu(), parameters_cpu[1])
        for data, error, floor, absolute in (record, second_record):
            payload.extend(data)
            errors.append(error)
            floors.append(floor)
            absolutes.append(absolute)
        if rows == 17:
            probe = (x, residual, tuple(value.cpu() for value in first))
        print("connected Dense rows", rows, "cache", before_second, flush=True)
    assert len(captures) == 2, captures
    before = gaudirt.recipe_cache_stats()
    again = compiler.compile(same_layer, dynamic_arg_indices=(0, 1))(probe[0], probe[1], *parameters[0])
    assert len(captures) == 3 and gaudirt.recipe_cache_stats() == before
    assert all(torch.equal(value.cpu(), expected) for value, expected in zip(again, probe[2]))
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert before[:2] == [0, 24], before
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert before[:2] == [24, 0], before
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("connected Dense", "cache", before, "max_row_relative_l2", max(errors),
          "max_fp32_floor", max(floors), "max_abs", max(absolutes),
          "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
