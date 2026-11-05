"""Frozen M=1 first-layer graph against direct-rounding CPU FP64 and FP32."""

import hashlib
import json
import math
import os
from pathlib import Path

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
import torch
import torch.nn.functional as F
from safetensors import safe_open
from fp64_reference import direct_bf16
import gaudirt
import gaudirt.compiler as compiler

EPS = 1e-6
EPS32 = torch.tensor(EPS, dtype=torch.float32).item()
SCALE = 1 / math.sqrt(128)


def layer(x, gamma1, gamma2, qkv_weight, out_weight, up_weight, down_weight,
          key_pool, value_pool, locations, requests, positions, mapping, header, rope_cache):
    normalized = torch.ops.gaudirt.rms_norm.default(x, gamma1, EPS)
    q, k, v = F.linear(normalized, qkv_weight).split([7168, 1024, 1024], dim=-1)
    q, k = torch.ops.gaudirt._rotary_embedding_validated.default(
        q, k, rope_cache, positions, header, 2, 128, 128, True
    )
    attention = torch.ops.gaudirt._paged_attn_validated.default(
        q, k, v, key_pool, value_pool, locations, requests, positions, mapping, header, 128, SCALE
    )
    projected = F.linear(attention, out_weight)
    normalized, residual = torch.ops.gaudirt.fused_add_rms_norm.default(
        projected, x, gamma2, EPS
    )
    activated = torch.ops.gaudirt.silu_and_mul.default(F.linear(normalized, up_weight))
    return F.linear(activated, down_weight), residual


def reference(x, weights, fp64):
    dtype = torch.float64 if fp64 else torch.float32
    rounded = direct_bf16 if fp64 else lambda value: value.to(torch.bfloat16).float()
    x = x.to(dtype)
    g1, g2, value_weight, out_weight, gate_weight, up_weight, down_weight = weights
    norm = rounded(x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS32) * g1.to(dtype))
    value = rounded(norm @ value_weight.to(dtype).t())
    attention = value.reshape(1, 8, 128).repeat_interleave(7, dim=1).reshape(1, 7168)
    projected = rounded(attention @ out_weight.to(dtype).t())
    residual = rounded(projected + x)
    norm = rounded(residual * torch.rsqrt((residual * residual).mean(-1, keepdim=True) + EPS32)
                   * g2.to(dtype))
    gate = rounded(norm @ gate_weight.to(dtype).t())
    up = rounded(norm @ up_weight.to(dtype).t())
    activated = rounded(rounded(gate / (1 + torch.exp(-gate))) * up)
    output = rounded(activated @ down_weight.to(dtype).t())
    final = rounded(output + residual)
    return tuple(value.double() for value in (output, residual, final))


def main():
    torch.set_num_threads(4)
    with safe_open(str(Path(os.environ["GAUDIRT_TARGET_MODEL"]) / "model.safetensors"),
                   framework="pt", device="cpu") as model:
        prefix = "model.layers.0."
        get = lambda name: model.get_tensor(prefix + name)
        x = model.get_tensor("model.embed_tokens.weight")[3:4].clone()
        g1 = get("input_layernorm.weight")
        g2 = get("post_attention_layernorm.weight")
        q, k, v = [get("self_attn." + name + "_proj.weight") for name in ("q", "k", "v")]
        out = get("self_attn.o_proj.weight")
        gate, up, down = [get("mlp." + name + "_proj.weight") for name in ("gate", "up", "down")]
        gaudirt.initialize(hbm_bytes=6 * 1024**3, pinned_bytes=8 * 1024**2)
        params = [x, g1, g2, torch.cat((q, k, v)), out, torch.cat((gate, up)), down]
        device_params = [p.to("gaudirt:0") for p in params]
        mapping = torch.tensor([[1, 2]], dtype=torch.int32)
        metadata = torch.classes.gaudirt.PagedMetadata(
            torch.tensor([1]), torch.tensor([0]), torch.tensor([0]), mapping, 3
        )
        key_pool = torch.zeros((3, 8, 128), dtype=torch.bfloat16, device="gaudirt:0")
        value_pool = torch.zeros_like(key_pool)
        cache = torch.cat((torch.ones(2, 64), torch.zeros(2, 64)), dim=-1).to(
            dtype=torch.bfloat16, device="gaudirt:0"
        )
        compiled = compiler.compile(layer, dynamic_arg_indices=(0, 9, 10, 11))
        output, residual = compiled(
            *device_params, key_pool, value_pool, metadata.row_locations(),
            metadata.row_requests(), metadata.rotary_positions(), metadata.request_mapping(),
            metadata.rotary_header(), cache
        )
        output_cpu, residual_cpu = output.cpu(), residual.cpu()
        key_cpu, value_cpu = key_pool.cpu(), value_pool.cpu()
        actual_out, actual_res = output_cpu.double(), residual_cpu.double()
        actual = (actual_out, actual_res, direct_bf16(actual_out + actual_res))
        reference_weights = (g1, g2, v, out, gate, up, down)
        truth = reference(x, reference_weights, True)
        floor = reference(x, reference_weights, False)
        metrics = {}
        for name, candidate, expected, baseline in zip(("mlp", "residual", "combined"),
                                                        actual, truth, floor):
            assert torch.isfinite(candidate).all() and torch.isfinite(expected).all() and torch.isfinite(baseline).all()
            denom = torch.linalg.vector_norm(expected, dim=-1).clamp_min(1e-30)
            err = (torch.linalg.vector_norm(candidate - expected, dim=-1) / denom).max().item()
            floor_err = (torch.linalg.vector_norm(baseline - expected, dim=-1) / denom).max().item()
            maximum = (candidate - expected).abs().max().item()
            limit = max(3e-3, 2 * floor_err)
            metrics[name] = dict(max_row_relative_l2=err, fp32_floor=floor_err,
                                 max_abs=maximum, threshold=limit)
            assert err <= limit and err <= 1e-2, (name, metrics[name])
        assert not torch.count_nonzero(key_cpu[[0, 2]]).item()
        assert not torch.count_nonzero(value_cpu[[0, 2]]).item()
        digest = hashlib.sha256()
        for tensor in (output_cpu, residual_cpu, key_cpu, value_cpu):
            digest.update(tensor.contiguous().view(torch.int16).numpy().tobytes())
        counts = gaudirt.recipe_cache_stats()
        if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
            assert counts[0] == 0 and counts[1] > 0, counts
        elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
            assert counts[0] > 0 and counts[1] == 0, counts
        print("FROZEN_LAYER_FP64", json.dumps(metrics, sort_keys=True), flush=True)
        print("RECIPE_STATS", counts, "OUTPUT_SHA256", digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
