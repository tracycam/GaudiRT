"""The frozen decoder reference rounds only its externally visible outputs."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F

from fp64_reference import direct_bf16
from frozen_same_geometry_layer_smoke import EPS32, reference
import bucket_fx_paged_partition_smoke as paged_partition


def independent(x, weights, fp64):
    dtype = torch.float64 if fp64 else torch.float32
    x = x.to(dtype)
    g1, g2, wk, wv, wo, wg, wu, wd = (value.to(dtype) for value in weights)
    norm = F.rms_norm(x, x.shape[-1:], g1, EPS32)
    key, value = norm @ wk.t(), norm @ wv.t()
    attention = value.reshape(1, 2, 2).repeat_interleave(2, dim=1).reshape(1, -1)
    residual = attention @ wo.t() + x
    norm2 = F.rms_norm(residual, residual.shape[-1:], g2, EPS32)
    output = (F.silu(norm2 @ wg.t()) * (norm2 @ wu.t())) @ wd.t()
    store = direct_bf16 if fp64 else lambda value: value.bfloat16().double()
    return tuple(store(value) for value in (output, residual, output + residual, key, value))


def old_mixed_output(x, weights):
    g1, g2, wk, wv, wo, wg, wu, wd = (value.float() for value in weights)
    bf = lambda value: value.bfloat16().float()
    x = x.float()
    norm = bf(F.rms_norm(x, x.shape[-1:], g1, EPS32))
    value = bf(norm @ wv.t())
    attention = value.reshape(1, 2, 2).repeat_interleave(2, dim=1).reshape(1, -1)
    residual = bf(bf(attention @ wo.t()) + x)
    norm2 = bf(F.rms_norm(residual, residual.shape[-1:], g2, EPS32))
    gate, up = bf(norm2 @ wg.t()), bf(norm2 @ wu.t())
    output = bf(bf(bf(F.silu(gate)) * up) @ wd.t())
    return bf(output + residual).double()


def main():
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(9163)
    hidden, kv_width, inter = 8, 4, 12
    x = torch.randn(1, hidden).bfloat16()
    shapes = [(hidden,), (hidden,), (kv_width, hidden), (kv_width, hidden), (hidden, hidden),
              (inter, hidden), (inter, hidden), (hidden, inter)]
    weights = tuple((torch.randn(shape) * 0.6).bfloat16() for shape in shapes)
    for fp64 in (False, True):
        got = reference(x, weights, fp64, heads=4, kv_heads=2, head_dim=2)
        expected = independent(x, weights, fp64)
        assert all(torch.isfinite(item).all() for item in got + expected)
        assert all(torch.equal(a, b) for a, b in zip(got, expected)), fp64
    pure32 = reference(x, weights, False, heads=4, kv_heads=2, head_dim=2)[2]
    assert not torch.equal(pure32, old_mixed_output(x, weights)), "test did not distinguish internal BF16 rounding"
    query_width, kv_width = 256, 128
    x = torch.randn(2, query_width).bfloat16()
    qkv = (torch.randn(query_width + 2 * kv_width, query_width) * 0.1).bfloat16()
    out_weight = (torch.randn(query_width, query_width) * 0.1).bfloat16()
    key_base = torch.randn(5, 2, 64).bfloat16()
    value_base = torch.randn_like(key_base).bfloat16()
    mapping = torch.tensor([[1, 2, 3, 4]])
    locations = torch.tensor([1, 3])
    requests, positions = torch.tensor([0, 0]), torch.tensor([0, 2])
    for fp64 in (False, True):
        dtype = torch.float64 if fp64 else torch.float32
        q, k, v = (x.to(dtype) @ qkv.to(dtype).t()).split([query_width, kv_width, kv_width], dim=-1)
        keys, values = key_base.to(dtype).clone(), value_base.to(dtype).clone()
        keys[locations], values[locations] = k.reshape(-1, 2, 64), v.reshape(-1, 2, 64)
        head_map = torch.arange(4) // 2
        attention = []
        for row in range(2):
            slots = mapping[0, :positions[row] + 1]
            key = keys[slots][:, head_map].permute(1, 0, 2)
            value = values[slots][:, head_map].permute(1, 0, 2)
            output = F.scaled_dot_product_attention(q[row].reshape(4, 1, 64), key, value,
                                                    scale=0.125)
            attention.append(output.reshape(query_width))
        projected = torch.stack(attention) @ out_weight.to(dtype).t()
        expected = direct_bf16(projected) if fp64 else projected.bfloat16().double()
        got = paged_partition.reference(x, qkv, out_weight, key_base, value_base, locations,
                                        requests, positions, mapping, fp64)
        assert torch.equal(got, expected), fp64
    print("frozen decoder reference matches pure FP32/FP64 and excludes intermediate BF16 rounding")


if __name__ == "__main__":
    main()
