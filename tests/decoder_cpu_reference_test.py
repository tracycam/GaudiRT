"""Exercise frozen full-layer CPU arithmetic with a nonempty online KV prefix."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F

from decoder_cpu_reference import decoder_layer, store_boundary


def independent(x, weights, prefix_keys, prefix_values, cosine, sine, heads, kv_heads, dim, eps, scale):
    weights = {name: value.to(x.dtype) for name, value in weights.items()}
    cosine, sine = cosine.to(x.dtype), sine.to(x.dtype)
    norm = F.rms_norm(x, x.shape[-1:], weights["in_norm"], eps)
    q = (norm @ weights["q"].t()).reshape(heads, dim)
    k = (norm @ weights["k"].t()).reshape(kv_heads, dim)
    v = (norm @ weights["v"].t()).reshape(kv_heads, dim)

    def rotary(input):
        a, b = input.chunk(2, dim=-1)
        return torch.cat((a * cosine - b * sine, b * cosine + a * sine), dim=-1)

    q, k = rotary(q), rotary(k)
    keys = torch.cat((prefix_keys.to(x.dtype), k[None]))
    values = torch.cat((prefix_values.to(x.dtype), v[None]))
    head_map = torch.arange(heads) // (heads // kv_heads)
    attended = F.scaled_dot_product_attention(
        q[:, None, :], keys[:, head_map, :].permute(1, 0, 2),
        values[:, head_map, :].permute(1, 0, 2), scale=scale
    ).reshape(1, -1)
    residual = attended @ weights["out"].t() + x
    norm2 = F.rms_norm(residual, residual.shape[-1:], weights["post_norm"], eps)
    output = (F.silu(norm2 @ weights["gate"].t()) * (norm2 @ weights["up"].t())) @ weights["down"].t()
    return output + residual, k, v


def main():
    torch.set_num_threads(2)
    torch.random.default_generator.manual_seed(1894)
    heads, kv_heads, dim, inter = 4, 2, 4, 24
    hidden, kv_width = heads * dim, kv_heads * dim
    shapes = {"in_norm": (hidden,), "post_norm": (hidden,), "q": (hidden, hidden),
              "k": (kv_width, hidden), "v": (kv_width, hidden), "out": (hidden, hidden),
              "gate": (inter, hidden), "up": (inter, hidden), "down": (hidden, inter)}
    weights = {name: (torch.randn(shape) * 0.25).bfloat16() for name, shape in shapes.items()}
    x = torch.randn(1, hidden).bfloat16()
    prefix_keys = torch.randn(7, kv_heads, dim).bfloat16()
    prefix_values = torch.randn_like(prefix_keys).bfloat16()
    cosine = torch.tensor([0.75, 0.5]).bfloat16()
    sine = torch.tensor([0.625, -0.875]).bfloat16()
    for dtype in (torch.float32, torch.float64):
        args = (x.to(dtype), weights, prefix_keys, prefix_values, cosine, sine, heads, kv_heads, dim, 1e-6, 0.5)
        got, expected = decoder_layer(*args), independent(*args)
        for a, b in zip(got, expected):
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            assert torch.equal(store_boundary(a), store_boundary(b)), dtype
        if dtype == torch.float32:
            for parts in (2, 4):
                split = decoder_layer(*args, parts=parts)
                assert max((a - b).abs().max().item() for a, b in zip(split, got)) < 1e-5

    midpoint = torch.tensor([1 + 2**-8 + 2**-32], dtype=torch.float64)
    assert store_boundary(midpoint).item() == 1 + 2**-7
    assert midpoint.float().bfloat16().double().item() == 1
    try:
        store_boundary(torch.tensor([float("nan")]))
    except AssertionError:
        pass
    else:
        raise AssertionError("nonfinite decoder reference was accepted")
    print("pure decoder reference matched independent nonempty-prefix attention for FP32 and FP64")


if __name__ == "__main__":
    main()
