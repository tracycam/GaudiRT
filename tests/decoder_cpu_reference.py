"""Pure CPU arithmetic for a frozen Dense decoder layer with an online KV prefix."""

import torch

from fp64_reference import direct_bf16


def project(value, weight, parts):
    if parts == 1:
        return value @ weight.t()
    assert parts > 0 and parts & (parts - 1) == 0 and parts <= weight.size(1)
    bounds = [(weight.size(1) * i // parts, weight.size(1) * (i + 1) // parts)
              for i in range(parts)]
    partial = [value[..., start:end] @ weight[:, start:end].t() for start, end in bounds]
    while len(partial) > 1:
        partial = [partial[i] + partial[i + 1] for i in range(0, len(partial), 2)]
    return partial[0]


def decoder_layer(x, weights, prefix_keys, prefix_values, cosine, sine,
                  heads, kv_heads, head_dim, epsilon, scale, parts=1):
    """Return unrounded output, current K and V in the input FP32/FP64 dtype."""
    assert x.dtype in (torch.float32, torch.float64) and x.shape == (1, heads * head_dim)
    assert heads % kv_heads == 0 and head_dim % 2 == 0
    assert prefix_keys.shape == prefix_values.shape
    assert prefix_keys.shape[1:] == (kv_heads, head_dim)
    assert cosine.shape == sine.shape == (head_dim // 2,)
    dtype = x.dtype
    weights = {name: value.to(dtype) for name, value in weights.items()}
    cosine, sine = cosine.to(dtype), sine.to(dtype)
    normalized = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + epsilon) * weights["in_norm"]
    query = project(normalized, weights["q"], parts).reshape(heads, head_dim)
    key = project(normalized, weights["k"], parts).reshape(kv_heads, head_dim)
    value = project(normalized, weights["v"], parts).reshape(kv_heads, head_dim)

    def rotate(input):
        first, second = input.chunk(2, dim=-1)
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine), dim=-1)

    query, key = rotate(query), rotate(key)
    keys = torch.cat((prefix_keys.to(dtype), key.unsqueeze(0)), dim=0)
    values = torch.cat((prefix_values.to(dtype), value.unsqueeze(0)), dim=0)
    head_map = torch.arange(heads) // (heads // kv_heads)
    scores = torch.einsum("hd,thd->ht", query, keys[:, head_map]) * scale
    attended = torch.einsum("ht,thd->hd", torch.softmax(scores, dim=-1), values[:, head_map]).reshape(1, -1)
    residual = project(attended, weights["out"], parts) + x
    normalized = (residual * torch.rsqrt((residual * residual).mean(-1, keepdim=True) + epsilon)
                  * weights["post_norm"])
    gate, up = project(normalized, weights["gate"], parts), project(normalized, weights["up"], parts)
    down = project((gate / (1 + torch.exp(-gate))) * up, weights["down"], parts)
    return down + residual, key, value


def store_boundary(value):
    assert value.dtype in (torch.float32, torch.float64) and torch.isfinite(value).all()
    return direct_bf16(value) if value.dtype == torch.float64 else value.bfloat16().double()
