"""Independent CPU references for isolated single-tile and multi-tile BF16-PV attention gates."""

import torch


def direct_bf16_pv(query, key_pool, value_pool, requests, positions, mapping,
                   heads, kv_heads, head_dim, scale):
    assert heads % kv_heads == 0
    query = query.reshape(-1, heads, head_dim).float()
    head_map = torch.arange(heads) // (heads // kv_heads)
    outputs = []
    for row in range(query.size(0)):
        indices = mapping[requests[row], :positions[row] + 1].long()
        assert indices.numel() > 0
        keys = key_pool[indices][:, head_map, :].float()
        values = value_pool[indices][:, head_map, :].float()
        scores = torch.einsum("hd,thd->ht", query[row], keys) * scale
        probabilities = torch.softmax(scores, dim=-1).to(torch.bfloat16).float()
        outputs.append(torch.einsum("ht,thd->hd", probabilities, values))
    return torch.stack(outputs).reshape(query.size(0), heads * head_dim).to(torch.bfloat16).double()


def tiled_bf16_pv(query, key_pool, value_pool, requests, positions, mapping,
                  heads, kv_heads, head_dim, scale, kv_tile):
    assert kv_tile > 0 and heads % kv_heads == 0
    query = query.reshape(-1, heads, head_dim).float()
    head_map = torch.arange(heads) // (heads // kv_heads)
    outputs = []
    for row in range(query.size(0)):
        indices = mapping[requests[row], :positions[row] + 1].long()
        assert indices.numel() > 0
        running_max = running_sum = running_output = None
        for start in range(0, indices.numel(), kv_tile):
            locations = indices[start:start + kv_tile]
            keys = key_pool[locations][:, head_map, :].float()
            values = value_pool[locations][:, head_map, :].float()
            scores = torch.einsum("hd,thd->ht", query[row], keys) * scale
            tile_max = scores.amax(dim=-1)
            probabilities = torch.exp(scores - tile_max[:, None])
            tile_sum = probabilities.sum(dim=-1)
            normalized = (probabilities / tile_sum[:, None]).to(torch.bfloat16).float()
            # This is a separate isolated-operator reference, not the pure
            # FP32 baseline used to determine a complete decoder-layer gate.
            tile_output = torch.einsum("ht,thd->hd", normalized, values)
            if running_max is None:
                running_max, running_sum = tile_max, tile_sum
                running_output = tile_output * tile_sum[:, None]
                continue
            next_max = torch.maximum(running_max, tile_max)
            old_scale = torch.exp(running_max - next_max)
            tile_scale = torch.exp(tile_max - next_max)
            running_output = (running_output * old_scale[:, None] +
                              tile_output * (tile_scale * tile_sum)[:, None])
            running_sum = running_sum * old_scale + tile_sum * tile_scale
            running_max = next_max
        outputs.append(running_output / running_sum[:, None])
    return torch.stack(outputs).reshape(query.size(0), heads * head_dim).to(torch.bfloat16).double()


def isolated_errors(actual, fp64, tiled):
    assert actual.shape == fp64.shape == tiled.shape and actual.ndim >= 2
    assert all(torch.isfinite(value).all() for value in (actual, fp64, tiled))
    actual, fp64, tiled = (value.double().reshape(value.size(0), -1) for value in (actual, fp64, tiled))
    error64 = (torch.linalg.vector_norm(actual - fp64, dim=-1) /
               torch.linalg.vector_norm(fp64, dim=-1).clamp_min(1e-30)).max().item()
    error_tiled = (torch.linalg.vector_norm(actual - tiled, dim=-1) /
                   torch.linalg.vector_norm(tiled, dim=-1).clamp_min(1e-30)).max().item()
    assert error64 <= 1e-2 and error_tiled <= 5e-3, (error64, error_tiled)
    return error64, error_tiled


def device_kv_tile(rows):
    import gaudirt

    probe = getattr(gaudirt._C, "_paged_attn_kv_tile", None)
    if probe is None:
        raise RuntimeError("GaudiRT attention must expose its actual KV tile for the independent reference")
    tile = probe(rows)
    if not isinstance(tile, int) or tile <= 0:
        raise RuntimeError("GaudiRT attention returned an invalid KV tile")
    return tile
