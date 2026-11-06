"""Check tiled attention reference arithmetic and both isolated gates on CPU."""

import torch

from paged_attention_reference import isolated_errors, tiled_bf16_pv


def direct_tiled(query, key_pool, value_pool, requests, positions, mapping,
                 heads, kv_heads, head_dim, scale, kv_tile):
    head_map = torch.arange(heads) // (heads // kv_heads)
    result = []
    for row in range(query.size(0)):
        indices = mapping[requests[row], :positions[row] + 1].long()
        keys = key_pool[indices][:, head_map, :].float()
        values = value_pool[indices][:, head_map, :].float()
        scores = torch.einsum("hd,thd->ht", query[row].reshape(heads, head_dim).float(), keys) * scale
        maximum = scores.amax(-1)
        denominator = torch.exp(scores - maximum[:, None]).sum(-1)
        numerator = torch.zeros(heads, head_dim)
        for start in range(0, len(indices), kv_tile):
            tile = scores[:, start:start + kv_tile]
            local_max = tile.amax(-1)
            p = torch.exp(tile - local_max[:, None]).to(torch.bfloat16).float()
            pv = torch.einsum("ht,thd->hd", p, values[start:start + kv_tile])
            numerator += pv * torch.exp(local_max - maximum)[:, None]
        result.append(numerator / denominator[:, None])
    return torch.stack(result).reshape(query.size(0), heads * head_dim).to(torch.bfloat16).double()


def rejects(actual, truth, tiled):
    try:
        isolated_errors(actual, truth, tiled)
    except AssertionError:
        return
    raise AssertionError("invalid isolated attention result passed")


def main():
    torch.set_num_threads(2)
    torch.manual_seed(771)
    heads, kv_heads, head_dim = 4, 2, 4
    query = torch.randn(3, heads * head_dim).bfloat16()
    keys = torch.randn(13, kv_heads, head_dim).bfloat16()
    values = torch.randn_like(keys).bfloat16()
    mapping = torch.tensor([[9, 2, 11, 5, 3, 10, 6, 1, 12, 4, 8, 7]], dtype=torch.int64)
    requests = torch.tensor([0, 0, 0])
    positions = torch.tensor([1, 7, 11])
    for tile in (2, 3, 5, 16):
        args = (query, keys, values, requests, positions, mapping, heads, kv_heads, head_dim, 0.5, tile)
        assert torch.equal(tiled_bf16_pv(*args), direct_tiled(*args)), tile

    fp64 = torch.tensor([[1.0, 0.2]], dtype=torch.float64)
    candidate = torch.tensor([[1.004, 0.2]], dtype=torch.float64)
    error64, error_tile = isolated_errors(candidate, fp64, candidate)
    assert 3e-3 < error64 < 1e-2 and error_tile == 0
    rejects(candidate, fp64, fp64)
    rejects(torch.tensor([[1.02, 0.2]], dtype=torch.float64), fp64,
            torch.tensor([[1.02, 0.2]], dtype=torch.float64))
    rejects(torch.tensor([[float("nan"), 0.2]], dtype=torch.float64), fp64, fp64)
    print("CPU tiled reference agrees with direct reduction; both isolated gates and finite check verified")


if __name__ == "__main__":
    main()
