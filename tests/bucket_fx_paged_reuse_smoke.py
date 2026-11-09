"""Three attention cuts must reuse identical intervening Dense programs."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F
import gaudirt
import gaudirt.compiler as compiler
import paged_attention_smoke as attention_ref
from bucket_fx_paged_partition_smoke import reference


def attention(x, weight, key_pool, value_pool, locations, requests, positions, mapping, header):
    query, key, value = F.linear(x, weight).split([256, 128, 128], dim=-1)
    return torch.ops.gaudirt._paged_attn_validated.default(
        query, key, value, key_pool, value_pool, locations, requests, positions, mapping, header, 64, 0.125,
    )


def connected(x, q0, q1, q2, o0, o1, o2, k0, v0, k1, v1, k2, v2,
              locations, requests, positions, mapping, header):
    y0 = attention(x, q0, k0, v0, locations, requests, positions, mapping, header)
    y1 = attention(F.linear(y0, o0), q1, k1, v1, locations, requests, positions, mapping, header)
    y2 = attention(F.linear(y1, o1), q2, k2, v2, locations, requests, positions, mapping, header)
    return F.linear(y2, o2)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(6531)
    gaudirt.initialize(hbm_bytes=2 * 1024**3, pinned_bytes=4 * 1024 * 1024)
    original = compiler.gaudirt_backend
    captures = []

    def capture(graph, inputs):
        captures.append(graph)
        assert sum(node.target == torch.ops.gaudirt._paged_attn_validated.default
                   for node in graph.graph.nodes) == 3
        return original(graph, inputs)

    compiler.gaudirt_backend = capture
    compiled = compiler.compile(connected, dynamic_arg_indices=(0, 13, 14, 15))
    weights = [(torch.randn(*shape) * 0.025).to(torch.bfloat16)
               for shape in [(512, 256)] * 3 + [(256, 256)] * 3]
    device_weights = [weight.to("gaudirt:0") for weight in weights]
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for rows in (1, 17, 137):
        metadata, locations, requests, positions, mapping = attention_ref.metadata(rows)
        x = torch.randn(rows, 256, dtype=torch.bfloat16)
        bases = [(torch.randn(attention_ref.SLOTS, 2, 64, dtype=torch.bfloat16),
                  torch.randn(attention_ref.SLOTS, 2, 64, dtype=torch.bfloat16)) for _ in range(3)]
        combined = [gaudirt._C._allocate_paged_kv(1, attention_ref.SLOTS, 2, 64) for _ in bases]
        pools = []
        for allocation, pair in zip(combined, bases):
            for index, tensor in enumerate(pair):
                allocation[0, index].copy_(tensor.transpose(0, 1).contiguous())
                pools.append(allocation[0, index].transpose(0, 1))
        plan = metadata.attention_plan(4, 2, 64)
        torch._dynamo.mark_dynamic(plan, 0, min=1, max=536870912)
        result = compiled(x.to("gaudirt:0"), *device_weights, *pools, metadata.row_locations(),
                          metadata.row_requests(), metadata.rotary_positions(),
                          plan, metadata.rotary_header())
        truth, floor = x.double(), x.double()
        for layer in range(3):
            args = (weights[layer], weights[3 + layer], *bases[layer], locations, requests, positions, mapping)
            truth = reference(truth, *args, True)
            floor = reference(floor, *args, False)
        actual = result.cpu().double()
        assert torch.isfinite(actual).all() and torch.isfinite(truth).all() and torch.isfinite(floor).all()
        denom = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-30)
        error = (torch.linalg.vector_norm(actual - truth, dim=-1) / denom).max().item()
        floor_error = (torch.linalg.vector_norm(floor - truth, dim=-1) / denom).max().item()
        absolute = (actual - truth).abs().max().item()
        assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2, (rows, error, floor_error, absolute)
        untouched = torch.ones(attention_ref.SLOTS, dtype=torch.bool)
        untouched[locations] = False
        for tensor, base in zip(pools, (value for pair in bases for value in pair)):
            actual_pool = tensor.cpu()
            assert torch.equal(actual_pool[untouched], base[untouched])
            payload.extend(actual_pool.view(torch.int16).numpy().tobytes())
        payload.extend(result.cpu().view(torch.int16).numpy().tobytes())
        errors.append(error); floors.append(floor_error); absolutes.append(absolute)
        print("three attention cuts rows", rows, "cache", gaudirt.recipe_cache_stats(), flush=True)
    assert len(captures) == 1, len(captures)
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 16], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [16, 0], counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("three attention cuts stats", counts, "max row error", max(errors), "FP32 floor", max(floors),
          "max abs", max(absolutes), "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
