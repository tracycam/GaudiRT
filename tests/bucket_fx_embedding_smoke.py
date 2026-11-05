"""Compile validated Embedding, RMSNorm and linear as one device graph."""

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


def graph(ids, table, gamma, weight):
    embedded = F.embedding(ids.long(), table)
    normalized = torch.ops.gaudirt.rms_norm.default(embedded, gamma, 1e-6)
    return embedded, F.linear(normalized, weight)


def check(outputs, ids, table, gamma, weight):
    embedded = table[ids.long()].double()
    eps = torch.tensor(1e-6, dtype=torch.float32).item()
    norm64 = direct_bf16(embedded * torch.rsqrt((embedded * embedded).mean(-1, keepdim=True) + eps)
                         * gamma.double())
    e32 = embedded.float()
    norm32 = (e32 * torch.rsqrt((e32 * e32).mean(-1, keepdim=True) + eps) * gamma.float()).to(
        torch.bfloat16
    ).float()
    result64 = direct_bf16(norm64 @ weight.double().t())
    result32 = (norm32 @ weight.float().t()).to(torch.bfloat16).double()
    actual_embed, actual = [value.cpu().double() for value in outputs]
    assert torch.equal(actual_embed, embedded)
    assert torch.isfinite(actual).all() and torch.isfinite(result64).all() and torch.isfinite(result32).all()
    denom = torch.linalg.vector_norm(result64, dim=-1).clamp_min(1e-30)
    error = (torch.linalg.vector_norm(actual - result64, dim=-1) / denom).max().item()
    floor = (torch.linalg.vector_norm(result32 - result64, dim=-1) / denom).max().item()
    absolute = (actual - result64).abs().max().item()
    assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (error, floor, absolute)
    payload = b"".join(value.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
                       for value in (actual_embed, actual))
    return payload, error, floor, absolute


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(24613)
    gaudirt.initialize(hbm_bytes=256 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    table_cpu = torch.randn(257, 64, dtype=torch.bfloat16)
    gamma_cpu = (1 + torch.randn(64) * 0.1).to(torch.bfloat16)
    weight_cpu = (torch.randn(64, 64) * 0.1).to(torch.bfloat16)
    table, gamma, weight = [value.to("gaudirt:0") for value in (table_cpu, gamma_cpu, weight_cpu)]
    captures = []
    original_backend = compiler.gaudirt_backend

    def count_backend(gm, example_inputs):
        captures.append(gm)
        return original_backend(gm, example_inputs)

    compiler.gaudirt_backend = count_backend
    compiled = compiler.compile(graph, dynamic_arg_indices=(0,))
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for rows in (1, 3, 17, 137, 513, 2048, 4096):
        ids_cpu = torch.arange(rows, dtype=torch.int64) % 257
        metadata, keep = metadata_reference.prepare(rows, ids_cpu, vocabulary=257)
        ids = gaudirt.validated_token_ids(metadata)
        if rows == 137:
            storage = torch.full((rows * 2,), 2147483647, dtype=torch.int32).to("gaudirt:0")
            view = storage[::2]
            view.copy_(ids)
            view._gaudirt_validated_vocab = metadata.vocabulary()
            ids = view
        actual = compiled(ids, table, gamma, weight)
        data, error, floor, absolute = check(actual, ids_cpu, table_cpu, gamma_cpu, weight_cpu)
        payload.extend(data); errors.append(error); floors.append(floor); absolutes.append(absolute)
        print("FX embedding rows", rows, "captures", len(captures),
              "recipes", gaudirt.recipe_cache_stats(), flush=True)
    assert len(captures) == 1, captures
    unvalidated = torch.tensor([1, 2, 3], dtype=torch.int32, device="gaudirt:0")
    try:
        compiled(unvalidated, table, gamma, weight)
    except RuntimeError as error:
        assert "must pass batch metadata validation" in str(error), str(error)
    else:
        raise AssertionError("unvalidated token IDs were accepted")
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 14], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [14, 0], counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("FX embedding stats", counts, "max row error", max(errors), "FP32 floor", max(floors),
          "max abs", max(absolutes), "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
