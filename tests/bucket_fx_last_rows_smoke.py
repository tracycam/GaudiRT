"""Validated last-row gather, output projection, and cast within one FX recipe."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import gaudirt.compiler as compiler
from fp64_reference import direct_bf16
import paged_device_metadata_smoke as metadata_test


def logits(hidden, indices, weight):
    selected = hidden[indices].to(torch.bfloat16)
    return (selected @ weight.T).float()


def cast_roundtrip(value):
    return value.to(torch.bfloat16).float()


def batch(rows, count):
    mapping = torch.arange(1, metadata_test.SLOTS, dtype=torch.int32).reshape(
        metadata_test.REQUESTS, metadata_test.CONTEXT
    )
    requests = torch.arange(count, dtype=torch.int64)
    lengths = [rows // count + int(index < rows % count) for index in range(count)]
    prefixes = [2 + index for index in range(count)]
    row_requests = torch.repeat_interleave(requests, torch.tensor(lengths))
    positions = torch.cat([torch.arange(prefix, prefix + length, dtype=torch.int64)
                           for prefix, length in zip(prefixes, lengths)])
    locations = mapping[row_requests, positions].long()
    return metadata_test.prepare(locations, requests, positions, mapping, prefixes, lengths)[0], lengths


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(24936)
    gaudirt.initialize(hbm_bytes=256 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    original = compiler.gaudirt_backend
    captures = []

    def capture(graph, inputs):
        captures.append(graph)
        return original(graph, inputs)

    compiler.gaudirt_backend = capture
    compiled = compiler.compile(logits)
    weight = (torch.randn(128, 256) * 0.05).to(torch.bfloat16)
    device_weight = weight.to("gaudirt:0")
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for count, rows_list in ((1, (1, 3, 17, 137, 513)), (4, (7, 33, 137))):
        for rows in rows_list:
            prepared, lengths = batch(rows, count)
            indices = gaudirt.validated_last_token_indices(prepared)
            torch._dynamo.mark_static(indices, 0)
            hidden = torch.randn(rows, 256, dtype=torch.bfloat16)
            result = compiled(hidden.to("gaudirt:0"), indices, device_weight)
            cpu_indices = torch.cumsum(torch.tensor(lengths), 0) - 1
            truth = direct_bf16(hidden[cpu_indices].double() @ weight.double().t())
            floor = (hidden[cpu_indices].float() @ weight.float().t()).to(torch.bfloat16).double()
            actual = result.cpu().double()
            assert result.shape == (count, 128) and result.dtype == torch.float32
            assert torch.isfinite(actual).all() and torch.isfinite(truth).all() and torch.isfinite(floor).all()
            denom = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-30)
            error = (torch.linalg.vector_norm(actual - truth, dim=-1) / denom).max().item()
            floor_error = (torch.linalg.vector_norm(floor - truth, dim=-1) / denom).max().item()
            absolute = (actual - truth).abs().max().item()
            assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2, (rows, count, error, floor_error)
            errors.append(error); floors.append(floor_error); absolutes.append(absolute)
            payload.extend(result.cpu().numpy().tobytes())
            print("last rows", rows, count, "cache", gaudirt.recipe_cache_stats(), flush=True)
    assert len(captures) == 2, len(captures)
    unvalidated = torch.tensor([0], dtype=torch.int32).to("gaudirt:0")
    torch._dynamo.mark_static(unvalidated, 0)
    try:
        compiled(torch.randn(1, 256, dtype=torch.bfloat16).to("gaudirt:0"), unvalidated, device_weight)
    except RuntimeError as error:
        assert "row indices must pass batch metadata validation" in str(error), str(error)
    else:
        raise AssertionError("unvalidated row indices were accepted")
    cast = compiler.compile(cast_roundtrip)
    boundaries = torch.tensor([1.00390625, 1.01171875, -1.00390625, -1.01171875,
                               2.0**-133, -2.0**-133, 2.0**-126, -2.0**-126,
                               0.0, -0.0, 2.0**-134, 3 * 2.0**-134])
    for width, rows in ((1, 3), (63, 1), (65, 3), (128, 3), (129, 17), (257, 3), (7168, 3), (22016, 3)):
        value = torch.randn(rows, width, dtype=torch.float32)
        size = min(value.numel(), boundaries.numel())
        value.view(-1)[:size] = boundaries[:size]
        device_value = value.to("gaudirt:0")
        torch._dynamo.mark_static(device_value, 1)
        converted = cast(device_value).cpu()
        expected = direct_bf16(value.double()).float()
        assert torch.equal(converted.view(torch.int32), expected.view(torch.int32)), (width, rows)
        payload.extend(converted.numpy().tobytes())
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[0] == 0 and counts[1] > 0, counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[0] > 0 and counts[1] == 0, counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("last rows stats", counts, "max row error", max(errors), "FP32 floor", max(floors),
          "max abs", max(absolutes), "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
