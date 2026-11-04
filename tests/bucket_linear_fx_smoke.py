"""A real FX-to-Synapse GEMM slice; this is not the full Dense compiler gate."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
torch.manual_seed(872)
import gaudirt
import gaudirt.compiler as compiler


def linear(x, weight):
    return torch.mm(x, weight.t())


def check(actual, input_cpu, weight_cpu):
    result = actual.cpu().double()
    ref64 = input_cpu.double() @ weight_cpu.double().t()
    ref32 = (input_cpu.float() @ weight_cpu.float().t()).double()
    assert torch.isfinite(result).all() and torch.isfinite(ref64).all() and torch.isfinite(ref32).all()
    denom = torch.linalg.vector_norm(ref64, dim=-1).clamp_min(1e-30)
    error = (torch.linalg.vector_norm(result - ref64, dim=-1) / denom).max().item()
    floor = (torch.linalg.vector_norm(ref32 - ref64, dim=-1) / denom).max().item()
    assert error <= max(3e-3, 2 * floor) and error <= 1e-2, (error, floor)
    return result.to(torch.bfloat16).view(torch.int16).numpy().tobytes(), error, floor


def run():
    gaudirt.initialize(hbm_bytes=128 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    captures = []
    backend = compiler.gaudirt_backend

    def capture_once(graph_module, example_inputs):
        captures.append(1)
        return backend(graph_module, example_inputs)

    compiler.gaudirt_backend = capture_once
    compiled = compiler.compile(linear)
    weight1_cpu = torch.randn(64, 64, dtype=torch.bfloat16)
    weight2_cpu = torch.randn(64, 64, dtype=torch.bfloat16)
    weight1 = weight1_cpu.to("gaudirt:0")
    weight2 = weight2_cpu.to("gaudirt:0")
    digest = hashlib.sha256()
    output_bytes = bytearray()
    errors, floors = [], []
    for m in (1, 2, 3, 5, 9, 17, 33, 65, 137, 257, 513, 2048, 4096):
        input_cpu = torch.randn(m, 64, dtype=torch.bfloat16)
        x = input_cpu.to("gaudirt:0")
        first = compiled(x, weight1)
        after_first = gaudirt.recipe_cache_stats()
        assert first.shape == (m, 64)
        assert first.is_contiguous() and first.data_ptr() % 256 == 0
        assert first.untyped_storage().nbytes() >= (1 << (m - 1).bit_length()) * 64 * 2
        payload, error, floor = check(first, input_cpu, weight1_cpu)
        digest.update(payload)
        output_bytes.extend(payload)
        errors.append(error)
        floors.append(floor)
        if m == 137:
            probe_input = x
            probe_many = first[-1].cpu().double()
        second = compiled(first, weight2)
        assert gaudirt.recipe_cache_stats() == after_first, "second layer recompiled the same recipe"
        payload, error, floor = check(second, first.cpu(), weight2_cpu)
        digest.update(payload)
        output_bytes.extend(payload)
        errors.append(error)
        floors.append(floor)
    before_probe = gaudirt.recipe_cache_stats()
    probe_single = compiled(probe_input[-1:], weight1).cpu().double().squeeze(0)
    assert gaudirt.recipe_cache_stats() == before_probe, "M=1 replay recompiled the recipe"
    assert torch.isfinite(probe_many).all() and torch.isfinite(probe_single).all()
    probe_error = (torch.linalg.vector_norm(probe_many - probe_single) /
                   torch.linalg.vector_norm(probe_single).clamp_min(1e-30)).item()
    assert probe_error <= 3e-3, probe_error
    input_storage_cpu = torch.randn(18, 128, dtype=torch.bfloat16)
    weight_storage_cpu = torch.randn(64, 128, dtype=torch.bfloat16)
    input_storage = input_storage_cpu.to("gaudirt:0")
    weight_storage = weight_storage_cpu.to("gaudirt:0")
    sliced_input = input_storage[1:, ::2]
    sliced_weight = weight_storage[:, ::2]
    before_slices = gaudirt.recipe_cache_stats()
    sliced_result = gaudirt._C.bucket_linear(sliced_input, sliced_weight)
    assert gaudirt.recipe_cache_stats() == before_slices, "sliced inputs recompiled the recipe"
    payload, error, floor = check(sliced_result, input_storage_cpu[1:, ::2], weight_storage_cpu[:, ::2])
    digest.update(payload)
    output_bytes.extend(payload)
    errors.append(error)
    floors.append(floor)
    counts = gaudirt.recipe_cache_stats()
    assert len(captures) == 1, ("unexpected FX capture count", len(captures))
    print("FX capture count:", len(captures), flush=True)
    print("bucket GEMM cache stats:", counts, flush=True)
    print("bucket GEMM output sha256:", digest.hexdigest(), "max_row_relative_l2:", max(errors),
          "max_cpu_fp32_floor:", max(floors), "M137_vs_M1_relative_l2:", probe_error, flush=True)
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(output_bytes)
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[:2] == [0, 13], counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[:2] == [13, 0], counts
    for invalid_m in (0, 4097):
        invalid = torch.empty((invalid_m, 64), dtype=torch.bfloat16, device="gaudirt:0")
        try:
            gaudirt._C.bucket_linear(invalid, weight1)
        except RuntimeError as error:
            assert "invalid bucket GEMM shape" in str(error)
        else:
            raise AssertionError("invalid token count was accepted")
    del x, first, second, weight1, weight2, invalid, input_storage, weight_storage, probe_input
    del sliced_input, sliced_weight, sliced_result
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("bucketed FX GEMM slice passed", flush=True)


if __name__ == "__main__":
    run()
