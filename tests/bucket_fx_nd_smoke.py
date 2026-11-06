"""Token-preserving 1D through 5D views, transposes and batched matmul."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import gaudirt.compiler as compiler
from fp64_reference import direct_bf16


def graph4(left, right):
    view = left.transpose(2, 3)
    return torch.matmul(view, right), view, left


def graph5(left, right):
    view = left.transpose(2, 3)
    return torch.matmul(view, right), view, left


def view5(value):
    shaped = value.view(-1, 2, 3, 2, 5)
    return shaped.transpose(1, 3), shaped


def view1(value):
    return value.view(-1, 1).view(-1)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(9041)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=4 * 1024**2)
    compiled = [compiler.compile(fn, dynamic_arg_indices=(0, 1)) for fn in (graph4, graph5)]
    compiled_view5, compiled_view1 = compiler.compile(view5), compiler.compile(view1)
    payload = bytearray()
    worst = worst_floor = 0.0
    for dtype in (torch.float32, torch.bfloat16):
        vector_program = gaudirt._C.BucketProgram([1], [[17, 0, 2, 1], [17, 1, 1]], [2])
        for rows in (1, 3, 17):
            for fn, left_shape, right_shape in (
                (compiled[0], (rows, 3, 4, 5), (rows, 3, 4, 6)),
                (compiled[1], (rows, 2, 3, 4, 5), (rows, 2, 4, 5, 6)),
            ):
                left = torch.randn(left_shape, dtype=dtype)
                right = torch.randn(right_shape, dtype=dtype)
                device_left = left.to("gaudirt:0")
                actual, view, original = fn(device_left, right.to("gaudirt:0"))
                expected_view = left.transpose(2, 3)
                assert view.data_ptr() == original.data_ptr() == device_left.data_ptr()
                assert view.shape == expected_view.shape and view.stride() == expected_view.stride()
                assert torch.equal(view.cpu(), expected_view)
                exact = torch.matmul(left.double().transpose(2, 3), right.double())
                truth = direct_bf16(exact) if dtype == torch.bfloat16 else exact.float().double()
                floor = torch.matmul(left.float().transpose(2, 3), right.float())
                floor = floor.to(dtype).double()
                value = actual.cpu().double()
                assert all(torch.isfinite(item).all() for item in (value, truth, floor))
                denominator = torch.linalg.vector_norm(truth.reshape(rows, -1), dim=-1).clamp_min(1e-30)
                error = (torch.linalg.vector_norm((value - truth).reshape(rows, -1), dim=-1) / denominator).max().item()
                floor_error = (torch.linalg.vector_norm((floor - truth).reshape(rows, -1), dim=-1) / denominator).max().item()
                assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2, (error, floor_error)
                worst, worst_floor = max(worst, error), max(worst_floor, floor_error)
                payload.extend(actual.cpu().contiguous().view(torch.uint8).numpy().tobytes())

            flat = torch.randn(rows, 60, dtype=dtype).to("gaudirt:0")
            transposed, shaped = compiled_view5(flat)
            assert transposed.data_ptr() == shaped.data_ptr() == flat.data_ptr()
            assert transposed.shape == (rows, 2, 3, 2, 5)
            assert torch.equal(transposed.cpu(), flat.cpu().view(rows, 2, 3, 2, 5).transpose(1, 3))
            vector = torch.randn(rows, dtype=dtype).to("gaudirt:0")
            alias = compiled_view1(vector)
            assert alias.data_ptr() == vector.data_ptr() and torch.equal(alias.cpu(), vector.cpu())
            vector_result = vector_program.run([vector])[0].cpu()
            assert torch.equal(vector_result, vector.cpu())
            payload.extend(vector_result.contiguous().view(torch.uint8).numpy().tobytes())
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("ND FX matmul max_row_relative_l2", worst, "FP32 floor", worst_floor,
          "recipe stats", gaudirt.recipe_cache_stats(), "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
