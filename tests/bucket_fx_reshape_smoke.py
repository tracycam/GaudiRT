"""Token-preserving views around dynamic batch GEMM retain storage semantics."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
from fp64_reference import direct_bf16

import gaudirt
import gaudirt.compiler as compiler


def view_only(x):
    return x.view(-1, 3, 7)


def round_trip(x):
    return x.view(-1, 3, 7).reshape(-1, 21)


def mixed_views(x):
    return x.view(-1, 3, 7).reshape(-1, 21)[:, 1:5]


def bmm(x, y):
    return torch.bmm(x.view(-1, 3, 7), y)


def bmm_flat(x, y):
    value = torch.bmm(x.view(-1, 3, 7), y)
    return value.view(-1, 15), value


def bmm_functional(x, y):
    return torch.bmm(torch.reshape(x, (-1, 3, 7)), y)


def bmm_aten(x, y):
    return torch.bmm(torch.ops.aten.view.default(x, [-1, 3, 7]), y)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(6372)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    captures = 0
    backend = compiler.gaudirt_backend

    def capture(graph, inputs):
        nonlocal captures
        captures += 1
        return backend(graph, inputs)

    compiler.gaudirt_backend = capture
    view = compiler.compile(view_only)
    roundtrip = compiler.compile(round_trip)
    mixed = compiler.compile(mixed_views)
    first = compiler.compile(bmm, dynamic_arg_indices=(0, 1))
    second = compiler.compile(bmm_flat, dynamic_arg_indices=(0, 1))
    functional = compiler.compile(bmm_functional, dynamic_arg_indices=(0, 1))
    aten = compiler.compile(bmm_aten, dynamic_arg_indices=(0, 1))
    digest = hashlib.sha256()
    worst = 0.0
    for dtype in (torch.bfloat16, torch.float32):
        for rows in (1, 3, 17, 137, 513, 4096):
            cpu = torch.randn(rows, 21).to(dtype)
            if rows == 3:
                cpu_backing = torch.randn(21, rows).to(dtype)
                cpu = cpu_backing.t()
                device = cpu_backing.to("gaudirt:0").t()
            elif rows == 17:
                cpu_backing = torch.empty((rows, 42), dtype=dtype)
                cpu_backing[:, ::2].copy_(cpu)
                cpu = cpu_backing[:, ::2]
                device_backing = torch.empty((rows, 42), dtype=dtype, device="gaudirt:0")
                device = device_backing[:, ::2]
                device.copy_(cpu)
            else:
                device = cpu.to("gaudirt:0")
            y_cpu = torch.randn(rows, 7, 5).to(dtype)
            y = y_cpu.to("gaudirt:0")
            viewed, back = view(device), roundtrip(device)
            sliced = mixed(device)
            assert viewed.data_ptr() == device.data_ptr() and viewed.stride() == cpu.view(-1, 3, 7).stride()
            assert back.data_ptr() == device.data_ptr() and back.stride() == cpu.stride()
            assert sliced.data_ptr() == device.data_ptr() + device.stride(1) * device.element_size()
            assert sliced.stride() == cpu[:, 1:5].stride() and torch.equal(sliced.cpu(), cpu[:, 1:5])
            assert torch.equal(viewed.cpu(), cpu.view(-1, 3, 7))
            if rows == 17:
                viewed[0, 0, 1].copy_(torch.tensor(2.5, dtype=dtype))
                cpu[0, 1] = 2.5
                assert torch.equal(device.cpu(), cpu), "view mutation did not reach its input"
            actual = first(device, y)
            before = gaudirt.recipe_cache_stats()
            flat, same = second(device, y)
            functional_output = functional(device, y)
            aten_output = aten(device, y)
            assert gaudirt.recipe_cache_stats() == before, "equivalent reshape graph recompiled"
            assert flat.data_ptr() == same.data_ptr() and torch.equal(flat.cpu(), same.cpu().view(-1, 15))
            assert torch.equal(actual.cpu(), same.cpu())
            assert torch.equal(actual.cpu(), functional_output.cpu()) and torch.equal(actual.cpu(), aten_output.cpu())
            truth = torch.bmm(cpu.double().view(-1, 3, 7), y_cpu.double())
            floor = torch.bmm(cpu.float().view(-1, 3, 7), y_cpu.float())
            if dtype == torch.bfloat16:
                truth = direct_bf16(truth)
                floor = floor.to(torch.bfloat16).double()
            else:
                truth, floor = truth.float().double(), floor.double()
            candidate = actual.cpu().double()
            assert torch.isfinite(candidate).all() and torch.isfinite(truth).all() and torch.isfinite(floor).all()
            denom = torch.linalg.vector_norm(truth.reshape(rows, -1), dim=-1).clamp_min(1e-30)
            error = (torch.linalg.vector_norm((candidate - truth).reshape(rows, -1), dim=-1) / denom).max().item()
            floor_error = (torch.linalg.vector_norm((floor - truth).reshape(rows, -1), dim=-1) / denom).max().item()
            assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2
            worst = max(worst, error)
            digest.update(actual.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    stats = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert stats[:2] == [0, 12], stats
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert stats[:2] == [12, 0], stats
    assert captures == 14, captures
    bad = gaudirt._C.BucketProgram([1], [gaudirt._C.FxNode.reshape(0, [4, 5])], [1])
    try:
        bad.run([torch.empty((3, 21), dtype=torch.float32, device="gaudirt:0")])
    except RuntimeError as error:
        assert "reshape changes the token width" in str(error)
    else:
        raise AssertionError("accepted a reshape that changes token width")
    print("FX reshape and bmm", "stats", stats, "max_row_relative_l2", worst,
          "output_sha256", digest.hexdigest(), flush=True)


if __name__ == "__main__":
    main()
