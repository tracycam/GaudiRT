"""Single-card PrivateUse1 allocation, view, copy, and shutdown smoke."""

import gc
import operator
import os
import random
import sys
import traceback

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules
with open("/proc/self/maps", encoding="utf-8") as maps:
    mapped = maps.read().lower()
assert not any(name in mapped for name in ("libhabana_pytorch", "gaudi-pytorch-bridge"))
import torch
import gaudirt


def run():
    cpu = torch.arange(48, dtype=torch.int32).reshape(6, 8)
    device = torch.empty(cpu.shape, dtype=cpu.dtype, device="gaudirt:0")
    device.copy_(cpu, non_blocking=True)
    if os.environ.get("GAUDIRT_TEST_RAISE") == "1":
        raise RuntimeError("injected lifecycle failure")
    assert torch.equal(device.cpu(), cpu)
    assert gaudirt.memory_allocated() == 256
    view = device.view(2, 3, 8)
    assert view.data_ptr() == device.data_ptr()
    tail = device.narrow(0, 1, 2)
    assert tail.data_ptr() == device.data_ptr() + 8 * cpu.element_size()
    assert torch.equal(tail.cpu(), cpu.narrow(0, 1, 2))
    transposed = device.transpose(0, 1)
    assert transposed.data_ptr() == device.data_ptr()
    assert transposed.stride() == (1, 8)
    selected = device.select(0, 2)
    assert selected.data_ptr() == device.data_ptr() + 16 * cpu.element_size()
    assert torch.equal(selected.cpu(), cpu.select(0, 2))
    permuted = device.view(2, 3, 8).permute(2, 0, 1)
    assert permuted.data_ptr() == device.data_ptr()
    assert torch.equal(permuted.cpu(), cpu.view(2, 3, 8).permute(2, 0, 1))
    alias = torch.ops.aten.alias.default(device)
    assert alias.data_ptr() == device.data_ptr()
    reshaped = torch.ops.aten._reshape_alias.default(device, [2, 3, 8], [24, 8, 1])
    assert reshaped.data_ptr() == device.data_ptr()
    assert torch.equal(reshaped.cpu(), cpu.view(2, 3, 8))
    strided = device.as_strided((3, 4), (8, 2), 1)
    assert strided.data_ptr() == device.data_ptr() + cpu.element_size()
    assert torch.equal(strided.cpu(), cpu.as_strided((3, 4), (8, 2), 1))
    other = torch.empty_like(device)
    other.copy_(device, non_blocking=True)
    assert torch.equal(other.cpu(), cpu)
    assert torch.equal(transposed.cpu(), cpu.t())
    transposed_copy = torch.empty((8, 6), dtype=cpu.dtype, device="gaudirt:0")
    transposed_copy.copy_(transposed, non_blocking=True)
    assert torch.equal(transposed_copy.cpu(), cpu.t())
    cpu_update = torch.arange(18, dtype=cpu.dtype).reshape(3, 6).t()
    device.narrow(1, 2, 3).copy_(cpu_update, non_blocking=True)
    cpu.narrow(1, 2, 3).copy_(cpu_update)
    assert torch.equal(device.cpu(), cpu)
    assert torch.equal(device.narrow(1, 2, 3).cpu(), cpu_update)

    cpu_row = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    broadcast = torch.empty((3, 4), dtype=torch.int32, device="gaudirt:0")
    broadcast.copy_(cpu_row, non_blocking=True)
    assert torch.equal(broadcast.cpu(), cpu_row.expand(3, 4))
    scalar = torch.empty((1,), dtype=torch.int32, device="gaudirt:0")
    scalar.fill_(17)
    broadcast.copy_(scalar, non_blocking=True)
    assert torch.equal(broadcast.cpu(), torch.full((3, 4), 17, dtype=torch.int32))
    row = torch.empty(cpu_row.shape, dtype=cpu_row.dtype, device="gaudirt:0")
    row.copy_(cpu_row)
    host_out = torch.empty((3, 4), dtype=cpu_row.dtype)
    host_out.copy_(row)
    assert torch.equal(host_out, cpu_row.expand(3, 4))

    pinned = torch.empty((6, 8), dtype=cpu.dtype, pin_memory=True)
    assert pinned.is_pinned()
    pinned.copy_(cpu)
    pinned_device = torch.empty((8, 6), dtype=cpu.dtype, device="gaudirt:0")
    pinned_device.copy_(pinned.t(), non_blocking=True)
    pinned_result = torch.empty((6, 8), dtype=cpu.dtype, pin_memory=True)
    pinned_result.t().copy_(pinned_device, non_blocking=True)
    pinned_device.copy_(torch.zeros((8, 6), dtype=cpu.dtype), non_blocking=True)
    gaudirt.synchronize()
    assert torch.equal(pinned_result, cpu)

    expanded = device.narrow(0, 0, 1).expand(3, 8)
    assert expanded.stride() == (0, 1)
    assert expanded.data_ptr() == device.data_ptr()
    assert device.unsqueeze(0).squeeze(0).data_ptr() == device.data_ptr()
    storage_view = torch.empty((0,), dtype=cpu.dtype, device="gaudirt:0")
    storage_view.set_(device.untyped_storage(), 8, (2, 8), (8, 1))
    assert storage_view.data_ptr() == device.data_ptr() + 8 * cpu.element_size()
    assert torch.equal(storage_view.cpu(), cpu[1:3])

    reuse_cpu = torch.arange(2 * 1024 * 1024, dtype=torch.int32)
    for _ in range(6):
        source = torch.empty(reuse_cpu.shape, dtype=reuse_cpu.dtype, device="gaudirt:0")
        target = torch.empty(reuse_cpu.shape, dtype=reuse_cpu.dtype, device="gaudirt:0")
        scalar_target = torch.empty((1,), dtype=reuse_cpu.dtype, device="gaudirt:0")
        source.copy_(reuse_cpu, non_blocking=True)
        target.copy_(source, non_blocking=True)
        del source
        scalar_target.fill_(0x76543210)
        actual = target.cpu()
        assert torch.equal(actual, reuse_cpu), "cross-stream staging reused an active allocation"
        del target, scalar_target, actual

    # More data than the remaining pinned pool, with both sides strided.
    large_cpu = torch.arange(128 * 4097, dtype=torch.int32).reshape(4097, 128).t()
    large_device = torch.empty((128, 4099), dtype=torch.int32, device="gaudirt:0")
    large_view = large_device.narrow(1, 1, 4097)
    large_view.copy_(large_cpu, non_blocking=True)
    assert torch.equal(large_view.cpu(), large_cpu)

    for dtype in (torch.bfloat16, torch.float32, torch.int32, torch.int64, torch.bool):
        source = torch.arange(60, dtype=torch.int32).reshape(6, 10).to(dtype)
        tensor = torch.empty((10, 6), dtype=dtype, device="gaudirt:0")
        tensor.copy_(source.t(), non_blocking=True)
        assert torch.equal(tensor.cpu(), source.t())
        copy = tensor.clone()
        assert torch.equal(copy.cpu(), source.t())
        values = {torch.bfloat16: -2.75, torch.float32: -3.125,
                  torch.int32: 0x76543210, torch.int64: 0x123456789abcdef,
                  torch.bool: True}
        value = values[dtype]
        tensor.narrow(1, 1, 3).fill_(value)
        reference = source.t().clone()
        reference.narrow(1, 1, 3).fill_(value)
        actual = tensor.cpu()
        if not torch.equal(actual, reference):
            print("fill mismatch:", dtype, "actual", actual[0].tolist(),
                  "expected", reference[0].tolist(), file=sys.stderr)
            raise AssertionError("fill mismatch")
        tensor.zero_()
        assert torch.equal(tensor.cpu(), torch.zeros_like(reference))
        singleton = torch.empty((1,), dtype=dtype, device="gaudirt:0")
        singleton.fill_(value)
        assert torch.equal(singleton.cpu(), torch.full((1,), value, dtype=dtype))

    for dtype in (torch.bfloat16, torch.float32, torch.int32, torch.int64, torch.bool):
        cpu_a = torch.arange(42, dtype=torch.int32).reshape(6, 7).to(dtype).t()
        cpu_b = torch.arange(42, 84, dtype=torch.int32).reshape(6, 7).to(dtype).t()
        a_storage = torch.empty((6, 7), dtype=dtype, device="gaudirt:0")
        b_storage = torch.empty((6, 7), dtype=dtype, device="gaudirt:0")
        a = a_storage.t()
        b = b_storage.t()
        a.copy_(cpu_a)
        b.copy_(cpu_b)
        for axis in (0, -1):
            expected = torch.cat((cpu_a, cpu_b), dim=axis)
            actual = torch.cat((a, b), dim=axis)
            assert torch.equal(actual.cpu(), expected), (dtype, axis)
            out = torch.empty(expected.shape, dtype=dtype, device="gaudirt:0")
            torch.cat((a, b), dim=axis, out=out)
            assert torch.equal(out.cpu(), expected), (dtype, axis, "out")
        empty = torch.empty((0,), dtype=dtype, device="gaudirt:0")
        assert torch.equal(torch.cat((empty, a), dim=0).cpu(), cpu_a)
        assert torch.cat((empty, empty), dim=0).numel() == 0
        assert torch.equal(torch.cat((a, a), dim=0).cpu(), torch.cat((cpu_a, cpu_a), dim=0))
        assert torch.equal(torch.cat((a_storage, a_storage), dim=1).cpu(),
                           torch.cat((cpu_a.t(), cpu_a.t()), dim=1))

    for dtype in (torch.bfloat16, torch.float32, torch.int32, torch.int64, torch.bool):
        cpu_a = torch.arange(-12, 23, dtype=torch.int32).reshape(5, 7).to(dtype).t()
        cpu_b = torch.tensor([-2, 0, 1, 5, 9], dtype=torch.int32).to(dtype)
        a_storage = torch.empty((5, 7), dtype=dtype, device="gaudirt:0")
        a = a_storage.t()
        a.copy_(cpu_a)
        b = torch.empty(cpu_b.shape, dtype=dtype, device="gaudirt:0")
        b.copy_(cpu_b)
        for op in (operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge):
            actual = op(a, b).cpu()
            expected = op(cpu_a, cpu_b)
            assert torch.isfinite(actual).all() and torch.equal(actual, expected), (dtype, op.__name__)
            actual_scalar = op(a, 1).cpu()
            expected_scalar = op(cpu_a, 1)
            assert torch.isfinite(actual_scalar).all() and torch.equal(actual_scalar, expected_scalar), (dtype, op.__name__, "scalar")
        condition = a > 0
        actual = torch.where(condition, a, b).cpu()
        expected = torch.where(cpu_a > 0, cpu_a, cpu_b)
        assert torch.isfinite(actual).all() and torch.equal(actual, expected), (dtype, "where")
        assert torch.equal((a == a).cpu(), cpu_a == cpu_a), (dtype, "alias comparison")
        assert torch.equal(torch.where(condition, a, a).cpu(), torch.where(cpu_a > 0, cpu_a, cpu_a))
        assert torch.equal((a_storage == a_storage).cpu(), cpu_a.t() == cpu_a.t()), (dtype, "contiguous alias")

    cpu_a = torch.tensor([-(2**63), -(2**40), -1, 0, 1, 2**40, 2**63 - 1], dtype=torch.int64)
    cpu_b = cpu_a.flip(0)
    cpu_condition = torch.tensor([True, False, True, False, True, False, True])
    a = torch.empty(cpu_a.shape, dtype=torch.int64, device="gaudirt:0")
    b = torch.empty(cpu_b.shape, dtype=torch.int64, device="gaudirt:0")
    condition = torch.empty(cpu_condition.shape, dtype=torch.bool, device="gaudirt:0")
    a.copy_(cpu_a)
    b.copy_(cpu_b)
    condition.copy_(cpu_condition)
    assert torch.equal(torch.where(condition, a, b).cpu(), torch.where(cpu_condition, cpu_a, cpu_b))

    for dtype in (torch.bfloat16, torch.float32):
        cpu_a = torch.tensor([float("nan"), float("inf"), -float("inf"),
                              0.0, -0.0, 2.0**-133, -2.0**-133, 2.0], dtype=dtype)
        cpu_b = torch.tensor([0.0, float("inf"), float("inf"), -0.0, 0.0,
                              0.0, 0.0, 3.0], dtype=dtype)
        a = torch.empty(cpu_a.shape, dtype=dtype, device="gaudirt:0")
        b = torch.empty(cpu_b.shape, dtype=dtype, device="gaudirt:0")
        a.copy_(cpu_a)
        b.copy_(cpu_b)
        for op in (operator.eq, operator.ne, operator.lt, operator.le, operator.gt, operator.ge):
            actual = op(a, b).cpu()
            expected = op(cpu_a, cpu_b)
            assert torch.isfinite(actual).all() and torch.equal(actual, expected), (dtype, op.__name__, "special")
        cpu_a = torch.tensor([0.0, -0.0, 1.0, -1.0, 2.0**-133], dtype=dtype)
        cpu_b = torch.tensor([-0.0, 0.0, -1.0, 1.0, -(2.0**-133)], dtype=dtype)
        cpu_condition = torch.tensor([True, True, False, False, True])
        a.copy_(torch.cat((cpu_a, cpu_a[:3])))
        b.copy_(torch.cat((cpu_b, cpu_b[:3])))
        condition = torch.empty(cpu_condition.shape, dtype=torch.bool, device="gaudirt:0")
        condition.copy_(cpu_condition)
        actual = torch.where(condition, a[:5], b[:5]).cpu()
        expected = torch.where(cpu_condition, cpu_a, cpu_b)
        bits = torch.int16 if dtype == torch.bfloat16 else torch.int32
        assert torch.isfinite(actual).all() and torch.equal(actual.view(bits), expected.view(bits)), (dtype, "where bits")

    cpu_a = torch.tensor([-(2**31), -1, 0, 1, 2**31 - 1], dtype=torch.int32)
    a = torch.empty(cpu_a.shape, dtype=cpu_a.dtype, device="gaudirt:0")
    a.copy_(cpu_a)
    for scalar in (1.5, -(2**40), 2**40):
        for op in (operator.eq, operator.lt, operator.gt):
            actual = op(a, scalar).cpu()
            expected = op(cpu_a, scalar)
            assert torch.isfinite(actual).all() and torch.equal(actual, expected), (scalar, op.__name__, "promotion")

    for left_dtype, right_dtype in ((torch.int32, torch.float32), (torch.bfloat16, torch.int64),
                                    (torch.bool, torch.int32), (torch.int64, torch.float32),
                                    (torch.bfloat16, torch.float32), (torch.bool, torch.bfloat16)):
        cpu_left = torch.tensor([-3, 0, 4], dtype=left_dtype)
        cpu_right = torch.tensor([2, 0, -1], dtype=right_dtype)
        left = torch.empty(cpu_left.shape, dtype=left_dtype, device="gaudirt:0")
        right = torch.empty(cpu_right.shape, dtype=right_dtype, device="gaudirt:0")
        left.copy_(cpu_left)
        right.copy_(cpu_right)
        actual = torch.cat((left, right)).cpu()
        expected = torch.cat((cpu_left, cpu_right))
        assert actual.dtype == expected.dtype and torch.equal(actual, expected), (left_dtype, right_dtype, "cat promotion")
        assert torch.equal((left < right).cpu(), cpu_left < cpu_right), (left_dtype, right_dtype, "comparison promotion")
        actual = torch.where(left != 0, left, right).cpu()
        expected = torch.where(cpu_left != 0, cpu_left, cpu_right)
        assert actual.dtype == expected.dtype and torch.equal(actual, expected), (left_dtype, right_dtype, "where promotion")
    cpu_condition = torch.tensor([True, False, True])
    cpu_value = torch.tensor([-2, 0, 5], dtype=torch.int32)
    condition = torch.empty(cpu_condition.shape, dtype=torch.bool, device="gaudirt:0")
    value = torch.empty(cpu_value.shape, dtype=torch.int32, device="gaudirt:0")
    condition.copy_(cpu_condition)
    value.copy_(cpu_value)
    for actual, expected in ((torch.where(condition, value, 1.5), torch.where(cpu_condition, cpu_value, 1.5)),
                             (torch.where(condition, 1.5, value), torch.where(cpu_condition, 1.5, cpu_value)),
                             (torch.where(condition, 1, 2.5), torch.where(cpu_condition, 1, 2.5))):
        assert actual.dtype == expected.dtype and torch.equal(actual.cpu(), expected)

    for dtype in (torch.int32, torch.int64):
        for start, end, step in ((0, 13, 1), (-11, 13, 3), (13, -11, -3),
                                 (-(2**31), -(2**31) + 8, 1),
                                 (2**31 - 8, 2**31 - 1, 1), (4, 4, 1),
                                 (-(2**31), 2**31 - 1, 2**31 - 1),
                                 (2**31 - 1, -(2**31), -(2**31))):
            expected = torch.arange(start, end, step, dtype=dtype)
            actual = torch.arange(start, end, step, dtype=dtype, device="gaudirt:0")
            host_actual = actual.cpu()
            if not torch.equal(host_actual, expected):
                print("range mismatch:", dtype, start, end, step,
                      "actual", host_actual.tolist(), "expected", expected.tolist(), file=sys.stderr)
                raise AssertionError("range mismatch")
            out = torch.empty((expected.numel() * 2,), dtype=dtype, device="gaudirt:0")[::2]
            torch.arange(start, end, step, out=out)
            assert torch.equal(out.cpu(), expected), (dtype, start, end, step, "out")
    for args in ((0.0, 2.0, 0.5), (0, 2**24 + 1, 1), (2**31, 2**31 + 1, 1)):
        try:
            torch.arange(*args, dtype=torch.int64, device="gaudirt:0")
        except RuntimeError as error:
            assert "GaudiRT: arange" in str(error)
        else:
            raise AssertionError("unsupported arange was accepted")
    for dtype in (torch.float32, torch.bfloat16):
        cases = ((0, 128, 2), (-100, 100, 1), (200, 256, 3)) if dtype == torch.bfloat16 else (
            (0, 128, 2), (-1025, 1025, 7), (2**24 - 32, 2**24, 3))
        for start, end, step in cases:
            expected = torch.arange(start, end, step, dtype=dtype)
            actual = torch.arange(start, end, step, dtype=dtype, device="gaudirt:0").cpu()
            bits = torch.int32 if dtype == torch.float32 else torch.int16
            if not torch.isfinite(actual).all() or not torch.equal(actual.view(bits), expected.view(bits)):
                bad = (actual.view(bits) != expected.view(bits)).nonzero().flatten()[:8]
                print("floating arange mismatch:", dtype, start, end, step, "indices", bad.tolist(),
                      "actual", actual[bad].tolist(), "expected", expected[bad].tolist(), file=sys.stderr)
                raise AssertionError("floating arange mismatch")
        invalid = ((0.1, 1.0, 0.2), (100000001, 100000199, 3))
        if dtype == torch.bfloat16:
            invalid += ((-1025, 1025, 7),)
        for args in invalid:
            try:
                torch.arange(*args, dtype=dtype, device="gaudirt:0")
            except RuntimeError as error:
                assert "GaudiRT: " in str(error)
            else:
                raise AssertionError("unchecked floating arange was accepted")
    rng = random.Random(2026)
    checked = 0
    while checked < 48:
        start = rng.randint(-(2**31), 2**31 - 1)
        step = rng.randint(-(2**31), 2**31 - 1)
        count = rng.randint(1, 6)
        end = start + step * count
        if not step or not -(2**31) <= end < 2**31:
            continue
        dtype = torch.int32 if checked % 2 else torch.int64
        expected = torch.arange(start, end, step, dtype=dtype)
        actual = torch.arange(start, end, step, dtype=dtype, device="gaudirt:0").cpu()
        if not torch.equal(actual, expected):
            print("random range mismatch:", dtype, start, end, step,
                  "actual", actual.tolist(), "expected", expected.tolist(), file=sys.stderr)
            raise AssertionError("random range mismatch")
        checked += 1

    for dtype in (torch.bfloat16, torch.float32, torch.int32, torch.int64, torch.bool):
        cpu_weight = torch.arange(7 * 11, dtype=torch.int32).to(dtype).reshape(7, 11)
        storage = torch.empty((11, 7), dtype=dtype, device="gaudirt:0")
        weight = storage.t()
        weight.copy_(cpu_weight)
        for index_dtype in (torch.int32, torch.int64):
            cpu_index = torch.tensor([[0, 6, 4], [2, 1, 1]], dtype=index_dtype)
            index = torch.empty(cpu_index.shape, dtype=index_dtype, device="gaudirt:0")
            index.copy_(cpu_index)
            actual = torch.nn.functional.embedding(index, weight).cpu()
            expected = torch.nn.functional.embedding(cpu_index, cpu_weight)
            assert torch.isfinite(actual).all() and torch.equal(actual, expected), (dtype, index_dtype, "embedding")
            for axis in (0, 1):
                cpu_ids = torch.tensor([0, 2, 1], dtype=index_dtype)
                ids = torch.empty(cpu_ids.shape, dtype=index_dtype, device="gaudirt:0")
                ids.copy_(cpu_ids)
                actual = torch.index_select(weight, axis, ids).cpu()
                expected = torch.index_select(cpu_weight, axis, cpu_ids)
                assert torch.isfinite(actual).all() and torch.equal(actual, expected), (dtype, index_dtype, axis)
        for bad_value in (-1, 7, 2**32):
            bad_cpu = torch.tensor([0, bad_value], dtype=torch.int64)
            bad = torch.empty(bad_cpu.shape, dtype=torch.int64, device="gaudirt:0")
            bad.copy_(bad_cpu)
            try:
                torch.nn.functional.embedding(bad, weight)
            except RuntimeError as error:
                assert "index out of bounds" in str(error)
            else:
                raise AssertionError("invalid embedding index was accepted")

    cpu_base = torch.arange(3 * 4 * 5, dtype=torch.float32).reshape(3, 4, 5)
    base = torch.empty(cpu_base.shape, dtype=cpu_base.dtype, device="gaudirt:0")
    base.copy_(cpu_base)
    cpu_source = cpu_base.permute(2, 1, 0)
    source = base.permute(2, 1, 0)
    cpu_ids = torch.tensor([2, 0, 1], dtype=torch.int64)
    ids = torch.empty(cpu_ids.shape, dtype=cpu_ids.dtype, device="gaudirt:0")
    ids.copy_(cpu_ids)
    for axis in (0, 1, 2, -1):
        actual = torch.index_select(source, axis, ids).cpu()
        expected = torch.index_select(cpu_source, axis, cpu_ids)
        assert torch.isfinite(actual).all() and torch.equal(actual, expected), (axis, "rank 3 gather")
    empty_ids = torch.empty((0,), dtype=torch.int64, device="gaudirt:0")
    assert torch.index_select(source, 1, empty_ids).shape == (5, 0, 3)
    cpu_alias = torch.tensor([2, 0, 1], dtype=torch.int32)
    device_alias = torch.empty(cpu_alias.shape, dtype=cpu_alias.dtype, device="gaudirt:0")
    device_alias.copy_(cpu_alias)
    assert torch.equal(torch.index_select(device_alias, 0, device_alias).cpu(),
                       torch.index_select(cpu_alias, 0, cpu_alias))

    for dtype in (torch.bfloat16, torch.float32, torch.int32):
        cpu_base = torch.tensor([3, 7, 7, -1, 4, 9, -2, 9, 0, 8, 4, 4,
                                 -1, 0, 2, 2, 6, 5, 1, 3, -3, 8, 0, 8], dtype=dtype).reshape(2, 3, 4)
        base = torch.empty(cpu_base.shape, dtype=dtype, device="gaudirt:0")
        base.copy_(cpu_base)
        cpu_source = cpu_base.permute(2, 0, 1)
        source = base.permute(2, 0, 1)
        for dim in (None, 0, 1, 2, -1):
            for keep in (False, True):
                expected = torch.argmax(cpu_source, dim=dim, keepdim=keep)
                actual = torch.argmax(source, dim=dim, keepdim=keep).cpu()
                if not torch.equal(actual, expected):
                    print("argmax mismatch:", dtype, dim, keep, "actual", actual.tolist(),
                          "expected", expected.tolist(), file=sys.stderr)
                    raise AssertionError("argmax mismatch")
        if dtype != torch.int32:
            cpu_source = torch.tensor([[1.0, float("nan"), 9.0, float("nan")],
                                       [-float("inf"), 2.0, float("inf"), float("inf")]], dtype=dtype)
            source = torch.empty(cpu_source.shape, dtype=dtype, device="gaudirt:0")
            source.copy_(cpu_source)
            actual = torch.argmax(source, dim=1).cpu()
            expected = torch.argmax(cpu_source, dim=1)
            if not torch.equal(actual, expected):
                print("argmax nonfinite mismatch:", dtype, "actual", actual.tolist(),
                      "expected", expected.tolist(), file=sys.stderr)
                raise AssertionError("argmax nonfinite mismatch")
            cpu_source = torch.tensor([[float("nan"), float("nan"), 1.0, 1.0],
                                       [-float("inf"), -float("inf"), -float("inf"), -float("inf")],
                                       [0.0, -0.0, -0.0, 0.0],
                                       [-float("inf"), -torch.finfo(dtype).max,
                                        -float("inf"), -torch.finfo(dtype).max]], dtype=dtype)
            source = torch.empty(cpu_source.shape, dtype=dtype, device="gaudirt:0")
            source.copy_(cpu_source)
            actual = torch.argmax(source, dim=1).cpu()
            expected = torch.argmax(cpu_source, dim=1)
            if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
                print("special argmax mismatch:", dtype, "actual", actual.tolist(),
                      "expected", expected.tolist(), file=sys.stderr)
                raise AssertionError("special argmax mismatch")
        else:
            cpu_source = torch.full((2, 4), -(2**31), dtype=dtype)
            source = torch.empty(cpu_source.shape, dtype=dtype, device="gaudirt:0")
            source.copy_(cpu_source)
            assert torch.equal(torch.argmax(source, dim=1).cpu(), torch.argmax(cpu_source, dim=1))

    for width in (1, 3, 127, 129, 1025):
        cpu_source = torch.zeros((2, width), dtype=torch.float32)
        cpu_source[:, min(1, width - 1)] = 2
        cpu_source[:, width - 1] = 2
        source = torch.empty(cpu_source.shape, dtype=cpu_source.dtype, device="gaudirt:0")
        source.copy_(cpu_source)
        actual = torch.argmax(source, dim=1).cpu()
        expected = torch.argmax(cpu_source, dim=1)
        assert torch.isfinite(actual).all() and torch.equal(actual, expected), (width, "argmax tie")
    cpu_source = torch.tensor([[-(2**63), 2**62, 2**62 + 1, 2**62],
                               [-1, -(2**63), -1, -(2**40)],
                               [2**63 - 1, 2**63 - 1, -(2**63), 0]], dtype=torch.int64)
    source = torch.empty(cpu_source.shape, dtype=cpu_source.dtype, device="gaudirt:0")
    source.copy_(cpu_source)
    for dim in (None, 0, 1, -1):
        for keep in (False, True):
            actual = torch.argmax(source, dim=dim, keepdim=keep).cpu()
            expected = torch.argmax(cpu_source, dim=dim, keepdim=keep)
            assert torch.isfinite(actual).all() and torch.equal(actual, expected), (dim, keep, "int64 argmax")
    cpu_source = torch.randint(-(2**63), 2**63 - 1, (7, 17), dtype=torch.int64,
                               generator=torch.Generator().manual_seed(85))
    cpu_source[0, 5] = cpu_source[0, 8] = 2**63 - 1
    source = torch.empty(cpu_source.shape, dtype=cpu_source.dtype, device="gaudirt:0")
    source.copy_(cpu_source)
    for dim in (0, 1):
        assert torch.equal(torch.argmax(source, dim=dim).cpu(), torch.argmax(cpu_source, dim=dim)), (dim, "random int64 argmax")
    random_logits = torch.randint(-10, 11, (16, 33), generator=torch.Generator().manual_seed(84)).float()
    random_logits[0, 3] = float("nan")
    random_logits[0, 7] = float("nan")
    random_logits[1, :] = -float("inf")
    random_logits[2, 9] = float("inf")
    for dtype in (torch.float32, torch.bfloat16):
        cpu_source = random_logits.to(dtype)
        source = torch.empty(cpu_source.shape, dtype=dtype, device="gaudirt:0")
        source.copy_(cpu_source)
        actual = torch.argmax(source, dim=1).cpu()
        expected = torch.argmax(cpu_source, dim=1)
        assert torch.isfinite(actual).all() and torch.equal(actual, expected), (dtype, "random argmax")

    casts = {
        torch.bfloat16: ([-12.75, -2.5, -0.5, 0, 1.5, 127.5], (torch.float32, torch.int32, torch.int64, torch.bool)),
        torch.float32: ([-12.75, -2.5, -0.5, 0, 1.5, 127.5], (torch.bfloat16, torch.int32, torch.int64, torch.bool)),
        torch.int32: ([-2147483648, -16777217, -1, 0, 16777217, 2147483647],
                      (torch.bfloat16, torch.float32, torch.int64, torch.bool)),
        torch.int64: ([-2**40, -2**31 - 1, -1, 0, 2**31 + 1, 2**40],
                      (torch.int32, torch.float32, torch.bfloat16, torch.bool)),
        torch.bool: ([True, False, True, True, False, False],
                     (torch.bfloat16, torch.float32, torch.int32, torch.int64)),
    }
    for source_dtype, (values, destinations) in casts.items():
        source = torch.tensor(values, dtype=source_dtype).reshape(2, 3).t()
        tensor = torch.empty(source.shape, dtype=source_dtype, device="gaudirt:0")
        tensor.copy_(source)
        for target_dtype in destinations:
            actual = tensor.to(target_dtype).cpu()
            expected = source.to(target_dtype)
            assert torch.isfinite(actual).all()
            if not torch.equal(actual, expected):
                print("cast mismatch:", source_dtype, target_dtype,
                      "actual", actual.tolist(), "expected", expected.tolist(), file=sys.stderr)
                raise AssertionError("cast mismatch")
            from_cpu = torch.empty(source.shape, dtype=target_dtype, device="gaudirt:0")
            from_cpu.copy_(source)
            assert torch.equal(from_cpu.cpu(), expected)
            to_cpu = torch.empty(source.shape, dtype=target_dtype)
            to_cpu.copy_(tensor)
            assert torch.equal(to_cpu, expected)

    edge_cases = [
        (torch.float32, torch.bfloat16,
         [0.0, 2.0**-133, -2.0**-133, 2.0**-126, 1.00390625, -1.00390625]),
        (torch.bfloat16, torch.float32,
         [0.0, 2.0**-133, -2.0**-133, 2.0**-126, 1.0, -1.0]),
        (torch.float32, torch.bool,
         [0.0, -0.0, 2.0**-133, -2.0**-133, 2.0**-126, -1.0]),
        (torch.bfloat16, torch.bool,
         [0.0, -0.0, 2.0**-133, -2.0**-133, 2.0**-126, -1.0]),
        (torch.float32, torch.int32,
         [0.0, 2**31 - 128, 2**31, -2**31, -2**31 - 256, 2**40, -2**40, 1.75]),
        (torch.bfloat16, torch.int32,
         [0.0, 2**31 - 2**23, 2**31, -2**31, -2**31 - 2**24, 2**40, -2**40, 1.75]),
        (torch.float32, torch.int64,
         [0.0, 2**63 - 2**39, 2**63, -2**63, -2**63 - 2**40, 2**70, -2**70, 1.75]),
        (torch.bfloat16, torch.int64,
         [0.0, 2**63 - 2**55, 2**63, -2**63, -2**63 - 2**56, 2**70, -2**70, 1.75]),
    ]
    for source_dtype, target_dtype, values in edge_cases:
        source = torch.tensor(values, dtype=source_dtype)
        assert torch.isfinite(source).all()
        tensor = torch.empty(source.shape, dtype=source_dtype, device="gaudirt:0")
        tensor.copy_(source)
        actual = tensor.to(target_dtype).cpu()
        expected = source.to(target_dtype)
        assert torch.isfinite(actual).all()
        if not torch.equal(actual, expected):
            print("cast edge mismatch:", source_dtype, target_dtype,
                  "actual", actual.tolist(), "expected", expected.tolist(), file=sys.stderr)
            raise AssertionError("cast edge mismatch")
    for dtype in (torch.float32, torch.bfloat16):
        source = torch.tensor([float("nan"), float("inf"), -float("inf"), -0.0], dtype=dtype)
        device = torch.empty(source.shape, dtype=dtype, device="gaudirt:0")
        device.copy_(source)
        for target in (torch.int32, torch.int64):
            actual = device.to(target).cpu()
            expected = source.to(target)
            if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
                print("nonfinite cast mismatch:", dtype, target, "actual", actual.tolist(),
                      "expected", expected.tolist(), file=sys.stderr)
                raise AssertionError("nonfinite cast mismatch")

    generator = torch.Generator().manual_seed(42)
    float_patterns = torch.randint(-(2**31), 2**31 - 1, (65536,), dtype=torch.int32,
                                   generator=generator).view(torch.float32)
    bf16_patterns = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    finite_float = float_patterns[torch.isfinite(float_patterns) &
                                  (float_patterns.abs() <= torch.finfo(torch.bfloat16).max)]
    finite_bf16 = bf16_patterns[torch.isfinite(bf16_patterns)]
    for source, target, bit_type in ((finite_float, torch.bfloat16, torch.int16),
                                     (finite_bf16, torch.float32, torch.int32)):
        device = torch.empty(source.shape, dtype=source.dtype, device="gaudirt:0")
        device.copy_(source)
        actual = device.to(target).cpu()
        expected = source.to(target)
        if not torch.isfinite(actual).all() or not torch.equal(actual.view(bit_type), expected.view(bit_type)):
            bad = (actual.view(bit_type) != expected.view(bit_type)).nonzero().flatten()[:4]
            print("floating cast pattern mismatch:", source.dtype, target, "input", source[bad].tolist(),
                  "actual", actual[bad].tolist(), "expected", expected[bad].tolist(), file=sys.stderr)
            raise AssertionError("floating cast pattern mismatch")
    for source in (float_patterns, bf16_patterns):
        device = torch.empty(source.shape, dtype=source.dtype, device="gaudirt:0")
        device.copy_(source)
        for target in (torch.int32, torch.int64):
            actual = device.to(target).cpu()
            expected = source.to(target)
            if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
                bad = (actual != expected).nonzero().flatten()[:4]
                print("float pattern cast mismatch:", source.dtype, target, "indices", bad.tolist(),
                      "input", source[bad].tolist(),
                      "actual", actual[bad].tolist(), "expected", expected[bad].tolist(), file=sys.stderr)
                raise AssertionError("float pattern cast mismatch")
    source = torch.randint(-(2**31), 2**31 - 1, (4096,), dtype=torch.int32, generator=generator)
    device = torch.empty(source.shape, dtype=source.dtype, device="gaudirt:0")
    device.copy_(source)
    for dtype in (torch.float32, torch.bfloat16):
        actual = device.to(dtype).cpu()
        expected = source.to(dtype)
        if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
            bad = (actual != expected).nonzero().flatten()[:4]
            print("int32 pattern cast mismatch:", dtype, "input", source[bad].tolist(),
                  "actual", actual[bad].tolist(), "expected", expected[bad].tolist(), file=sys.stderr)
            raise AssertionError("int32 pattern cast mismatch")
    source = torch.randint(-(2**63), 2**63 - 1, (16384,), dtype=torch.int64, generator=generator)
    device = torch.empty(source.shape, dtype=source.dtype, device="gaudirt:0")
    device.copy_(source)
    for dtype in (torch.int32, torch.float32, torch.bfloat16):
        actual = device.to(dtype).cpu()
        expected = source.to(dtype)
        if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
            bad = (actual != expected).nonzero().flatten()[:4]
            print("random cast mismatch:", dtype, "input", source[bad].tolist(),
                  "actual", actual[bad].tolist(), "expected", expected[bad].tolist(), file=sys.stderr)
            raise AssertionError("random cast mismatch")
    boundaries = [-(2**63), -(2**63) + 1, -(2**63) + 2**38 - 1,
                  -(2**63) + 2**38, 2**63 - 2**38 - 1, 2**63 - 2**38,
                  2**63 - 1]
    for exponent in (24, 25, 31, 32, 40, 48, 55, 62):
        midpoint = (2**exponent) + (2**(exponent - 24))
        for offset in (-2, -1, 0, 1, 2):
            boundaries.extend((midpoint + offset, -midpoint - offset))
        bf16_midpoint = (2**exponent) + (2**(exponent - 8))
        for offset in (-2, -1, 0, 1, 2):
            boundaries.extend((bf16_midpoint + offset, -bf16_midpoint - offset))
    source = torch.tensor(boundaries, dtype=torch.int64)
    device = torch.empty(source.shape, dtype=source.dtype, device="gaudirt:0")
    device.copy_(source)
    actual = device.to(torch.float32).cpu()
    expected = source.to(torch.float32)
    if not torch.isfinite(actual).all() or not torch.equal(actual.view(torch.int32), expected.view(torch.int32)):
        bad = (actual.view(torch.int32) != expected.view(torch.int32)).nonzero().flatten()[:8]
        print("int64 midpoint mismatch:", "input", source[bad].tolist(), "actual", actual[bad].tolist(),
              "expected", expected[bad].tolist(), file=sys.stderr)
        raise AssertionError("int64 midpoint mismatch")
    actual = device.to(torch.bfloat16).cpu()
    expected = source.to(torch.bfloat16)
    if not torch.isfinite(actual).all() or not torch.equal(actual.view(torch.int16), expected.view(torch.int16)):
        bad = (actual.view(torch.int16) != expected.view(torch.int16)).nonzero().flatten()[:8]
        print("int64 bf16 midpoint mismatch:", "input", source[bad].tolist(), "actual", actual[bad].tolist(),
              "expected", expected[bad].tolist(), file=sys.stderr)
        raise AssertionError("int64 bf16 midpoint mismatch")
    middle = 2**30 + 2**22
    source = torch.tensor([middle - 1, middle, middle + 1, -middle - 1,
                           -middle, -middle + 1], dtype=torch.int32)
    device = torch.empty(source.shape, dtype=source.dtype, device="gaudirt:0")
    device.copy_(source)
    assert torch.equal(device.to(torch.float32).cpu(), source.to(torch.float32))
    assert torch.equal(source.to(torch.bfloat16), source.to(torch.float32).to(torch.bfloat16))
    actual = device.to(torch.bfloat16).cpu()
    expected = source.to(torch.bfloat16)
    if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
        print("int32 bf16 midpoint mismatch:", source.tolist(), "actual", actual.tolist(),
              "expected", expected.tolist(), file=sys.stderr)
        raise AssertionError("int32 bf16 midpoint mismatch")


gaudirt.initialize(hbm_bytes=32 * 1024 * 1024, pinned_bytes=1024 * 1024)
error = None
try:
    run()
except Exception:
    error = traceback.format_exc()
gc.collect()
assert gaudirt.memory_allocated() == 0
stats = gaudirt.recipe_cache_stats()
if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
    assert stats[0] == 0 and stats[1] > 0 and stats[2] == stats[1], stats
if os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
    assert stats[0] > 0 and stats[1] == 0 and stats[2] == stats[0], stats
print("cold recipe cache stats:", stats)
gaudirt.empty_cache()
gaudirt.shutdown()
assert "habana_frameworks" not in sys.modules
if error:
    print(error, file=sys.stderr)
    sys.exit(1)
print("generic PrivateUse1 lifecycle, views and strided copies passed")
