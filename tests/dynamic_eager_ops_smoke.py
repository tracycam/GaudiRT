#!/usr/bin/env python3
"""Verify dynamic 1D eager cold recipes eliminate per-length recompilation and preserve accuracy."""

import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


torch.set_num_threads(4)


def _sync_and_stats():
    gaudirt.synchronize()
    return tuple(gaudirt.recipe_cache_stats())


def _cold_keys(prefix):
    return sorted(k for k in gaudirt._C._recipe_cache_keys() if k.startswith(prefix))


def test_cast_dynamic(device):
    pairs = [
        (torch.int64, torch.int32),
        (torch.int32, torch.int64),
        (torch.bool, torch.float32),
        (torch.float32, torch.bool),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.float32, torch.int32),
        (torch.bfloat16, torch.bool),
        (torch.int32, torch.bool),
        (torch.int64, torch.bool),
        (torch.bool, torch.int64),
    ]

    def make_input(dtype, n, seed):
        gen = torch.Generator().manual_seed(seed)
        if dtype == torch.bool:
            t = torch.randint(0, 2, (n,), generator=gen, dtype=torch.int32).bool()
        elif dtype in (torch.int32, torch.int64):
            t = torch.randint(-50, 50, (n,), generator=gen, dtype=dtype)
        elif dtype == torch.bfloat16:
            t = (torch.randn(n, generator=gen, dtype=torch.float32) * 3.5).to(torch.bfloat16)
        else:
            t = torch.randn(n, generator=gen, dtype=torch.float32) * 10.0
        if n >= 5:
            if dtype in (torch.float32, torch.bfloat16, torch.int32, torch.int64):
                t[0] = 0
        return t

    def check_pair(src_dtype, dst_dtype, n, seed):
        x_cpu = make_input(src_dtype, n, seed)
        if src_dtype == torch.float32 and dst_dtype == torch.int32 and n >= 5:
            x_cpu[1] = float("nan")
            x_cpu[2] = 3.0e9
            x_cpu[3] = -17.75
            x_cpu[4] = 42.9
        elif src_dtype == torch.int64 and dst_dtype == torch.bool and n >= 5:
            # Verify full 64-bit comparison (high-bit set with zero low 32 bits must be True).
            x_cpu[1] = 1 << 40
            x_cpu[2] = -(1 << 40)
        x_dev = x_cpu.to(device)
        got = x_dev.to(dst_dtype).cpu()
        if src_dtype == torch.float32 and dst_dtype == torch.int32 and n >= 5:
            assert got[1].item() == -2147483648, got[1]
            assert got[2].item() == -2147483648, got[2]
            in_range = torch.tensor([0, 3, 4], dtype=torch.long)
            assert torch.equal(got[in_range], x_cpu[in_range].to(torch.int32))
            if n > 5:
                assert torch.equal(got[5:], x_cpu[5:].to(torch.int32))
        else:
            assert torch.equal(got, x_cpu.to(dst_dtype)), (src_dtype, dst_dtype, n)

    for idx, (s, d) in enumerate(pairs):
        check_pair(s, d, 5, 100 + idx)
    warm_stats = _sync_and_stats()
    warm_cast_keys = _cold_keys("cold_cast_")

    for n in (17, 65, 130):
        for idx, (s, d) in enumerate(pairs):
            check_pair(s, d, n, 200 + n + idx)
    # Strided non-contiguous cast also reuses cached recipes cleanly.
    s_cpu = torch.arange(0, 66, dtype=torch.int64)[::2]
    s_dev = s_cpu.to(device)
    assert torch.equal(s_dev.to(torch.int32).cpu(), s_cpu.to(torch.int32))

    after_stats = _sync_and_stats()
    assert after_stats[1] == warm_stats[1] and after_stats[2] == warm_stats[2], (
        warm_stats, after_stats
    )
    assert _cold_keys("cold_cast_") == warm_cast_keys


def test_index_select_dynamic(device):
    def run_cases(idx_len):
        gen = torch.Generator().manual_seed(300 + idx_len)
        for idx_dtype in (torch.int32, torch.int64):
            # 1D
            src1 = torch.arange(200, dtype=torch.bfloat16)
            ids1 = torch.randint(0, 200, (idx_len,), generator=gen, dtype=idx_dtype)
            got1 = src1.to(device).index_select(0, ids1.to(device)).cpu()
            assert torch.equal(got1, src1.index_select(0, ids1.long()))

            # 2D axis=0 with fixed cols and varying rows within first 65536 bucket
            rows = 120 + idx_len
            src2 = torch.randn(rows, 16, generator=gen, dtype=torch.bfloat16)
            ids2 = torch.randint(0, rows, (idx_len,), generator=gen, dtype=idx_dtype)
            got2 = src2.to(device).index_select(0, ids2.to(device)).cpu()
            assert torch.equal(got2, src2.index_select(0, ids2.long()))

            # 2D axis=1 with fixed rows
            src3 = torch.randn(16, 150, generator=gen, dtype=torch.float32)
            ids3 = torch.randint(0, 150, (idx_len,), generator=gen, dtype=idx_dtype)
            got3 = src3.to(device).index_select(1, ids3.to(device)).cpu()
            assert torch.equal(got3, src3.index_select(1, ids3.long()))

    run_cases(5)
    warm_stats = _sync_and_stats()
    for n in (1, 17, 65):
        run_cases(n)
    after_stats = _sync_and_stats()
    assert after_stats[1] == warm_stats[1] and after_stats[2] == warm_stats[2], (
        warm_stats, after_stats
    )

    # OOB checks: -1, upper, and (1 << 40) to catch int64 high-bit truncation.
    base = torch.arange(32, dtype=torch.bfloat16, device=device)
    for bad in (-1, 32, 1 << 40):
        idx_bad = torch.tensor([0, bad], dtype=torch.int64, device=device)
        try:
            _ = base.index_select(0, idx_bad)
            gaudirt.synchronize()
            raise AssertionError(f"expected RuntimeError for OOB index {bad}")
        except RuntimeError as exc:
            assert "out of bounds" in str(exc) or "index" in str(exc), str(exc)


def test_index_put_dynamic(device):
    slots = 128

    def run_put(count, idx_dtype, seed):
        gen = torch.Generator().manual_seed(seed)
        dst_cpu = torch.arange(slots, dtype=torch.int64) * 10
        idx_cpu = torch.randperm(slots, generator=gen)[:count].to(idx_dtype)
        val_cpu = torch.randint(-100000, 100000, (count,), generator=gen, dtype=torch.int64)
        dst_dev = dst_cpu.to(device)
        dst_dev.index_put_((idx_cpu.to(device),), val_cpu.to(device))
        dst_cpu.index_put_((idx_cpu.long(),), val_cpu)
        assert torch.equal(dst_dev.cpu(), dst_cpu)

    for dtype in (torch.int32, torch.int64):
        run_put(5, dtype, 400)
    warm_stats = _sync_and_stats()
    for n in (1, 17, 32):
        for dtype in (torch.int32, torch.int64):
            run_put(n, dtype, 500 + n)
    after_stats = _sync_and_stats()
    assert after_stats[1] == warm_stats[1] and after_stats[2] == warm_stats[2], (
        warm_stats, after_stats
    )

    # Duplicate indices: last-write-wins sequential semantics.
    dup_cpu = torch.zeros(16, dtype=torch.int64)
    dup_dev = dup_cpu.to(device)
    dup_idx = torch.tensor([3, 1, 3, 2, 1], dtype=torch.int64)
    dup_val = torch.tensor([100, 200, 300, 400, 500], dtype=torch.int64)
    dup_dev.index_put_((dup_idx.to(device),), dup_val.to(device))
    expected = torch.zeros(16, dtype=torch.int64)
    for k, v in zip(dup_idx.tolist(), dup_val.tolist()):
        expected[k] = v
    assert torch.equal(dup_dev.cpu(), expected)


def test_cat_dim0_dynamic(device):
    before = _sync_and_stats()
    gen = torch.Generator().manual_seed(600)
    for step in range(4):
        n1, n2 = 5 + step * 7, 3 + step * 11
        # 1D
        a1 = torch.randint(-20, 20, (n1,), generator=gen, dtype=torch.int64)
        b1 = torch.randint(-20, 20, (n2,), generator=gen, dtype=torch.int64)
        assert torch.equal(torch.cat([a1.to(device), b1.to(device)], dim=0).cpu(), torch.cat([a1, b1], dim=0))

        # 2D (including strided)
        a2 = torch.randn(n1 * 2, 24, generator=gen, dtype=torch.bfloat16)[::2]
        b2 = torch.randn(n2, 24, generator=gen, dtype=torch.bfloat16)
        a2_dev = torch.randn(n1 * 2, 24, generator=gen, dtype=torch.bfloat16)
        a2_cpu = a2_dev[::2]
        assert torch.equal(
            torch.cat([a2_dev.to(device)[::2], b2.to(device)], dim=0).cpu(),
            torch.cat([a2_cpu, b2], dim=0),
        )

        # 3D
        a3 = torch.randn(n1, 4, 8, generator=gen, dtype=torch.float32)
        b3 = torch.randn(n2, 4, 8, generator=gen, dtype=torch.float32)
        assert torch.equal(torch.cat([a3.to(device), b3.to(device)], dim=0).cpu(), torch.cat([a3, b3], dim=0))

    # P-RT-95 regression: strided D2D synMemCopyAsyncMultiple followed by contiguous synMemCopyAsync.
    for dt in (torch.bfloat16, torch.float32):
        cpu_a = torch.arange(42, dtype=torch.int32).reshape(6, 7).to(dt).t()
        cpu_b = torch.arange(42, 84, dtype=torch.int32).reshape(6, 7).to(dt).t()
        expected = torch.cat((cpu_a, cpu_b), dim=0)
        a_s = torch.empty((6, 7), dtype=dt, device=device)
        b_s = torch.empty((6, 7), dtype=dt, device=device)
        a_t, b_t = a_s.t(), b_s.t()
        a_t.copy_(cpu_a)
        b_t.copy_(cpu_b)
        out_cat = torch.empty(expected.shape, dtype=dt, device=device)
        tmp = torch.empty(expected.shape, dtype=dt, device=device)
        out_nb = torch.empty(expected.shape, dtype=dt, device=device)
        for _ in range(25):
            torch.cat((a_t, b_t), dim=0, out=out_cat)
            assert torch.equal(out_cat.cpu(), expected)
            tmp[:7].copy_(a_t, non_blocking=True)
            tmp[7:].copy_(b_t, non_blocking=True)
            out_nb.copy_(tmp, non_blocking=True)
            assert torch.equal(out_nb.cpu(), expected)

    after = _sync_and_stats()
    assert after == before, (before, after)


def test_advance_decode_lengths_dynamic(device):
    def run(bsz):
        seq = torch.arange(10, 10 + bsz, dtype=torch.int64)
        orig = torch.arange(5, 5 + bsz, dtype=torch.int32)
        next_s, next_o = torch.ops.gaudirt._advance_decode_lengths(seq.to(device), orig.to(device))
        assert torch.equal(next_s.cpu(), seq + 1)
        assert torch.equal(next_o.cpu(), orig + 1)

    run(5)
    warm_stats = _sync_and_stats()
    for bsz in (1, 17, 33):
        run(bsz)
    after_stats = _sync_and_stats()
    assert after_stats[1] == warm_stats[1] and after_stats[2] == warm_stats[2], (
        warm_stats, after_stats
    )


def test_pointwise_dynamic(device):
    # cast_device(f32 -> i64) invokes pointwise({mult, sub, add, or_fwd_u8}) plus cold_where.
    # Warm up length 5, then verify cold_pw_* keys (pw_v2 dynamic) do NOT increase at 17, 65, 130.
    def run_f32_to_i64(n, seed):
        gen = torch.Generator().manual_seed(seed)
        x_cpu = (torch.rand(n, generator=gen, dtype=torch.float32) - 0.5) * 1.0e5
        got = x_cpu.to(device).to(torch.int64).cpu()
        assert torch.equal(got, x_cpu.to(torch.int64))

    run_f32_to_i64(5, 700)
    gaudirt.synchronize()
    pw_keys_warm = _cold_keys("cold_pw_")
    assert pw_keys_warm, "expected pw_v2 recipes to be created"
    for n in (17, 65, 130):
        run_f32_to_i64(n, 700 + n)
    gaudirt.synchronize()
    pw_keys_after = _cold_keys("cold_pw_")
    assert pw_keys_after == pw_keys_warm, (pw_keys_warm, pw_keys_after)


def test_binary_ewise_dynamic(device):
    def _check_all(n, seed):
        gen = torch.Generator().manual_seed(seed)

        # int64: avoid minimum/maximum (routed via static cold_where)
        i64_a = torch.randint(-100, 100, (n,), generator=gen, dtype=torch.int64)
        i64_b = torch.randint(-100, 100, (n,), generator=gen, dtype=torch.int64)
        if n >= 5:
            # Exercise full 64-bit dynamic comparisons (zero low-32 bits on high-bit values).
            i64_a[0], i64_b[0] = 1 << 40, 1 << 40
            i64_a[1], i64_b[1] = 1 << 40, 0
            i64_a[2], i64_b[2] = -(1 << 40), -(1 << 40)
            i64_a[3], i64_b[3] = 0, 1 << 40
        i64_s = torch.tensor([1 << 40], dtype=torch.int64)
        for la, lb in ((i64_a, i64_b), (i64_a, i64_s), (i64_s, i64_b)):
            for fn in (
                lambda x, y: x + y,
                lambda x, y: x - y,
                lambda x, y: x * y,
                lambda x, y: x == y,
                lambda x, y: x != y,
                lambda x, y: x < y,
                lambda x, y: x <= y,
                lambda x, y: x > y,
                lambda x, y: x >= y,
            ):
                got = fn(la.to(device), lb.to(device)).cpu()
                assert torch.equal(got, fn(la, lb)), (n, fn)

        # int32
        i32_a = torch.randint(-100, 100, (n,), generator=gen, dtype=torch.int32)
        i32_b = torch.randint(-100, 100, (n,), generator=gen, dtype=torch.int32)
        i32_s = torch.tensor([11], dtype=torch.int32)
        for la, lb in ((i32_a, i32_b), (i32_a, i32_s), (i32_s, i32_b)):
            for fn in (
                lambda x, y: x + y,
                lambda x, y: x - y,
                lambda x, y: x * y,
                torch.minimum,
                torch.maximum,
                lambda x, y: x & y,
                lambda x, y: x | y,
                lambda x, y: x == y,
                lambda x, y: x != y,
                lambda x, y: x < y,
                lambda x, y: x <= y,
                lambda x, y: x > y,
                lambda x, y: x >= y,
            ):
                got = fn(la.to(device), lb.to(device)).cpu()
                assert torch.equal(got, fn(la, lb))

        # f32 and bf16
        for fdtype, rtol, atol in ((torch.float32, 1e-5, 1e-5), (torch.bfloat16, 2e-2, 2e-2)):
            fa = (torch.randn(n, generator=gen, dtype=torch.float32) * 2.0).to(fdtype)
            fb = (torch.randn(n, generator=gen, dtype=torch.float32) * 2.0).to(fdtype)
            fb = torch.where(fb.abs() < 0.25, torch.full_like(fb, 1.5), fb)
            fs = torch.tensor([1.5], dtype=fdtype)
            for la, lb in ((fa, fb), (fa, fs), (fs, fb)):
                for fn, is_cmp in (
                    (lambda x, y: x + y, False),
                    (lambda x, y: x - y, False),
                    (lambda x, y: x * y, False),
                    (lambda x, y: x / y, False),
                    (torch.minimum, False),
                    (torch.maximum, False),
                    (lambda x, y: x == y, True),
                    (lambda x, y: x != y, True),
                    (lambda x, y: x < y, True),
                    (lambda x, y: x <= y, True),
                    (lambda x, y: x > y, True),
                    (lambda x, y: x >= y, True),
                ):
                    got = fn(la.to(device), lb.to(device)).cpu()
                    ref = fn(la, lb)
                    if is_cmp:
                        assert torch.equal(got, ref)
                    else:
                        torch.testing.assert_close(got, ref, rtol=rtol, atol=atol)

        # bool
        ba = torch.randint(0, 2, (n,), generator=gen, dtype=torch.int32).bool()
        bb = torch.randint(0, 2, (n,), generator=gen, dtype=torch.int32).bool()
        bs = torch.tensor([True], dtype=torch.bool)
        for la, lb in ((ba, bb), (ba, bs), (bs, bb)):
            for fn in (
                torch.logical_and,
                torch.logical_or,
                lambda x, y: x & y,
                lambda x, y: x | y,
                lambda x, y: x == y,
                lambda x, y: x != y,
            ):
                got = fn(la.to(device), lb.to(device)).cpu()
                assert torch.equal(got, fn(la, lb))

        # Non-contiguous slice operands (staggered strides before flatten view).
        raw = torch.randn(n * 2, generator=gen, dtype=torch.float32)
        assert torch.allclose((raw.to(device)[::2] + 1.0).cpu(), raw[::2] + 1.0, atol=1e-5, rtol=1e-5)

    # Warm at length 5 and length 1 (count==1 covers scalar-output edge combinations).
    _check_all(5, 800)
    _check_all(1, 801)
    warm_stats = _sync_and_stats()
    warm_bin_keys = _cold_keys("cold_bin_")

    for n in (17, 65):
        _check_all(n, 810 + n)
    after_stats = _sync_and_stats()
    assert after_stats[1] == warm_stats[1] and after_stats[2] == warm_stats[2], (
        warm_stats, after_stats
    )
    assert _cold_keys("cold_bin_") == warm_bin_keys


def test_allocate_paged_slots_unified_bucket(device):
    def run_alloc(prefixes, lengths, num_pages):
        page_size = 16
        prev = [-1 for _ in prefixes]
        pages = list(range(100, 100 + num_pages))
        slots, used = gaudirt._C._allocate_paged_slots(
            torch.tensor(prev, dtype=torch.int64, device=device),
            torch.tensor(pages, dtype=torch.int64, device=device),
            prefixes,
            lengths,
            page_size,
        )
        expected, off = [], 0
        for p, l, last in zip(prefixes, lengths, prev):
            ex = min(l, (-p) % page_size)
            expected.extend(last + 1 + i for i in range(ex))
            rem = l - ex
            for i in range(rem):
                expected.append(pages[off + i // page_size] * page_size + i % page_size)
            off += (rem + page_size - 1) // page_size
        assert used == off and slots.cpu().tolist() == expected

    # All combinations satisfy max(reqs, total_tokens, num_pages) in (9..16] -> unified bucket=16.
    run_alloc([0, 3], [5, 7], 12)
    warm_stats = _sync_and_stats()
    run_alloc([0, 16, 4], [4, 5, 5], 18)
    run_alloc([1, 2, 3, 4, 5], [2, 3, 2, 2, 2], 30)
    after_stats = _sync_and_stats()
    assert after_stats[1] == warm_stats[1] and after_stats[2] == warm_stats[2], (
        warm_stats, after_stats
    )


def main():
    assert "habana_frameworks" not in sys.modules
    gaudirt.initialize(hbm_bytes=512 * 1024 * 1024, pinned_bytes=16 * 1024 * 1024)
    device = torch.device("gaudirt:0")
    test_cast_dynamic(device)
    test_index_select_dynamic(device)
    test_index_put_dynamic(device)
    test_cat_dim0_dynamic(device)
    test_advance_decode_lengths_dynamic(device)
    test_pointwise_dynamic(device)
    test_binary_ewise_dynamic(device)
    test_allocate_paged_slots_unified_bucket(device)
    print("DYNAMIC_EAGER_OPS_SMOKE_PASSED", "stats=", _sync_and_stats(), flush=True)


if __name__ == "__main__":
    main()
