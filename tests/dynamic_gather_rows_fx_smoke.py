#!/usr/bin/env python3
"""Verify dynamic 2-extent gather_rows FX partitioning, argmax bucketing, and Fail-Fast contracts."""

import os
import sys

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F
import gaudirt
import gaudirt.compiler as compiler
from fp64_reference import direct_bf16


torch.set_num_threads(4)
torch.random.default_generator.manual_seed(5809)


def _reference_bf16(x_cpu, idx_cpu, w1_cpu, b1_cpu, w_lm_cpu):
    h = direct_bf16(x_cpu.double() @ w1_cpu.double().t() + b1_cpu.double())
    h = direct_bf16(F.silu(h.double()))
    g = h[idx_cpu.long()]
    return direct_bf16(g.double() @ w_lm_cpu.double().t()).float()


def main():
    assert "habana_frameworks" not in sys.modules
    gaudirt.initialize(hbm_bytes=512 * 1024 * 1024, pinned_bytes=8 * 1024 * 1024)
    device = torch.device("gaudirt:0")
    hidden = 64
    vocab = 4096

    w1_cpu = (torch.randn(hidden, hidden, dtype=torch.bfloat16) * 0.2).contiguous()
    b1_cpu = (torch.randn(hidden, dtype=torch.bfloat16) * 0.05).contiguous()
    w_lm_cpu = (torch.randn(vocab, hidden, dtype=torch.bfloat16) * 0.1).contiguous()
    w1_dev = w1_cpu.to(device)
    b1_dev = b1_cpu.to(device)
    w_lm_dev = w_lm_cpu.to(device)

    captures = []
    real_backend = compiler.gaudirt_backend

    def counting_backend(graph, inputs):
        captures.append(graph)
        return real_backend(graph, inputs)

    compiler.gaudirt_backend = counting_backend

    try:
        def forward_xy(x, idx, w1, b1, w_lm):
            h = F.silu(F.linear(x, w1, b1))
            gathered = h[idx]
            return F.linear(gathered, w_lm)

        def forward_yx(idx, x, w1, b1, w_lm):
            h = F.silu(F.linear(x, w1, b1))
            gathered = h.index_select(0, idx)
            return F.linear(gathered, w_lm)

        compiled_xy = compiler.compile(forward_xy, dynamic_arg_indices=(0, 1))
        compiled_yx = compiler.compile(forward_yx, dynamic_arg_indices=(0, 1))

        m_schedule = [1, 17, 64, 192]

        def make_inputs(m_rows, b_rows, idx_dtype, seed):
            gen = torch.Generator().manual_seed(seed)
            x_cpu = (torch.randn((m_rows, hidden), generator=gen, dtype=torch.float32) * 0.4).to(torch.bfloat16)
            idx_vals = torch.randperm(m_rows, generator=gen)[:b_rows]
            if b_rows > m_rows:
                idx_vals = torch.randint(0, m_rows, (b_rows,), generator=gen)
            idx_cpu = idx_vals.to(idx_dtype)
            x_dev = x_cpu.to(device)
            idx_dev = idx_cpu.to(device)
            idx_dev._gaudirt_validated_rows = m_rows
            return x_cpu, idx_cpu, x_dev, idx_dev

        # Warm up power-of-2 buckets for both M and B, and compile dynamic recipes once.
        for m_warm in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            for b_warm in (1, 2, 4, 8, 16):
                _, _, x_dev, idx_dev = make_inputs(m_warm, b_warm, torch.int32, 100 + m_warm * 31 + b_warm)
                out = compiled_xy(x_dev, idx_dev, w1_dev, b1_dev, w_lm_dev)
                _ = torch.argmax(out, dim=-1)
        gaudirt.synchronize()

        counts_after_warm = gaudirt.recipe_cache_stats()
        gaudirt.synchronize()

        # Sweep all B = 16..1 (16 distinct batch sizes > dynamo recompile_limit=8) across varying M.
        for step, b_rows in enumerate(range(16, 0, -1)):
            m_rows = m_schedule[step % len(m_schedule)]
            x_cpu, idx_cpu, x_dev, idx_dev = make_inputs(m_rows, b_rows, torch.int32, 1000 + step)
            out_dev = compiled_xy(x_dev, idx_dev, w1_dev, b1_dev, w_lm_dev)
            argmax_dev = torch.argmax(out_dev, dim=-1)
            gaudirt.synchronize()

            out_actual = out_dev.cpu().float()
            out_ref = _reference_bf16(x_cpu, idx_cpu, w1_cpu, b1_cpu, w_lm_cpu)
            torch.testing.assert_close(out_actual, out_ref, rtol=3e-2, atol=4e-2)
            assert torch.equal(argmax_dev.cpu(), torch.argmax(out_actual, dim=-1)), (
                b_rows, argmax_dev.cpu(), torch.argmax(out_actual, dim=-1)
            )

        assert len(captures) == 1, f"expected 1 FX capture for forward_xy, got {len(captures)}"
        counts_after_xy = gaudirt.recipe_cache_stats()
        assert counts_after_xy[1] == counts_after_warm[1] and counts_after_xy[2] == counts_after_warm[2], (
            f"unexpected steady-state compilation during forward_xy sweep: {counts_after_warm} -> {counts_after_xy}"
        )

        # Now verify reversed outer arg order (idx, x), index_select, and int64 validated indices.
        captures.clear()
        for m_warm in (1, 17, 64, 192):
            for b_warm in (1, 2, 4, 8, 16):
                _, _, x_dev, idx_dev = make_inputs(m_warm, b_warm, torch.int64, 2000 + m_warm * 17 + b_warm)
                out = compiled_yx(idx_dev, x_dev, w1_dev, b1_dev, w_lm_dev)
                _ = torch.argmax(out, dim=-1)
        gaudirt.synchronize()
        counts_after_yx_warm = gaudirt.recipe_cache_stats()

        for step, b_rows in enumerate(range(16, 0, -1)):
            m_rows = m_schedule[(step + 1) % len(m_schedule)]
            x_cpu, idx_cpu, x_dev, idx_dev = make_inputs(m_rows, b_rows, torch.int64, 3000 + step)
            out_dev = compiled_yx(idx_dev, x_dev, w1_dev, b1_dev, w_lm_dev)
            argmax_dev = torch.argmax(out_dev, dim=-1)
            gaudirt.synchronize()

            out_actual = out_dev.cpu().float()
            out_ref = _reference_bf16(x_cpu, idx_cpu, w1_cpu, b1_cpu, w_lm_cpu)
            torch.testing.assert_close(out_actual, out_ref, rtol=3e-2, atol=4e-2)
            assert torch.equal(argmax_dev.cpu(), torch.argmax(out_actual, dim=-1))

        assert len(captures) == 1, f"expected 1 FX capture for forward_yx, got {len(captures)}"
        counts_after_yx = gaudirt.recipe_cache_stats()
        assert counts_after_yx[1] == counts_after_yx_warm[1] and counts_after_yx[2] == counts_after_yx_warm[2], (
            f"unexpected steady-state compilation during forward_yx sweep: {counts_after_yx_warm} -> {counts_after_yx}"
        )

        # Verify tight unpadded eager logits (requiring tail staging inside argmax) across B=1..16 (bf16 & fp32).
        for dtype in (torch.bfloat16, torch.float32):
            for b_warm in (1, 2, 4, 8, 16):
                t = torch.randn((b_warm, vocab), dtype=dtype).to(device)
                _ = torch.argmax(t, dim=-1)
            gaudirt.synchronize()
            before_eager = gaudirt.recipe_cache_stats()
            for b_rows in range(16, 0, -1):
                t_cpu = torch.randn((b_rows, vocab), dtype=dtype)
                t_dev = t_cpu.to(device)
                got = torch.argmax(t_dev, dim=-1)
                gaudirt.synchronize()
                assert torch.equal(got.cpu(), torch.argmax(t_cpu.float(), dim=-1))
            after_eager = gaudirt.recipe_cache_stats()
            assert after_eager[1] == before_eager[1] and after_eager[2] == before_eager[2], (
                dtype, before_eager, after_eager
            )

        # Verify Fail-Fast rejection for unvalidated and mismatched row indices.
        _, _, x_dev, idx_bad = make_inputs(17, 5, torch.int32, 4001)
        delattr(idx_bad, "_gaudirt_validated_rows")
        for args in ((x_dev, idx_bad),):
            try:
                compiled_xy(args[0], args[1], w1_dev, b1_dev, w_lm_dev)
                raise AssertionError("expected unvalidated indices to fail")
            except RuntimeError as exc:
                assert "validation" in str(exc), str(exc)

        idx_bad._gaudirt_validated_rows = 18  # != 17
        try:
            compiled_xy(x_dev, idx_bad, w1_dev, b1_dev, w_lm_dev)
            raise AssertionError("expected mismatched row validation to fail")
        except RuntimeError as exc:
            assert "validation" in str(exc), str(exc)

        # Verify Fail-Fast rejection if one BucketProgram mixes two distinct dynamic dim0 extents.
        def bad_mixed_segment(x, y):
            return torch.cat([x, y], dim=0)

        compiled_bad = compiler.compile(bad_mixed_segment, dynamic_arg_indices=(0, 1))
        x_bad = torch.randn((7, 8), dtype=torch.bfloat16).to(device)
        y_bad = torch.randn((3, 8), dtype=torch.bfloat16).to(device)
        try:
            compiled_bad(x_bad, y_bad)
            raise AssertionError("expected mixed-extent BucketProgram to fail")
        except (RuntimeError, torch._dynamo.exc.BackendCompilerFailed) as exc:
            assert "one dynamic outer extent" in str(exc), str(exc)

        print("DYNAMIC_GATHER_ROWS_FX_SMOKE_PASSED", "stats=", gaudirt.recipe_cache_stats(), flush=True)
    finally:
        compiler.gaudirt_backend = real_backend


if __name__ == "__main__":
    main()
