"""Device verification for ISSUE-02 & ISSUE-03: generalized head_dim/kv_heads, SWA, Softcap, MIXED, allocator, 4k baseline timing."""

import math
import os
import statistics
import sys
import time

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def cpu_ref_attention(q, k_pool, v_pool, seq_metas, scale, softcap=0.0, sliding_window=-1):
    # seq_metas: list of (row_start, prefix, length, slot_list)
    heads, head_dim = q.shape[1], q.shape[2]
    kv_heads = k_pool.shape[1]
    group = heads // kv_heads
    head_map = torch.arange(heads) // group
    kf = k_pool.float()
    vf = v_pool.float()
    qf = q.float()
    out = torch.zeros_like(qf)
    for row_start, prefix, length, slots in seq_metas:
        loc = torch.tensor(slots[: prefix + length], dtype=torch.long)
        ks = kf[loc][:, head_map, :]  # [T, H, D]
        vs = vf[loc][:, head_map, :]
        for i in range(length):
            pos = prefix + i
            left = 0 if sliding_window <= 0 else max(0, pos - sliding_window + 1)
            scores = torch.einsum("hd,thd->ht", qf[row_start + i], ks[left : pos + 1]) * scale
            if softcap > 0.0:
                scores = softcap * torch.tanh(scores / softcap)
            probs = torch.softmax(scores, dim=-1)
            out[row_start + i] = torch.einsum("ht,thd->hd", probs, vs[left : pos + 1])
    return out.to(torch.bfloat16)


def make_kv_pool(total_slots, kv_heads, head_dim, k_cpu=None, v_cpu=None):
    combined = gaudirt._C._allocate_paged_kv(1, total_slots, kv_heads, head_dim)
    k_pool = combined[0, 0].transpose(0, 1)
    v_pool = combined[0, 1].transpose(0, 1)
    if k_cpu is not None:
        k_pool.copy_(k_cpu)
    if v_cpu is not None:
        v_pool.copy_(v_cpu)
    return k_pool, v_pool


def main():
    torch.set_num_threads(4)
    torch.manual_seed(7721)
    gaudirt.initialize(hbm_bytes=4096 * 1024**2, pinned_bytes=16 * 1024**2)
    dev = torch.device("gaudirt:0")

    # 1. Generalized head_dim (64, 128, 256), non-pow2 kv_heads (1, 3), MIXED batching, SWA, Softcap
    cases = [
        # (q_heads, kv_heads, head_dim, [(prefix, length), ...], softcap, sliding_window)
        (9, 3, 64, [(0, 48), (19, 1)], 0.0, -1),          # SmolLM2-like d=64, kv_heads=3, MIXED
        (16, 8, 128, [(0, 96), (32, 1), (45, 1)], 0.0, -1),
        (8, 4, 256, [(0, 140), (70, 1)], 50.0, 64),       # Gemma2-like d=256 + Softcap + SWA crossing window=64
        (8, 1, 256, [(0, 300), (150, 1)], 30.0, 96),      # MQA d=256 + SWA crossing window=96 + MIXED
    ]
    for q_heads, kv_heads, head_dim, seqs, softcap, swa in cases:
        scale = 1.0 / math.sqrt(head_dim)
        total_slots = 4096
        k_pool_cpu = torch.randn(total_slots, kv_heads, head_dim, dtype=torch.bfloat16)
        v_pool_cpu = torch.randn(total_slots, kv_heads, head_dim, dtype=torch.bfloat16)
        k_pool, v_pool = make_kv_pool(total_slots, kv_heads, head_dim, k_pool_cpu, v_pool_cpu)

        max_ctx = max(p + l for p, l in seqs)
        req_count = len(seqs)
        mapping = torch.zeros((req_count, max_ctx + 16), dtype=torch.int32)
        locations_list, seq_metas = [], []
        cursor = 1
        row_cursor = 0
        for req_idx, (prefix, length) in enumerate(seqs):
            slots = [cursor + i for i in range(prefix + length)]
            cursor += prefix + length + 3
            mapping[req_idx, : prefix + length] = torch.tensor(slots, dtype=torch.int32)
            locations_list.extend(slots[prefix : prefix + length])
            seq_metas.append((row_cursor, prefix, length, slots))
            row_cursor += length

        rows = row_cursor
        q_cpu = torch.randn(rows, q_heads * head_dim, dtype=torch.bfloat16)
        k_cpu = torch.randn(rows, kv_heads * head_dim, dtype=torch.bfloat16)
        v_cpu = torch.randn(rows, kv_heads * head_dim, dtype=torch.bfloat16)
        locations = torch.tensor(locations_list, dtype=torch.int64, device=dev)
        req_pool_indices = torch.arange(req_count, dtype=torch.int64, device=dev)
        positions = torch.tensor([p + i for p, l in seqs for i in range(l)], dtype=torch.int64, device=dev)
        prefixes = [p for p, _ in seqs]
        lengths = [l for _, l in seqs]

        meta = torch.classes.gaudirt.PagedMetadata._from_scheduler(
            locations, req_pool_indices, positions, mapping.to(dev), prefixes, lengths, total_slots, None, 0
        )
        out = torch.ops.gaudirt.paged_attn.default(
            q_cpu.to(dev), k_cpu.to(dev), v_cpu.to(dev), k_pool, v_pool,
            meta, head_dim, scale, softcap, swa
        )
        # Update CPU reference pools cleanly
        for r_idx, loc in enumerate(locations_list):
            k_pool_cpu[loc] = k_cpu[r_idx].view(kv_heads, head_dim)
            v_pool_cpu[loc] = v_cpu[r_idx].view(kv_heads, head_dim)
        ref = cpu_ref_attention(q_cpu.view(rows, q_heads, head_dim), k_pool_cpu, v_pool_cpu,
                                seq_metas, scale, softcap, swa).reshape(rows, q_heads * head_dim)
        rel_l2 = (torch.linalg.vector_norm(out.cpu().float() - ref.float(), dim=-1) /
                  torch.linalg.vector_norm(ref.float(), dim=-1).clamp_min(1e-6)).max().item()
        assert torch.isfinite(out.cpu()).all() and rel_l2 < 1.5e-2, (
            q_heads, kv_heads, head_dim, softcap, swa, rel_l2
        )
        print(f"case qh={q_heads} kvh={kv_heads} d={head_dim} softcap={softcap} swa={swa} "
              f"rows={rows} rel_l2={rel_l2:.5f}", flush=True)
        del k_pool, v_pool, out, meta, q_cpu, k_cpu, v_cpu, locations, req_pool_indices, positions

    # 2. Request count > 256 (300 requests, DECODE)
    req_count = 300
    mapping300 = torch.arange(1, 1 + req_count * 8, dtype=torch.int32).reshape(req_count, 8)
    k_pool300, v_pool300 = make_kv_pool(req_count * 8 + 16, 2, 64)
    locs300 = mapping300[:, 4].to(torch.int64).to(dev)
    reqs300 = torch.arange(req_count, dtype=torch.int64, device=dev)
    pos300 = torch.full((req_count,), 4, dtype=torch.int64)
    meta300 = torch.classes.gaudirt.PagedMetadata._from_scheduler_fused(
        locs300, reqs300, pos300, mapping300.to(dev), [4] * req_count, [1] * req_count,
        req_count * 8 + 16, None, 0, 4, 2, 64
    )
    q300 = torch.randn(req_count, 4 * 64, dtype=torch.bfloat16).to(dev)
    k300 = torch.randn(req_count, 2 * 64, dtype=torch.bfloat16).to(dev)
    v300 = torch.randn(req_count, 2 * 64, dtype=torch.bfloat16).to(dev)
    out300 = torch.ops.gaudirt.paged_attn.default(q300, k300, v300, k_pool300, v_pool300, meta300, 64, 1.0 / 8.0)
    assert out300.shape == (req_count, 256) and torch.isfinite(out300.cpu()).all()
    print("request_count=300 PASSED", flush=True)

    # 3. Generalized slot allocator: page_size in {1, 8, 16, 32, 64, 128} and >8192 pages (max_pages=16384)
    for page_size in (1, 8, 16, 32, 64, 128):
        prefixes, lengths = [0, 3, page_size], [5, 7, page_size + 2]
        prev = [-1, 100 * page_size + (2 % page_size), 200 * page_size + (page_size - 1)]
        pages = list(range(10, 80))
        expected, off = [], 0
        for p, l, last in zip(prefixes, lengths, prev):
            ex = min(l, (-p) % page_size)
            expected.extend(last + 1 + i for i in range(ex))
            rem = l - ex
            for i in range(rem):
                expected.append(pages[off + i // page_size] * page_size + i % page_size)
            off += (rem + page_size - 1) // page_size
        slots, used = gaudirt._C._allocate_paged_slots(
            torch.tensor(prev, dtype=torch.int64, device=dev),
            torch.tensor(pages, dtype=torch.int64, device=dev),
            prefixes, lengths, page_size,
        )
        assert used == off and slots.cpu().tolist() == expected, (page_size, used, off)
        rel_in = [9000 * page_size, 9000 * page_size + (page_size - 1), 12000 * page_size]
        uniq = gaudirt._C._release_paged_slots(torch.tensor(rel_in, dtype=torch.int64, device=dev), page_size, 16384)
        assert uniq.cpu().tolist() == [9000, 12000], (page_size, uniq.cpu().tolist())
    print("allocator page_sizes {1,8,16,32,64,128} & max_pages=16384 PASSED", flush=True)

    # 4. Isolated plain-causal M=4096 (56/8, d=128) non-regression check
    q_h, kv_h, d = 56, 8, 128
    m_rows = 4096
    slots4k = 8192
    map4k = torch.arange(1, 1 + m_rows, dtype=torch.int32).reshape(1, m_rows).to(dev)
    k_p4k, v_p4k = make_kv_pool(slots4k, kv_h, d)
    q4k = torch.randn(m_rows, q_h * d, dtype=torch.bfloat16).to(dev)
    k4k = torch.randn(m_rows, kv_h * d, dtype=torch.bfloat16).to(dev)
    v4k = torch.randn(m_rows, kv_h * d, dtype=torch.bfloat16).to(dev)
    locs4k = torch.arange(1, 1 + m_rows, dtype=torch.int64, device=dev)
    reqs4k = torch.zeros(1, dtype=torch.int64, device=dev)
    pos4k = torch.arange(m_rows, dtype=torch.int64, device=dev)
    meta4k = torch.classes.gaudirt.PagedMetadata._from_scheduler(
        locs4k, reqs4k, pos4k, map4k, [0], [m_rows], slots4k, None, 0
    )
    # Warmup
    _ = torch.ops.gaudirt.paged_attn.default(q4k, k4k, v4k, k_p4k, v_p4k, meta4k, d, 1.0 / math.sqrt(d))
    gaudirt.synchronize()
    timings = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = torch.ops.gaudirt.paged_attn.default(q4k, k4k, v4k, k_p4k, v_p4k, meta4k, d, 1.0 / math.sqrt(d))
        gaudirt.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)
    print(f"plain_causal M=4096 56/8 d=128 warm median_ms={statistics.median(timings):.2f} "
          f"samples={[round(x, 2) for x in timings]}", flush=True)


if __name__ == "__main__":
    main()
