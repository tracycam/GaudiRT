"""Two-device HCCL collective, sharded embedding, and MME/NIC overlap verification."""

import argparse
import json
import os
import socket
import statistics
import subprocess
import sys
import time

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _rel_l2(actual, expected) -> float:
    diff = (actual.double() - expected.double()).norm().item()
    denom = expected.double().norm().item()
    return 0.0 if denom == 0.0 and diff == 0.0 else diff / max(denom, 1e-12)


def _worker(rank: int, world_size: int, port: int, result_file: str) -> None:
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import gaudirt
    import torch
    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol
    from gaudirt.compiler import compile as gaudirt_compile

    assert gaudirt.device_count() >= world_size, (
        f"expected >= {world_size} visible devices, got {gaudirt.device_count()}"
    )
    assert not gaudirt.is_initialized(), "device was acquired eagerly before set_device"
    gaudirt.configure_default_runtime(hbm_bytes=8 * 1024**3)
    gaudirt.set_device(rank)
    assert not gaudirt.is_initialized(), "set_device must remain lazy"

    dist.init_process_group(
        backend="gaudirt",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    pg = dist.group.WORLD
    dev = torch.device(f"gaudirt:{rank}")

    # 1. Eager collectives across non-power-of-two and bucketed token counts.
    for m in (1, 3, 17, 513, 1024, 4096):
        g0 = torch.Generator(device="cpu").manual_seed(1000 + m)
        g1 = torch.Generator(device="cpu").manual_seed(2000 + m)
        cpu_r0 = torch.randn((m, 256), generator=g0, dtype=torch.bfloat16)
        cpu_r1 = torch.randn((m, 256), generator=g1, dtype=torch.bfloat16)
        local_cpu = cpu_r0 if rank == 0 else cpu_r1

        x = local_cpu.to(dev)
        dist.all_reduce(x, group=pg)
        ref_ar = (cpu_r0.float() + cpu_r1.float()).to(torch.bfloat16)
        x_cpu = x.cpu()
        assert _rel_l2(x_cpu, ref_ar) < 1e-3, f"all_reduce mismatch at M={m}"

        # Cross-rank bit-exactness check via all_gather.
        gathered_ar = torch.empty((world_size * m, 256), dtype=torch.bfloat16, device=dev)
        dist.all_gather_into_tensor(gathered_ar, x, group=pg)
        gar_cpu = gathered_ar.cpu()
        assert torch.equal(gar_cpu[:m], gar_cpu[m:]), f"cross-rank all_reduce drift at M={m}"

        # ReduceScatter (even M) & AllToAll (even M).
        x2 = torch.cat([cpu_r0, cpu_r1], dim=0).to(dev) if rank == 0 else torch.cat([cpu_r1, cpu_r0], dim=0).to(dev)
        rs_out = torch.empty((m, 256), dtype=torch.bfloat16, device=dev)
        dist.reduce_scatter_tensor(rs_out, x2, group=pg)
        ref_rs = (cpu_r0.float() + cpu_r1.float()).to(torch.bfloat16)
        assert _rel_l2(rs_out.cpu(), ref_rs) < 1e-3, f"reduce_scatter mismatch at M={m}"

        a2a_in = torch.full((2 * m, 64), float(rank + 1), dtype=torch.bfloat16, device=dev)
        a2a_out = torch.empty_like(a2a_in)
        dist.all_to_all_single(a2a_out, a2a_in, group=pg)
        a2a_cpu = a2a_out.cpu()
        assert torch.all(a2a_cpu[:m] == 1.0) and torch.all(a2a_cpu[m:] == 2.0), f"all_to_all mismatch at M={m}"

    # Variable-split eager all_to_all + broadcast + send/recv + barrier.
    in_splits = [3, 5] if rank == 0 else [7, 2]
    out_splits = [3, 7] if rank == 0 else [5, 2]
    v_in = torch.full((sum(in_splits), 16), float(rank + 10), dtype=torch.bfloat16, device=dev)
    v_out = torch.empty((sum(out_splits), 16), dtype=torch.bfloat16, device=dev)
    dist.all_to_all_single(v_out, v_in, output_split_sizes=out_splits, input_split_sizes=in_splits, group=pg)
    v_cpu = v_out.cpu()
    assert torch.all(v_cpu[:out_splits[0]] == 10.0) and torch.all(v_cpu[out_splits[0]:] == 11.0)

    bcast = torch.full((8, 32), 42.0 if rank == 0 else -1.0, dtype=torch.bfloat16, device=dev)
    dist.broadcast(bcast, src=0, group=pg)
    assert torch.all(bcast.cpu() == 42.0)

    if rank == 0:
        dist.send(torch.full((4, 16), 7.5, dtype=torch.bfloat16, device=dev), dst=1, group=pg)
    else:
        p2p_buf = torch.zeros((4, 16), dtype=torch.bfloat16, device=dev)
        dist.recv(p2p_buf, src=0, group=pg)
        assert torch.all(p2p_buf.cpu() == 7.5)
    dist.barrier(group=pg)

    # 2. Compiled VocabParallelEmbedding (padded base + added tokens) + AllReduce + AllGatherDimLast.
    # Base vocab = 1010 -> per-rank 505 padded to 512 (pad=7); added vocab = 16 -> per-rank 8 padded to 64.
    g_emb = torch.Generator(device="cpu").manual_seed(777)
    full_base = torch.randn((1010, 128), generator=g_emb, dtype=torch.bfloat16)
    full_added = torch.randn((16, 128), generator=g_emb, dtype=torch.bfloat16)
    full_vocab = torch.cat([full_base, full_added], dim=0)  # 1026 total tokens

    shard_weight_cpu = torch.zeros((576, 128), dtype=torch.bfloat16)
    org_start, org_end = rank * 505, (rank + 1) * 505
    add_start, add_end = 1010 + rank * 8, 1010 + (rank + 1) * 8
    num_org_pad = 512 - 505  # 7
    shard_weight_cpu[:505] = full_base[org_start:org_end]
    shard_weight_cpu[512:520] = full_added[rank * 8:(rank + 1) * 8]
    shard_weight = shard_weight_cpu.to(dev)

    g_head = torch.Generator(device="cpu").manual_seed(888)
    full_head = torch.randn((512, 128), generator=g_head, dtype=torch.bfloat16)
    head_shard = full_head[rank * 256:(rank + 1) * 256].contiguous().to(dev)

    def sharded_embed_and_head(tokens, last_idx, emb_w, lm_w):
        org_mask = (tokens >= org_start) & (tokens < org_end)
        add_mask = (tokens >= add_start) & (tokens < add_end)
        add_off = add_start - (org_end - org_start) - num_org_pad
        valid_off = (org_start * org_mask) + (add_off * add_mask)
        vocab_mask = org_mask | add_mask
        masked = vocab_mask * (tokens - valid_off)
        emb = torch.nn.functional.embedding(masked.long(), emb_w)
        emb.masked_fill_((~vocab_mask).unsqueeze(-1), 0)
        hidden = funcol.wait_tensor(funcol.all_reduce(emb, "sum", pg))
        pruned = hidden[last_idx]
        local_logits = torch.nn.functional.linear(pruned, lm_w)
        gathered = funcol.wait_tensor(funcol.all_gather_tensor(local_logits, 0, pg))
        logits = gathered.reshape((world_size, local_logits.shape[0], local_logits.shape[1]))
        logits = logits.movedim(0, 1).reshape((local_logits.shape[0], world_size * local_logits.shape[1]))
        return hidden, logits

    state_holder = {}
    compiled_sharded = gaudirt_compile(
        sharded_embed_and_head,
        dynamic_arg_indices=(0,),
        prepare_capture=lambda: torch._dynamo.mark_static(state_holder["last_idx"], 0),
    )

    for b_sz in (1, 4):
        for idx_step, m in enumerate((3, 17, 513)):
            gen_t = torch.Generator(device="cpu").manual_seed(5000 + b_sz * 100 + m)
            tok_cpu = torch.randint(0, 1026, (m,), generator=gen_t, dtype=torch.int32)
            # Ensure boundary tokens from both shards and added range are exercised.
            tok_cpu[0] = 0
            tok_cpu[min(1, m - 1)] = 505
            tok_cpu[min(2, m - 1)] = 1020
            last_cpu = torch.linspace(0, m - 1, steps=b_sz, dtype=torch.int64).to(torch.int32)
            tok_dev = tok_cpu.to(dev)
            tok_dev._gaudirt_validated_vocab = 1026
            last_dev = last_cpu.to(dev)
            last_dev._gaudirt_validated_rows = m
            state_holder["last_idx"] = last_dev

            stats_before = gaudirt.recipe_cache_stats()
            h_dev, lg_dev = compiled_sharded(tok_dev, last_dev, shard_weight, head_shard)
            stats_after = gaudirt.recipe_cache_stats()
            if idx_step == 2:
                # m=17 -> bucket 32, m=513 -> bucket 1024; re-run m=17 to assert 0 warm recompiles.
                s0 = gaudirt.recipe_cache_stats()
                h_warm, lg_warm = compiled_sharded(tok_dev, last_dev, shard_weight, head_shard)
                s1 = gaudirt.recipe_cache_stats()
                assert s1[1] == s0[1], f"unexpected warm recompile: {s0} -> {s1}"
                assert torch.equal(h_dev.cpu(), h_warm.cpu()) and torch.equal(lg_dev.cpu(), lg_warm.cpu())
            del stats_before, stats_after

            ref_h = full_vocab[tok_cpu.long()]
            ref_lg = torch.nn.functional.linear(ref_h[last_cpu.long()].float(), full_head.float()).to(torch.bfloat16)
            assert torch.equal(h_dev.cpu(), ref_h), f"sharded embedding mismatch at B={b_sz}, M={m}"
            assert _rel_l2(lg_dev.cpu(), ref_lg) < 2e-3, f"AllGatherDimLast logits mismatch at B={b_sz}, M={m}"

    # 3. Convex DAG Partitioner + ASAP Start / ALAP Wait + Hardware Overlap benchmark.
    def fn_ab_c(x, w1, w2):
        a = torch.nn.functional.linear(x, w1)
        b = funcol.wait_tensor(funcol.all_reduce(a, "sum", pg))
        c = torch.nn.functional.linear(x, w2)
        return b[:, :4096] + c

    def fn_c_ab(x, w1, w2):
        c = torch.nn.functional.linear(x, w2)
        a = torch.nn.functional.linear(x, w1)
        b = funcol.wait_tensor(funcol.all_reduce(a, "sum", pg))
        return b[:, :4096] + c

    def fn_serial(x, w1, w2):
        a = torch.nn.functional.linear(x, w1)
        b = funcol.wait_tensor(funcol.all_reduce(a, "sum", pg))
        return torch.nn.functional.linear(b[:, :4096], w2)

    comp_ab_c = gaudirt_compile(fn_ab_c, dynamic_arg_indices=(0,))
    comp_c_ab = gaudirt_compile(fn_c_ab, dynamic_arg_indices=(0,))
    comp_serial = gaudirt_compile(fn_serial, dynamic_arg_indices=(0,))

    g_ov = torch.Generator(device="cpu").manual_seed(900 + rank)
    x_ov = torch.randn((4096, 4096), generator=g_ov, dtype=torch.bfloat16).to(dev)
    w1_ov = (torch.randn((16384, 4096), generator=g_ov, dtype=torch.bfloat16) * 0.02).to(dev)
    w2_ov = (torch.randn((4096, 4096), generator=g_ov, dtype=torch.bfloat16) * 0.02).to(dev)

    out_1 = comp_ab_c(x_ov, w1_ov, w2_ov)
    out_2 = comp_c_ab(x_ov, w1_ov, w2_ov)
    _ = comp_serial(x_ov, w1_ov, w2_ov)
    gaudirt.synchronize()

    sched_1 = comp_ab_c._gaudirt_last_execution.runner.schedule_summary()
    sched_2 = comp_c_ab._gaudirt_last_execution.runner.schedule_summary()
    sched_ser = comp_serial._gaudirt_last_execution.runner.schedule_summary()
    expected_overlap = [
        ("program", 0, False),
        ("collective_start", 0, True),
        ("program", 0, False),
        ("collective_wait", 0, True),
        ("program", 0, False),
    ]
    expected_serial = [
        ("program", 0, False),
        ("collective_start", 0, False),
        ("collective_wait", 0, False),
        ("program", 0, False),
    ]
    assert sched_1 == expected_overlap, f"unexpected schedule for fn_ab_c: {sched_1}"
    assert sched_2 == expected_overlap, f"unexpected schedule for fn_c_ab: {sched_2}"
    assert sched_ser == expected_serial, f"unexpected schedule for fn_serial: {sched_ser}"
    assert torch.equal(out_1.cpu(), out_2.cpu()), "lexical order changed overlapped output"

    # Measure isolated Comm (128 MiB AllReduce), isolated Compute (4096x4096x4096 GEMM),
    # vs overlapped (a_tiny + 128 MiB AllReduce overlapped with heavy w2 GEMM).
    w_heavy = (torch.randn((16384, 4096), generator=g_ov, dtype=torch.bfloat16) * 0.02).to(dev)
    comm_buf = torch.randn((4096, 16384), generator=g_ov, dtype=torch.bfloat16).to(dev)

    def only_compute(x, w):
        return torch.nn.functional.linear(x, w)

    def only_comm(buf):
        return funcol.wait_tensor(funcol.all_reduce(buf, "sum", pg))

    def overlapped_pair(x, buf, w):
        b = funcol.wait_tensor(funcol.all_reduce(buf, "sum", pg))
        c = torch.nn.functional.linear(x, w)
        return b + c

    c_only_compute = gaudirt_compile(only_compute, dynamic_arg_indices=(0,))
    c_only_comm = gaudirt_compile(only_comm, dynamic_arg_indices=(0,))
    c_overlapped = gaudirt_compile(overlapped_pair, dynamic_arg_indices=(0, 1))

    for _ in range(5):
        out_tmp = c_only_compute(x_ov, w_heavy)
        del out_tmp
        out_tmp = c_only_comm(comm_buf)
        del out_tmp
        out_tmp = c_overlapped(x_ov, comm_buf, w_heavy)
        del out_tmp
    gaudirt.synchronize()
    steady_mem_before = gaudirt.memory_allocated()

    ov_sched = c_overlapped._gaudirt_last_execution.runner.schedule_summary()
    assert ov_sched == [
        ("collective_start", 0, True),
        ("program", 0, False),
        ("collective_wait", 0, True),
        ("program", 0, False),
    ], f"unexpected overlapped_pair schedule: {ov_sched}"

    def bench(fn):
        samples = []
        for _ in range(20):
            dist.barrier(group=pg)
            gaudirt.synchronize()
            t0 = time.perf_counter()
            out_tmp = fn()
            gaudirt.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
            del out_tmp
        return statistics.median(samples)

    compute_ms = bench(lambda: c_only_compute(x_ov, w_heavy))
    comm_ms = bench(lambda: c_only_comm(comm_buf))
    overlapped_ms = bench(lambda: c_overlapped(x_ov, comm_buf, w_heavy))
    steady_mem_after = gaudirt.memory_allocated()
    assert steady_mem_after == steady_mem_before, (
        f"steady-state HBM leak detected: before={steady_mem_before}, after={steady_mem_after}"
    )

    assert overlapped_ms < compute_ms + comm_ms, (
        f"expected hardware overlap: overlapped={overlapped_ms:.3f}ms >= "
        f"compute={compute_ms:.3f}ms + comm={comm_ms:.3f}ms"
    )

    if rank == 0 and result_file:
        with open(result_file, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "world_size": world_size,
                    "visible_modules": [gaudirt.visible_module_for_device(i) for i in range(world_size)],
                    "schedule_ab_c": sched_1,
                    "schedule_c_ab": sched_2,
                    "schedule_serial": sched_ser,
                    "compute_only_median_ms": round(compute_ms, 4),
                    "comm_only_median_ms": round(comm_ms, 4),
                    "serial_sum_ms": round(compute_ms + comm_ms, 4),
                    "overlapped_median_ms": round(overlapped_ms, 4),
                    "saved_ms": round((compute_ms + comm_ms) - overlapped_ms, 4),
                },
                handle,
                indent=2,
            )

    dist.destroy_process_group()


def _abort_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import gaudirt
    import torch
    import torch.distributed as dist

    gaudirt.configure_default_runtime(hbm_bytes=512 * 1024**2)
    gaudirt.set_device(rank)
    dist.init_process_group(
        backend="gaudirt",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    t = torch.ones((16, 16), dtype=torch.bfloat16, device=f"gaudirt:{rank}")
    dist.all_reduce(t)
    gaudirt.synchronize()
    if rank == 1:
        raise RuntimeError("intentional rank-1 failure to verify hcclCommAbort path")


def _run_abort_subtest() -> float:
    port = _free_port()
    code = (
        "import sys; "
        "from tp_comm_overlap_smoke import _abort_worker; "
        "_abort_worker(int(sys.argv[1]), 2, int(sys.argv[2]))"
    )
    env = os.environ.copy()
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    env["PYTHONPATH"] = tests_dir + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    t0 = time.perf_counter()
    p0 = subprocess.Popen([sys.executable, "-c", code, "0", str(port)], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p1 = subprocess.Popen([sys.executable, "-c", code, "1", str(port)], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err1 = p1.communicate(timeout=20)
    assert p1.returncode != 0 and b"intentional rank-1 failure" in err1
    # Rank 0 stayed idle after sync and exits normally once work is done.
    _, err0 = p0.communicate(timeout=20)
    assert p0.returncode == 0, f"rank 0 failed during abort subtest: {err0.decode(errors='replace')}"
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    import torch.multiprocessing as mp

    port = _free_port()
    result_file = args.json_out or f"/tmp/gaudirt_tp_comm_{os.getpid()}.json"
    mp.spawn(_worker, args=(2, port, result_file), nprocs=2, join=True)

    abort_elapsed_s = _run_abort_subtest()
    with open(result_file, encoding="utf-8") as handle:
        summary = json.load(handle)
    summary["abort_subtest_elapsed_s"] = round(abort_elapsed_s, 4)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
    else:
        os.remove(result_file)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
