"""TP=4 gather_reduce / dynamic rank_sum device verification (P-RT-93 regression)."""

import os

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import socket


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _canonical_rank_sum(tensors):
    acc = tensors[0].float()
    for t in tensors[1:]:
        acc = acc + t.float()
    return acc.to(tensors[0].dtype)


def _run_checks(rank: int, world_size: int, port: int, gaudirt, torch, dist, funcol, gaudirt_compile) -> None:
    pg = dist.group.WORLD
    dev = torch.device(f"gaudirt:{rank}")

    # 1. Eager all_reduce on non-power-of-two exact < bucket elements for BF16 and FP32.
    for dtype in (torch.bfloat16, torch.float32):
        for exact in (128, 384, 1536, 24576, 49152):
            rank_inputs = []
            for r in range(world_size):
                g = torch.Generator(device="cpu").manual_seed(7000 + exact + r * 31 + (0 if dtype == torch.bfloat16 else 999))
                rank_inputs.append(torch.randn((exact,), generator=g, dtype=dtype))
            ref = _canonical_rank_sum(rank_inputs)

            x = rank_inputs[rank].to(dev)
            dist.all_reduce(x, group=pg)
            x_cpu = x.cpu()
            assert torch.equal(x_cpu.view(torch.uint8), ref.view(torch.uint8)), (
                f"eager rank_sum mismatch at dtype={dtype}, exact={exact}"
            )

            gathered = torch.empty((world_size * exact,), dtype=dtype, device=dev)
            dist.all_gather_into_tensor(gathered, x, group=pg)
            gar_cpu = gathered.cpu().view(world_size, exact)
            for r in range(1, world_size):
                assert torch.equal(gar_cpu[0].view(torch.uint8), gar_cpu[r].view(torch.uint8)), (
                    f"cross-rank bitwise drift at dtype={dtype}, exact={exact}, rank={r}"
                )

        # 2. Adjacent sentinel check: ProcessGroup slice [0:1536] inside [0:2048] parent buffer.
        # Static bucket_elements=2048 would overwrite tail [1536:2048]; dynamic rank_sum must preserve it.
        sentinel_val = -7.5
        parent_inputs = []
        for r in range(world_size):
            g = torch.Generator(device="cpu").manual_seed(9100 + r * 17)
            parent_inputs.append(torch.randn((1536,), generator=g, dtype=dtype))
        ref_slice = _canonical_rank_sum(parent_inputs)

        parent = torch.full((2048,), sentinel_val, dtype=dtype, device=dev)
        parent[:1536].copy_(parent_inputs[rank].to(dev))
        view = parent[:1536]
        dist.all_reduce(view, group=pg)
        parent_cpu = parent.cpu()
        assert torch.equal(parent_cpu[:1536].view(torch.uint8), ref_slice.view(torch.uint8)), (
            f"sentinel slice value mismatch at dtype={dtype}"
        )
        assert torch.all(parent_cpu[1536:] == sentinel_val), (
            f"OOB write detected past exact=1536 at dtype={dtype}"
        )

    # 3. Compiled FX functional collective (SegmentedRunner path) with dynamic M in {3, 48, 49}, W=512.
    for dtype in (torch.bfloat16, torch.float32):
        scale_t = torch.full((512,), 1.5, dtype=dtype, device=dev)
        bias_t = torch.full((512,), 0.5, dtype=dtype, device=dev)

        def _fx_segment(x, s, b):
            y = x * s
            z = funcol.wait_tensor(funcol.all_reduce(y, "sum", pg))
            return z - b

        compiled = gaudirt_compile(_fx_segment, dynamic_arg_indices=(0,))
        for m in (3, 48, 49):
            rank_inputs = []
            for r in range(world_size):
                g = torch.Generator(device="cpu").manual_seed(12000 + m * 101 + r * 23)
                rank_inputs.append(torch.randn((m, 512), generator=g, dtype=dtype))
            scaled = [(t * 1.5).to(dtype) for t in rank_inputs]
            ref_fx = (_canonical_rank_sum(scaled) - 0.5).to(dtype)

            out = compiled(rank_inputs[rank].to(dev), scale_t, bias_t)
            out_cpu = out.cpu()
            assert torch.equal(out_cpu.view(torch.uint8), ref_fx.view(torch.uint8)), (
                f"compiled FX rank_sum mismatch at dtype={dtype}, m={m}"
            )

            gar = torch.empty((world_size * m, 512), dtype=dtype, device=dev)
            dist.all_gather_into_tensor(gar, out, group=pg)
            gar_cpu = gar.cpu().view(world_size, m, 512)
            for r in range(1, world_size):
                assert torch.equal(gar_cpu[0].view(torch.uint8), gar_cpu[r].view(torch.uint8)), (
                    f"cross-rank FX drift at dtype={dtype}, m={m}, rank={r}"
                )

    gaudirt.synchronize()


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import gaudirt
    import torch
    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol
    from gaudirt.compiler import compile as gaudirt_compile

    assert gaudirt.device_count() >= world_size, (
        f"expected >= {world_size} visible devices, got {gaudirt.device_count()}"
    )
    gaudirt.configure_default_runtime(hbm_bytes=4 * 1024**3)
    gaudirt.set_device(rank)

    dist.init_process_group(
        backend="gaudirt",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    err = None
    try:
        _run_checks(rank, world_size, port, gaudirt, torch, dist, funcol, gaudirt_compile)
    except BaseException as exc:
        err = exc
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    if err is not None:
        raise err
    gaudirt.shutdown()

    if rank == 0:
        print("tp4_gather_reduce_smoke PASSED (bf16/f32 eager, adjacent sentinel, FX dynamic M=3/48/49)", flush=True)


def main() -> None:
    import torch.multiprocessing as mp

    port = _free_port()
    mp.spawn(_worker, args=(4, port), nprocs=4, join=True)


if __name__ == "__main__":
    main()
