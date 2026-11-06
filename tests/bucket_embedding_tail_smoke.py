"""Embedding isolates unused bucket rows without altering or copying aligned inputs."""

import ctypes
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import threading

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    driver = ctypes.CDLL(None)
    counter = driver.gaudirt_test_d2d_copy_count
    created = driver.gaudirt_test_event_create_count
    destroyed = driver.gaudirt_test_event_destroy_count
    counter.restype = created.restype = destroyed.restype = ctypes.c_uint64
    gaudirt.initialize(hbm_bytes=128 * 1024**2, pinned_bytes=4 * 1024**2)
    table_cpu = torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16).to(torch.bfloat16)
    table = table_cpu.to("gaudirt:0")
    ids_cpu = torch.tensor([1, 2, 3, 5, 6, 7, 8, 9], dtype=torch.int32)
    full_ids = ids_cpu.to("gaudirt:0")
    program = gaudirt._C.BucketProgram(
        [1, 0], [gaudirt._C.FxNode.binary(gaudirt._C.FxOp.embedding, 0, 1)], [2]
    )

    def check(rows):
        ids = full_ids[:rows]
        bucket = 1 << (rows - 1).bit_length()
        result = program.run([ids, table])[0]
        padded = result.as_strided((bucket, 16), (16, 1)).cpu()
        assert torch.equal(padded[:rows], table_cpu[ids_cpu[:rows].long()])
        assert torch.equal(padded[rows:], table_cpu[0].expand(bucket - rows, -1))
        return result

    streams = [gaudirt.Stream() for _ in range(4)]
    gate = threading.Barrier(len(streams))

    def concurrent(index):
        with gaudirt.stream(streams[index]):
            gate.wait(timeout=30)
            for _ in range(8):
                value = check((3, 4, 5, 7)[index])
                del value

    with ThreadPoolExecutor(max_workers=len(streams)) as executor:
        list(executor.map(concurrent, range(len(streams))))
    gaudirt.synchronize()
    first = check(3)
    counts = gaudirt.recipe_cache_stats()
    second = check(4)
    assert gaudirt.recipe_cache_stats() == counts, "same bucket recompiled"
    third = check(5)
    first_row = check(1)
    large_cpu = torch.arange(4096, dtype=torch.int32) % 16
    large_ids = large_cpu.to("gaudirt:0")
    almost_full = program.run([large_ids[:4095], table])[0]
    physical = almost_full.as_strided((4096, 16), (16, 1)).cpu()
    assert torch.equal(physical[:4095], table_cpu[large_cpu[:4095].long()])
    assert torch.equal(physical[4095], table_cpu[0])
    large_counts = gaudirt.recipe_cache_stats()
    full_result = program.run([large_ids, table])[0]
    assert gaudirt.recipe_cache_stats() == large_counts, "maximum bucket recompiled"
    assert torch.equal(full_result.cpu(), table_cpu[large_cpu.long()])
    assert torch.equal(large_ids.cpu(), large_cpu)
    assert torch.equal(full_ids.cpu(), ids_cpu), "embedding modified the input storage"
    gaudirt.synchronize()
    before, before_created, before_destroyed = counter(), created(), destroyed()
    repeated = program.run([full_ids[:3], table])[0]
    gaudirt.synchronize()
    assert counter() == before, "aligned embedding input was staged with D2D"
    assert (created(), destroyed()) == (before_created, before_destroyed), "events changed during warm launch"
    assert torch.equal(repeated.cpu(), table_cpu[ids_cpu[:3].long()])
    del first, second, third, first_row, almost_full, full_result, large_ids
    del repeated, full_ids, table, program, streams
    gaudirt.empty_cache()
    gaudirt.shutdown()
    gaudirt.initialize(hbm_bytes=8 * 1024**2, pinned_bytes=1024**2)
    assert gaudirt.memory_allocated() == 0
    gaudirt.shutdown()
    print("embedding tail isolated across four streams, same-bucket reuse, aligned input D2D copies 0, warm event changes 0, clean reinit", flush=True)


if __name__ == "__main__":
    main()
