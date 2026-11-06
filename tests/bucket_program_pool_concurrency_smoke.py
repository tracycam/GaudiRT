"""Graph staging remains valid while another thread releases idle cache pools."""

from concurrent.futures import ThreadPoolExecutor
import os
import sys
import threading

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    program = gaudirt._C.BucketProgram([1, 1], [gaudirt._C.FxNode.binary(gaudirt._C.FxOp.add, 0, 1)], [2])
    streams = [gaudirt.Stream(), gaudirt.Stream()]
    host = [torch.full((3, 64), i + 1, dtype=torch.bfloat16) for i in range(2)]
    inputs = [(x.to("gaudirt:0"), (x + 2).to("gaudirt:0")) for x in host]
    gaudirt.synchronize()
    gate = threading.Barrier(3)

    def run(index):
        with gaudirt.stream(streams[index]):
            gate.wait(timeout=30)
            for _ in range(64):
                result = program.run(list(inputs[index]))[0]
                assert torch.equal(result.cpu(), host[index] * 2 + 2)

    def release_idle_pools():
        gate.wait(timeout=30)
        for _ in range(64):
            gaudirt.empty_cache()

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run, i) for i in range(2)]
        futures.append(executor.submit(release_idle_pools))
        for future in futures:
            future.result(timeout=120)
    gaudirt.synchronize()
    del inputs, streams, program
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("two streams and repeated concurrent empty_cache preserved staged inputs", flush=True)


if __name__ == "__main__":
    main()
