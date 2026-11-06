"""Two warmed FX segments can reach the driver concurrently on separate streams."""

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
    probe = ctypes.CDLL(None)
    probe.gaudirt_test_launch_gate_passed.restype = ctypes.c_bool
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    program = gaudirt._C.BucketProgram([1, 1], [gaudirt._C.FxNode.binary(gaudirt._C.FxOp.add, 0, 1)], [2])
    streams = [gaudirt.Stream(), gaudirt.Stream()]
    host = [torch.full((4, 64), i + 1, dtype=torch.bfloat16) for i in range(2)]
    inputs = [(x.to("gaudirt:0"), (x + 2).to("gaudirt:0")) for x in host]
    warm = program.run(list(inputs[0]))[0]
    gaudirt.synchronize()
    del warm

    barrier = threading.Barrier(2)

    def run(index):
        with gaudirt.stream(streams[index]):
            barrier.wait(timeout=30)
            return program.run(list(inputs[index]))[0]

    probe.gaudirt_test_launch_gate_arm()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, range(2)))
    gaudirt.synchronize()
    assert probe.gaudirt_test_launch_gate_passed(), "driver launches were serialized"
    for index, result in enumerate(results):
        assert torch.equal(result.cpu(), host[index] * 2 + 2)
    assert gaudirt.recipe_cache_stats()[2] == 1
    del result, results, inputs, streams, program
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("two warmed FX segments reached synLaunch concurrently and produced correct output", flush=True)


if __name__ == "__main__":
    main()
