"""Reused intermediate buffers stay stream-local and do not own weights."""

from concurrent.futures import ThreadPoolExecutor
import gc
import os
import sys
import threading
import weakref

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    node = gaudirt._C.FxNode.binary(gaudirt._C.FxOp.add, 0, 1)
    program = gaudirt._C.BucketProgram([1, 1], [node], [2])
    runner = gaudirt._C.SegmentedRunner(2)
    route = [gaudirt._C.SegmentRoute(False, 0, [])]
    intermediate = runner.add_program(program, [0, 1], route)
    result = runner.add_program(program, [intermediate[0], 1], route)
    runner.set_outputs(result)
    weight = torch.ones((3, 64), dtype=torch.bfloat16).to("gaudirt")
    runner.bind_static([1], [weight])
    weight_ref = weakref.ref(weight)
    streams = [gaudirt.Stream(), gaudirt.Stream()]
    inputs = [torch.full((3, 64), value, dtype=torch.bfloat16).to("gaudirt") for value in (2, 5)]
    gaudirt.synchronize()
    barrier = threading.Barrier(2)

    def run(index):
        with gaudirt.stream(streams[index]):
            barrier.wait(timeout=30)
            for _ in range(32):
                output = runner.run_bound([inputs[index]])[0]
                assert torch.equal(output.cpu(), torch.full((3, 64), (2, 5)[index] + 2,
                                                            dtype=torch.bfloat16))

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run, index) for index in range(2)]
        for future in futures:
            future.result(timeout=120)
    gaudirt.synchronize()
    del inputs, streams, weight
    gc.collect()
    assert weight_ref() is None, "runner must not extend the weight's lifetime"
    gaudirt.empty_cache()
    gaudirt.shutdown()  # The runner and its cached plan may outlive the context.
    print("two streams, reusable outputs, weak weights and shutdown passed", flush=True)


if __name__ == "__main__":
    main()
