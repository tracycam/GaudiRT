"""One compiled program can initialize and run from separate Python streams."""

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
    program = gaudirt._C.BucketProgram([1, 1], [[1, 0, 1, -1]], [2])
    streams = [gaudirt.Stream() for _ in range(4)]
    inputs = []
    for worker, rows in enumerate((3, 17, 3, 17)):
        first = torch.full((rows, 64), worker + 1, dtype=torch.bfloat16)
        second = torch.full((rows, 64), 2 * worker + 3, dtype=torch.bfloat16)
        inputs.append((first.to("gaudirt:0"), second.to("gaudirt:0"), first + second))
    gaudirt.synchronize()
    gate = threading.Barrier(len(streams))

    def run(worker):
        first, second, expected = inputs[worker]
        with gaudirt.stream(streams[worker]):
            gate.wait(timeout=30)
            for _ in range(8):
                result = program.run([first, second])[0]
                assert torch.equal(result.cpu(), expected)
        return worker

    with ThreadPoolExecutor(max_workers=len(streams)) as executor:
        assert sorted(executor.map(run, range(len(streams)))) == list(range(len(streams)))
    stats = gaudirt.recipe_cache_stats()
    assert stats[1] == 2, stats
    del inputs, streams, program
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("shared program concurrent initialization and four-stream execution passed", stats, flush=True)


if __name__ == "__main__":
    main()
