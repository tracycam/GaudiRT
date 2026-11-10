"""Share dead segment outputs without overwriting live views or retained results."""

import gc
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def add_program():
    node = gaudirt._C.FxNode.binary(gaudirt._C.FxOp.add, 0, 1)
    return gaudirt._C.BucketProgram([1, 0], [node], [2])


def linear_runner(program, weight, count):
    runner = gaudirt._C.SegmentedRunner(2)
    current = 0
    for _ in range(count):
        current, = runner.add_program(program, [current, 1], [gaudirt._C.SegmentRoute(False, 0, [])])
    runner.set_outputs([current])
    runner.bind_static([1], [weight])
    return runner


def main():
    gaudirt.initialize(hbm_bytes=512 * 1024**2, pinned_bytes=8 * 1024**2)
    rows, width = 257, 512
    weight = torch.ones(width, dtype=torch.bfloat16).to("gaudirt")
    value = torch.full((rows, width), 3, dtype=torch.bfloat16).to("gaudirt")
    program = add_program()
    short = linear_runner(program, weight, 3)
    result = short.run_bound([value])[0]
    assert torch.equal(result.cpu(), torch.full((rows, width), 6, dtype=torch.bfloat16))
    del result
    gaudirt.synchronize()
    baseline = gaudirt.memory_allocated()

    long = linear_runner(program, weight, 32)
    result = long.run_bound([value])[0]
    assert torch.equal(result.cpu(), torch.full((rows, width), 35, dtype=torch.bfloat16))
    del result
    gaudirt.synchronize()
    retained = gaudirt.memory_allocated() - baseline
    bucket_bytes = 512 * width * 2
    assert retained <= 5 * bucket_bytes, (retained, bucket_bytes)
    result = long.run_bound([value])[0]
    assert torch.equal(result.cpu(), torch.full((rows, width), 35, dtype=torch.bfloat16))
    assert gaudirt.memory_allocated() - baseline <= 6 * bucket_bytes

    # Two routes share one computed output. The narrow route stays live after
    # the full route is consumed and must survive later same-shaped writes.
    half = torch.ones(width // 2, dtype=torch.bfloat16).to("gaudirt")
    half_program = add_program()
    alias_runner = gaudirt._C.SegmentedRunner(3)
    view = gaudirt._C.SegmentView(gaudirt._C.SegmentViewKind.feature, [0, width // 2])
    whole, narrow = alias_runner.add_program(
        program, [0, 1], [gaudirt._C.SegmentRoute(False, 0, []),
                          gaudirt._C.SegmentRoute(False, 0, [view])])
    mid, = alias_runner.add_program(program, [whole, 1], [gaudirt._C.SegmentRoute(False, 0, [])])
    final, = alias_runner.add_program(program, [mid, 1], [gaudirt._C.SegmentRoute(False, 0, [])])
    part, = alias_runner.add_program(half_program, [narrow, 2], [gaudirt._C.SegmentRoute(False, 0, [])])
    alias_runner.set_outputs([final, part])
    alias_runner.bind_static([1, 2], [weight, half])
    outputs, traces = alias_runner.run_bound_traced([value], [0])
    saved_view = traces[0][1][1]
    saved = saved_view.cpu().clone()
    assert torch.equal(outputs[0].cpu(), torch.full((rows, width), 6, dtype=torch.bfloat16))
    assert torch.equal(outputs[1].cpu(), torch.full((rows, width // 2), 5, dtype=torch.bfloat16))
    small = torch.full((1, width), 9, dtype=torch.bfloat16).to("gaudirt")
    small_outputs = alias_runner.run_bound([small])
    assert torch.equal(small_outputs[0].cpu(), torch.full((1, width), 12, dtype=torch.bfloat16))
    assert torch.equal(small_outputs[1].cpu(), torch.full((1, width // 2), 11, dtype=torch.bfloat16))
    assert torch.equal(saved_view.cpu(), saved)
    assert torch.equal(result.cpu(), torch.full((rows, width), 35, dtype=torch.bfloat16))

    del result, outputs, traces, saved_view, small_outputs, small, alias_runner
    del long, short, program, half_program, value, half, weight
    gc.collect()
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print(f"cross-segment allocation, views, bucket switch and retained outputs passed; retained={retained}", flush=True)


if __name__ == "__main__":
    main()
