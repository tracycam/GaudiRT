"""Aligned repeated-word fills use stream-ordered memset without compiling a graph."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt


def bytes_of(tensor):
    return tensor.contiguous().view(torch.uint8)


def main():
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    values = ((torch.bool, True), (torch.bfloat16, -7.5), (torch.float32, -0.0),
              (torch.int32, 0x12345678), (torch.int64, -1))
    for dtype, value in values:
        tensor = torch.empty(1024, dtype=dtype, device="gaudirt:0")
        compiled = gaudirt.recipe_cache_stats()[1]
        tensor.fill_(value)
        assert gaudirt.recipe_cache_stats()[1] == compiled, (dtype, gaudirt.recipe_cache_stats())
        assert torch.equal(bytes_of(tensor.cpu()), bytes_of(torch.full((1024,), value, dtype=dtype))), dtype

    guard = torch.full((640,), -17, dtype=torch.int32, device="gaudirt:0")
    compiled = gaudirt.recipe_cache_stats()[1]
    guard[64:576].fill_(41)
    assert gaudirt.recipe_cache_stats()[1] == compiled
    expected = torch.full((640,), -17, dtype=torch.int32)
    expected[64:576] = 41
    assert torch.equal(guard.cpu(), expected)

    # A 64-bit pattern with unlike halves and a strided view still require the
    # general fill implementation, including preservation of untouched bytes.
    large = 0x123456789ABCDEF
    tensor = torch.empty(256, dtype=torch.int64, device="gaudirt:0")
    tensor.fill_(large)
    assert torch.equal(tensor.cpu(), torch.full((256,), large, dtype=torch.int64))
    base = torch.full((512,), -7, dtype=torch.int32, device="gaudirt:0")
    base[1::2].fill_(9)
    expected = torch.full((512,), -7, dtype=torch.int32)
    expected[1::2] = 9
    assert torch.equal(base.cpu(), expected)

    # A different consumer stream must see the complete memset.
    stream = gaudirt.Stream()
    tensor = torch.empty(1024, dtype=torch.int32, device="gaudirt:0")
    with gaudirt.stream(stream):
        tensor.fill_(37)
        ready = gaudirt.Event()
        ready.record(stream)
    gaudirt.current_stream().wait_event(ready)
    assert torch.equal(tensor.cpu(), torch.full((1024,), 37, dtype=torch.int32))
    del tensor, base, guard, stream, ready
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("aligned fill: exact bytes, no recipe; strided/64-bit and cross-stream cases passed", flush=True)


if __name__ == "__main__":
    main()
