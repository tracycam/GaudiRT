"""Strided graph input staging survives aliasing and cross-stream reuse."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch

import gaudirt
import gaudirt.compiler as compiler


def identity(x):
    return x


def pair(x):
    return x, x


def main():
    gaudirt.initialize(hbm_bytes=128 * 1024**2, pinned_bytes=4 * 1024**2)
    cpu = torch.arange(3 * 34, dtype=torch.float32).reshape(3, 34)
    base = cpu.to("gaudirt:0")
    view = base[:, ::2]
    assert gaudirt._C._prepare_strided_input(base) is None
    singleton = base[::4]
    assert singleton.is_contiguous() and singleton.stride(0) != base.stride(0)
    canonical = gaudirt._C._prepare_strided_input(singleton)
    assert canonical is not None and canonical.data_ptr() == singleton.data_ptr()
    assert canonical.stride() == (34, 1) and torch.equal(canonical.cpu(), cpu[::4])
    del singleton, canonical
    ids_cpu = torch.tensor([[4, 9], [5, 9], [6, 9]], dtype=torch.int32)
    ids = ids_cpu.to("gaudirt:0")
    staged_ids = gaudirt._C._prepare_strided_input(ids[:, 0])
    assert torch.equal(staged_ids.as_strided((4,), (1,), 0).cpu(),
                       torch.tensor([4, 5, 6, 0], dtype=torch.int32))
    del staged_ids, ids
    gaudirt.synchronize()
    first, second = gaudirt.Stream(), gaudirt.Stream()
    with gaudirt.stream(first):
        staged = gaudirt._C._prepare_strided_input(view)
        assert staged is not None and staged.is_contiguous()
        assert staged.shape == (3, 17) and staged.untyped_storage().nbytes() >= 4 * 17 * 4
        event = gaudirt.Event()
        event.record(first)
    with gaudirt.stream(second):
        second.wait_event(event)
        output = torch.empty_like(staged)
        output.copy_(staged, non_blocking=True)
        staged.record_stream(second)
        del staged
    with gaudirt.stream(first):
        for _ in range(8):
            temporary = gaudirt._C._prepare_strided_input(view)
            del temporary
    second.synchronize()
    assert torch.equal(output.cpu(), cpu[:, ::2])
    first.synchronize()

    result = compiler.compile(identity)(view)
    assert result.data_ptr() == view.data_ptr() and result.stride() == view.stride()
    assert torch.equal(result.cpu(), cpu[:, ::2])
    aliases = compiler.compile(pair)(view)
    assert all(value.data_ptr() == view.data_ptr() and value.stride() == view.stride() for value in aliases)
    staged = gaudirt._C._prepare_strided_input(view)
    reshaped = staged.as_strided((51,), (1,), 0)
    restored = gaudirt._C._restore_staged_alias(view, staged, reshaped)
    assert restored.data_ptr() == view.data_ptr() and restored.stride() == (2,)
    assert torch.equal(restored.cpu(), cpu[:, ::2].reshape(-1))
    transposed = base.t()
    staged_transposed = gaudirt._C._prepare_strided_input(transposed)
    restored_transposed = gaudirt._C._restore_staged_alias(transposed, staged_transposed, staged_transposed.t())
    assert restored_transposed.data_ptr() == base.data_ptr() and restored_transposed.stride() == base.stride()
    assert torch.equal(restored_transposed.cpu(), cpu)
    try:
        gaudirt._C._restore_staged_alias(transposed, staged_transposed, staged_transposed.view(-1))
    except RuntimeError as error:
        assert "cannot preserve the original layout" in str(error)
    else:
        raise AssertionError("accepted an unrepresentable staged alias")
    del aliases, staged, reshaped, restored, transposed, staged_transposed, restored_transposed
    live = gaudirt._C._prepare_strided_input(view)
    try:
        gaudirt.shutdown()
    except RuntimeError as error:
        assert "graph tensors must be destroyed" in str(error)
    else:
        raise AssertionError("shutdown accepted a live graph-pool tensor")
    del live, result, view, base, output, event
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("graph input staging, alias lifetime and cross-stream reuse passed", flush=True)


if __name__ == "__main__":
    main()
