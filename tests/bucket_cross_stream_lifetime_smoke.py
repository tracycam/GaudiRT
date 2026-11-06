"""Compiled input storage remains live across streams, including graph-pool views."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import gaudirt.compiler as compiler


def project(x, weight):
    return torch.mm(x, weight)


def main():
    torch.random.default_generator.manual_seed(53)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    producer, consumer = gaudirt.Stream(), gaudirt.Stream()
    x_cpu = (torch.randint(-8, 9, (32, 128)).float() / 16).to(torch.bfloat16)
    weight_cpu = (torch.randint(-8, 9, (128, 96)).float() / 16).to(torch.bfloat16)
    expected = (x_cpu.float() @ weight_cpu.float()).to(torch.bfloat16)
    weight = weight_cpu.to("gaudirt:0")
    compiled = compiler.compile(project)
    warm = compiled(x_cpu.to("gaudirt:0"), weight)
    assert torch.equal(warm.cpu(), expected)
    del warm
    gaudirt.synchronize()

    reused = 0
    for graph_pool in (False, True):
        for _ in range(16):
            with gaudirt.stream(producer):
                if graph_pool:
                    host_base = torch.empty((32, 256), dtype=torch.bfloat16)
                    host_base[:, ::2] = x_cpu
                    host_base[:, 1::2] = 11
                    base = host_base.to("gaudirt:0")
                    x = gaudirt._C._prepare_strided_input(base[:, ::2])
                else:
                    x = x_cpu.to("gaudirt:0")
                address = x.data_ptr()
                ready = gaudirt.Event()
                ready.record(producer)
            with gaudirt.stream(consumer):
                consumer.wait_event(ready)
                output = compiled(x, weight)
            del x
            with gaudirt.stream(producer):
                if graph_pool:
                    scratch = gaudirt._C._prepare_strided_input(base[:, 1::2])
                else:
                    scratch = torch.empty((32, 128), dtype=torch.bfloat16, device="gaudirt:0")
                    scratch.fill_(11)
                reused += scratch.data_ptr() == address
            assert torch.equal(output.cpu(), expected)
            del output, scratch, ready
            if graph_pool:
                del base
    assert reused > 0, "test did not exercise reuse of input storage"
    gaudirt.synchronize()
    del weight
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("compiled cross-stream allocation lifetime passed, reused", reused, flush=True)


if __name__ == "__main__":
    main()
