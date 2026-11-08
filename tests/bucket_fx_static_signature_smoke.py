"""Stable weights reuse their capture signatures without holding storage alive."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F
import gaudirt
import gaudirt.compiler as compiler


def projection(x, weight):
    return F.linear(x, weight)


def main():
    torch.set_num_threads(4)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=8 * 1024**2)
    captures = []
    backend = compiler.gaudirt_backend

    def capture(graph, inputs):
        captures.append(1)
        return backend(graph, inputs)

    compiler.gaudirt_backend = capture
    fn = compiler.compile(projection)
    rng = torch.Generator().manual_seed(2187)
    host = [torch.randn(64, 64, generator=rng), torch.randn(64, 64, generator=rng)]
    weights = [value.to("gaudirt:0") for value in host]
    calls = [0, 0]

    def profile(frame, event, argument):
        if (event == "call" and frame.f_code.co_name == "signature" and
                frame.f_globals.get("__name__") == compiler.__name__):
            value = frame.f_locals.get("value")
            for index, weight in enumerate(weights):
                if value is weight:
                    calls[index] += 1

    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        for rows, variant in ((1, 0), (3, 0), (17, 0), (3, 1), (3, 0), (3, 1)):
            x_cpu = torch.randn(rows, 64, generator=rng)
            actual = fn(x_cpu.to("gaudirt:0"), weights[variant]).cpu()
            assert torch.allclose(actual, F.linear(x_cpu, host[variant]), atol=2e-4, rtol=2e-4)
        assert calls == [1, 1], calls
        host[0].mul_(0.5)
        weights[0].copy_(host[0])
        x_cpu = torch.randn(3, 64, generator=rng)
        actual = fn(x_cpu.to("gaudirt:0"), weights[0]).cpu()
        assert torch.allclose(actual, F.linear(x_cpu, host[0]), atol=2e-4, rtol=2e-4)
        assert calls == [2, 1], calls
        with torch.inference_mode():
            host.append(torch.randn(64, 64, generator=rng))
            weights.append(host[-1].to("gaudirt:0"))
            calls.append(0)
            for rows in (3, 17, 1):
                x_cpu = torch.randn(rows, 64, generator=rng)
                actual = fn(x_cpu.to("gaudirt:0"), weights[2]).cpu()
                assert torch.allclose(actual, F.linear(x_cpu, host[2]), atol=2e-4, rtol=2e-4)
            host[2].mul_(0.5)
            weights[2].copy_(host[2])
            x_cpu = torch.randn(3, 64, generator=rng)
            actual = fn(x_cpu.to("gaudirt:0"), weights[2]).cpu()
            assert torch.allclose(actual, F.linear(x_cpu, host[2]), atol=2e-4, rtol=2e-4)
            address = weights[2].data_ptr()
            weights[2].set_(weights[2].untyped_storage(), 0, (32, 64), (64, 1))
            host[2] = host[2][:32]
            assert weights[2].data_ptr() == address
            actual = fn(x_cpu.to("gaudirt:0"), weights[2]).cpu()
            assert torch.allclose(actual, F.linear(x_cpu, host[2]), atol=2e-4, rtol=2e-4)
        assert calls == [2, 1, 1], calls
    finally:
        sys.setprofile(previous)
    assert len(captures) == 3, captures
    print("static tensor signature evaluations", calls, "FX captures", len(captures), flush=True)


if __name__ == "__main__":
    main()
