"""Capture once and keep Dynamo configuration patches out of later calls."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F

import gaudirt
import gaudirt.compiler as compiler


def projection(value, weight):
    return F.linear(value, weight)


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(248)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=4 * 1024**2)
    captures = []
    backend = compiler.gaudirt_backend

    def capture(graph, inputs):
        captures.append(1)
        return backend(graph, inputs)

    compiler.gaudirt_backend = capture
    compiled = compiler.compile(projection)
    weight_cpu = torch.randn(16, 16)
    weight = weight_cpu.to("gaudirt:0")
    configs = (torch._dynamo.config, torch.fx.experimental._config)
    patch_code = type(configs[0]).patch.__code__
    calls = [0, 0]

    def profile(frame, event, arg):
        if event == "call" and frame.f_code is patch_code:
            caller = frame.f_back
            if caller and caller.f_globals.get("__name__") == compiler.__name__ and caller.f_code.co_name == "execute":
                for index, config in enumerate(configs):
                    if frame.f_locals.get("self") is config:
                        calls[index] += 1

    previous_profile = sys.getprofile()
    sys.setprofile(profile)
    try:
        for rows in (1, 3, 17, 137, 1):
            value = torch.randn(rows, 16)
            actual = compiled(value.to("gaudirt:0"), weight).cpu()
            expected = projection(value, weight_cpu)
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    finally:
        sys.setprofile(previous_profile)
    assert len(captures) == 1, captures
    assert calls == [1, 1], calls
    print("FX captures", len(captures), "capture configuration patch calls", calls, flush=True)


if __name__ == "__main__":
    main()
