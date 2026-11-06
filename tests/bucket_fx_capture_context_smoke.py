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
    weights_cpu = [torch.randn(16, width) for width in (16, 20)]
    weights = [value.to("gaudirt:0") for value in weights_cpu]
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
        for rows, variant in ((1, 0), (3, 0), (1, 1), (17, 1), (137, 0)):
            value = torch.randn(rows, weights_cpu[variant].size(1))
            actual = compiled(value.to("gaudirt:0"), weights[variant]).cpu()
            expected = projection(value, weights_cpu[variant])
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    finally:
        sys.setprofile(previous_profile)
    assert len(captures) == 2, captures
    assert calls == [2, 2], calls
    print("FX captures", len(captures), "capture configuration patch calls", calls, flush=True)


if __name__ == "__main__":
    main()
