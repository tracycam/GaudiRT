"""Check the linear output boundary before a connected residual addition."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import gaudirt.compiler as compiler


def biased(x, weight, bias, residual):
    return torch.nn.functional.linear(x, weight, bias) + residual


def without_bias(x, weight, residual):
    return torch.nn.functional.linear(x, weight) + residual


def main():
    torch.random.default_generator.manual_seed(877)
    gaudirt.initialize(hbm_bytes=128 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
    for dtype in (torch.bfloat16, torch.float32):
        for has_bias in (False, True):
            fn = compiler.compile(biased if has_bias else without_bias,
                                  dynamic_arg_indices=(0, 3 if has_bias else 2))
            weight = torch.randn(64, 64, dtype=dtype).to("gaudirt:0")
            bias = torch.randn(64, dtype=dtype).to("gaudirt:0") if has_bias else None
            for m in (1, 3, 17, 137):
                x_cpu = torch.randn(m, 64, dtype=dtype)
                r_cpu = torch.randn(m, 64, dtype=dtype)
                x, residual = x_cpu.to("gaudirt:0"), r_cpu.to("gaudirt:0")
                actual = (fn(x, weight, bias, residual) if has_bias else fn(x, weight, residual)).cpu()
                linear = gaudirt._C.bucket_linear(x, weight, bias).cpu()
                expected = (linear.float() + r_cpu.float()).to(dtype)
                assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                if dtype == torch.bfloat16:
                    bad = torch.count_nonzero(actual.view(torch.int16) != expected.view(torch.int16)).item()
                    assert bad == 0, (dtype, has_bias, m, bad)
                else:
                    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
            print("dtype", dtype, "bias", has_bias, "boundary passed", flush=True)
    del x, residual, weight, bias, fn
    gaudirt.empty_cache()
    gaudirt.shutdown()
    print("connected linear residual boundary passed", flush=True)


if __name__ == "__main__":
    main()
