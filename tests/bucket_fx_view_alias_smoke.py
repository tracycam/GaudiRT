"""FX feature slices and splits preserve PyTorch storage aliasing."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F

import gaudirt
import gaudirt.compiler as compiler


def input_views(x):
    left, right = x.split([4, 13], dim=-1)
    return x[:, 1:4], left, right, right[:, 1:3]


def projection_views(x, weight):
    value = F.linear(x, weight)
    return value[:, 1:5], value


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(1911)
    gaudirt.initialize(hbm_bytes=256 * 1024**2, pinned_bytes=4 * 1024**2)
    compiled_input = compiler.compile(input_views)
    compiled_projection = compiler.compile(projection_views)
    weight = torch.randn(17, 17, dtype=torch.bfloat16).to("gaudirt:0")
    for rows in (1, 3, 17, 137):
        cpu = torch.randn(rows, 17, dtype=torch.bfloat16)
        if rows == 3:
            cpu_backing = torch.randn(17, rows, dtype=torch.bfloat16)
            cpu = cpu_backing.t()
            backing = cpu_backing.to("gaudirt:0")
            device = backing.t()
        elif rows == 17:
            cpu_backing = torch.empty((rows, 34), dtype=torch.bfloat16)
            cpu_backing[:, ::2].copy_(cpu)
            cpu = cpu_backing[:, ::2]
            backing = torch.empty((rows, 34), dtype=torch.bfloat16, device="gaudirt:0")
            device = backing[:, ::2]
            device.copy_(cpu)
        else:
            device = cpu.to("gaudirt:0")
        outputs = compiled_input(device)
        expected = input_views(cpu)
        assert all(a.untyped_storage().data_ptr() == device.untyped_storage().data_ptr() for a in outputs)
        assert [a.stride() for a in outputs] == [a.stride() for a in expected]
        assert all(torch.equal(a.cpu(), b) for a, b in zip(outputs, expected))
        outputs[0][0, 0].copy_(torch.tensor(7, dtype=torch.bfloat16))
        expected[0][0, 0] = 7
        assert torch.equal(device.cpu(), cpu), "slice mutation did not reach its input"
        assert torch.equal(outputs[1].cpu(), expected[1])

        part, whole = compiled_projection(device, weight)
        assert part.untyped_storage().data_ptr() == whole.untyped_storage().data_ptr()
        assert part.data_ptr() == whole.data_ptr() + whole.element_size()
        assert part.stride() == whole.stride()
        expected_whole = whole.cpu()
        part[0, 0].copy_(torch.tensor(3, dtype=torch.bfloat16))
        expected_whole[0, 1] = 3
        assert torch.equal(whole.cpu(), expected_whole), "slice mutation did not reach its producer"
    print("FX feature output views preserve input and computed storage aliases", flush=True)


if __name__ == "__main__":
    main()
