"""Keep visible aliases while compacting an internal, unshared feature slice."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.fx
import torch.nn.functional as F
import gaudirt.compiler as compiler


def compact(x, weight):
    q, k, v = F.linear(x, weight).split([8, 4, 4], dim=-1)
    return q * 2, k * 2, v


def with_source(x, weight):
    qkv = F.linear(x, weight)
    q, k, v = qkv.split([8, 4, 4], dim=-1)
    return q * 2, k * 2, v, qkv


def shared_slices(x, weight):
    q, k, v = F.linear(x, weight).split([8, 4, 4], dim=-1)
    return q, k, v


class CaptureValues(torch.fx.Interpreter):
    def run_node(self, node):
        result = super().run_node(node)
        node.meta["example_value"] = result
        return result


def routes(function, indices):
    x, weight = torch.randn(3, 8), torch.randn(16, 8)
    x._dynamo_dynamic_indices = {0}
    graph = torch.fx.symbolic_trace(function)
    CaptureValues(graph).run(x, weight)
    return compiler._program_backend(graph, [x, weight], indices)._gaudirt_segment[2]


def main():
    ordinary = routes(compact, ())
    internal = routes(compact, (2,))
    assert ordinary[2][2] == (("feature", 12, 4),)
    assert internal[2][2] == ()
    shared = routes(with_source, (2,))
    assert shared[2][2] == (("feature", 12, 4),) and shared[2][1] == shared[3][1]
    shared = routes(shared_slices, (0, 1, 2))
    assert all(route[2] for route in shared) and len({route[1] for route in shared}) == 1
    print("internal QKV slice is compact; visible aliases remain shared", flush=True)


if __name__ == "__main__":
    main()
