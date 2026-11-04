"""FX entry point for the currently supported bucketed GEMM subgraph."""

import functools

import torch
import torch.fx.experimental._config

from . import _C


def gaudirt_backend(graph_module, example_inputs):
    del example_inputs
    nodes = tuple(graph_module.graph.nodes)
    placeholders = tuple(node for node in nodes if node.op == "placeholder")
    outputs = tuple(node for node in nodes if node.op == "output")
    if len(outputs) != 1:
        raise RuntimeError("GaudiRT FX: expected one graph output")
    returned = outputs[0].args[0]
    if not isinstance(returned, (tuple, list)) or len(returned) != 1:
        raise RuntimeError("GaudiRT FX: only a single GEMM output is implemented")
    gemm = returned[0]
    mm_targets = (torch.mm, torch.ops.aten.mm.default)
    if (not isinstance(gemm, torch.fx.Node) or gemm.op != "call_function" or
            gemm.target not in mm_targets or len(gemm.args) != 2):
        raise RuntimeError("GaudiRT FX: unsupported subgraph")
    input_node, transposed = gemm.args
    if (not isinstance(transposed, torch.fx.Node) or transposed.op != "call_method" or
            transposed.target != "t" or
            len(transposed.args) != 1):
        raise RuntimeError("GaudiRT FX: GEMM weight must be transposed with t()")
    weight_node = transposed.args[0]
    if input_node not in placeholders or weight_node not in placeholders:
        raise RuntimeError("GaudiRT FX: GEMM operands must be external inputs")
    if any(node.op not in ("placeholder", "output") and node not in (gemm, transposed)
           for node in nodes):
        raise RuntimeError("GaudiRT FX: unsupported additional operation")
    input_index = placeholders.index(input_node)
    weight_index = placeholders.index(weight_node)

    def execute(*args):
        return (_C.bucket_linear(args[input_index], args[weight_index]),)

    return execute


def compile(module, dynamic_arg_indices=(0,)):
    """Capture once with unpadded token tensors and dispatch the supported FX graph."""
    compiled = torch.compile(module, backend=gaudirt_backend, dynamic=True, fullgraph=True)

    @functools.wraps(module)
    def execute(*args, **kwargs):
        for index in dynamic_arg_indices:
            torch._dynamo.mark_dynamic(args[index], 0, min=1, max=4096)
        with torch.fx.experimental._config.patch(backed_size_oblivious=True):
            return compiled(*args, **kwargs)

    return execute
