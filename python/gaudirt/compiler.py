"""FX entry point for the currently supported bucketed GEMM subgraph."""

import functools
import operator

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
    mm_targets = (torch.mm, torch.ops.aten.mm.default, torch.matmul,
                  torch.ops.aten.matmul.default, operator.matmul)
    addmm_targets = (torch.addmm, torch.ops.aten.addmm.default)
    linear_targets = (torch.nn.functional.linear, torch.ops.aten.linear.default)
    if not isinstance(gemm, torch.fx.Node) or gemm.op != "call_function":
        raise RuntimeError("GaudiRT FX: unsupported subgraph")
    accepted = {gemm}
    bias_node = None

    def transposed_weight(node):
        if not isinstance(node, torch.fx.Node) or node.kwargs:
            raise RuntimeError("GaudiRT FX: GEMM weight must be transposed with t()")
        if node.op == "call_method" and node.target == "t" and len(node.args) == 1:
            pass
        elif ((node.op == "call_method" and node.target == "transpose") or
              (node.op == "call_function" and node.target in (torch.transpose, torch.ops.aten.transpose.int))):
            if (len(node.args) != 3 or any(not isinstance(dim, int) or dim < -2 or dim > 1
                                           for dim in node.args[1:]) or
                    node.args[1] % 2 == node.args[2] % 2):
                raise RuntimeError("GaudiRT FX: GEMM requires a two-dimensional weight transpose")
        else:
            raise RuntimeError("GaudiRT FX: GEMM weight must be transposed with t()")
        accepted.add(node)
        return node.args[0]

    if gemm.target in mm_targets and len(gemm.args) == 2:
        if gemm.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported GEMM arguments")
        input_node, transposed = gemm.args
        weight_node = transposed_weight(transposed)
    elif gemm.target in addmm_targets and len(gemm.args) == 3:
        if set(gemm.kwargs) - {"alpha", "beta"} or any(
            not isinstance(value, (int, float)) or value != 1 for value in gemm.kwargs.values()
        ):
            raise RuntimeError("GaudiRT FX: addmm requires alpha=beta=1")
        bias_node, input_node, transposed = gemm.args
        if bias_node not in placeholders:
            raise RuntimeError("GaudiRT FX: addmm bias must be an external tensor")
        weight_node = transposed_weight(transposed)
    elif gemm.target in linear_targets and len(gemm.args) in (2, 3):
        input_node, weight_node = gemm.args[:2]
        if set(gemm.kwargs) - {"bias"} or (len(gemm.args) == 3 and "bias" in gemm.kwargs):
            raise RuntimeError("GaudiRT FX: unsupported linear arguments")
        bias_node = gemm.args[2] if len(gemm.args) == 3 else gemm.kwargs.get("bias")
        if bias_node is not None and bias_node not in placeholders:
            raise RuntimeError("GaudiRT FX: linear bias must be an external tensor")
    else:
        raise RuntimeError("GaudiRT FX: unsupported subgraph")
    if input_node not in placeholders or weight_node not in placeholders:
        raise RuntimeError("GaudiRT FX: GEMM operands must be external inputs")
    if any(node.op not in ("placeholder", "output") and node not in accepted
           for node in nodes):
        raise RuntimeError("GaudiRT FX: unsupported additional operation")
    input_index = placeholders.index(input_node)
    weight_index = placeholders.index(weight_node)
    bias_index = placeholders.index(bias_node) if bias_node is not None else None

    def execute(*args):
        bias = None if bias_index is None else args[bias_index]
        return (_C.bucket_linear(args[input_index], args[weight_index], bias),)

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
