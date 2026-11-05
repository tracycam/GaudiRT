"""FX entry point for the supported bucketed device subgraphs."""

import functools
import math
import operator
import struct

import torch
import torch.fx.experimental._config

from . import _C


def _program_backend(graph_module, example_inputs):
    nodes = tuple(graph_module.graph.nodes)
    placeholders = tuple(node for node in nodes if node.op == "placeholder")
    if len(placeholders) != len(example_inputs):
        raise RuntimeError("GaudiRT FX: placeholder count mismatch")
    tensor_indices, roles, ids = [], [], {}
    for index, (node, value) in enumerate(zip(placeholders, example_inputs)):
        if not isinstance(value, torch.Tensor):
            continue
        dynamic = getattr(value, "_dynamo_dynamic_indices", set()) or set()
        if dynamic and dynamic != {0}:
            raise RuntimeError("GaudiRT FX: only the outer token dimension may be dynamic")
        ids[node] = len(tensor_indices)
        tensor_indices.append(index)
        roles.append(int(bool(dynamic)))
    if not any(roles):
        raise RuntimeError("GaudiRT FX: mark each dynamic token input before capture")
    operations, transposes, splits, fused_norms, rotations = [], {}, {}, {}, {}
    linear_targets = (torch.nn.functional.linear, torch.ops.aten.linear.default)
    mm_targets = (torch.mm, torch.ops.aten.mm.default, torch.matmul,
                  torch.ops.aten.matmul.default, operator.matmul)
    addmm_targets = (torch.addmm, torch.ops.aten.addmm.default)
    add_targets = (operator.add, torch.add, torch.ops.aten.add.Tensor)
    norm_targets = (torch.ops.gaudirt.rms_norm, torch.ops.gaudirt.rms_norm.default)
    fused_norm_targets = (torch.ops.gaudirt.fused_add_rms_norm,
                          torch.ops.gaudirt.fused_add_rms_norm.default)
    silu_targets = (torch.ops.gaudirt.silu_and_mul, torch.ops.gaudirt.silu_and_mul.default)
    rotary_targets = (torch.ops.gaudirt._rotary_embedding_validated,
                      torch.ops.gaudirt._rotary_embedding_validated.default)
    split_targets = (torch.split, torch.ops.aten.split_with_sizes.default)

    def tensor_id(value):
        if value not in ids:
            raise RuntimeError("GaudiRT FX: unsupported tensor or scalar input")
        return ids[value]

    def weight_id(value):
        if value not in transposes:
            raise RuntimeError("GaudiRT FX: GEMM weight must be transposed")
        return tensor_id(transposes[value])

    output = None
    for node in nodes:
        if node.op == "placeholder":
            continue
        if node.op == "output":
            returned = node.args[0]
            if not isinstance(returned, (tuple, list)) or not returned:
                raise RuntimeError("GaudiRT FX: program output must be a nonempty tensor sequence")
            output = returned
            continue
        if node.op == "call_method" and node.target == "t" and len(node.args) == 1 and not node.kwargs:
            transposes[node] = node.args[0]
            continue
        if ((node.op == "call_method" and node.target == "split") or
                (node.op == "call_function" and node.target in split_targets)):
            if len(node.args) not in (2, 3) or set(node.kwargs) - {"dim"} or (
                len(node.args) == 3 and "dim" in node.kwargs
            ):
                raise RuntimeError("GaudiRT FX: unsupported split arguments")
            dim = node.args[2] if len(node.args) == 3 else node.kwargs.get("dim", 0)
            sizes = node.args[1]
            if dim not in (-1, 1) or not isinstance(sizes, (tuple, list)) or not sizes or any(
                not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in sizes
            ):
                raise RuntimeError("GaudiRT FX: split requires positive feature sizes")
            splits[node] = (tensor_id(node.args[0]), tuple(sizes))
            continue
        if node.op != "call_function":
            raise RuntimeError(f"GaudiRT FX: unsupported program node {node.op}: {node.target}")
        if node.target in rotary_targets:
            if len(node.args) != 9 or node.kwargs:
                raise RuntimeError("GaudiRT FX: unsupported rotary arguments")
            context, head_dim, rotary_dim, is_neox = node.args[5:]
            if (any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in (context, head_dim, rotary_dim)) or not isinstance(is_neox, bool)):
                raise RuntimeError("GaudiRT FX: invalid rotary configuration")
            producer = len(tensor_indices) + len(operations)
            operations.append([7, *(tensor_id(value) for value in node.args[:5]),
                               context, head_dim, rotary_dim, int(is_neox)])
            rotations[node] = [producer, None]
            continue
        if node.target is operator.getitem and len(node.args) == 2 and node.args[0] in rotations:
            producer, key_output = rotations[node.args[0]]
            index = node.args[1]
            if node.kwargs or index not in (0, 1):
                raise RuntimeError("GaudiRT FX: invalid rotary output index")
            if index == 0:
                ids[node] = producer
                continue
            if key_output is None:
                key_output = len(tensor_indices) + len(operations)
                rotations[node.args[0]][1] = key_output
                operations.append([8, producer, 0, -1])
            ids[node] = key_output
            continue
        if node.target in fused_norm_targets:
            if len(node.args) != 4 or node.kwargs:
                raise RuntimeError("GaudiRT FX: unsupported fused RMSNorm arguments")
            epsilon = node.args[3]
            if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
                raise RuntimeError("GaudiRT FX: fused RMSNorm epsilon must be positive and finite")
            bits = struct.unpack("<I", struct.pack("<f", epsilon))[0]
            producer = len(tensor_indices) + len(operations)
            operations.append([5, tensor_id(node.args[0]), tensor_id(node.args[1]), tensor_id(node.args[2]), bits])
            fused_norms[node] = [producer, None]
            continue
        if node.target is operator.getitem and len(node.args) == 2 and node.args[0] in fused_norms:
            producer, residual = fused_norms[node.args[0]]
            index = node.args[1]
            if node.kwargs or index not in (0, 1):
                raise RuntimeError("GaudiRT FX: invalid fused RMSNorm output index")
            if index == 0:
                ids[node] = producer
                continue
            if residual is not None:
                ids[node] = residual
                continue
            fused_norms[node.args[0]][1] = len(tensor_indices) + len(operations)
            operation = [6, producer, 0, -1]
        elif node.target is operator.getitem and len(node.args) == 2 and node.args[0] in splits:
            source, sizes = splits[node.args[0]]
            index = node.args[1]
            if node.kwargs or not isinstance(index, int) or index < 0 or index >= len(sizes):
                raise RuntimeError("GaudiRT FX: invalid split output index")
            if len(sizes) == 1:
                ids[node] = source
                continue
            operation = [4, source, sum(sizes[:index]), sizes[index], sum(sizes)]
        elif node.target in linear_targets and len(node.args) in (2, 3):
            if set(node.kwargs) - {"bias"} or (len(node.args) == 3 and "bias" in node.kwargs):
                raise RuntimeError("GaudiRT FX: unsupported linear arguments")
            bias = node.args[2] if len(node.args) == 3 else node.kwargs.get("bias")
            operation = [0, tensor_id(node.args[0]), tensor_id(node.args[1]),
                         -1 if bias is None else tensor_id(bias)]
        elif node.target in mm_targets and len(node.args) == 2 and not node.kwargs:
            operation = [0, tensor_id(node.args[0]), weight_id(node.args[1]), -1]
        elif node.target in addmm_targets and len(node.args) == 3:
            if set(node.kwargs) - {"alpha", "beta"} or any(
                not isinstance(value, (int, float)) or value != 1 for value in node.kwargs.values()
            ):
                raise RuntimeError("GaudiRT FX: addmm requires alpha=beta=1")
            operation = [0, tensor_id(node.args[1]), weight_id(node.args[2]), tensor_id(node.args[0])]
        elif node.target in add_targets and len(node.args) == 2:
            if set(node.kwargs) - {"alpha"} or node.kwargs.get("alpha", 1) != 1:
                raise RuntimeError("GaudiRT FX: add requires alpha=1")
            operation = [1, tensor_id(node.args[0]), tensor_id(node.args[1]), -1]
        elif node.target in norm_targets and len(node.args) == 3 and not node.kwargs:
            epsilon = node.args[2]
            if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
                raise RuntimeError("GaudiRT FX: RMSNorm epsilon must be positive and finite")
            bits = struct.unpack("<I", struct.pack("<f", epsilon))[0]
            operation = [2, tensor_id(node.args[0]), tensor_id(node.args[1]), bits]
        elif node.target in silu_targets and len(node.args) == 1 and not node.kwargs:
            source = tensor_id(node.args[0])
            operation = [3, source, source, -1]
        else:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        ids[node] = len(tensor_indices) + len(operations)
        operations.append(operation)
    if not operations or output is None or any(value not in ids for value in output):
        raise RuntimeError("GaudiRT FX: unsupported program output")
    output_ids = [ids[value] for value in output]
    if any(value < len(tensor_indices) for value in output_ids) or len(set(output_ids)) != len(output_ids):
        raise RuntimeError("GaudiRT FX: outputs must be distinct program operations")
    program = _C.BucketProgram(roles, operations, output_ids)

    def execute(*args):
        return tuple(program.run([args[index] for index in tensor_indices]))

    return execute


def gaudirt_backend(graph_module, example_inputs):
    nodes = tuple(graph_module.graph.nodes)
    placeholders = tuple(node for node in nodes if node.op == "placeholder")
    outputs = tuple(node for node in nodes if node.op == "output")
    if len(outputs) != 1:
        raise RuntimeError("GaudiRT FX: expected one graph output")
    returned = outputs[0].args[0]
    if not isinstance(returned, (tuple, list)) or not returned:
        raise RuntimeError("GaudiRT FX: program output must be a nonempty tensor sequence")
    if len(returned) != 1:
        return _program_backend(graph_module, example_inputs)
    gemm = returned[0]
    mm_targets = (torch.mm, torch.ops.aten.mm.default, torch.matmul,
                  torch.ops.aten.matmul.default, operator.matmul)
    addmm_targets = (torch.addmm, torch.ops.aten.addmm.default)
    linear_targets = (torch.nn.functional.linear, torch.ops.aten.linear.default)
    if not isinstance(gemm, torch.fx.Node) or gemm.op != "call_function":
        return _program_backend(graph_module, example_inputs)
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
        return _program_backend(graph_module, example_inputs)
    if input_node not in placeholders or weight_node not in placeholders:
        return _program_backend(graph_module, example_inputs)
    if any(node.op not in ("placeholder", "output") and node not in accepted
           for node in nodes):
        return _program_backend(graph_module, example_inputs)
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
        with torch._dynamo.config.patch(specialize_float=True), \
             torch.fx.experimental._config.patch(backed_size_oblivious=True):
            return compiled(*args, **kwargs)

    return execute
