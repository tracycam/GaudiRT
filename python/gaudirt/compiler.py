"""FX entry point for the supported bucketed device subgraphs."""

import builtins
import functools
import math
import operator
import struct
import threading
import weakref

import torch
import torch.fx.experimental._config

from . import _C


def _finite_float32(value):
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise RuntimeError("GaudiRT FX: scalar is outside FP32 range") from error
    if not math.isfinite(result):
        raise RuntimeError("GaudiRT FX: scalar must be finite")
    return result


def _restore_staged_outputs(value, replacements):
    if isinstance(value, torch.Tensor):
        for original, staged in replacements:
            alias = _C._restore_staged_alias(original, staged, value)
            if alias is not None:
                return alias
    elif isinstance(value, tuple):
        values = tuple(_restore_staged_outputs(item, replacements) for item in value)
        return type(value)(*values) if hasattr(value, "_fields") else values
    elif isinstance(value, list):
        return [_restore_staged_outputs(item, replacements) for item in value]
    elif isinstance(value, dict):
        return {key: _restore_staged_outputs(item, replacements) for key, item in value.items()}
    return value


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
        roles.append(int(node.meta.get("gaudirt_dynamic", bool(dynamic))))
    if not any(roles):
        raise RuntimeError("GaudiRT FX: mark each dynamic token input before capture")
    operations, transposes, splits, fused_norms, rotations, index_casts = [], {}, {}, {}, {}, {}
    Node, Op = _C.FxNode, _C.FxOp
    alias_views = {}
    validated_token_bindings, validated_row_bindings = [], []
    linear_targets = (torch.nn.functional.linear, torch.ops.aten.linear.default)
    mm_targets = (torch.mm, torch.ops.aten.mm.default, torch.matmul,
                  torch.ops.aten.matmul.default, operator.matmul)
    bmm_targets = (torch.bmm, torch.ops.aten.bmm.default)
    addmm_targets = (torch.addmm, torch.ops.aten.addmm.default)
    add_targets = (operator.add, torch.add, torch.ops.aten.add.Tensor, torch.ops.aten.add.Scalar)
    sub_targets = (operator.sub, torch.sub, torch.ops.aten.sub.Tensor, torch.ops.aten.sub.Scalar)
    mul_targets = (operator.mul, torch.mul, torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)
    div_targets = (operator.truediv, torch.div, torch.true_divide,
                   torch.ops.aten.div.Tensor, torch.ops.aten.div.Scalar)
    norm_targets = (torch.ops.gaudirt.rms_norm, torch.ops.gaudirt.rms_norm.default)
    fused_norm_targets = (torch.ops.gaudirt.fused_add_rms_norm,
                          torch.ops.gaudirt.fused_add_rms_norm.default)
    silu_targets = (torch.ops.gaudirt.silu_and_mul, torch.ops.gaudirt.silu_and_mul.default)
    rotary_targets = (torch.ops.gaudirt._rotary_embedding_validated,
                      torch.ops.gaudirt._rotary_embedding_validated.default)
    split_targets = (torch.split, torch.ops.aten.split_with_sizes.default)
    cat_targets = (torch.cat, torch.concat, torch.concatenate, torch.ops.aten.cat.default)
    embedding_targets = (torch.nn.functional.embedding, torch.ops.aten.embedding.default)
    index_select_targets = (torch.index_select, torch.ops.aten.index_select.default)
    reshape_targets = (torch.reshape, torch.ops.aten.view.default, torch.ops.aten.reshape.default)
    transpose_targets = (torch.transpose, torch.ops.aten.transpose.int)

    def tensor_id(value):
        if value not in ids:
            raise RuntimeError("GaudiRT FX: unsupported tensor or scalar input")
        return ids[value]

    def weight_id(value):
        if value not in transposes:
            raise RuntimeError("GaudiRT FX: GEMM weight must be transposed")
        return tensor_id(transposes[value])

    def binary_operation(args, tensor_op, scalar_op):
        first, second = args
        first_scalar = isinstance(first, (int, float))
        second_scalar = isinstance(second, (int, float))
        if not first_scalar and not second_scalar:
            return Node.binary(tensor_op, tensor_id(first), tensor_id(second))
        if first_scalar == second_scalar:
            raise RuntimeError("GaudiRT FX: binary operation needs a tensor input")
        scalar, tensor = (first, second) if first_scalar else (second, first)
        return Node.scalar(scalar_op, tensor_id(tensor), _finite_float32(scalar), first_scalar)

    def add_alias(node, source, operation):
        source_id = tensor_id(source)
        base, chain = alias_views.get(source, (source_id, ()))
        alias_views[node] = (base, chain + (operation,))

    def is_reshape_node(node):
        return ((node.op == "call_method" and node.target in ("view", "reshape")) or
                (node.op == "call_function" and node.target in reshape_targets))

    def feature_slice(node, source, index):
        example = source.meta.get("example_value")
        if not isinstance(example, torch.Tensor) or example.ndim != 2 or not isinstance(example.shape[1], int):
            raise RuntimeError("GaudiRT FX: feature slice requires a static width")
        start, stop, step = index.indices(example.shape[1])
        if step != 1 or stop <= start:
            raise RuntimeError("GaudiRT FX: feature slice requires a nonempty unit step")
        source_id = tensor_id(source)
        if start == 0 and stop == example.shape[1]:
            ids[node] = source_id
            if source in alias_views:
                alias_views[node] = alias_views[source]
        else:
            ids[node] = len(tensor_indices) + len(operations)
            operations.append(Node.slice(source_id, start, stop - start, example.shape[1]))
            add_alias(node, source, ("feature", start, stop - start))

    output, single_output = None, False
    for node in nodes:
        if node.op == "placeholder":
            continue
        if node.op == "output":
            returned = node.args[0]
            if isinstance(returned, torch.fx.Node):
                returned, single_output = (returned,), True
            if not isinstance(returned, (tuple, list)) or not returned:
                raise RuntimeError("GaudiRT FX: program output must be a nonempty tensor sequence")
            output = returned
            continue
        if node.op == "call_method" and node.target == "t" and len(node.args) == 1 and not node.kwargs:
            transposes[node] = node.args[0]
            continue
        if (node.op == "call_function" and node.target is builtins.getattr and len(node.args) == 2 and
                node.args[1] == "T" and not node.kwargs):
            transposes[node] = node.args[0]
            continue
        if ((node.op == "call_method" and node.target == "transpose") or
                (node.op == "call_function" and node.target in transpose_targets)):
            if len(node.args) != 3 or node.kwargs:
                raise RuntimeError("GaudiRT FX: unsupported transpose arguments")
            source, dim0, dim1 = node.args
            example = source.meta.get("example_value")
            if (not isinstance(example, torch.Tensor) or
                    any(not isinstance(dim, int) or isinstance(dim, bool) or
                        dim < -example.ndim or dim >= example.ndim for dim in (dim0, dim1))):
                raise RuntimeError("GaudiRT FX: invalid transpose dimensions")
            dim0, dim1 = dim0 % example.ndim, dim1 % example.ndim
            if dim0 == dim1:
                ids[node] = tensor_id(source)
                if source in alias_views:
                    alias_views[node] = alias_views[source]
            elif example.ndim == 2:
                transposes[node] = source
            elif 3 <= example.ndim <= 5 and dim0 != 0 and dim1 != 0:
                source_id = tensor_id(source)
                add_alias(node, source, ("transpose", dim0, dim1))
                if all(user.op == "output" for user in node.users):
                    ids[node] = source_id
                else:
                    ids[node] = len(tensor_indices) + len(operations)
                    operations.append(Node.transpose(source_id, dim0, dim1))
            else:
                raise RuntimeError("GaudiRT FX: transpose must keep the outer token dimension")
            continue
        if node.op == "call_method" and node.target == "long" and len(node.args) == 1 and not node.kwargs:
            index_casts[node] = node.args[0]
            continue
        if is_reshape_node(node) and not node.kwargs:
            source = node.args[0]
            source_id = tensor_id(source)
            original = source.meta.get("example_value")
            result = node.meta.get("example_value")
            if not isinstance(original, torch.Tensor) or not isinstance(result, torch.Tensor):
                raise RuntimeError("GaudiRT FX: missing view shape metadata")
            if (not 1 <= original.ndim <= 5 or not 1 <= result.ndim <= 5 or
                    original.dtype != result.dtype or not result.is_contiguous() or
                    str(original.shape[0]) != str(result.shape[0]) or
                    any(not isinstance(size, int) or size <= 0 for size in result.shape[1:])):
                raise RuntimeError("GaudiRT FX: view must keep the outer token dimension")
            shape = (-1, *(int(size) for size in result.shape[1:]))
            same_shape = tuple(map(str, original.shape)) == tuple(map(str, result.shape))
            if original.is_contiguous():
                add_alias(node, source, ("reshape", shape))
            if original.is_contiguous() and (
                same_shape or all(user.op == "output" or is_reshape_node(user) for user in node.users)
            ):
                ids[node] = source_id
            else:
                ids[node] = len(tensor_indices) + len(operations)
                operations.append(Node.reshape(source_id, list(shape[1:])))
            continue
        if ((node.op == "call_method" and node.target in ("to", "float")) or
                (node.op == "call_function" and node.target is torch.ops.aten._to_copy.default)):
            explicit_copy = node.op == "call_function"
            if explicit_copy:
                if len(node.args) != 1 or set(node.kwargs) - {"dtype"}:
                    raise RuntimeError("GaudiRT FX: unsupported copy arguments")
                dtype = node.kwargs.get("dtype")
            elif node.target == "float":
                if len(node.args) != 1 or node.kwargs:
                    raise RuntimeError("GaudiRT FX: unsupported cast arguments")
                dtype = torch.float32
            else:
                if (len(node.args) not in (1, 2) or set(node.kwargs) - {"dtype"} or
                        (len(node.args) == 2 and "dtype" in node.kwargs)):
                    raise RuntimeError("GaudiRT FX: unsupported cast arguments")
                dtype = node.args[1] if len(node.args) == 2 else node.kwargs.get("dtype")
            source = node.args[0]
            example = source.meta.get("example_value")
            if not isinstance(example, torch.Tensor):
                raise RuntimeError("GaudiRT FX: missing cast metadata")
            if explicit_copy and dtype is None:
                dtype = example.dtype
            if dtype not in (torch.bfloat16, torch.float32):
                raise RuntimeError("GaudiRT FX: cast supports BF16 and FP32")
            if example.dtype == dtype:
                if explicit_copy:
                    ids[node] = len(tensor_indices) + len(operations)
                    operations.append(Node.unary(Op.copy, tensor_id(source)))
                else:
                    ids[node] = tensor_id(source)
                    if source in alias_views:
                        alias_views[node] = alias_views[source]
            else:
                ids[node] = len(tensor_indices) + len(operations)
                operations.append(Node.cast(tensor_id(source), dtype == torch.float32))
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
            example = node.args[0].meta.get("example_value")
            if (not isinstance(example, torch.Tensor) or example.ndim != 2 or
                    not isinstance(example.shape[1], int) or sum(sizes) != example.shape[1]):
                raise RuntimeError("GaudiRT FX: split sizes do not match the static feature width")
            splits[node] = (tensor_id(node.args[0]), tuple(sizes), node.args[0])
            continue
        if node.op != "call_function":
            raise RuntimeError(f"GaudiRT FX: unsupported program node {node.op}: {node.target}")
        if node.target in cat_targets:
            if len(node.args) not in (1, 2) or set(node.kwargs) - {"dim"} or (
                len(node.args) == 2 and "dim" in node.kwargs
            ):
                raise RuntimeError("GaudiRT FX: unsupported cat arguments")
            tensors = node.args[0]
            dim = node.args[1] if len(node.args) == 2 else node.kwargs.get("dim", 0)
            if not isinstance(tensors, (tuple, list)) or len(tensors) < 2 or dim not in (-1, 1):
                raise RuntimeError("GaudiRT FX: cat requires at least two feature tensors")
            ids[node] = len(tensor_indices) + len(operations)
            operations.append(Node.concat([tensor_id(value) for value in tensors]))
            continue
        if (node.target is operator.getitem and len(node.args) == 2 and not node.kwargs and
                isinstance(node.args[1], tuple)):
            source, index = node.args
            if (len(index) != 2 or index[0] != slice(None) or not isinstance(index[1], slice)):
                raise RuntimeError("GaudiRT FX: only feature slices with all token rows are supported")
            feature_slice(node, source, index[1])
            continue
        if node.target is torch.ops.aten.slice.Tensor:
            names = ("dim", "start", "end", "step")
            if not 1 <= len(node.args) <= 5 or set(node.kwargs) - set(names):
                raise RuntimeError("GaudiRT FX: unsupported slice arguments")
            values = [0, None, None, 1]
            for i, value in enumerate(node.args[1:]):
                if names[i] in node.kwargs:
                    raise RuntimeError("GaudiRT FX: duplicate slice argument")
                values[i] = value
            for i, name in enumerate(names):
                if name in node.kwargs:
                    values[i] = node.kwargs[name]
            dim, start, end, step = values
            if dim not in (-1, 1) or any(value is not None and not isinstance(value, int)
                                         for value in (start, end, step)):
                raise RuntimeError("GaudiRT FX: invalid feature slice arguments")
            feature_slice(node, node.args[0], slice(start, end, step))
            continue
        if node.target in embedding_targets:
            if node.target is torch.nn.functional.embedding:
                if len(node.args) != 2 or set(node.kwargs) - {
                    "padding_idx", "max_norm", "norm_type", "scale_grad_by_freq", "sparse"
                } or node.kwargs.get("max_norm") is not None or node.kwargs.get("scale_grad_by_freq", False) \
                        or node.kwargs.get("sparse", False):
                    raise RuntimeError("GaudiRT FX: unsupported embedding arguments")
                indices, weight = node.args
            else:
                if (len(node.args) < 2 or len(node.args) > 5 or node.kwargs or
                        (len(node.args) > 3 and node.args[3]) or (len(node.args) > 4 and node.args[4])):
                    raise RuntimeError("GaudiRT FX: unsupported embedding arguments")
                weight, indices = node.args[:2]
            indices = index_casts.get(indices, indices)
            index_id, weight_id_value = tensor_id(indices), tensor_id(weight)
            if index_id >= len(tensor_indices) or weight_id_value >= len(tensor_indices):
                raise RuntimeError("GaudiRT FX: embedding requires validated external IDs and weights")
            validated_token_bindings.append((index_id, weight_id_value))
            ids[node] = len(tensor_indices) + len(operations)
            operations.append(Node.binary(Op.embedding, index_id, weight_id_value))
            continue
        if node.target in rotary_targets:
            if len(node.args) != 9 or node.kwargs:
                raise RuntimeError("GaudiRT FX: unsupported rotary arguments")
            context, head_dim, rotary_dim, is_neox = node.args[5:]
            if (any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in (context, head_dim, rotary_dim)) or not isinstance(is_neox, bool)):
                raise RuntimeError("GaudiRT FX: invalid rotary configuration")
            producer = len(tensor_indices) + len(operations)
            operations.append(Node.rotary(*(tensor_id(value) for value in node.args[:5]),
                                          context, head_dim, rotary_dim, is_neox))
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
                operations.append(Node.unary(Op.rotary_key, producer))
            ids[node] = key_output
            continue
        if node.target in fused_norm_targets:
            if len(node.args) != 4 or node.kwargs:
                raise RuntimeError("GaudiRT FX: unsupported fused RMSNorm arguments")
            epsilon = node.args[3]
            if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
                raise RuntimeError("GaudiRT FX: fused RMSNorm epsilon must be positive and finite")
            producer = len(tensor_indices) + len(operations)
            operations.append(Node.fused_norm(tensor_id(node.args[0]), tensor_id(node.args[1]),
                                              tensor_id(node.args[2]), _finite_float32(epsilon)))
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
            operation = Node.unary(Op.fused_residual, producer)
        elif node.target is operator.getitem and len(node.args) == 2 and node.args[0] in splits:
            source, sizes, source_node = splits[node.args[0]]
            index = node.args[1]
            if node.kwargs or not isinstance(index, int) or index < 0 or index >= len(sizes):
                raise RuntimeError("GaudiRT FX: invalid split output index")
            if len(sizes) == 1:
                ids[node] = source
                if source_node in alias_views:
                    alias_views[node] = alias_views[source_node]
                continue
            start = sum(sizes[:index])
            add_alias(node, source_node, ("feature", start, sizes[index]))
            operation = Node.slice(source, start, sizes[index], sum(sizes))
        elif (node.target is operator.getitem and len(node.args) == 2 and not node.kwargs and
              isinstance(node.args[1], torch.fx.Node)) or node.target in index_select_targets:
            if node.target in index_select_targets:
                if len(node.args) != 3 or node.args[1] != 0 or node.kwargs:
                    raise RuntimeError("GaudiRT FX: only outer-dimension index_select is supported")
                source, index = node.args[0], node.args[2]
            else:
                source, index = node.args
            index_id = tensor_id(index)
            if index_id >= len(tensor_indices) or roles[index_id]:
                raise RuntimeError("GaudiRT FX: row gather requires external validated indices")
            validated_row_bindings.append(index_id)
            operation = Node.binary(Op.gather_rows, tensor_id(source), index_id)
        elif node.target in linear_targets and len(node.args) in (2, 3):
            if set(node.kwargs) - {"bias"} or (len(node.args) == 3 and "bias" in node.kwargs):
                raise RuntimeError("GaudiRT FX: unsupported linear arguments")
            bias = node.args[2] if len(node.args) == 3 else node.kwargs.get("bias")
            operation = Node.linear(tensor_id(node.args[0]), tensor_id(node.args[1]),
                                    None if bias is None else tensor_id(bias), True)
        elif node.target in bmm_targets and len(node.args) == 2 and not node.kwargs:
            operation = Node.binary(Op.bmm, tensor_id(node.args[0]), tensor_id(node.args[1]))
        elif node.target in mm_targets and len(node.args) == 2 and not node.kwargs:
            left = node.args[0].meta.get("example_value")
            right = node.args[1].meta.get("example_value")
            if (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and
                    left.ndim == right.ndim and 3 <= left.ndim <= 5):
                operation = Node.binary(Op.bmm, tensor_id(node.args[0]), tensor_id(node.args[1]))
            elif node.args[1] in transposes:
                operation = Node.linear(tensor_id(node.args[0]), weight_id(node.args[1]), None, True)
            else:
                operation = Node.linear(tensor_id(node.args[0]), tensor_id(node.args[1]), None, False)
        elif node.target in addmm_targets and len(node.args) == 3:
            if set(node.kwargs) - {"alpha", "beta"} or any(
                not isinstance(value, (int, float)) or value != 1 for value in node.kwargs.values()
            ):
                raise RuntimeError("GaudiRT FX: addmm requires alpha=beta=1")
            if node.args[2] in transposes:
                operation = Node.linear(tensor_id(node.args[1]), weight_id(node.args[2]),
                                        tensor_id(node.args[0]), True)
            else:
                operation = Node.linear(tensor_id(node.args[1]), tensor_id(node.args[2]),
                                        tensor_id(node.args[0]), False)
        elif node.target in add_targets and len(node.args) == 2:
            if set(node.kwargs) - {"alpha"} or node.kwargs.get("alpha", 1) != 1:
                raise RuntimeError("GaudiRT FX: add requires alpha=1")
            operation = binary_operation(node.args, Op.add, Op.scalar_add)
        elif node.target in sub_targets and len(node.args) == 2:
            if set(node.kwargs) - {"alpha"} or node.kwargs.get("alpha", 1) != 1:
                raise RuntimeError("GaudiRT FX: sub requires alpha=1")
            operation = binary_operation(node.args, Op.sub, Op.scalar_sub)
        elif node.target in mul_targets and len(node.args) == 2 and not node.kwargs:
            operation = binary_operation(node.args, Op.mul, Op.scalar_mul)
        elif node.target in div_targets and len(node.args) == 2 and not node.kwargs:
            operation = binary_operation(node.args, Op.div, Op.scalar_div)
        elif node.target in norm_targets and len(node.args) == 3 and not node.kwargs:
            epsilon = node.args[2]
            if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
                raise RuntimeError("GaudiRT FX: RMSNorm epsilon must be positive and finite")
            operation = Node.norm(tensor_id(node.args[0]), tensor_id(node.args[1]), _finite_float32(epsilon))
        elif node.target in silu_targets and len(node.args) == 1 and not node.kwargs:
            source = tensor_id(node.args[0])
            operation = Node.unary(Op.silu_and_mul, source)
        else:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        ids[node] = len(tensor_indices) + len(operations)
        operations.append(operation)
    if output is None or any(value not in ids for value in output):
        raise RuntimeError("GaudiRT FX: unsupported program output")
    output_ids = [alias_views[value][0] if value in alias_views else ids[value] for value in output]
    computed_ids = list(dict.fromkeys(value for value in output_ids if value >= len(tensor_indices)))
    if operations and not computed_ids and any(
        operation.op not in (Op.feature_slice, Op.reshape, Op.transpose) for operation in operations
    ):
        raise RuntimeError("GaudiRT FX: program operations have no output")
    program = _C.BucketProgram(roles, operations, computed_ids) if computed_ids else None
    if program is None and (computed_ids or validated_token_bindings):
        raise RuntimeError("GaudiRT FX: unsupported identity graph")
    computed_slots = {value: index for index, value in enumerate(computed_ids)}
    result_routes = tuple((value < len(tensor_indices),
                           tensor_indices[value] if value < len(tensor_indices) else computed_slots[value],
                           alias_views.get(node, (None, ()))[1])
                          for node, value in zip(output, output_ids))
    dynamic_argument = tensor_indices[roles.index(1)] if validated_row_bindings else None

    def execute(*args):
        for index_id, weight_id_value in validated_token_bindings:
            index = args[tensor_indices[index_id]]
            weight = args[tensor_indices[weight_id_value]]
            vocabulary = getattr(index, "_gaudirt_validated_vocab", None)
            if not isinstance(vocabulary, int) or vocabulary <= 0 or vocabulary > weight.shape[0]:
                raise RuntimeError("GaudiRT FX: embedding token IDs must pass batch metadata validation")
        for index_id in validated_row_bindings:
            index = args[tensor_indices[index_id]]
            dynamic = args[dynamic_argument]
            if getattr(index, "_gaudirt_validated_rows", None) != dynamic.shape[0]:
                raise RuntimeError("GaudiRT FX: row indices must pass batch metadata validation")
        computed = program.run([args[index] for index in tensor_indices]) if program else ()
        result = []
        for is_input, index, views in result_routes:
            tensor = args[index] if is_input else computed[index]
            for operation in views:
                if operation[0] == "feature":
                    tensor = _C._feature_view(tensor, operation[1], operation[2])
                elif operation[0] == "reshape":
                    tensor = _C._reshape_view(tensor, operation[1])
                else:
                    tensor = _C._transpose_feature_view(tensor, operation[1], operation[2])
            result.append(tensor)
        result = tuple(result)
        return result[0] if single_output else result

    execute._gaudirt_segment = (program, tuple(tensor_indices), result_routes, single_output,
                                tuple((tensor_indices[index], tensor_indices[weight])
                                      for index, weight in validated_token_bindings),
                                tuple((tensor_indices[index], dynamic_argument)
                                      for index in validated_row_bindings))
    return execute


class _SegmentExecution:
    def __init__(self, split, runner, input_indices, output_template, checks, records):
        self.split = split
        self.runner = runner
        self.input_indices = input_indices
        self.output_template = output_template
        self.token_checks, self.row_checks = checks
        self.records = records
        self.tracing = False
        self.trace_result = None
        self.trace_deltas = ()
        self.profile_events = None
        self.profile_host_ms = None

    def enable_trace(self):
        """Observe actual C++ segment results for correctness audits."""
        if self.profile_events is not None:
            raise RuntimeError("GaudiRT FX: tracing and segment profiling are mutually exclusive")
        self.tracing = True
        for segment, *_ in self.records:
            segment.forward = lambda *args: self.trace_result

    def named_children(self):
        if not self.tracing:
            raise RuntimeError("GaudiRT FX: enable segment tracing before attaching audit callbacks")
        return self.split.named_children()

    def trace_compile_delta(self, index):
        return self.trace_deltas[index]

    def profile(self, events):
        """Use initialized timing events; the caller synchronizes before reuse."""
        if events is not None and (self.tracing or len(events) != len(self.records)):
            raise RuntimeError("GaudiRT FX: invalid segment profiling events")
        self.profile_events = None if events is None else tuple(event.event_id for pair in events for event in pair)

    def forward(self, *args):
        for index, weight in self.token_checks:
            vocabulary = getattr(args[index], "_gaudirt_validated_vocab", None)
            if not isinstance(vocabulary, int) or vocabulary <= 0 or vocabulary > args[weight].shape[0]:
                raise RuntimeError("GaudiRT FX: embedding token IDs must pass batch metadata validation")
        for index, dynamic in self.row_checks:
            if getattr(args[index], "_gaudirt_validated_rows", None) != args[dynamic].shape[0]:
                raise RuntimeError("GaudiRT FX: row indices must pass batch metadata validation")
        inputs = [args[index] for index in self.input_indices]
        if self.tracing:
            result, traces = self.runner.run_traced(inputs, list(range(len(self.records))))
            self.trace_deltas = tuple(trace[2] for trace in traces)
            for i, (segment, host_arguments, tensor_arguments, single) in enumerate(self.records):
                tensor_values, outputs, _ = traces[i]
                actual = [args[value] if kind == "input" else value for kind, value in host_arguments]
                for index, tensor in zip(tensor_arguments, tensor_values):
                    actual[index] = tensor
                self.trace_result = outputs[0] if single else tuple(outputs)
                segment.forward(*actual)
                traces[i] = None
            self.trace_result = None
        elif self.profile_events is not None:
            result, self.profile_host_ms = self.runner.run_profiled(inputs, self.profile_events)
        else:
            result = self.runner.run(inputs)

        def restore(template):
            kind, value = template
            if kind == "tensor":
                return result[value]
            if kind == "input":
                return args[value]
            if kind == "tuple":
                return tuple(restore(item) for item in value)
            if kind == "list":
                return [restore(item) for item in value]
            return value

        return restore(self.output_template)


def _segment_executor(split, example_inputs, paged):
    graph_nodes = tuple(split.graph.nodes)
    placeholders = tuple(node for node in graph_nodes if node.op == "placeholder")
    if len(placeholders) != len(example_inputs):
        raise RuntimeError("GaudiRT FX: split placeholder count mismatch")
    positions = {node: index for index, node in enumerate(placeholders)}
    input_indices = tuple(index for index, value in enumerate(example_inputs) if isinstance(value, torch.Tensor))
    values = {placeholders[index]: slot for slot, index in enumerate(input_indices)}
    runner = _C.SegmentedRunner(len(input_indices))
    token_checks, row_checks, records = set(), set(), []
    dynamic_inputs = [index for index in input_indices if placeholders[index].meta.get("gaudirt_dynamic")]
    dynamic_input = dynamic_inputs[0] if dynamic_inputs else None

    def tensor_slot(value):
        if not isinstance(value, torch.fx.Node) or value not in values or isinstance(values[value], tuple):
            raise RuntimeError("GaudiRT FX: unsupported tensor connection between segments")
        return values[value]

    def external_index(value):
        if not isinstance(value, torch.fx.Node) or value not in positions:
            raise RuntimeError("GaudiRT FX: validated metadata must enter the captured graph directly")
        return positions[value]

    def host_arguments(arguments, tensor_arguments):
        tensor_arguments = set(tensor_arguments)
        result = []
        for index, value in enumerate(arguments):
            if index in tensor_arguments:
                result.append(("literal", None))
            elif isinstance(value, torch.fx.Node):
                result.append(("input", external_index(value)))
            else:
                result.append(("literal", value))
        return tuple(result)

    for node in graph_nodes:
        if node.op == "placeholder":
            continue
        if node.op == "call_function" and node.target is operator.getitem:
            source, index = node.args
            if source not in values or not isinstance(values[source], tuple) or not isinstance(index, int):
                raise RuntimeError("GaudiRT FX: unsupported split graph indexing")
            values[node] = values[source][index]
            continue
        if node.op == "output":
            output_ids = []

            def output_template(value):
                if isinstance(value, torch.fx.Node):
                    if value in values:
                        slots = values[value]
                        if isinstance(slots, tuple):
                            return ("tuple", tuple(output_template_slot(slot) for slot in slots))
                        return output_template_slot(slots)
                    return ("input", external_index(value))
                if isinstance(value, (tuple, list)):
                    return ("tuple" if isinstance(value, tuple) else "list",
                            tuple(output_template(item) for item in value))
                return ("literal", value)

            def output_template_slot(slot):
                output_ids.append(slot)
                return ("tensor", len(output_ids) - 1)

            template = output_template(node.args[0])
            runner.set_outputs(output_ids)
            return _SegmentExecution(split, runner, input_indices, template,
                                     (tuple(sorted(token_checks)), tuple(sorted(row_checks))), records).forward
        if node.op != "call_module" or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported operation between segments")
        segment = split.get_submodule(node.target)
        segment_nodes = tuple(segment.graph.nodes)
        segment_inputs = tuple(item for item in segment_nodes if item.op == "placeholder")
        if len(segment_inputs) != len(node.args):
            raise RuntimeError("GaudiRT FX: segment argument count mismatch")
        stateful = [item for item in segment_nodes if item.op == "call_function" and item.target in paged]
        if stateful:
            if len(stateful) != 1 or any(item.op not in ("placeholder", "output") and item is not stateful[0]
                                         for item in segment_nodes):
                raise RuntimeError("GaudiRT FX: invalid paged-attention partition")
            attention = stateful[0]
            if len(attention.args) != 12 or attention.kwargs:
                raise RuntimeError("GaudiRT FX: unsupported paged-attention arguments")
            try:
                tensor_arguments = tuple(segment_inputs.index(item) for item in attention.args[:10])
            except ValueError as error:
                raise RuntimeError("GaudiRT FX: attention tensors must be segment inputs") from error
            head_dim, scale = attention.args[10:]
            if not isinstance(head_dim, int) or not isinstance(scale, (int, float)):
                raise RuntimeError("GaudiRT FX: attention geometry must be static")
            slots = [tensor_slot(node.args[index]) for index in tensor_arguments]
            values[node] = runner.add_attention(slots, head_dim, float(scale))
            records.append((segment, host_arguments(node.args, tensor_arguments), tensor_arguments, True))
            continue
        examples = [item.meta.get("example_value") for item in segment_inputs]
        program, tensor_arguments, routes, single, token_bindings, row_bindings = \
            _program_backend(segment, examples)._gaudirt_segment
        slots = [tensor_slot(node.args[index]) for index in tensor_arguments]
        compiled_routes = []
        for is_input, index, operations in routes:
            source = tensor_arguments.index(index) if is_input else index
            views = []
            for operation in operations:
                kind = getattr(_C.SegmentViewKind, operation[0])
                arguments = operation[1] if operation[0] == "reshape" else operation[1:]
                views.append(_C.SegmentView(kind, list(arguments)))
            compiled_routes.append(_C.SegmentRoute(is_input, source, views))
        emitted = tuple(runner.add_program(program, slots, compiled_routes))
        values[node] = emitted[0] if single else emitted
        for index, weight in token_bindings:
            token_checks.add((external_index(node.args[index]), external_index(node.args[weight])))
        for index, _ in row_bindings:
            if dynamic_input is None:
                raise RuntimeError("GaudiRT FX: missing dynamic input for validated row indices")
            row_checks.add((external_index(node.args[index]), dynamic_input))
        records.append((segment, host_arguments(node.args, tensor_arguments), tensor_arguments, single))
    raise RuntimeError("GaudiRT FX: split graph has no output")


def _partition_backend(graph_module, example_inputs):
    """Split a fully captured graph only at the stateful paged-attention operator."""
    from torch.fx.passes.split_module import split_module

    nodes = tuple(graph_module.graph.nodes)
    placeholders = [node for node in nodes if node.op == "placeholder"]
    if len(placeholders) != len(example_inputs):
        raise RuntimeError("GaudiRT FX: placeholder count mismatch")
    extents = set()
    for node, value in zip(placeholders, example_inputs):
        if not isinstance(value, torch.Tensor):
            continue
        dynamic = getattr(value, "_dynamo_dynamic_indices", set()) or set()
        if dynamic and dynamic != {0}:
            raise RuntimeError("GaudiRT FX: only the outer token dimension may be dynamic")
        example = node.meta.get("example_value")
        if dynamic:
            if not isinstance(example, torch.Tensor) or example.ndim == 0:
                raise RuntimeError("GaudiRT FX: missing dynamic input shape")
            extents.add(str(example.shape[0]))
    if not extents:
        raise RuntimeError("GaudiRT FX: mark each dynamic token input before capture")
    for node in nodes:
        example = node.meta.get("example_value")
        if isinstance(example, torch.Tensor) and example.ndim:
            node.meta["gaudirt_dynamic"] = str(example.shape[0]) in extents

    paged = (torch.ops.gaudirt._paged_attn_validated, torch.ops.gaudirt._paged_attn_validated.default)
    partitions, partition = {}, 0
    for node in nodes:
        if node.op in ("placeholder", "output"):
            continue
        if node.op == "call_function" and node.target in paged:
            partition += 1
            partitions[node] = partition
            partition += 1
        else:
            partitions[node] = partition
    split = split_module(graph_module, graph_module, lambda node: partitions[node], keep_original_order=True)
    return _segment_executor(split, example_inputs, paged)


def gaudirt_backend(graph_module, example_inputs):
    nodes = tuple(graph_module.graph.nodes)
    if any(node.op == "call_function" and node.target in (
        torch.ops.gaudirt._paged_attn_validated, torch.ops.gaudirt._paged_attn_validated.default
    ) for node in nodes):
        return _partition_backend(graph_module, example_inputs)
    return _program_backend(graph_module, example_inputs)


def compile(module, dynamic_arg_indices=(0,)):
    """Capture once with unpadded token tensors and dispatch the supported FX graph."""
    class CaptureConfigRequired(Exception):
        pass

    capture = threading.local()

    def guarded_backend(graph, inputs):
        # A changed Dynamo guard can require capture even when the inexpensive
        # outer signature is unchanged. Never build that graph without the
        # configuration used for the first capture.
        if not getattr(capture, "enabled", False):
            raise CaptureConfigRequired()
        return gaudirt_backend(graph, inputs)

    compiled = torch.compile(module, backend=guarded_backend, dynamic=True, fullgraph=True)
    captured_signatures = set()
    static_signatures = []
    signature_tokens = {}

    def signature(value, dynamic=False):
        if isinstance(value, torch.Tensor):
            sizes = value.shape[1:] if dynamic else value.shape
            return (torch.Tensor, value.dtype, value.device, tuple(sizes), tuple(value.stride()),
                    value.requires_grad)
        if isinstance(value, (str, int, float, bool, type(None))):
            return (type(value), value)
        if isinstance(value, (tuple, list)):
            return (type(value), tuple(signature(item) for item in value))
        if isinstance(value, dict):
            return (dict, tuple((key, signature(item)) for key, item in sorted(value.items())))
        # Dynamo guards actual object identity and the attributes it consumed.
        # A freshly created scheduler metadata object need not force a config
        # patch on every step when those guards still hit.
        return (type(value),)

    def static_tensor_token(value, index):
        # Parameters and buffers are normally the same Tensor objects at every
        # forward step. Keep only weak references so compilation does not
        # extend the lifetime of device storage.
        try:
            version = value._version
        except RuntimeError:
            # Inference tensors have no version counter. Dynamo still guards
            # their metadata; changing the storage address invalidates this
            # token, while in-place value updates are consumed by the graph.
            version = None
        address = value.data_ptr()
        while len(static_signatures) <= index:
            static_signatures.append([])
        entries = static_signatures[index]
        for position, cached in enumerate(entries):
            if cached[0]() is value and cached[1:3] == (version, address):
                if position:
                    entries.insert(0, entries.pop(position))
                return cached[3]
        torch._dynamo.mark_static(value)
        normalized = signature(value)
        token = signature_tokens.setdefault(normalized, object())
        if len(entries) >= 64:
            entries[:] = [entry for entry in entries if entry[0]() is not None][:63]
        entries.insert(0, (weakref.ref(value), version, address, token))
        return token

    @functools.wraps(module)
    def execute(*args, **kwargs):
        prepared_args = list(args)
        staged_inputs = {}
        replacements = []
        dynamic_ids = {id(args[index]) for index in dynamic_arg_indices}
        static_tokens = [None] * len(args)
        declared_dynamic = set()
        for index, value in enumerate(args):
            if isinstance(value, torch.Tensor) and id(value) not in dynamic_ids:
                dimensions = getattr(value, "_dynamo_dynamic_indices", set()) or set()
                if dimensions == {0}:
                    declared_dynamic.add(index)
                elif dimensions:
                    raise RuntimeError("GaudiRT FX: only the outer dimension may be dynamic")
                else:
                    static_tokens[index] = static_tensor_token(value, index)
        for index in dynamic_arg_indices:
            original = args[index]
            if id(original) not in staged_inputs:
                if (isinstance(original, torch.Tensor) and original.ndim == 1 and original.stride(0) == 1 and
                        original.device.type == "gaudirt" and original.device.index == 0 and
                        0 < original.shape[0] <= 4096):
                    staged = None
                else:
                    staged = _C._prepare_strided_input(original)
                if staged is not None:
                    replacements.append((original, staged))
                    for name in ("_gaudirt_validated_vocab", "_gaudirt_validated_rows"):
                        if hasattr(original, name):
                            setattr(staged, name, getattr(original, name))
                staged_inputs[id(original)] = original if staged is None else staged
            prepared_args[index] = staged_inputs[id(original)]
            if prepared_args[index].ndim > 1:
                torch._dynamo.mark_static(prepared_args[index], list(range(1, prepared_args[index].ndim)))
            torch._dynamo.mark_dynamic(prepared_args[index], 0, min=1, max=4096)
        key = (tuple(static_tokens[index] if static_tokens[index] is not None else
                     signature(value, index in dynamic_arg_indices or index in declared_dynamic)
                     for index, value in enumerate(prepared_args)), signature(kwargs))
        needs_capture = key not in captured_signatures
        while True:
            if needs_capture:
                with torch._dynamo.config.patch(specialize_float=True, suppress_errors=False,
                                               fail_on_recompile_limit_hit=True), \
                     torch.fx.experimental._config.patch(backed_size_oblivious=True, use_duck_shape=False), \
                     torch._dynamo.error_on_graph_break(True):
                    capture.enabled = True
                    try:
                        result = compiled(*prepared_args, **kwargs)
                    finally:
                        capture.enabled = False
                captured_signatures.add(key)
                break
            try:
                result = compiled(*prepared_args, **kwargs)
                break
            except torch._dynamo.exc.BackendCompilerFailed as error:
                if not isinstance(error.inner_exception, CaptureConfigRequired):
                    raise
                needs_capture = True

        return _restore_staged_outputs(result, replacements) if replacements else result

    return execute
