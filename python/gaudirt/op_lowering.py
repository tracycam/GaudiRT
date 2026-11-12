"""FX subgraph lowering to C++ BucketProgram."""

import builtins
import math
import operator
import struct

import torch

from . import _C
from .fx_passes import _fx_shard_embedding


def _finite_float32(value):
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise RuntimeError("GaudiRT FX: scalar is outside FP32 range") from error
    if not math.isfinite(result):
        raise RuntimeError("GaudiRT FX: scalar must be finite")
    return result


def _program_backend(graph_module, example_inputs, compact_outputs=()):
    nodes = tuple(graph_module.graph.nodes)
    placeholders = tuple(node for node in nodes if node.op == "placeholder")
    if len(placeholders) != len(example_inputs):
        raise RuntimeError("GaudiRT FX: placeholder count mismatch")
    placeholder_tensors = []
    for index, (node, value) in enumerate(zip(placeholders, example_inputs)):
        if not isinstance(value, torch.Tensor):
            continue
        dynamic = getattr(value, "_dynamo_dynamic_indices", set()) or set()
        if dynamic and dynamic != {0}:
            raise RuntimeError("GaudiRT FX: only the outer token dimension may be dynamic")
        role = int(node.meta.get("gaudirt_dynamic", bool(dynamic)))
        placeholder_tensors.append((index, node, role))
    placeholder_tensors.sort(key=lambda item: not item[2])
    tensor_indices, roles, ids = [], [], {}
    for index, node, role in placeholder_tensors:
        ids[node] = len(tensor_indices)
        tensor_indices.append(index)
        roles.append(role)
    if not any(roles):
        raise RuntimeError("GaudiRT FX: mark each dynamic token input before capture")

    operations, transposes, splits, fused_norms, rotations, index_casts = [], {}, {}, {}, {}, {}
    Node, Op = _C.FxNode, _C.FxOp
    alias_views = {}
    validated_token_bindings, validated_row_bindings = [], []

    linear_targets = (torch.nn.functional.linear, torch.ops.aten.linear.default)
    block_fp8_targets = (torch.ops.gaudirt.block_fp8_linear, torch.ops.gaudirt.block_fp8_linear.default)
    fp8_targets = (torch.ops.gaudirt.fp8_linear, torch.ops.gaudirt.fp8_linear.default)
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

    def append_op(node, operation):
        ids[node] = len(tensor_indices) + len(operations)
        operations.append(operation)

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
            append_op(node, Node.slice(source_id, start, stop - start, example.shape[1]))
            add_alias(node, source, ("feature", start, stop - start))

    def lower_transpose(node):
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
                append_op(node, Node.transpose(source_id, dim0, dim1))
        else:
            raise RuntimeError("GaudiRT FX: transpose must keep the outer token dimension")

    def lower_reshape(node):
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
            append_op(node, Node.reshape(source_id, list(shape[1:])))

    def lower_cast_or_copy(node):
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
                append_op(node, Node.unary(Op.copy, tensor_id(source)))
            else:
                ids[node] = tensor_id(source)
                if source in alias_views:
                    alias_views[node] = alias_views[source]
        else:
            append_op(node, Node.cast(tensor_id(source), dtype == torch.float32))

    def lower_split(node):
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

    def lower_cat(node):
        if len(node.args) not in (1, 2) or set(node.kwargs) - {"dim"} or (
            len(node.args) == 2 and "dim" in node.kwargs
        ):
            raise RuntimeError("GaudiRT FX: unsupported cat arguments")
        tensors = node.args[0]
        dim = node.args[1] if len(node.args) == 2 else node.kwargs.get("dim", 0)
        if not isinstance(tensors, (tuple, list)) or len(tensors) < 2 or dim not in (-1, 1):
            raise RuntimeError("GaudiRT FX: cat requires at least two feature tensors")
        append_op(node, Node.concat([tensor_id(value) for value in tensors]))

    def lower_aten_slice(node):
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

    def lower_shard_emb(node):
        if len(node.args) != 7 or node.kwargs:
            raise RuntimeError("GaudiRT FX: invalid _fx_shard_embedding arguments")
        indices, weight, org_start, org_end, add_start, add_end, add_offset = node.args
        indices = index_casts.get(indices, indices)
        index_id, weight_id_value = tensor_id(indices), tensor_id(weight)
        if index_id >= len(tensor_indices) or weight_id_value >= len(tensor_indices):
            raise RuntimeError("GaudiRT FX: shard_embedding requires validated external IDs and weights")
        validated_token_bindings.append((index_id, weight_id_value, True))
        append_op(node, Node.shard_embedding(
            index_id, weight_id_value,
            int(org_start), int(org_end), int(add_start), int(add_end), int(add_offset),
        ))

    def lower_embedding(node):
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
        validated_token_bindings.append((index_id, weight_id_value, False))
        append_op(node, Node.binary(Op.embedding, index_id, weight_id_value))

    def lower_rotary(node):
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

    def lower_fused_norm(node):
        if len(node.args) != 4 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported fused RMSNorm arguments")
        epsilon = node.args[3]
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            raise RuntimeError("GaudiRT FX: fused RMSNorm epsilon must be positive and finite")
        producer = len(tensor_indices) + len(operations)
        operations.append(Node.fused_norm(tensor_id(node.args[0]), tensor_id(node.args[1]),
                                          tensor_id(node.args[2]), _finite_float32(epsilon)))
        fused_norms[node] = [producer, None]

    def lower_getitem(node):
        if len(node.args) == 2 and not node.kwargs and isinstance(node.args[1], tuple):
            source, index = node.args
            if len(index) != 2 or index[0] != slice(None) or not isinstance(index[1], slice):
                raise RuntimeError("GaudiRT FX: only feature slices with all token rows are supported")
            feature_slice(node, source, index[1])
            return
        if len(node.args) == 2 and node.args[0] in rotations:
            producer, key_output = rotations[node.args[0]]
            index = node.args[1]
            if node.kwargs or index not in (0, 1):
                raise RuntimeError("GaudiRT FX: invalid rotary output index")
            if index == 0:
                ids[node] = producer
                return
            if key_output is None:
                key_output = len(tensor_indices) + len(operations)
                rotations[node.args[0]][1] = key_output
                operations.append(Node.unary(Op.rotary_key, producer))
            ids[node] = key_output
            return
        if len(node.args) == 2 and node.args[0] in fused_norms:
            producer, residual = fused_norms[node.args[0]]
            index = node.args[1]
            if node.kwargs or index not in (0, 1):
                raise RuntimeError("GaudiRT FX: invalid fused RMSNorm output index")
            if index == 0:
                ids[node] = producer
                return
            if residual is not None:
                ids[node] = residual
                return
            fused_norms[node.args[0]][1] = len(tensor_indices) + len(operations)
            append_op(node, Node.unary(Op.fused_residual, producer))
            return
        if len(node.args) == 2 and node.args[0] in splits:
            source, sizes, source_node = splits[node.args[0]]
            index = node.args[1]
            if node.kwargs or not isinstance(index, int) or index < 0 or index >= len(sizes):
                raise RuntimeError("GaudiRT FX: invalid split output index")
            if len(sizes) == 1:
                ids[node] = source
                if source_node in alias_views:
                    alias_views[node] = alias_views[source_node]
                return
            start = sum(sizes[:index])
            add_alias(node, source_node, ("feature", start, sizes[index]))
            append_op(node, Node.slice(source, start, sizes[index], sum(sizes)))
            return
        if len(node.args) == 2 and not node.kwargs and isinstance(node.args[1], torch.fx.Node):
            source, index = node.args
            index_id = tensor_id(index)
            if index_id >= len(tensor_indices) or roles[index_id]:
                raise RuntimeError("GaudiRT FX: row gather requires external validated indices")
            validated_row_bindings.append(index_id)
            append_op(node, Node.binary(Op.gather_rows, tensor_id(source), index_id))
            return
        raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")

    def lower_index_select(node):
        if len(node.args) != 3 or node.args[1] != 0 or node.kwargs:
            raise RuntimeError("GaudiRT FX: only outer-dimension index_select is supported")
        source, index = node.args[0], node.args[2]
        index_id = tensor_id(index)
        if index_id >= len(tensor_indices) or roles[index_id]:
            raise RuntimeError("GaudiRT FX: row gather requires external validated indices")
        validated_row_bindings.append(index_id)
        append_op(node, Node.binary(Op.gather_rows, tensor_id(source), index_id))

    def lower_quant_linear(node, is_block):
        if len(node.args) not in (3, 4):
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        label = "block FP8" if is_block else "FP8"
        if set(node.kwargs) - {"bias"} or (len(node.args) == 4 and "bias" in node.kwargs):
            raise RuntimeError(f"GaudiRT FX: unsupported {label} linear arguments")
        bias = node.args[3] if len(node.args) == 4 else node.kwargs.get("bias")
        ctor = Node.block_fp8 if is_block else Node.fp8
        append_op(node, ctor(*(tensor_id(value) for value in node.args[:3]),
                             None if bias is None else tensor_id(bias)))

    def lower_linear(node):
        if len(node.args) not in (2, 3):
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        if set(node.kwargs) - {"bias"} or (len(node.args) == 3 and "bias" in node.kwargs):
            raise RuntimeError("GaudiRT FX: unsupported linear arguments")
        bias = node.args[2] if len(node.args) == 3 else node.kwargs.get("bias")
        append_op(node, Node.linear(tensor_id(node.args[0]), tensor_id(node.args[1]),
                                    None if bias is None else tensor_id(bias), True))

    def lower_bmm(node):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        append_op(node, Node.binary(Op.bmm, tensor_id(node.args[0]), tensor_id(node.args[1])))

    def lower_mm(node):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        left = node.args[0].meta.get("example_value")
        right = node.args[1].meta.get("example_value")
        if (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and
                left.ndim == right.ndim and 3 <= left.ndim <= 5):
            append_op(node, Node.binary(Op.bmm, tensor_id(node.args[0]), tensor_id(node.args[1])))
        elif node.args[1] in transposes:
            append_op(node, Node.linear(tensor_id(node.args[0]), weight_id(node.args[1]), None, True))
        else:
            append_op(node, Node.linear(tensor_id(node.args[0]), tensor_id(node.args[1]), None, False))

    def lower_addmm(node):
        if len(node.args) != 3:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        if set(node.kwargs) - {"alpha", "beta"} or any(
            not isinstance(value, (int, float)) or value != 1 for value in node.kwargs.values()
        ):
            raise RuntimeError("GaudiRT FX: addmm requires alpha=beta=1")
        if node.args[2] in transposes:
            append_op(node, Node.linear(tensor_id(node.args[1]), weight_id(node.args[2]),
                                        tensor_id(node.args[0]), True))
        else:
            append_op(node, Node.linear(tensor_id(node.args[1]), tensor_id(node.args[2]),
                                        tensor_id(node.args[0]), False))

    def lower_add_or_sub(node, tensor_op, scalar_op, label):
        if len(node.args) != 2:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        if set(node.kwargs) - {"alpha"} or node.kwargs.get("alpha", 1) != 1:
            raise RuntimeError(f"GaudiRT FX: {label} requires alpha=1")
        append_op(node, binary_operation(node.args, tensor_op, scalar_op))

    def lower_mul_or_div(node, tensor_op, scalar_op):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        append_op(node, binary_operation(node.args, tensor_op, scalar_op))

    def lower_rms_norm(node):
        if len(node.args) != 3 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        epsilon = node.args[2]
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            raise RuntimeError("GaudiRT FX: RMSNorm epsilon must be positive and finite")
        append_op(node, Node.norm(tensor_id(node.args[0]), tensor_id(node.args[1]), _finite_float32(epsilon)))

    def lower_silu_and_mul(node):
        if len(node.args) != 1 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        append_op(node, Node.unary(Op.silu_and_mul, tensor_id(node.args[0])))

    target_handlers = {
        torch.ops.aten._to_copy.default: lower_cast_or_copy,
        torch.ops.aten.slice.Tensor: lower_aten_slice,
        _fx_shard_embedding: lower_shard_emb,
        operator.getitem: lower_getitem,
    }
    for t in transpose_targets:
        target_handlers[t] = lower_transpose
    for t in split_targets:
        target_handlers[t] = lower_split
    for t in cat_targets:
        target_handlers[t] = lower_cat
    for t in embedding_targets:
        target_handlers[t] = lower_embedding
    for t in rotary_targets:
        target_handlers[t] = lower_rotary
    for t in fused_norm_targets:
        target_handlers[t] = lower_fused_norm
    for t in index_select_targets:
        target_handlers[t] = lower_index_select
    for t in block_fp8_targets:
        target_handlers[t] = lambda n: lower_quant_linear(n, True)
    for t in fp8_targets:
        target_handlers[t] = lambda n: lower_quant_linear(n, False)
    for t in linear_targets:
        target_handlers[t] = lower_linear
    for t in bmm_targets:
        target_handlers[t] = lower_bmm
    for t in mm_targets:
        target_handlers[t] = lower_mm
    for t in addmm_targets:
        target_handlers[t] = lower_addmm
    for t in add_targets:
        target_handlers[t] = lambda n: lower_add_or_sub(n, Op.add, Op.scalar_add, "add")
    for t in sub_targets:
        target_handlers[t] = lambda n: lower_add_or_sub(n, Op.sub, Op.scalar_sub, "sub")
    for t in mul_targets:
        target_handlers[t] = lambda n: lower_mul_or_div(n, Op.mul, Op.scalar_mul)
    for t in div_targets:
        target_handlers[t] = lambda n: lower_mul_or_div(n, Op.div, Op.scalar_div)
    for t in norm_targets:
        target_handlers[t] = lower_rms_norm
    for t in silu_targets:
        target_handlers[t] = lower_silu_and_mul

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
        if node.op == "call_method" and node.target == "long" and len(node.args) == 1 and not node.kwargs:
            index_casts[node] = node.args[0]
            continue
        if is_reshape_node(node) and not node.kwargs:
            lower_reshape(node)
            continue
        if node.op == "call_method":
            if node.target == "transpose":
                lower_transpose(node)
                continue
            if node.target in ("to", "float"):
                lower_cast_or_copy(node)
                continue
            if node.target == "split":
                lower_split(node)
                continue
            raise RuntimeError(f"GaudiRT FX: unsupported program node {node.op}: {node.target}")
        if node.op != "call_function":
            raise RuntimeError(f"GaudiRT FX: unsupported program node {node.op}: {node.target}")
        handler = target_handlers.get(node.target)
        if handler is None:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        handler(node)

    if output is None or any(value not in ids for value in output):
        raise RuntimeError("GaudiRT FX: unsupported program output")
    output_bases = [alias_views[value][0] if value in alias_views else ids[value] for value in output]
    output_ids, output_views = [], []
    for index, value in enumerate(output):
        base, views = alias_views.get(value, (ids[value], ()))
        # Only internal values consumed exclusively by attention may lose the
        # producer's stride. Public outputs and shared views retain aliasing.
        compact = (index in compact_outputs and base >= len(tensor_indices) and
                   output_bases.count(base) == 1 and len(views) == 1 and
                   views[0][0] == "feature" and ids[value] != base)
        output_ids.append(ids[value] if compact else base)
        output_views.append(() if compact else views)
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
                           views)
                          for value, views in zip(output_ids, output_views))
    dynamic_argument = tensor_indices[roles.index(1)] if validated_row_bindings else None

    def execute(*args):
        for index_id, weight_id_value, is_sharded in validated_token_bindings:
            index = args[tensor_indices[index_id]]
            weight = args[tensor_indices[weight_id_value]]
            vocabulary = getattr(index, "_gaudirt_validated_vocab", None)
            if (not isinstance(vocabulary, int) or vocabulary <= 0 or
                    (not is_sharded and vocabulary > weight.shape[0])):
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
                                tuple((tensor_indices[index], tensor_indices[weight], is_sharded)
                                      for index, weight, is_sharded in validated_token_bindings),
                                tuple((tensor_indices[index], dynamic_argument)
                                      for index in validated_row_bindings))
    return execute
