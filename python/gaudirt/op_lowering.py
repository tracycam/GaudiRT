"""FX subgraph lowering to C++ BucketProgram."""

import builtins
import math
import operator
import struct

import torch

from . import _C
from .fx_passes import _fx_shard_embedding, _fx_softcap


# c10::ScalarType numeric enum values (verified in torch/headeronly/core/ScalarType.h)
_SCALAR_TYPE_CODES = {
    torch.int32: 3,
    torch.int64: 4,
    torch.float32: 6,
    torch.bool: 11,
    torch.bfloat16: 15,
}


def _scalar_type_code(dtype: torch.dtype) -> int:
    code = _SCALAR_TYPE_CODES.get(dtype)
    if code is None:
        raise RuntimeError(f"GaudiRT FX: unsupported scalar dtype {dtype}")
    return code


def _resolve_scalar_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, torch.SymInt):
        return int(value)
    if isinstance(value, torch.SymFloat):
        return float(value)
    return None


def _finite_float32(value):
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise RuntimeError("GaudiRT FX: scalar is outside FP32 range") from error
    if not math.isfinite(result):
        raise RuntimeError("GaudiRT FX: scalar must be finite")
    return result


def _node_example(node):
    if not isinstance(node, torch.fx.Node):
        return None
    ex = node.meta.get("example_value", node.meta.get("val"))
    if isinstance(ex, torch.Tensor):
        return ex
    return None


def _program_backend(graph_module, example_inputs, compact_outputs=(), _active_tensor_args=None):
    nodes = tuple(graph_module.graph.nodes)
    placeholders = tuple(node for node in nodes if node.op == "placeholder")
    if len(placeholders) != len(example_inputs):
        raise RuntimeError("GaudiRT FX: placeholder count mismatch")
    placeholder_tensors = []
    passthrough_placeholders = []
    for index, (node, value) in enumerate(zip(placeholders, example_inputs)):
        if not isinstance(value, torch.Tensor):
            continue
        dynamic = getattr(value, "_dynamo_dynamic_indices", set()) or set()
        if dynamic and dynamic != {0}:
            raise RuntimeError("GaudiRT FX: only the outer token dimension may be dynamic")
        role = int(node.meta.get("gaudirt_dynamic", bool(dynamic)))
        if _active_tensor_args is not None and index not in _active_tensor_args:
            passthrough_placeholders.append((index, node, role, value))
        else:
            placeholder_tensors.append((index, node, role, value))
    placeholder_tensors.sort(key=lambda item: not item[2])
    tensor_indices, roles, ids = [], [], {}
    passthrough_args = {}
    node_shapes, node_dtypes, node_dynamic = {}, {}, {}
    dynamic_dim0_token = None
    for index, node, role, value in placeholder_tensors:
        ids[node] = len(tensor_indices)
        tensor_indices.append(index)
        roles.append(role)
        ex = _node_example(node)
        ref = ex if ex is not None else value
        shape = tuple(ref.shape)
        if role:
            dynamic_dim0_token = str(shape[0])
            node_shapes[node] = (-1, *(int(s) for s in shape[1:]))
        else:
            node_shapes[node] = tuple(int(s) for s in shape)
        node_dtypes[node] = ref.dtype
        node_dynamic[node] = bool(role)
    for index, node, role, value in passthrough_placeholders:
        sentinel = -1 - index
        ids[node] = sentinel
        passthrough_args[sentinel] = index
        ex = _node_example(node)
        ref = ex if ex is not None else value
        shape = tuple(ref.shape)
        if role:
            if dynamic_dim0_token is None:
                dynamic_dim0_token = str(shape[0])
            node_shapes[node] = (-1, *(int(s) for s in shape[1:]))
        else:
            node_shapes[node] = tuple(int(s) for s in shape)
        node_dtypes[node] = ref.dtype
        node_dynamic[node] = bool(role)
    if _active_tensor_args is None and not any(roles):
        raise RuntimeError("GaudiRT FX: mark each dynamic token input before capture")

    operations, transposes, splits, fused_norms, rotations, layer_norms, index_casts = (
        [], {}, {}, {}, {}, {}, {}
    )
    Node, Op = _C.FxNode, _C.FxOp
    alias_views = {}
    validated_token_bindings, validated_row_bindings = [], []

    def _op(name):
        if not hasattr(Op, name):
            raise RuntimeError(f"GaudiRT FX: _C.FxOp.{name} is unavailable in this build")
        return getattr(Op, name)

    def _node_ctor(name):
        if not hasattr(Node, name):
            raise RuntimeError(f"GaudiRT FX: _C.FxNode.{name} is unavailable in this build")
        return getattr(Node, name)

    def _infer_meta(node):
        ex = _node_example(node)
        if ex is not None:
            is_dyn = bool(ex.ndim >= 1 and (isinstance(ex.shape[0], torch.SymInt) or (
                dynamic_dim0_token is not None and str(ex.shape[0]) == dynamic_dim0_token
                and any(isinstance(a, torch.fx.Node) and node_dynamic.get(a, False) for a in node.args)
            )))
            if is_dyn:
                return (-1, *(int(s) for s in ex.shape[1:])), ex.dtype, True
            return tuple(int(s) for s in ex.shape), ex.dtype, False
        return None, None, False

    def shape_of(node):
        if node in node_shapes:
            return node_shapes[node]
        sh, dt, dyn = _infer_meta(node)
        if sh is not None:
            node_shapes[node] = sh
            node_dtypes[node] = dt
            node_dynamic[node] = dyn
        return sh

    def dtype_of(node):
        if node in node_dtypes:
            return node_dtypes[node]
        sh, dt, dyn = _infer_meta(node)
        if dt is not None:
            node_shapes[node] = sh
            node_dtypes[node] = dt
            node_dynamic[node] = dyn
        return dt

    def is_dyn_of(node):
        if node in node_dynamic:
            return node_dynamic[node]
        sh, dt, dyn = _infer_meta(node)
        if sh is not None:
            node_shapes[node] = sh
            node_dtypes[node] = dt
            node_dynamic[node] = dyn
        return dyn

    def record_derived(node, shape=None, dtype=None, dynamic=None):
        inf_sh, inf_dt, inf_dyn = _infer_meta(node)
        if shape is None:
            shape = inf_sh
        if dtype is None:
            dtype = inf_dt
        if dynamic is None:
            dynamic = (shape[0] == -1) if shape else inf_dyn
        if shape is not None:
            node_shapes[node] = shape
        if dtype is not None:
            node_dtypes[node] = dtype
        node_dynamic[node] = bool(dynamic)

    aten = torch.ops.aten
    linear_targets = (torch.nn.functional.linear, aten.linear.default)
    block_fp8_targets = (torch.ops.gaudirt.block_fp8_linear, torch.ops.gaudirt.block_fp8_linear.default)
    fp8_targets = (torch.ops.gaudirt.fp8_linear, torch.ops.gaudirt.fp8_linear.default)
    mm_targets = (torch.mm, aten.mm.default, torch.matmul, aten.matmul.default, operator.matmul)
    bmm_targets = (torch.bmm, aten.bmm.default)
    addmm_targets = (torch.addmm, aten.addmm.default)
    add_targets = (operator.add, torch.add, aten.add.Tensor, aten.add.Scalar)
    sub_targets = (operator.sub, torch.sub, aten.sub.Tensor, aten.sub.Scalar)
    mul_targets = (operator.mul, torch.mul, aten.mul.Tensor, aten.mul.Scalar)
    div_targets = (operator.truediv, torch.div, torch.true_divide, aten.div.Tensor, aten.div.Scalar)
    norm_targets = (torch.ops.gaudirt.rms_norm, torch.ops.gaudirt.rms_norm.default)
    fused_norm_targets = (torch.ops.gaudirt.fused_add_rms_norm, torch.ops.gaudirt.fused_add_rms_norm.default)
    gemma_norm_targets = (
        (torch.ops.gaudirt.gemma_rms_norm, torch.ops.gaudirt.gemma_rms_norm.default)
        if hasattr(torch.ops.gaudirt, "gemma_rms_norm") else ()
    )
    fused_gemma_norm_targets = (
        (torch.ops.gaudirt.fused_add_gemma_rms_norm, torch.ops.gaudirt.fused_add_gemma_rms_norm.default)
        if hasattr(torch.ops.gaudirt, "fused_add_gemma_rms_norm") else ()
    )
    silu_mul_targets = (torch.ops.gaudirt.silu_and_mul, torch.ops.gaudirt.silu_and_mul.default)
    gelu_mul_targets = (
        (torch.ops.gaudirt.gelu_and_mul, torch.ops.gaudirt.gelu_and_mul.default)
        if hasattr(torch.ops.gaudirt, "gelu_and_mul") else ()
    )
    rotary_targets = (torch.ops.gaudirt._rotary_embedding_validated, torch.ops.gaudirt._rotary_embedding_validated.default)
    rotary_pos_targets = (
        (torch.ops.gaudirt.rotary_embedding_positions, torch.ops.gaudirt.rotary_embedding_positions.default)
        if hasattr(torch.ops.gaudirt, "rotary_embedding_positions") else ()
    )
    split_targets = (torch.split, aten.split_with_sizes.default, aten.split.Tensor)
    cat_targets = (torch.cat, torch.concat, torch.concatenate, aten.cat.default)
    embedding_targets = (torch.nn.functional.embedding, aten.embedding.default)
    index_select_targets = (torch.index_select, aten.index_select.default)
    reshape_targets = (torch.reshape, aten.view.default, aten.reshape.default)
    transpose_targets = (torch.transpose, aten.transpose.int)
    unsqueeze_targets = (torch.unsqueeze, aten.unsqueeze.default)
    squeeze_targets = (torch.squeeze, aten.squeeze.default, aten.squeeze.dim, aten.squeeze.dims)
    expand_targets = (aten.expand.default,)

    exp_targets = (torch.exp, aten.exp.default)
    tanh_targets = (torch.tanh, aten.tanh.default)
    sigmoid_targets = (torch.sigmoid, aten.sigmoid.default)
    silu_targets = (torch.nn.functional.silu, aten.silu.default)
    gelu_targets = (torch.nn.functional.gelu, aten.gelu.default)
    clamp_targets = (torch.clamp, aten.clamp.default)
    clamp_min_targets = (aten.clamp_min.default,)
    clamp_max_targets = (aten.clamp_max.default,)
    rsqrt_targets = (torch.rsqrt, aten.rsqrt.default)
    sqrt_targets = (torch.sqrt, aten.sqrt.default)
    neg_targets = (operator.neg, torch.neg, aten.neg.default)
    abs_targets = (operator.abs, torch.abs, aten.abs.default)

    isnan_targets = (torch.isnan, aten.isnan.default)
    where_targets = (torch.where, aten.where.self)
    min_targets = (torch.minimum, aten.minimum.default)
    max_targets = (torch.maximum, aten.maximum.default)
    logical_and_targets = (torch.logical_and, aten.logical_and.default)
    logical_or_targets = (torch.logical_or, aten.logical_or.default)
    logical_not_targets = (operator.not_, torch.logical_not, aten.logical_not.default)
    bitwise_and_targets = (operator.and_, torch.bitwise_and, aten.bitwise_and.Tensor)
    bitwise_or_targets = (operator.or_, torch.bitwise_or, aten.bitwise_or.Tensor)
    eq_targets = (operator.eq, torch.eq, aten.eq.Tensor, aten.eq.Scalar)
    ne_targets = (operator.ne, torch.ne, aten.ne.Tensor, aten.ne.Scalar)
    lt_targets = (operator.lt, torch.lt, aten.lt.Tensor, aten.lt.Scalar)
    le_targets = (operator.le, torch.le, aten.le.Tensor, aten.le.Scalar)
    gt_targets = (operator.gt, torch.gt, aten.gt.Tensor, aten.gt.Scalar)
    ge_targets = (operator.ge, torch.ge, aten.ge.Tensor, aten.ge.Scalar)
    sum_targets = (torch.sum, aten.sum.default, aten.sum.dim_IntList)
    mean_targets = (torch.mean, aten.mean.default, aten.mean.dim)
    amax_targets = (torch.amax, aten.amax.default)
    softmax_targets = (torch.softmax, torch.nn.functional.softmax, aten.softmax.int, aten._softmax.default)
    log_softmax_targets = (torch.log_softmax, torch.nn.functional.log_softmax, aten.log_softmax.int, aten._log_softmax.default)
    layer_norm_targets = (torch.nn.functional.layer_norm, aten.layer_norm.default)
    native_layer_norm_targets = (aten.native_layer_norm.default,)
    full_like_targets = (torch.full_like, aten.full_like.default)

    consumed_inputs = set()

    def mark_consumed(tid):
        if tid < 0:
            raise RuntimeError("GaudiRT FX: internal error - passthrough tensor unexpectedly consumed")
        if tid < len(tensor_indices):
            consumed_inputs.add(tid)
        return tid

    def tensor_id(value, consume=True):
        if value not in ids:
            raise RuntimeError(f"GaudiRT FX: unsupported tensor or scalar input {value!r}")
        tid = ids[value]
        if consume:
            mark_consumed(tid)
        return tid

    def weight_id(value):
        if value not in transposes:
            raise RuntimeError("GaudiRT FX: GEMM weight must be transposed")
        return tensor_id(transposes[value])

    def append_op(node, operation, shape=None, dtype=None, dynamic=None):
        ids[node] = len(tensor_indices) + len(operations)
        operations.append(operation)
        record_derived(node, shape, dtype, dynamic)

    def binary_operation(args, tensor_op, scalar_op):
        first, second = args
        first_scalar = _resolve_scalar_value(first)
        second_scalar = _resolve_scalar_value(second)
        if first_scalar is None and second_scalar is None:
            return Node.binary(tensor_op, tensor_id(first), tensor_id(second))
        if first_scalar is not None and second_scalar is not None:
            raise RuntimeError("GaudiRT FX: binary operation needs a tensor input")
        if first_scalar is not None:
            return Node.scalar(scalar_op, tensor_id(second), _finite_float32(float(first_scalar)), True)
        return Node.scalar(scalar_op, tensor_id(first), _finite_float32(float(second_scalar)), False)

    def _broadcast_shape(s1, s2):
        if s1 is None or s2 is None:
            return None
        r = max(len(s1), len(s2))
        a = (1,) * (r - len(s1)) + s1
        b = (1,) * (r - len(s2)) + s2
        out = []
        for x, y in zip(a, b):
            if x == y:
                out.append(x)
            elif x == 1:
                out.append(y)
            elif y == 1:
                out.append(x)
            else:
                out.append(max(x, y))
        return tuple(out)

    def _binary_meta(args, force_dtype=None):
        first, second = args
        s1 = shape_of(first) if isinstance(first, torch.fx.Node) else ()
        s2 = shape_of(second) if isinstance(second, torch.fx.Node) else ()
        d1 = dtype_of(first) if isinstance(first, torch.fx.Node) else None
        d2 = dtype_of(second) if isinstance(second, torch.fx.Node) else None
        out_sh = _broadcast_shape(s1, s2)
        if force_dtype is not None:
            out_dt = force_dtype
        elif d1 is not None and d2 is not None:
            out_dt = torch.promote_types(d1, d2)
        else:
            out_dt = d1 if d1 is not None else d2
        return out_sh, out_dt

    def add_alias(node, source, operation):
        source_id = tensor_id(source, consume=False)
        base, chain = alias_views.get(source, (source_id, ()))
        alias_views[node] = (base, chain + (operation,))

    def is_reshape_node(node):
        return ((node.op == "call_method" and node.target in ("view", "reshape")) or
                (node.op == "call_function" and node.target in reshape_targets))

    def _apply_slice(node, source, dim, index):
        src_shape = shape_of(source)
        if src_shape is None or not 1 <= len(src_shape) <= 5:
            raise RuntimeError("GaudiRT FX: slice requires known tensor rank 1..5")
        ndim = len(src_shape)
        wdim = dim % ndim
        if is_dyn_of(source) and wdim == 0:
            raise RuntimeError("GaudiRT FX: cannot slice along dynamic outer token dimension")
        dim_size = src_shape[wdim]
        if not isinstance(dim_size, int) or dim_size <= 0:
            raise RuntimeError("GaudiRT FX: slice requires static dimension width")
        start, stop, step = index.indices(dim_size)
        if step != 1 or stop <= start:
            raise RuntimeError("GaudiRT FX: slice requires a nonempty unit step")
        source_id = tensor_id(source, consume=False)
        out_shape = list(src_shape)
        out_shape[wdim] = stop - start
        if start == 0 and stop == dim_size:
            ids[node] = source_id
            record_derived(node, tuple(out_shape), dtype_of(source), is_dyn_of(source))
            if source in alias_views:
                alias_views[node] = alias_views[source]
        elif ndim == 2 and wdim == 1:
            mark_consumed(source_id)
            append_op(node, Node.slice(source_id, start, stop - start, dim_size),
                      tuple(out_shape), dtype_of(source), is_dyn_of(source))
            add_alias(node, source, ("feature", start, stop - start))
        else:
            mark_consumed(source_id)
            ctor = _node_ctor("slice_dim")
            append_op(node, ctor(source_id, wdim, start, stop - start),
                      tuple(out_shape), dtype_of(source), is_dyn_of(source))

    def lower_transpose(node):
        if len(node.args) != 3 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported transpose arguments")
        source, dim0, dim1 = node.args
        src_shape = shape_of(source)
        if (src_shape is None or any(not isinstance(dim, int) or isinstance(dim, bool) or
                dim < -len(src_shape) or dim >= len(src_shape) for dim in (dim0, dim1))):
            raise RuntimeError("GaudiRT FX: invalid transpose dimensions")
        ndim = len(src_shape)
        dim0, dim1 = dim0 % ndim, dim1 % ndim
        out_shape = list(src_shape)
        out_shape[dim0], out_shape[dim1] = out_shape[dim1], out_shape[dim0]
        if dim0 == dim1:
            ids[node] = tensor_id(source, consume=False)
            record_derived(node, src_shape, dtype_of(source), is_dyn_of(source))
            if source in alias_views:
                alias_views[node] = alias_views[source]
        elif ndim == 2 and not is_dyn_of(source) and all(u.target in (aten.mm.default, torch.mm, aten.addmm.default, torch.addmm, aten.matmul.default, torch.matmul, operator.matmul) for u in node.users):
            transposes[node] = source
            record_derived(node, tuple(out_shape), dtype_of(source), False)
        elif not is_dyn_of(source) and ndim == 2:
            # Check if used as weight in mm/addmm vs genuine data transpose
            if any(u.op == "call_function" and u.target in (aten.mm.default, torch.mm, aten.addmm.default, torch.addmm) for u in node.users):
                transposes[node] = source
                record_derived(node, tuple(out_shape), dtype_of(source), False)
            else:
                append_op(node, Node.transpose(tensor_id(source), dim0, dim1),
                          tuple(out_shape), dtype_of(source), False)
        elif 2 <= ndim <= 5 and (not is_dyn_of(source) or (dim0 != 0 and dim1 != 0)):
            source_id = tensor_id(source, consume=False)
            if is_dyn_of(source):
                add_alias(node, source, ("transpose", dim0, dim1))
            if is_dyn_of(source) and all(user.op == "output" for user in node.users):
                ids[node] = source_id
                record_derived(node, tuple(out_shape), dtype_of(source), is_dyn_of(source))
            else:
                mark_consumed(source_id)
                append_op(node, Node.transpose(source_id, dim0, dim1),
                          tuple(out_shape), dtype_of(source), is_dyn_of(source))
        else:
            raise RuntimeError("GaudiRT FX: transpose must keep the dynamic outer token dimension")

    def lower_reshape(node):
        source = node.args[0]
        source_id = tensor_id(source, consume=False)
        src_shape = shape_of(source)
        src_dtype = dtype_of(source)
        src_dyn = is_dyn_of(source)
        if src_shape is None or not 1 <= len(src_shape) <= 5:
            raise RuntimeError("GaudiRT FX: missing view source shape metadata")
        ex_res = _node_example(node)
        if ex_res is not None:
            if ex_res.dtype != src_dtype or (source.op == "placeholder" and not ex_res.is_contiguous()):
                raise RuntimeError("GaudiRT FX: view must preserve dtype and contiguity")
            target_dims = [(-1 if (src_dyn and i == 0) else int(s)) for i, s in enumerate(ex_res.shape)]
        else:
            raw_shape = node.args[1] if len(node.args) == 2 and isinstance(node.args[1], (list, tuple)) else node.args[1:]
            raw_shape = [(-1 if isinstance(d, torch.SymInt) else int(d)) for d in raw_shape]
            if src_dyn:
                src_width = 1
                for s in src_shape[1:]:
                    src_width *= int(s)
                trailing = list(raw_shape[1:])
                neg_idx = next((i for i, d in enumerate(trailing) if d == -1), None)
                if neg_idx is not None:
                    known = 1
                    for i, d in enumerate(trailing):
                        if i != neg_idx:
                            known *= d
                    trailing[neg_idx] = src_width // known
                target_dims = [-1] + trailing
            else:
                src_numel = 1
                for s in src_shape:
                    src_numel *= int(s)
                known = 1
                neg_idx = None
                for i, d in enumerate(raw_shape):
                    if d == -1:
                        neg_idx = i
                    else:
                        known *= d
                if neg_idx is not None:
                    raw_shape[neg_idx] = src_numel // known
                target_dims = raw_shape
        if not 1 <= len(target_dims) <= 5:
            raise RuntimeError("GaudiRT FX: invalid reshape rank")
        if src_dyn:
            if target_dims[0] != -1:
                raise RuntimeError("GaudiRT FX: dynamic view must keep the outer token dimension")
            if any(size <= 0 for size in target_dims[1:]):
                raise RuntimeError("GaudiRT FX: dynamic view trailing dimensions must be positive")
            fx_dims = list(target_dims[1:])
        else:
            if any(size <= 0 for size in target_dims):
                raise RuntimeError("GaudiRT FX: static reshape dimensions must be positive")
            fx_dims = list(target_dims)
        out_shape = tuple(target_dims)
        same_shape = src_shape == out_shape
        ex_orig = _node_example(source)
        orig_contig = ex_orig.is_contiguous() if ex_orig is not None else True
        if orig_contig and src_dyn:
            add_alias(node, source, ("reshape", out_shape))
        if orig_contig and (
            same_shape or (src_dyn and all(user.op == "output" or is_reshape_node(user) for user in node.users))
        ):
            ids[node] = source_id
            record_derived(node, out_shape, src_dtype, src_dyn)
        else:
            mark_consumed(source_id)
            append_op(node, Node.reshape(source_id, fx_dims), out_shape, src_dtype, src_dyn)

    def lower_unsqueeze(node):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported unsqueeze arguments")
        source, dim = node.args
        src_shape = shape_of(source)
        if src_shape is None or not isinstance(dim, int) or isinstance(dim, bool):
            raise RuntimeError("GaudiRT FX: unsqueeze requires known shape and int dim")
        out_rank = len(src_shape) + 1
        wdim = dim % out_rank
        if is_dyn_of(source) and wdim == 0:
            raise RuntimeError("GaudiRT FX: cannot unsqueeze dynamic dim 0")
        out_shape = list(src_shape)
        out_shape.insert(wdim, 1)
        fx_dims = out_shape[1:] if is_dyn_of(source) else out_shape
        append_op(node, Node.reshape(tensor_id(source), fx_dims),
                  tuple(out_shape), dtype_of(source), is_dyn_of(source))

    def lower_squeeze(node):
        if not 1 <= len(node.args) <= 2 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported squeeze arguments")
        source = node.args[0]
        src_shape = shape_of(source)
        if src_shape is None:
            raise RuntimeError("GaudiRT FX: squeeze requires known shape")
        ndim = len(src_shape)
        if len(node.args) == 1:
            drop = {i for i, s in enumerate(src_shape) if s == 1}
        else:
            raw = node.args[1]
            dims = raw if isinstance(raw, (list, tuple)) else (raw,)
            drop = set()
            for d in dims:
                wd = int(d) % ndim
                if src_shape[wd] == 1:
                    drop.add(wd)
        if not drop:
            ids[node] = tensor_id(source, consume=False)
            record_derived(node, src_shape, dtype_of(source), is_dyn_of(source))
            return
        if is_dyn_of(source) and 0 in drop:
            raise RuntimeError("GaudiRT FX: cannot squeeze dynamic outer dimension")
        out_shape = [s for i, s in enumerate(src_shape) if i not in drop]
        if not out_shape:
            out_shape = [1]
        fx_dims = out_shape[1:] if is_dyn_of(source) else out_shape
        append_op(node, Node.reshape(tensor_id(source), fx_dims),
                  tuple(out_shape), dtype_of(source), is_dyn_of(source))

    def lower_expand(node):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported expand arguments")
        source, sizes = node.args
        src_shape = shape_of(source)
        src_dyn = is_dyn_of(source)
        if src_shape is None or not isinstance(sizes, (list, tuple)) or not 1 <= len(sizes) <= 5:
            raise RuntimeError("GaudiRT FX: expand requires rank 1..5")
        out_rank = len(sizes)
        offset = out_rank - len(src_shape)
        if offset < 0:
            raise RuntimeError("GaudiRT FX: expand target rank must be >= source rank")
        ex_res = _node_example(node)
        resolved = []
        for i, sz in enumerate(sizes):
            src_dim = src_shape[i - offset] if i >= offset else 1
            if i == 0 and (isinstance(sz, torch.SymInt) or (
                dynamic_dim0_token is not None and str(sz) == dynamic_dim0_token and (src_dyn or src_dim == 1)
            ) or (sz == -1 and src_dyn)):
                resolved.append(-1)
            elif i == 0 and ex_res is not None and (
                isinstance(ex_res.shape[0], torch.SymInt) or
                (dynamic_dim0_token is not None and str(ex_res.shape[0]) == dynamic_dim0_token and (src_dyn or src_dim == 1))
            ):
                resolved.append(-1)
            elif isinstance(sz, int) and sz == -1:
                if src_dim == -1:
                    resolved.append(-1)
                else:
                    resolved.append(int(src_dim))
            else:
                sval = _resolve_scalar_value(sz)
                if sval is None or isinstance(sval, bool) or int(sval) <= 0:
                    raise RuntimeError("GaudiRT FX: expand requires positive sizes")
                resolved.append(int(sval))
        if resolved == list(src_shape):
            ids[node] = tensor_id(source, consume=False)
            record_derived(node, tuple(resolved), dtype_of(source), resolved[0] == -1)
            return
        ctor = _node_ctor("expand")
        append_op(node, ctor(tensor_id(source), resolved),
                  tuple(resolved), dtype_of(source), resolved[0] == -1)

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
            if (len(node.args) not in (1, 2) or set(node.kwargs) - {"dtype", "non_blocking"} or
                    (len(node.args) == 2 and "dtype" in node.kwargs)):
                raise RuntimeError("GaudiRT FX: unsupported cast arguments")
            dtype = node.args[1] if len(node.args) == 2 else node.kwargs.get("dtype")
        source = node.args[0]
        src_dtype = dtype_of(source)
        src_shape = shape_of(source)
        src_dyn = is_dyn_of(source)
        if src_dtype is None:
            raise RuntimeError("GaudiRT FX: missing cast dtype metadata")
        if dtype is None:
            dtype = src_dtype
        if dtype not in _SCALAR_TYPE_CODES:
            raise RuntimeError(f"GaudiRT FX: unsupported cast target dtype {dtype}")
        if src_dtype == dtype:
            if explicit_copy and node.target is aten._to_copy.default:
                ids[node] = tensor_id(source, consume=False)
                record_derived(node, src_shape, dtype, src_dyn)
                if source in alias_views:
                    alias_views[node] = alias_views[source]
            elif explicit_copy:
                append_op(node, Node.unary(Op.copy, tensor_id(source)), src_shape, dtype, src_dyn)
            else:
                ids[node] = tensor_id(source, consume=False)
                record_derived(node, src_shape, dtype, src_dyn)
                if source in alias_views:
                    alias_views[node] = alias_views[source]
        elif (src_dtype, dtype) in ((torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)):
            append_op(node, Node.cast(tensor_id(source), dtype == torch.float32),
                      src_shape, dtype, src_dyn)
        else:
            ctor = _node_ctor("cast_dtype")
            append_op(node, ctor(tensor_id(source), _scalar_type_code(dtype)),
                      src_shape, dtype, src_dyn)

    def lower_split(node):
        if len(node.args) not in (2, 3) or set(node.kwargs) - {"dim"} or (
            len(node.args) == 3 and "dim" in node.kwargs
        ):
            raise RuntimeError("GaudiRT FX: unsupported split arguments")
        source = node.args[0]
        src_shape = shape_of(source)
        if src_shape is None or not 1 <= len(src_shape) <= 5:
            raise RuntimeError("GaudiRT FX: split requires known source shape")
        ndim = len(src_shape)
        dim = node.args[2] if len(node.args) == 3 else node.kwargs.get("dim", 0)
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise RuntimeError("GaudiRT FX: split dim must be an int")
        wdim = dim % ndim
        if is_dyn_of(source) and wdim == 0:
            raise RuntimeError("GaudiRT FX: cannot split along dynamic outer token axis")
        dim_size = src_shape[wdim]
        if not isinstance(dim_size, int) or dim_size <= 0:
            raise RuntimeError("GaudiRT FX: split requires static dimension width")
        size_arg = node.args[1]
        if isinstance(size_arg, int) and not isinstance(size_arg, bool) and size_arg > 0:
            sizes = []
            rem = dim_size
            while rem > 0:
                take = min(size_arg, rem)
                sizes.append(take)
                rem -= take
            sizes = tuple(sizes)
        elif isinstance(size_arg, (tuple, list)) and size_arg and all(
            isinstance(s, int) and not isinstance(s, bool) and s > 0 for s in size_arg
        ):
            sizes = tuple(size_arg)
        else:
            raise RuntimeError("GaudiRT FX: split requires positive sizes")
        if sum(sizes) != dim_size:
            raise RuntimeError("GaudiRT FX: split sizes do not match dimension width")
        splits[node] = (tensor_id(source, consume=False), sizes, source, wdim, src_shape, dtype_of(source), is_dyn_of(source))

    def lower_cat(node):
        if len(node.args) not in (1, 2) or set(node.kwargs) - {"dim"} or (
            len(node.args) == 2 and "dim" in node.kwargs
        ):
            raise RuntimeError("GaudiRT FX: unsupported cat arguments")
        tensors = node.args[0]
        dim = node.args[1] if len(node.args) == 2 else node.kwargs.get("dim", 0)
        if not isinstance(tensors, (tuple, list)) or len(tensors) < 2:
            raise RuntimeError("GaudiRT FX: cat requires at least two tensors")
        ref_shape = shape_of(tensors[0])
        if ref_shape is None:
            raise RuntimeError("GaudiRT FX: cat requires known input shapes")
        ndim = len(ref_shape)
        wdim = int(dim) % ndim
        if is_dyn_of(tensors[0]) and wdim == 0:
            raise RuntimeError("GaudiRT FX: cannot concat along dynamic outer axis")
        out_shape = list(ref_shape)
        total_cat = 0
        for t in tensors:
            sh = shape_of(t)
            if sh is None or len(sh) != ndim or not isinstance(sh[wdim], int):
                raise RuntimeError("GaudiRT FX: cat tensor shape mismatch")
            total_cat += sh[wdim]
        out_shape[wdim] = total_cat
        if wdim == ndim - 1:
            append_op(node, Node.concat([tensor_id(value) for value in tensors]),
                      tuple(out_shape), dtype_of(tensors[0]), is_dyn_of(tensors[0]))
        else:
            ctor = _node_ctor("concat_dim")
            append_op(node, ctor([tensor_id(value) for value in tensors], wdim),
                      tuple(out_shape), dtype_of(tensors[0]), is_dyn_of(tensors[0]))

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
        if not isinstance(dim, int) or isinstance(dim, bool) or any(
            value is not None and not isinstance(value, int) for value in (start, end, step)
        ):
            raise RuntimeError("GaudiRT FX: invalid slice arguments")
        _apply_slice(node, node.args[0], dim, slice(start, end, step))

    def lower_shard_emb(node):
        if len(node.args) != 7 or node.kwargs:
            raise RuntimeError("GaudiRT FX: invalid _fx_shard_embedding arguments")
        indices, weight, org_start, org_end, add_start, add_end, add_offset = node.args
        indices = index_casts.get(indices, indices)
        index_id, weight_id_value = tensor_id(indices), tensor_id(weight)
        if index_id >= len(tensor_indices) or weight_id_value >= len(tensor_indices):
            raise RuntimeError("GaudiRT FX: shard_embedding requires validated external IDs and weights")
        validated_token_bindings.append((index_id, weight_id_value, True))
        w_sh = shape_of(weight)
        idx_sh = shape_of(indices)
        out_sh = (idx_sh[0], w_sh[1]) if idx_sh and w_sh else None
        append_op(node, Node.shard_embedding(
            index_id, weight_id_value,
            int(org_start), int(org_end), int(add_start), int(add_end), int(add_offset),
        ), out_sh, dtype_of(weight), is_dyn_of(indices))

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
        w_sh = shape_of(weight)
        idx_sh = shape_of(indices)
        out_sh = (idx_sh[0], w_sh[1]) if idx_sh and w_sh else None
        append_op(node, Node.binary(Op.embedding, index_id, weight_id_value),
                  out_sh, dtype_of(weight), is_dyn_of(indices))

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
        rotations[node] = [producer, None, shape_of(node.args[0]), shape_of(node.args[1]),
                           dtype_of(node.args[0]), is_dyn_of(node.args[0])]

    def lower_rotary_pos(node):
        if len(node.args) != 7 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported rotary_embedding_positions arguments")
        q, k, cache, pos, head_dim, rotary_dim, is_neox = node.args
        if (any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in (head_dim, rotary_dim)) or not isinstance(is_neox, bool)):
            raise RuntimeError("GaudiRT FX: invalid rotary_embedding_positions configuration")
        ctor = _node_ctor("rotary_pos")
        producer = len(tensor_indices) + len(operations)
        operations.append(ctor(tensor_id(q), tensor_id(k), tensor_id(cache), tensor_id(pos),
                               head_dim, rotary_dim, is_neox))
        rotations[node] = [producer, None, shape_of(q), shape_of(k), dtype_of(q), is_dyn_of(q)]

    def lower_fused_norm(node, gemma=False):
        if len(node.args) != 4 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported fused RMSNorm arguments")
        epsilon = node.args[3]
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            raise RuntimeError("GaudiRT FX: fused RMSNorm epsilon must be positive and finite")
        producer = len(tensor_indices) + len(operations)
        op_node = (_node_ctor("fused_gemma_norm")(tensor_id(node.args[0]), tensor_id(node.args[1]),
                                                  tensor_id(node.args[2]), _finite_float32(epsilon))
                   if gemma else
                   Node.fused_norm(tensor_id(node.args[0]), tensor_id(node.args[1]),
                                   tensor_id(node.args[2]), _finite_float32(epsilon)))
        operations.append(op_node)
        fused_norms[node] = [producer, None, shape_of(node.args[0]), dtype_of(node.args[0]), is_dyn_of(node.args[0])]

    def lower_getitem(node):
        if len(node.args) == 2 and not node.kwargs and isinstance(node.args[1], (tuple, slice)):
            source, raw_index = node.args
            index = (raw_index,) if isinstance(raw_index, slice) else raw_index
            src_shape = shape_of(source)
            if src_shape is None or len(index) > len(src_shape):
                raise RuntimeError("GaudiRT FX: invalid tensor slice tuple")
            slice_dim = None
            slice_obj = None
            for d, item in enumerate(index):
                if item == slice(None):
                    continue
                if slice_dim is not None or not isinstance(item, slice):
                    raise RuntimeError("GaudiRT FX: only single-axis full slices are supported")
                slice_dim = d
                slice_obj = item
            if slice_dim is None:
                ids[node] = tensor_id(source, consume=False)
                record_derived(node, src_shape, dtype_of(source), is_dyn_of(source))
                return
            _apply_slice(node, source, slice_dim, slice_obj)
            return
        if len(node.args) == 2 and node.args[0] in rotations:
            entry = rotations[node.args[0]]
            producer, key_output, q_sh, k_sh, dt, dyn = entry
            index = node.args[1]
            if node.kwargs or index not in (0, 1):
                raise RuntimeError("GaudiRT FX: invalid rotary output index")
            if index == 0:
                ids[node] = producer
                record_derived(node, q_sh, dt, dyn)
                return
            if key_output is None:
                key_output = len(tensor_indices) + len(operations)
                entry[1] = key_output
                operations.append(Node.unary(Op.rotary_key, producer))
            ids[node] = key_output
            record_derived(node, k_sh, dt, dyn)
            return
        if len(node.args) == 2 and node.args[0] in fused_norms:
            entry = fused_norms[node.args[0]]
            producer, residual, sh, dt, dyn = entry
            index = node.args[1]
            if node.kwargs or index not in (0, 1):
                raise RuntimeError("GaudiRT FX: invalid fused RMSNorm output index")
            if index == 0:
                ids[node] = producer
                record_derived(node, sh, dt, dyn)
                return
            if residual is not None:
                ids[node] = residual
                record_derived(node, sh, dt, dyn)
                return
            entry[1] = len(tensor_indices) + len(operations)
            append_op(node, Node.unary(Op.fused_residual, producer), sh, dt, dyn)
            return
        if len(node.args) == 2 and node.args[0] in layer_norms:
            producer, sh, dt, dyn = layer_norms[node.args[0]]
            index = node.args[1]
            if node.kwargs or index != 0:
                raise RuntimeError("GaudiRT FX: only normalized output (index 0) of layer_norm is supported")
            ids[node] = producer
            record_derived(node, sh, dt, dyn)
            return
        if len(node.args) == 2 and node.args[0] in splits:
            source, sizes, source_node, wdim, src_sh, dt, dyn = splits[node.args[0]]
            index = node.args[1]
            if node.kwargs or not isinstance(index, int) or index < 0 or index >= len(sizes):
                raise RuntimeError("GaudiRT FX: invalid split output index")
            out_sh = list(src_sh)
            out_sh[wdim] = sizes[index]
            out_sh = tuple(out_sh)
            if len(sizes) == 1:
                ids[node] = source
                record_derived(node, out_sh, dt, dyn)
                if source_node in alias_views:
                    alias_views[node] = alias_views[source_node]
                return
            start = sum(sizes[:index])
            mark_consumed(source)
            if len(src_sh) == 2 and wdim == 1:
                add_alias(node, source_node, ("feature", start, sizes[index]))
                append_op(node, Node.slice(source, start, sizes[index], sum(sizes)), out_sh, dt, dyn)
            else:
                append_op(node, _node_ctor("slice_dim")(source, wdim, start, sizes[index]), out_sh, dt, dyn)
            return
        if len(node.args) == 2 and not node.kwargs and isinstance(node.args[1], torch.fx.Node):
            source, index = node.args
            index_id = tensor_id(index)
            if index_id >= len(tensor_indices) or roles[index_id]:
                raise RuntimeError("GaudiRT FX: row gather requires external validated indices")
            validated_row_bindings.append(index_id)
            src_sh = shape_of(source)
            out_sh = (-1, *src_sh[1:]) if src_sh else None
            append_op(node, Node.binary(Op.gather_rows, tensor_id(source), index_id),
                      out_sh, dtype_of(source), True)
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
        src_sh = shape_of(source)
        out_sh = (-1, *src_sh[1:]) if src_sh else None
        append_op(node, Node.binary(Op.gather_rows, tensor_id(source), index_id),
                  out_sh, dtype_of(source), True)

    def lower_quant_linear(node, is_block):
        if len(node.args) not in (3, 4):
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        label = "block FP8" if is_block else "FP8"
        if set(node.kwargs) - {"bias"} or (len(node.args) == 4 and "bias" in node.kwargs):
            raise RuntimeError(f"GaudiRT FX: unsupported {label} linear arguments")
        bias = node.args[3] if len(node.args) == 4 else node.kwargs.get("bias")
        ctor = Node.block_fp8 if is_block else Node.fp8
        x_sh, w_sh = shape_of(node.args[0]), shape_of(node.args[1])
        out_cols = (w_sh[1] if len(w_sh) == 3 else w_sh[0]) if (is_block and w_sh) else (w_sh[0] if w_sh else None)
        out_sh = (*x_sh[:-1], out_cols) if x_sh and out_cols is not None else None
        append_op(node, ctor(*(tensor_id(value) for value in node.args[:3]),
                             None if bias is None else tensor_id(bias)),
                  out_sh, dtype_of(node.args[0]), is_dyn_of(node.args[0]))

    def lower_linear(node):
        if len(node.args) not in (2, 3):
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        if set(node.kwargs) - {"bias"} or (len(node.args) == 3 and "bias" in node.kwargs):
            raise RuntimeError("GaudiRT FX: unsupported linear arguments")
        bias = node.args[2] if len(node.args) == 3 else node.kwargs.get("bias")
        x_sh, w_sh = shape_of(node.args[0]), shape_of(node.args[1])
        out_sh = (*x_sh[:-1], w_sh[0]) if x_sh and w_sh else None
        append_op(node, Node.linear(tensor_id(node.args[0]), tensor_id(node.args[1]),
                                    None if bias is None else tensor_id(bias), True),
                  out_sh, dtype_of(node.args[0]), is_dyn_of(node.args[0]))

    def lower_bmm(node):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        l_sh, r_sh = shape_of(node.args[0]), shape_of(node.args[1])
        out_sh = (*l_sh[:-1], r_sh[-1]) if l_sh and r_sh else None
        append_op(node, Node.binary(Op.bmm, tensor_id(node.args[0]), tensor_id(node.args[1])),
                  out_sh, dtype_of(node.args[0]), is_dyn_of(node.args[0]))

    def lower_mm(node):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        l_sh, r_sh = shape_of(node.args[0]), shape_of(node.args[1])
        if l_sh is not None and r_sh is not None and len(l_sh) == len(r_sh) and 3 <= len(l_sh) <= 5:
            out_sh = (*l_sh[:-1], r_sh[-1])
            append_op(node, Node.binary(Op.bmm, tensor_id(node.args[0]), tensor_id(node.args[1])),
                      out_sh, dtype_of(node.args[0]), is_dyn_of(node.args[0]))
        elif node.args[1] in transposes:
            w = transposes[node.args[1]]
            w_sh = shape_of(w)
            out_sh = (*l_sh[:-1], w_sh[0]) if l_sh and w_sh else None
            append_op(node, Node.linear(tensor_id(node.args[0]), tensor_id(w), None, True),
                      out_sh, dtype_of(node.args[0]), is_dyn_of(node.args[0]))
        else:
            out_sh = (*l_sh[:-1], r_sh[1]) if l_sh and r_sh else None
            append_op(node, Node.linear(tensor_id(node.args[0]), tensor_id(node.args[1]), None, False),
                      out_sh, dtype_of(node.args[0]), is_dyn_of(node.args[0]))

    def lower_addmm(node):
        if len(node.args) != 3:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        if set(node.kwargs) - {"alpha", "beta"} or any(
            not isinstance(value, (int, float)) or value != 1 for value in node.kwargs.values()
        ):
            raise RuntimeError("GaudiRT FX: addmm requires alpha=beta=1")
        l_sh = shape_of(node.args[1])
        if node.args[2] in transposes:
            w = transposes[node.args[2]]
            w_sh = shape_of(w)
            out_sh = (*l_sh[:-1], w_sh[0]) if l_sh and w_sh else None
            append_op(node, Node.linear(tensor_id(node.args[1]), tensor_id(w),
                                        tensor_id(node.args[0]), True),
                      out_sh, dtype_of(node.args[1]), is_dyn_of(node.args[1]))
        else:
            r_sh = shape_of(node.args[2])
            out_sh = (*l_sh[:-1], r_sh[1]) if l_sh and r_sh else None
            append_op(node, Node.linear(tensor_id(node.args[1]), tensor_id(node.args[2]),
                                        tensor_id(node.args[0]), False),
                      out_sh, dtype_of(node.args[1]), is_dyn_of(node.args[1]))

    def lower_add_or_sub(node, tensor_op, scalar_op, label):
        if len(node.args) != 2:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        if set(node.kwargs) - {"alpha"} or node.kwargs.get("alpha", 1) != 1:
            raise RuntimeError(f"GaudiRT FX: {label} requires alpha=1")
        out_sh, out_dt = _binary_meta(node.args)
        append_op(node, binary_operation(node.args, tensor_op, scalar_op), out_sh, out_dt)

    def lower_mul_or_div(node, tensor_op, scalar_op, is_div=False):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        out_sh, out_dt = _binary_meta(node.args)
        if is_div and out_dt in (torch.int32, torch.int64, torch.bool):
            out_dt = torch.float32
        append_op(node, binary_operation(node.args, tensor_op, scalar_op), out_sh, out_dt)

    def lower_cmp(node, op_name, scalar_name):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported comparison {node.target}")
        out_sh, _ = _binary_meta(node.args, force_dtype=torch.bool)
        append_op(node, binary_operation(node.args, _op(op_name), _op(scalar_name)),
                  out_sh, torch.bool)

    def lower_binary_op(node, op_name, force_dtype=None):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported binary operation {node.target}")
        a, b = node.args
        sa, sb = _resolve_scalar_value(a), _resolve_scalar_value(b)
        if op_name in ("minimum", "maximum") and (sa is not None or sb is not None):
            if sa is not None and sb is not None:
                raise RuntimeError("GaudiRT FX: minimum/maximum needs a tensor input")
            t, sc = (b, float(sa)) if sa is not None else (a, float(sb))
            ctor = _node_ctor("ewise_unary")
            lo, hi = (sc, float("inf")) if op_name == "maximum" else (float("-inf"), sc)
            append_op(node, ctor(tensor_id(t), 7, lo, hi), shape_of(t), dtype_of(t), is_dyn_of(t))
            return
        out_sh, out_dt = _binary_meta(node.args, force_dtype=force_dtype)
        append_op(node, Node.binary(_op(op_name), tensor_id(a), tensor_id(b)), out_sh, out_dt)

    def lower_full_like_scalar(ref_node, value, dtype):
        is_int = dtype in (torch.int32, torch.int64)
        fval = 0.0 if is_int else float(value)
        ival = int(value) if is_int else (1 if (dtype == torch.bool and bool(value)) else 0)
        ctor = _node_ctor("full_like")
        op = ctor(tensor_id(ref_node), fval, ival, _scalar_type_code(dtype), is_int or dtype == torch.bool)
        new_id = len(tensor_indices) + len(operations)
        operations.append(op)
        return new_id

    def lower_where(node):
        if len(node.args) != 3 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported where arguments")
        cond, x, y = node.args
        sx, sy = _resolve_scalar_value(x), _resolve_scalar_value(y)
        if isinstance(x, torch.fx.Node):
            ref = x
            dtype = dtype_of(x)
        elif isinstance(y, torch.fx.Node):
            ref = y
            dtype = dtype_of(y)
        else:
            ref = cond
            dtype = torch.float32
        if isinstance(x, torch.fx.Node) and isinstance(y, torch.fx.Node):
            dtype = torch.promote_types(dtype_of(x), dtype_of(y))
        x_id = lower_full_like_scalar(ref, sx, dtype) if sx is not None else tensor_id(x)
        y_id = lower_full_like_scalar(ref, sy, dtype) if sy is not None else tensor_id(y)
        out_sh = _broadcast_shape(
            _broadcast_shape(shape_of(cond), shape_of(x) if isinstance(x, torch.fx.Node) else ()),
            shape_of(y) if isinstance(y, torch.fx.Node) else (),
        )
        append_op(node, Node.ternary(_op("where"), tensor_id(cond), x_id, y_id), out_sh, dtype)

    def lower_full_like(node):
        if len(node.args) < 2:
            raise RuntimeError("GaudiRT FX: unsupported full_like arguments")
        ref = node.args[0]
        fill_val = _resolve_scalar_value(node.args[1])
        if fill_val is None:
            raise RuntimeError("GaudiRT FX: full_like requires a constant scalar fill value")
        dtype = node.kwargs.get("dtype") or dtype_of(ref)
        is_int = dtype in (torch.int32, torch.int64)
        fval = 0.0 if is_int else float(fill_val)
        ival = int(fill_val) if is_int else (1 if (dtype == torch.bool and bool(fill_val)) else 0)
        ctor = _node_ctor("full_like")
        append_op(node, ctor(tensor_id(ref), fval, ival, _scalar_type_code(dtype),
                             is_int or dtype == torch.bool),
                  shape_of(ref), dtype, is_dyn_of(ref))

    def lower_unary_ewise(node, op_code, alpha=0.0, beta=0.0):
        src = node.args[0]
        ctor = _node_ctor("ewise_unary")
        append_op(node, ctor(tensor_id(src), op_code, alpha, beta),
                  shape_of(src), dtype_of(src), is_dyn_of(src))

    def lower_gelu(node):
        if len(node.args) < 1:
            raise RuntimeError("GaudiRT FX: unsupported gelu arguments")
        approx = node.args[1] if len(node.args) > 1 else node.kwargs.get("approximate", "none")
        if approx not in ("none", "tanh"):
            raise RuntimeError(f"GaudiRT FX: invalid gelu approximate={approx!r}")
        lower_unary_ewise(node, 5 if approx == "tanh" else 4)

    def lower_clamp(node):
        if not 1 <= len(node.args) <= 3:
            raise RuntimeError("GaudiRT FX: unsupported clamp arguments")
        min_v = node.args[1] if len(node.args) > 1 else node.kwargs.get("min")
        max_v = node.args[2] if len(node.args) > 2 else node.kwargs.get("max")
        lo = float(min_v) if min_v is not None else float("-inf")
        hi = float(max_v) if max_v is not None else float("inf")
        if lo > hi:
            lo = hi
        lower_unary_ewise(node, 7, lo, hi)

    def lower_clamp_min(node):
        lo = float(node.args[1] if len(node.args) > 1 else node.kwargs["min"])
        lower_unary_ewise(node, 7, lo, float("inf"))

    def lower_clamp_max(node):
        hi = float(node.args[1] if len(node.args) > 1 else node.kwargs["max"])
        lower_unary_ewise(node, 7, float("-inf"), hi)

    def lower_softcap(node):
        cap = float(node.args[1])
        lower_unary_ewise(node, 6, _finite_float32(cap), 0.0)

    def lower_rms_norm(node, gemma=False):
        if len(node.args) != 3 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported RMSNorm arguments")
        epsilon = node.args[2]
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) or epsilon <= 0:
            raise RuntimeError("GaudiRT FX: RMSNorm epsilon must be positive and finite")
        ctor = _node_ctor("gemma_norm") if gemma else Node.norm
        append_op(node, ctor(tensor_id(node.args[0]), tensor_id(node.args[1]), _finite_float32(epsilon)),
                  shape_of(node.args[0]), dtype_of(node.args[0]), is_dyn_of(node.args[0]))

    def lower_silu_and_mul(node):
        if len(node.args) != 1 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported silu_and_mul arguments")
        sh = shape_of(node.args[0])
        out_sh = (*sh[:-1], sh[-1] // 2) if sh else None
        append_op(node, Node.unary(Op.silu_and_mul, tensor_id(node.args[0])),
                  out_sh, dtype_of(node.args[0]), is_dyn_of(node.args[0]))

    def lower_gelu_and_mul(node):
        if len(node.args) not in (1, 2):
            raise RuntimeError("GaudiRT FX: unsupported gelu_and_mul arguments")
        approx = node.args[1] if len(node.args) == 2 else node.kwargs.get("approximate", "none")
        if approx not in ("none", "tanh"):
            raise RuntimeError(f"GaudiRT FX: invalid gelu_and_mul approximate={approx!r}")
        sh = shape_of(node.args[0])
        out_sh = (*sh[:-1], sh[-1] // 2) if sh else None
        ctor = _node_ctor("gelu_and_mul")
        append_op(node, ctor(tensor_id(node.args[0]), 1 if approx == "tanh" else 0),
                  out_sh, dtype_of(node.args[0]), is_dyn_of(node.args[0]))

    def lower_reduction(node, op_name):
        src = node.args[0]
        src_shape = shape_of(src)
        src_dtype = dtype_of(src)
        src_dyn = is_dyn_of(src)
        if src_shape is None or not 1 <= len(src_shape) <= 5:
            raise RuntimeError("GaudiRT FX: reduction requires known rank 1..5")
        ndim = len(src_shape)
        if node.target in (aten.sum.default, aten.mean.default) or (
            node.target in (torch.sum, torch.mean) and len(node.args) <= 1 and "dim" not in node.kwargs
        ):
            dims = list(range(ndim))
            keepdim = bool(node.kwargs.get("keepdim", False))
            dtype = node.kwargs.get("dtype")
        elif node.target is aten.amax.default or node.target is torch.amax:
            raw_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", [])
            keepdim = bool(node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False))
            dims = list(range(ndim)) if (raw_dim is None or raw_dim == [] or raw_dim == ()) else (
                [raw_dim] if isinstance(raw_dim, int) else list(raw_dim)
            )
            dtype = None
        else:
            raw_dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim")
            keepdim = bool(node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False))
            dtype = node.args[3] if len(node.args) > 3 else node.kwargs.get("dtype")
            dims = list(range(ndim)) if raw_dim is None else (
                [raw_dim] if isinstance(raw_dim, int) else list(raw_dim)
            )
        wdims = sorted({int(d) % ndim for d in dims}, reverse=True)
        if not wdims:
            ids[node] = tensor_id(src)
            record_derived(node, src_shape, src_dtype, src_dyn)
            return
        if src_dyn and 0 in wdims:
            raise RuntimeError("GaudiRT FX: cannot reduce along dynamic outer token axis")
        cur_id = tensor_id(src)
        cur_shape = list(src_shape)
        cur_dtype = dtype or src_dtype
        if op_name == "sum_dim" and cur_dtype == torch.bool:
            cur_dtype = torch.int32
        if cur_dtype != src_dtype:
            cur_id_new = len(tensor_indices) + len(operations)
            operations.append(_node_ctor("cast_dtype")(cur_id, _scalar_type_code(cur_dtype)))
            cur_id = cur_id_new
        reduce_ctor = _node_ctor("reduce")
        op_enum = _op(op_name)
        for i, d in enumerate(wdims):
            if keepdim:
                cur_shape[d] = 1
            else:
                cur_shape.pop(d)
            out_id = len(tensor_indices) + len(operations)
            operations.append(reduce_ctor(op_enum, cur_id, d, keepdim))
            cur_id = out_id
        ids[node] = cur_id
        record_derived(node, tuple(cur_shape), cur_dtype, src_dyn)

    def lower_softmax(node, is_log):
        src = node.args[0]
        src_shape = shape_of(src)
        src_dtype = dtype_of(src)
        src_dyn = is_dyn_of(src)
        if src_shape is None:
            raise RuntimeError("GaudiRT FX: softmax requires known input shape")
        dim = node.args[1] if len(node.args) > 1 else node.kwargs["dim"]
        wdim = int(dim) % len(src_shape)
        if src_dyn and wdim == 0:
            raise RuntimeError("GaudiRT FX: softmax along dynamic outer token axis is unsupported")
        target_dtype = None
        if node.target in (aten._softmax.default, aten._log_softmax.default):
            half_to_float = bool(node.args[2] if len(node.args) > 2 else node.kwargs.get("half_to_float", False))
            if half_to_float:
                target_dtype = torch.float32
        else:
            target_dtype = node.args[2] if len(node.args) > 2 else node.kwargs.get("dtype")
        cur_id = tensor_id(src)
        cur_dtype = src_dtype
        if target_dtype is not None and target_dtype != src_dtype:
            cur_id_new = len(tensor_indices) + len(operations)
            operations.append(Node.cast(cur_id, target_dtype == torch.float32))
            cur_id = cur_id_new
            cur_dtype = target_dtype
        op_enum = _op("log_softmax" if is_log else "softmax")
        append_op(node, _node_ctor("softmax")(op_enum, cur_id, wdim),
                  src_shape, cur_dtype, src_dyn)

    def lower_layer_norm(node, is_native):
        src = node.args[0]
        norm_shape = tuple(int(s) for s in (node.args[1] if len(node.args) > 1 else node.kwargs["normalized_shape"]))
        weight = node.args[2] if len(node.args) > 2 else node.kwargs.get("weight")
        bias = node.args[3] if len(node.args) > 3 else node.kwargs.get("bias")
        eps = float(node.args[4] if len(node.args) > 4 else node.kwargs.get("eps", 1e-5))
        src_shape = shape_of(src)
        src_dtype = dtype_of(src)
        src_dyn = is_dyn_of(src)
        if src_shape is None or len(norm_shape) > len(src_shape):
            raise RuntimeError("GaudiRT FX: invalid layer_norm normalized_shape")
        if len(norm_shape) != 1 or src_shape[-1] != norm_shape[0]:
            raise RuntimeError("GaudiRT FX: layer_norm normalized_shape must match the trailing dimension")
        producer = len(tensor_indices) + len(operations)
        op = _node_ctor("layer_norm")(
            tensor_id(src),
            None if weight is None else tensor_id(weight),
            None if bias is None else tensor_id(bias),
            _finite_float32(eps),
        )
        operations.append(op)
        if is_native:
            layer_norms[node] = (producer, src_shape, src_dtype, src_dyn)
        else:
            ids[node] = producer
            record_derived(node, src_shape, src_dtype, src_dyn)

    target_handlers = {
        aten._to_copy.default: lower_cast_or_copy,
        aten.to.dtype: lower_cast_or_copy,
        aten.to.dtype_layout: lower_cast_or_copy,
        aten.slice.Tensor: lower_aten_slice,
        _fx_shard_embedding: lower_shard_emb,
        _fx_softcap: lower_softcap,
        operator.getitem: lower_getitem,
    }
    if hasattr(aten, "type_as"):
        target_handlers[aten.type_as.default] = lambda n: (
            ids.__setitem__(n, tensor_id(n.args[0], consume=False)) or record_derived(n, shape_of(n.args[0]), dtype_of(n.args[0]), is_dyn_of(n.args[0]))
            if dtype_of(n.args[0]) == dtype_of(n.args[1]) else
            append_op(n, _node_ctor("cast_dtype")(tensor_id(n.args[0]), _scalar_type_code(dtype_of(n.args[1]))),
                      shape_of(n.args[0]), dtype_of(n.args[1]), is_dyn_of(n.args[0]))
        )
    for t in transpose_targets:
        target_handlers[t] = lower_transpose
    for t in unsqueeze_targets:
        target_handlers[t] = lower_unsqueeze
    for t in squeeze_targets:
        target_handlers[t] = lower_squeeze
    for t in expand_targets:
        target_handlers[t] = lower_expand
    for t in split_targets:
        target_handlers[t] = lower_split
    for t in cat_targets:
        target_handlers[t] = lower_cat
    for t in embedding_targets:
        target_handlers[t] = lower_embedding
    for t in rotary_targets:
        target_handlers[t] = lower_rotary
    for t in rotary_pos_targets:
        target_handlers[t] = lower_rotary_pos
    for t in fused_norm_targets:
        target_handlers[t] = lambda n: lower_fused_norm(n, False)
    for t in fused_gemma_norm_targets:
        target_handlers[t] = lambda n: lower_fused_norm(n, True)
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
        target_handlers[t] = lambda n: lower_mul_or_div(n, Op.div, Op.scalar_div, True)
    for t in norm_targets:
        target_handlers[t] = lambda n: lower_rms_norm(n, False)
    for t in gemma_norm_targets:
        target_handlers[t] = lambda n: lower_rms_norm(n, True)
    for t in silu_mul_targets:
        target_handlers[t] = lower_silu_and_mul
    for t in gelu_mul_targets:
        target_handlers[t] = lower_gelu_and_mul

    for t in exp_targets:
        target_handlers[t] = lambda n: lower_unary_ewise(n, 0)
    for t in tanh_targets:
        target_handlers[t] = lambda n: lower_unary_ewise(n, 1)
    for t in sigmoid_targets:
        target_handlers[t] = lambda n: lower_unary_ewise(n, 2)
    for t in silu_targets:
        target_handlers[t] = lambda n: lower_unary_ewise(n, 3)
    for t in gelu_targets:
        target_handlers[t] = lower_gelu
    for t in clamp_targets:
        target_handlers[t] = lower_clamp
    for t in clamp_min_targets:
        target_handlers[t] = lower_clamp_min
    for t in clamp_max_targets:
        target_handlers[t] = lower_clamp_max
    for t in rsqrt_targets:
        target_handlers[t] = lambda n: lower_unary_ewise(n, 8)
    for t in sqrt_targets:
        target_handlers[t] = lambda n: lower_unary_ewise(n, 9)
    for t in neg_targets:
        target_handlers[t] = lambda n: lower_unary_ewise(n, 10)
    for t in abs_targets:
        target_handlers[t] = lambda n: lower_unary_ewise(n, 11)

    for t in isnan_targets:
        target_handlers[t] = lambda n: append_op(
            n, Node.unary(_op("isnan"), tensor_id(n.args[0])),
            shape_of(n.args[0]), torch.bool, is_dyn_of(n.args[0]),
        )
    for t in logical_not_targets:
        target_handlers[t] = lambda n: append_op(
            n, Node.unary(_op("logical_not"), tensor_id(n.args[0])),
            shape_of(n.args[0]), torch.bool, is_dyn_of(n.args[0]),
        )
    for t in where_targets:
        target_handlers[t] = lower_where
    for t in min_targets:
        target_handlers[t] = lambda n: lower_binary_op(n, "minimum")
    for t in max_targets:
        target_handlers[t] = lambda n: lower_binary_op(n, "maximum")
    for t in logical_and_targets:
        target_handlers[t] = lambda n: lower_binary_op(n, "logical_and", torch.bool)
    for t in logical_or_targets:
        target_handlers[t] = lambda n: lower_binary_op(n, "logical_or", torch.bool)
    for t in bitwise_and_targets:
        target_handlers[t] = lambda n: lower_binary_op(n, "bitwise_and")
    for t in bitwise_or_targets:
        target_handlers[t] = lambda n: lower_binary_op(n, "bitwise_or")
    for t in eq_targets:
        target_handlers[t] = lambda n: lower_cmp(n, "eq", "scalar_eq")
    for t in ne_targets:
        target_handlers[t] = lambda n: lower_cmp(n, "ne", "scalar_ne")
    for t in lt_targets:
        target_handlers[t] = lambda n: lower_cmp(n, "lt", "scalar_lt")
    for t in le_targets:
        target_handlers[t] = lambda n: lower_cmp(n, "le", "scalar_le")
    for t in gt_targets:
        target_handlers[t] = lambda n: lower_cmp(n, "gt", "scalar_gt")
    for t in ge_targets:
        target_handlers[t] = lambda n: lower_cmp(n, "ge", "scalar_ge")
    for t in sum_targets:
        target_handlers[t] = lambda n: lower_reduction(n, "sum_dim")
    for t in mean_targets:
        target_handlers[t] = lambda n: lower_reduction(n, "mean_dim")
    for t in amax_targets:
        target_handlers[t] = lambda n: lower_reduction(n, "amax_dim")
    for t in softmax_targets:
        target_handlers[t] = lambda n: lower_softmax(n, False)
    for t in log_softmax_targets:
        target_handlers[t] = lambda n: lower_softmax(n, True)
    for t in layer_norm_targets:
        target_handlers[t] = lambda n: lower_layer_norm(n, False)
    for t in native_layer_norm_targets:
        target_handlers[t] = lambda n: lower_layer_norm(n, True)
    for t in full_like_targets:
        target_handlers[t] = lower_full_like

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
            if node.target == "unsqueeze":
                lower_unsqueeze(node)
                continue
            if node.target == "squeeze":
                lower_squeeze(node)
                continue
            if node.target == "expand":
                lower_expand(node)
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
    if _active_tensor_args is None and computed_ids and len(consumed_inputs) < len(tensor_indices):
        active = frozenset(tensor_indices[i] for i in consumed_inputs)
        return _program_backend(graph_module, example_inputs, compact_outputs, _active_tensor_args=active)
    program = _C.BucketProgram(roles, operations, computed_ids) if computed_ids else None
    if program is None and (computed_ids or validated_token_bindings):
        raise RuntimeError("GaudiRT FX: unsupported identity graph")
    computed_slots = {value: index for index, value in enumerate(computed_ids)}
    result_routes = tuple(
        (True, passthrough_args[value], views)
        if value < 0
        else (
            value < len(tensor_indices),
            tensor_indices[value] if value < len(tensor_indices) else computed_slots[value],
            views,
        )
        for value, views in zip(output_ids, output_views)
    )
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
