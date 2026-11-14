"""FX subgraph lowering to C++ BucketProgram."""

import builtins
import math
import operator

import torch

from . import _C
from .fx_passes import _fx_shard_embedding, _fx_softcap
from .op_lowering_helpers import (
    _SCALAR_TYPE_CODES,
    _broadcast_shape,
    _finite_float32,
    _node_example,
    _resolve_scalar_value,
    _scalar_type_code,
    build_op_target_groups,
)

Node = _C.FxNode
Op = _C.FxOp


def _op(name):
    if not hasattr(Op, name):
        raise RuntimeError(f"GaudiRT FX: _C.FxOp.{name} is unavailable in this build")
    return getattr(Op, name)


def _node_ctor(name):
    if not hasattr(Node, name):
        raise RuntimeError(f"GaudiRT FX: _C.FxNode.{name} is unavailable in this build")
    return getattr(Node, name)


class _ProgramLoweringBuilder:
    """Lower an FX GraphModule into a C++ BucketProgram + view/validation epilogue."""

    def __init__(self, graph_module, example_inputs, compact_outputs=(), _active_tensor_args=None):
        self.graph_module = graph_module
        self.example_inputs = example_inputs
        self.compact_outputs = compact_outputs
        self._active_tensor_args = _active_tensor_args
        self.nodes = tuple(graph_module.graph.nodes)
        self.tensor_indices = []
        self.roles = []
        self.ids = {}
        self.passthrough_args = {}
        self.node_shapes = {}
        self.node_dtypes = {}
        self.node_dynamic = {}
        self.dynamic_dim0_token = None
        self.operations = []
        self.transposes = {}
        self.splits = {}
        self.fused_norms = {}
        self.rotations = {}
        self.layer_norms = {}
        self.index_casts = {}
        self.alias_views = {}
        self.validated_token_bindings = []
        self.validated_row_bindings = []
        self.consumed_inputs = set()
        self.groups = build_op_target_groups()
        self.reshape_targets = self.groups["reshape"]
        self.target_handlers = self._build_target_handlers()

    def _infer_meta(self, node):
        ex = _node_example(node)
        if ex is not None:
            is_dyn = bool(ex.ndim >= 1 and (isinstance(ex.shape[0], torch.SymInt) or (
                self.dynamic_dim0_token is not None and str(ex.shape[0]) == self.dynamic_dim0_token
                and any(isinstance(a, torch.fx.Node) and self.node_dynamic.get(a, False) for a in node.args)
            )))
            if is_dyn:
                return (-1, *(int(s) for s in ex.shape[1:])), ex.dtype, True
            return tuple(int(s) for s in ex.shape), ex.dtype, False
        return None, None, False

    def shape_of(self, node):
        if node in self.node_shapes:
            return self.node_shapes[node]
        sh, dt, dyn = self._infer_meta(node)
        if sh is not None:
            self.node_shapes[node] = sh
            self.node_dtypes[node] = dt
            self.node_dynamic[node] = dyn
        return sh

    def dtype_of(self, node):
        if node in self.node_dtypes:
            return self.node_dtypes[node]
        sh, dt, dyn = self._infer_meta(node)
        if dt is not None:
            self.node_shapes[node] = sh
            self.node_dtypes[node] = dt
            self.node_dynamic[node] = dyn
        return dt

    def is_dyn_of(self, node):
        if node in self.node_dynamic:
            return self.node_dynamic[node]
        sh, dt, dyn = self._infer_meta(node)
        if sh is not None:
            self.node_shapes[node] = sh
            self.node_dtypes[node] = dt
            self.node_dynamic[node] = dyn
        return dyn

    def record_derived(self, node, shape=None, dtype=None, dynamic=None):
        inf_sh, inf_dt, inf_dyn = self._infer_meta(node)
        if shape is None:
            shape = inf_sh
        if dtype is None:
            dtype = inf_dt
        if dynamic is None:
            dynamic = (shape[0] == -1) if shape else inf_dyn
        if shape is not None:
            self.node_shapes[node] = shape
        if dtype is not None:
            self.node_dtypes[node] = dtype
        self.node_dynamic[node] = bool(dynamic)

    def mark_consumed(self, tid):
        if tid < 0:
            raise RuntimeError("GaudiRT FX: internal error - passthrough tensor unexpectedly consumed")
        if tid < len(self.tensor_indices):
            self.consumed_inputs.add(tid)
        return tid

    def tensor_id(self, value, consume=True):
        if value not in self.ids:
            raise RuntimeError(f"GaudiRT FX: unsupported tensor or scalar input {value!r}")
        tid = self.ids[value]
        if consume:
            self.mark_consumed(tid)
        return tid

    def append_op(self, node, operation, shape=None, dtype=None, dynamic=None):
        self.ids[node] = len(self.tensor_indices) + len(self.operations)
        self.operations.append(operation)
        self.record_derived(node, shape, dtype, dynamic)

    def binary_operation(self, args, tensor_op, scalar_op):
        first, second = args
        first_scalar = _resolve_scalar_value(first)
        second_scalar = _resolve_scalar_value(second)
        if first_scalar is None and second_scalar is None:
            return Node.binary(tensor_op, self.tensor_id(first), self.tensor_id(second))
        if first_scalar is not None and second_scalar is not None:
            raise RuntimeError("GaudiRT FX: binary operation needs a tensor input")
        if first_scalar is not None:
            return Node.scalar(scalar_op, self.tensor_id(second), _finite_float32(float(first_scalar)), True)
        return Node.scalar(scalar_op, self.tensor_id(first), _finite_float32(float(second_scalar)), False)

    def _binary_meta(self, args, force_dtype=None):
        first, second = args
        s1 = self.shape_of(first) if isinstance(first, torch.fx.Node) else ()
        s2 = self.shape_of(second) if isinstance(second, torch.fx.Node) else ()
        d1 = self.dtype_of(first) if isinstance(first, torch.fx.Node) else None
        d2 = self.dtype_of(second) if isinstance(second, torch.fx.Node) else None
        out_sh = _broadcast_shape(s1, s2)
        if force_dtype is not None:
            out_dt = force_dtype
        elif d1 is not None and d2 is not None:
            out_dt = torch.promote_types(d1, d2)
        else:
            out_dt = d1 if d1 is not None else d2
        return out_sh, out_dt

    def add_alias(self, node, source, operation):
        source_id = self.tensor_id(source, consume=False)
        base, chain = self.alias_views.get(source, (source_id, ()))
        self.alias_views[node] = (base, chain + (operation,))

    def is_reshape_node(self, node):
        return ((node.op == "call_method" and node.target in ("view", "reshape")) or
                (node.op == "call_function" and node.target in self.reshape_targets))

    def _apply_slice(self, node, source, dim, index):
        src_shape = self.shape_of(source)
        if src_shape is None or not 1 <= len(src_shape) <= 5:
            raise RuntimeError("GaudiRT FX: slice requires known tensor rank 1..5")
        ndim = len(src_shape)
        wdim = dim % ndim
        if self.is_dyn_of(source) and wdim == 0:
            raise RuntimeError("GaudiRT FX: cannot slice along dynamic outer token dimension")
        dim_size = src_shape[wdim]
        if not isinstance(dim_size, int) or dim_size <= 0:
            raise RuntimeError("GaudiRT FX: slice requires static dimension width")
        start, stop, step = index.indices(dim_size)
        if step != 1 or stop <= start:
            raise RuntimeError("GaudiRT FX: slice requires a nonempty unit step")
        source_id = self.tensor_id(source, consume=False)
        out_shape = list(src_shape)
        out_shape[wdim] = stop - start
        if start == 0 and stop == dim_size:
            self.ids[node] = source_id
            self.record_derived(node, tuple(out_shape), self.dtype_of(source), self.is_dyn_of(source))
            if source in self.alias_views:
                self.alias_views[node] = self.alias_views[source]
        elif ndim == 2 and wdim == 1:
            self.mark_consumed(source_id)
            self.append_op(node, Node.slice(source_id, start, stop - start, dim_size),
                           tuple(out_shape), self.dtype_of(source), self.is_dyn_of(source))
            self.add_alias(node, source, ("feature", start, stop - start))
        else:
            self.mark_consumed(source_id)
            self.append_op(node, _node_ctor("slice_dim")(source_id, wdim, start, stop - start),
                           tuple(out_shape), self.dtype_of(source), self.is_dyn_of(source))

    def lower_transpose(self, node):
        aten = torch.ops.aten
        if len(node.args) != 3 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported transpose arguments")
        source, dim0, dim1 = node.args
        src_shape = self.shape_of(source)
        if (src_shape is None or any(not isinstance(dim, int) or isinstance(dim, bool) or
                dim < -len(src_shape) or dim >= len(src_shape) for dim in (dim0, dim1))):
            raise RuntimeError("GaudiRT FX: invalid transpose dimensions")
        ndim = len(src_shape)
        dim0, dim1 = dim0 % ndim, dim1 % ndim
        out_shape = list(src_shape)
        out_shape[dim0], out_shape[dim1] = out_shape[dim1], out_shape[dim0]
        if dim0 == dim1:
            self.ids[node] = self.tensor_id(source, consume=False)
            self.record_derived(node, src_shape, self.dtype_of(source), self.is_dyn_of(source))
            if source in self.alias_views:
                self.alias_views[node] = self.alias_views[source]
        elif ndim == 2 and not self.is_dyn_of(source) and all(
            u.target in (aten.mm.default, torch.mm, aten.addmm.default, torch.addmm, aten.matmul.default, torch.matmul, operator.matmul)
            for u in node.users
        ):
            self.transposes[node] = source
            self.record_derived(node, tuple(out_shape), self.dtype_of(source), False)
        elif not self.is_dyn_of(source) and ndim == 2:
            if any(u.op == "call_function" and u.target in (aten.mm.default, torch.mm, aten.addmm.default, torch.addmm) for u in node.users):
                self.transposes[node] = source
                self.record_derived(node, tuple(out_shape), self.dtype_of(source), False)
            else:
                self.append_op(node, Node.transpose(self.tensor_id(source), dim0, dim1),
                               tuple(out_shape), self.dtype_of(source), False)
        elif 2 <= ndim <= 5 and (not self.is_dyn_of(source) or (dim0 != 0 and dim1 != 0)):
            source_id = self.tensor_id(source, consume=False)
            if self.is_dyn_of(source):
                self.add_alias(node, source, ("transpose", dim0, dim1))
            if self.is_dyn_of(source) and all(user.op == "output" for user in node.users):
                self.ids[node] = source_id
                self.record_derived(node, tuple(out_shape), self.dtype_of(source), self.is_dyn_of(source))
            else:
                self.mark_consumed(source_id)
                self.append_op(node, Node.transpose(source_id, dim0, dim1),
                               tuple(out_shape), self.dtype_of(source), self.is_dyn_of(source))
        else:
            raise RuntimeError("GaudiRT FX: transpose must keep the dynamic outer token dimension")

    def lower_reshape(self, node):
        source = node.args[0]
        source_id = self.tensor_id(source, consume=False)
        src_shape = self.shape_of(source)
        src_dtype = self.dtype_of(source)
        src_dyn = self.is_dyn_of(source)
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
            self.add_alias(node, source, ("reshape", out_shape))
        if orig_contig and (
            same_shape or (src_dyn and all(user.op == "output" or self.is_reshape_node(user) for user in node.users))
        ):
            self.ids[node] = source_id
            self.record_derived(node, out_shape, src_dtype, src_dyn)
        else:
            self.mark_consumed(source_id)
            self.append_op(node, Node.reshape(source_id, fx_dims), out_shape, src_dtype, src_dyn)

    def lower_unsqueeze(self, node):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported unsqueeze arguments")
        source, dim = node.args
        src_shape = self.shape_of(source)
        if src_shape is None or not isinstance(dim, int) or isinstance(dim, bool):
            raise RuntimeError("GaudiRT FX: unsqueeze requires known shape and int dim")
        out_rank = len(src_shape) + 1
        wdim = dim % out_rank
        if self.is_dyn_of(source) and wdim == 0:
            raise RuntimeError("GaudiRT FX: cannot unsqueeze dynamic dim 0")
        out_shape = list(src_shape)
        out_shape.insert(wdim, 1)
        fx_dims = out_shape[1:] if self.is_dyn_of(source) else out_shape
        self.append_op(node, Node.reshape(self.tensor_id(source), fx_dims),
                       tuple(out_shape), self.dtype_of(source), self.is_dyn_of(source))

    def lower_squeeze(self, node):
        if not 1 <= len(node.args) <= 2 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported squeeze arguments")
        source = node.args[0]
        src_shape = self.shape_of(source)
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
            self.ids[node] = self.tensor_id(source, consume=False)
            self.record_derived(node, src_shape, self.dtype_of(source), self.is_dyn_of(source))
            return
        if self.is_dyn_of(source) and 0 in drop:
            raise RuntimeError("GaudiRT FX: cannot squeeze dynamic outer dimension")
        out_shape = [s for i, s in enumerate(src_shape) if i not in drop]
        if not out_shape:
            out_shape = [1]
        fx_dims = out_shape[1:] if self.is_dyn_of(source) else out_shape
        self.append_op(node, Node.reshape(self.tensor_id(source), fx_dims),
                       tuple(out_shape), self.dtype_of(source), self.is_dyn_of(source))

    def lower_expand(self, node):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported expand arguments")
        source, sizes = node.args
        src_shape = self.shape_of(source)
        src_dyn = self.is_dyn_of(source)
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
                self.dynamic_dim0_token is not None and str(sz) == self.dynamic_dim0_token and (src_dyn or src_dim == 1)
            ) or (sz == -1 and src_dyn)):
                resolved.append(-1)
            elif i == 0 and ex_res is not None and (
                isinstance(ex_res.shape[0], torch.SymInt) or
                (self.dynamic_dim0_token is not None and str(ex_res.shape[0]) == self.dynamic_dim0_token and (src_dyn or src_dim == 1))
            ):
                resolved.append(-1)
            elif isinstance(sz, int) and sz == -1:
                resolved.append(-1 if src_dim == -1 else int(src_dim))
            else:
                sval = _resolve_scalar_value(sz)
                if sval is None or isinstance(sval, bool) or int(sval) <= 0:
                    raise RuntimeError("GaudiRT FX: expand requires positive sizes")
                resolved.append(int(sval))
        if resolved == list(src_shape):
            self.ids[node] = self.tensor_id(source, consume=False)
            self.record_derived(node, tuple(resolved), self.dtype_of(source), resolved[0] == -1)
            return
        self.append_op(node, _node_ctor("expand")(self.tensor_id(source), resolved),
                       tuple(resolved), self.dtype_of(source), resolved[0] == -1)

    def lower_cast_or_copy(self, node):
        aten = torch.ops.aten
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
        src_dtype = self.dtype_of(source)
        src_shape = self.shape_of(source)
        src_dyn = self.is_dyn_of(source)
        if src_dtype is None:
            raise RuntimeError("GaudiRT FX: missing cast dtype metadata")
        if dtype is None:
            dtype = src_dtype
        if dtype not in _SCALAR_TYPE_CODES:
            raise RuntimeError(f"GaudiRT FX: unsupported cast target dtype {dtype}")
        if src_dtype == dtype:
            if explicit_copy and node.target is not aten._to_copy.default:
                self.append_op(node, Node.unary(Op.copy, self.tensor_id(source)), src_shape, dtype, src_dyn)
            else:
                self.ids[node] = self.tensor_id(source, consume=False)
                self.record_derived(node, src_shape, dtype, src_dyn)
                if source in self.alias_views:
                    self.alias_views[node] = self.alias_views[source]
        elif (src_dtype, dtype) in ((torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16)):
            self.append_op(node, Node.cast(self.tensor_id(source), dtype == torch.float32),
                           src_shape, dtype, src_dyn)
        else:
            self.append_op(node, _node_ctor("cast_dtype")(self.tensor_id(source), _scalar_type_code(dtype)),
                           src_shape, dtype, src_dyn)

    def lower_split(self, node):
        if len(node.args) not in (2, 3) or set(node.kwargs) - {"dim"} or (
            len(node.args) == 3 and "dim" in node.kwargs
        ):
            raise RuntimeError("GaudiRT FX: unsupported split arguments")
        source = node.args[0]
        src_shape = self.shape_of(source)
        if src_shape is None or not 1 <= len(src_shape) <= 5:
            raise RuntimeError("GaudiRT FX: split requires known source shape")
        ndim = len(src_shape)
        dim = node.args[2] if len(node.args) == 3 else node.kwargs.get("dim", 0)
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise RuntimeError("GaudiRT FX: split dim must be an int")
        wdim = dim % ndim
        if self.is_dyn_of(source) and wdim == 0:
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
        self.splits[node] = (
            self.tensor_id(source, consume=False), sizes, source, wdim, src_shape,
            self.dtype_of(source), self.is_dyn_of(source),
        )

    def lower_cat(self, node):
        if len(node.args) not in (1, 2) or set(node.kwargs) - {"dim"} or (
            len(node.args) == 2 and "dim" in node.kwargs
        ):
            raise RuntimeError("GaudiRT FX: unsupported cat arguments")
        tensors = node.args[0]
        dim = node.args[1] if len(node.args) == 2 else node.kwargs.get("dim", 0)
        if not isinstance(tensors, (tuple, list)) or len(tensors) < 2:
            raise RuntimeError("GaudiRT FX: cat requires at least two tensors")
        ref_shape = self.shape_of(tensors[0])
        if ref_shape is None:
            raise RuntimeError("GaudiRT FX: cat requires known input shapes")
        ndim = len(ref_shape)
        wdim = int(dim) % ndim
        if self.is_dyn_of(tensors[0]) and wdim == 0:
            raise RuntimeError("GaudiRT FX: cannot concat along dynamic outer axis")
        out_shape = list(ref_shape)
        total_cat = 0
        for t in tensors:
            sh = self.shape_of(t)
            if sh is None or len(sh) != ndim or not isinstance(sh[wdim], int):
                raise RuntimeError("GaudiRT FX: cat tensor shape mismatch")
            total_cat += sh[wdim]
        out_shape[wdim] = total_cat
        tids = [self.tensor_id(value) for value in tensors]
        op = Node.concat(tids) if wdim == ndim - 1 else _node_ctor("concat_dim")(tids, wdim)
        self.append_op(node, op, tuple(out_shape), self.dtype_of(tensors[0]), self.is_dyn_of(tensors[0]))

    def lower_aten_slice(self, node):
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
        self._apply_slice(node, node.args[0], dim, slice(start, end, step))

    def lower_shard_emb(self, node):
        if len(node.args) != 7 or node.kwargs:
            raise RuntimeError("GaudiRT FX: invalid _fx_shard_embedding arguments")
        indices, weight, org_start, org_end, add_start, add_end, add_offset = node.args
        indices = self.index_casts.get(indices, indices)
        index_id, weight_id_value = self.tensor_id(indices), self.tensor_id(weight)
        if index_id >= len(self.tensor_indices) or weight_id_value >= len(self.tensor_indices):
            raise RuntimeError("GaudiRT FX: shard_embedding requires validated external IDs and weights")
        self.validated_token_bindings.append((index_id, weight_id_value, True))
        w_sh = self.shape_of(weight)
        idx_sh = self.shape_of(indices)
        out_sh = (idx_sh[0], w_sh[1]) if idx_sh and w_sh else None
        self.append_op(node, Node.shard_embedding(
            index_id, weight_id_value,
            int(org_start), int(org_end), int(add_start), int(add_end), int(add_offset),
        ), out_sh, self.dtype_of(weight), self.is_dyn_of(indices))

    def lower_embedding(self, node):
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
        indices = self.index_casts.get(indices, indices)
        index_id, weight_id_value = self.tensor_id(indices), self.tensor_id(weight)
        if index_id >= len(self.tensor_indices) or weight_id_value >= len(self.tensor_indices):
            raise RuntimeError("GaudiRT FX: embedding requires validated external IDs and weights")
        self.validated_token_bindings.append((index_id, weight_id_value, False))
        w_sh = self.shape_of(weight)
        idx_sh = self.shape_of(indices)
        out_sh = (idx_sh[0], w_sh[1]) if idx_sh and w_sh else None
        self.append_op(node, Node.binary(Op.embedding, index_id, weight_id_value),
                       out_sh, self.dtype_of(weight), self.is_dyn_of(indices))

    def lower_rotary(self, node):
        if len(node.args) != 9 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported rotary arguments")
        context, head_dim, rotary_dim, is_neox = node.args[5:]
        if (any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in (context, head_dim, rotary_dim)) or not isinstance(is_neox, bool)):
            raise RuntimeError("GaudiRT FX: invalid rotary configuration")
        producer = len(self.tensor_indices) + len(self.operations)
        self.operations.append(Node.rotary(*(self.tensor_id(value) for value in node.args[:5]),
                                           context, head_dim, rotary_dim, is_neox))
        self.rotations[node] = [producer, None, self.shape_of(node.args[0]), self.shape_of(node.args[1]),
                                self.dtype_of(node.args[0]), self.is_dyn_of(node.args[0])]

    def lower_rotary_pos(self, node):
        if len(node.args) != 7 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported rotary_embedding_positions arguments")
        q, k, cache, pos, head_dim, rotary_dim, is_neox = node.args
        if (any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in (head_dim, rotary_dim)) or not isinstance(is_neox, bool)):
            raise RuntimeError("GaudiRT FX: invalid rotary_embedding_positions configuration")
        producer = len(self.tensor_indices) + len(self.operations)
        self.operations.append(_node_ctor("rotary_pos")(
            self.tensor_id(q), self.tensor_id(k), self.tensor_id(cache), self.tensor_id(pos),
            head_dim, rotary_dim, is_neox,
        ))
        self.rotations[node] = [producer, None, self.shape_of(q), self.shape_of(k),
                                self.dtype_of(q), self.is_dyn_of(q)]

    def lower_fused_norm(self, node):
        if len(node.args) not in (4, 5) or set(node.kwargs) - {"weight_offset"} or (
            len(node.args) == 5 and "weight_offset" in node.kwargs
        ):
            raise RuntimeError("GaudiRT FX: unsupported fused RMSNorm arguments")
        epsilon = node.args[3]
        weight_offset = node.args[4] if len(node.args) == 5 else node.kwargs.get("weight_offset", 0.0)
        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise RuntimeError("GaudiRT FX: fused RMSNorm epsilon must be positive and finite")
        if not isinstance(weight_offset, (int, float)) or isinstance(weight_offset, bool) or not math.isfinite(weight_offset):
            raise RuntimeError("GaudiRT FX: fused RMSNorm weight_offset must be finite")
        producer = len(self.tensor_indices) + len(self.operations)
        self.operations.append(Node.fused_norm(
            self.tensor_id(node.args[0]), self.tensor_id(node.args[1]), self.tensor_id(node.args[2]),
            _finite_float32(epsilon), _finite_float32(float(weight_offset)),
        ))
        self.fused_norms[node] = [
            producer, None, self.shape_of(node.args[0]),
            self.dtype_of(node.args[0]), self.is_dyn_of(node.args[0]),
        ]

    def lower_getitem(self, node):
        if len(node.args) == 2 and not node.kwargs and isinstance(node.args[1], (tuple, slice)):
            source, raw_index = node.args
            index = (raw_index,) if isinstance(raw_index, slice) else raw_index
            src_shape = self.shape_of(source)
            if src_shape is None or len(index) > len(src_shape):
                raise RuntimeError("GaudiRT FX: invalid tensor slice tuple")
            slice_dim = slice_obj = None
            for d, item in enumerate(index):
                if item == slice(None):
                    continue
                if slice_dim is not None or not isinstance(item, slice):
                    raise RuntimeError("GaudiRT FX: only single-axis full slices are supported")
                slice_dim, slice_obj = d, item
            if slice_dim is None:
                self.ids[node] = self.tensor_id(source, consume=False)
                self.record_derived(node, src_shape, self.dtype_of(source), self.is_dyn_of(source))
                return
            self._apply_slice(node, source, slice_dim, slice_obj)
            return
        if len(node.args) == 2 and node.args[0] in self.rotations:
            entry = self.rotations[node.args[0]]
            producer, key_output, q_sh, k_sh, dt, dyn = entry
            index = node.args[1]
            if node.kwargs or index not in (0, 1):
                raise RuntimeError("GaudiRT FX: invalid rotary output index")
            if index == 0:
                self.ids[node] = producer
                self.record_derived(node, q_sh, dt, dyn)
                return
            if key_output is None:
                key_output = len(self.tensor_indices) + len(self.operations)
                entry[1] = key_output
                self.operations.append(Node.unary(Op.rotary_key, producer))
            self.ids[node] = key_output
            self.record_derived(node, k_sh, dt, dyn)
            return
        if len(node.args) == 2 and node.args[0] in self.fused_norms:
            entry = self.fused_norms[node.args[0]]
            producer, residual, sh, dt, dyn = entry
            index = node.args[1]
            if node.kwargs or index not in (0, 1):
                raise RuntimeError("GaudiRT FX: invalid fused RMSNorm output index")
            if index == 0:
                self.ids[node] = producer
                self.record_derived(node, sh, dt, dyn)
                return
            if residual is not None:
                self.ids[node] = residual
                self.record_derived(node, sh, dt, dyn)
                return
            entry[1] = len(self.tensor_indices) + len(self.operations)
            self.append_op(node, Node.unary(Op.fused_residual, producer), sh, dt, dyn)
            return
        if len(node.args) == 2 and node.args[0] in self.layer_norms:
            producer, sh, dt, dyn = self.layer_norms[node.args[0]]
            if node.kwargs or node.args[1] != 0:
                raise RuntimeError("GaudiRT FX: only normalized output (index 0) of layer_norm is supported")
            self.ids[node] = producer
            self.record_derived(node, sh, dt, dyn)
            return
        if len(node.args) == 2 and node.args[0] in self.splits:
            source, sizes, source_node, wdim, src_sh, dt, dyn = self.splits[node.args[0]]
            index = node.args[1]
            if node.kwargs or not isinstance(index, int) or index < 0 or index >= len(sizes):
                raise RuntimeError("GaudiRT FX: invalid split output index")
            out_sh = list(src_sh)
            out_sh[wdim] = sizes[index]
            out_sh = tuple(out_sh)
            if len(sizes) == 1:
                self.ids[node] = source
                self.record_derived(node, out_sh, dt, dyn)
                if source_node in self.alias_views:
                    self.alias_views[node] = self.alias_views[source_node]
                return
            start = sum(sizes[:index])
            self.mark_consumed(source)
            if len(src_sh) == 2 and wdim == 1:
                self.add_alias(node, source_node, ("feature", start, sizes[index]))
                self.append_op(node, Node.slice(source, start, sizes[index], sum(sizes)), out_sh, dt, dyn)
            else:
                self.append_op(node, _node_ctor("slice_dim")(source, wdim, start, sizes[index]), out_sh, dt, dyn)
            return
        if len(node.args) == 2 and not node.kwargs and isinstance(node.args[1], torch.fx.Node):
            source, index = node.args
            index_id = self.tensor_id(index)
            if index_id >= len(self.tensor_indices) or self.roles[index_id]:
                raise RuntimeError("GaudiRT FX: row gather requires external validated indices")
            self.validated_row_bindings.append(index_id)
            src_sh = self.shape_of(source)
            out_sh = (-1, *src_sh[1:]) if src_sh else None
            self.append_op(node, Node.binary(Op.gather_rows, self.tensor_id(source), index_id),
                           out_sh, self.dtype_of(source), True)
            return
        raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")

    def lower_index_select(self, node):
        if len(node.args) != 3 or node.args[1] != 0 or node.kwargs:
            raise RuntimeError("GaudiRT FX: only outer-dimension index_select is supported")
        source, index = node.args[0], node.args[2]
        index_id = self.tensor_id(index)
        if index_id >= len(self.tensor_indices) or self.roles[index_id]:
            raise RuntimeError("GaudiRT FX: row gather requires external validated indices")
        self.validated_row_bindings.append(index_id)
        src_sh = self.shape_of(source)
        out_sh = (-1, *src_sh[1:]) if src_sh else None
        self.append_op(node, Node.binary(Op.gather_rows, self.tensor_id(source), index_id),
                       out_sh, self.dtype_of(source), True)

    def lower_quant_linear(self, node, is_block):
        if len(node.args) not in (3, 4):
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        label = "block FP8" if is_block else "FP8"
        if set(node.kwargs) - {"bias"} or (len(node.args) == 4 and "bias" in node.kwargs):
            raise RuntimeError(f"GaudiRT FX: unsupported {label} linear arguments")
        bias = node.args[3] if len(node.args) == 4 else node.kwargs.get("bias")
        ctor = Node.block_fp8 if is_block else Node.fp8
        x_sh, w_sh = self.shape_of(node.args[0]), self.shape_of(node.args[1])
        out_cols = (w_sh[1] if len(w_sh) == 3 else w_sh[0]) if (is_block and w_sh) else (w_sh[0] if w_sh else None)
        out_sh = (*x_sh[:-1], out_cols) if x_sh and out_cols is not None else None
        self.append_op(node, ctor(*(self.tensor_id(value) for value in node.args[:3]),
                                  None if bias is None else self.tensor_id(bias)),
                       out_sh, self.dtype_of(node.args[0]), self.is_dyn_of(node.args[0]))

    def lower_linear(self, node):
        if len(node.args) not in (2, 3):
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        if set(node.kwargs) - {"bias"} or (len(node.args) == 3 and "bias" in node.kwargs):
            raise RuntimeError("GaudiRT FX: unsupported linear arguments")
        bias = node.args[2] if len(node.args) == 3 else node.kwargs.get("bias")
        x_sh, w_sh = self.shape_of(node.args[0]), self.shape_of(node.args[1])
        out_sh = (*x_sh[:-1], w_sh[0]) if x_sh and w_sh else None
        self.append_op(node, Node.linear(self.tensor_id(node.args[0]), self.tensor_id(node.args[1]),
                                         None if bias is None else self.tensor_id(bias), True),
                       out_sh, self.dtype_of(node.args[0]), self.is_dyn_of(node.args[0]))

    def lower_bmm(self, node):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        l_sh, r_sh = self.shape_of(node.args[0]), self.shape_of(node.args[1])
        out_sh = (*l_sh[:-1], r_sh[-1]) if l_sh and r_sh else None
        self.append_op(node, Node.binary(Op.bmm, self.tensor_id(node.args[0]), self.tensor_id(node.args[1])),
                       out_sh, self.dtype_of(node.args[0]), self.is_dyn_of(node.args[0]))

    def lower_mm(self, node):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        l_sh, r_sh = self.shape_of(node.args[0]), self.shape_of(node.args[1])
        if l_sh is not None and r_sh is not None and len(l_sh) == len(r_sh) and 3 <= len(l_sh) <= 5:
            out_sh = (*l_sh[:-1], r_sh[-1])
            self.append_op(node, Node.binary(Op.bmm, self.tensor_id(node.args[0]), self.tensor_id(node.args[1])),
                           out_sh, self.dtype_of(node.args[0]), self.is_dyn_of(node.args[0]))
        elif node.args[1] in self.transposes:
            w = self.transposes[node.args[1]]
            w_sh = self.shape_of(w)
            out_sh = (*l_sh[:-1], w_sh[0]) if l_sh and w_sh else None
            self.append_op(node, Node.linear(self.tensor_id(node.args[0]), self.tensor_id(w), None, True),
                           out_sh, self.dtype_of(node.args[0]), self.is_dyn_of(node.args[0]))
        else:
            out_sh = (*l_sh[:-1], r_sh[1]) if l_sh and r_sh else None
            self.append_op(node, Node.linear(self.tensor_id(node.args[0]), self.tensor_id(node.args[1]), None, False),
                           out_sh, self.dtype_of(node.args[0]), self.is_dyn_of(node.args[0]))

    def lower_addmm(self, node):
        if len(node.args) != 3:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        if set(node.kwargs) - {"alpha", "beta"} or any(
            not isinstance(value, (int, float)) or value != 1 for value in node.kwargs.values()
        ):
            raise RuntimeError("GaudiRT FX: addmm requires alpha=beta=1")
        l_sh = self.shape_of(node.args[1])
        if node.args[2] in self.transposes:
            w = self.transposes[node.args[2]]
            w_sh = self.shape_of(w)
            out_sh = (*l_sh[:-1], w_sh[0]) if l_sh and w_sh else None
            self.append_op(node, Node.linear(self.tensor_id(node.args[1]), self.tensor_id(w),
                                             self.tensor_id(node.args[0]), True),
                           out_sh, self.dtype_of(node.args[1]), self.is_dyn_of(node.args[1]))
        else:
            r_sh = self.shape_of(node.args[2])
            out_sh = (*l_sh[:-1], r_sh[1]) if l_sh and r_sh else None
            self.append_op(node, Node.linear(self.tensor_id(node.args[1]), self.tensor_id(node.args[2]),
                                             self.tensor_id(node.args[0]), False),
                           out_sh, self.dtype_of(node.args[1]), self.is_dyn_of(node.args[1]))

    def lower_add_or_sub(self, node, tensor_op, scalar_op, label):
        if len(node.args) != 2:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        if set(node.kwargs) - {"alpha"} or node.kwargs.get("alpha", 1) != 1:
            raise RuntimeError(f"GaudiRT FX: {label} requires alpha=1")
        out_sh, out_dt = self._binary_meta(node.args)
        self.append_op(node, self.binary_operation(node.args, tensor_op, scalar_op), out_sh, out_dt)

    def lower_mul_or_div(self, node, tensor_op, scalar_op, is_div=False):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
        out_sh, out_dt = self._binary_meta(node.args)
        if is_div and out_dt in (torch.int32, torch.int64, torch.bool):
            out_dt = torch.float32
        self.append_op(node, self.binary_operation(node.args, tensor_op, scalar_op), out_sh, out_dt)

    def lower_cmp(self, node, op_name, scalar_name):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported comparison {node.target}")
        out_sh, _ = self._binary_meta(node.args, force_dtype=torch.bool)
        self.append_op(node, self.binary_operation(node.args, _op(op_name), _op(scalar_name)),
                       out_sh, torch.bool)

    def lower_binary_op(self, node, op_name, force_dtype=None):
        if len(node.args) != 2 or node.kwargs:
            raise RuntimeError(f"GaudiRT FX: unsupported binary operation {node.target}")
        a, b = node.args
        sa, sb = _resolve_scalar_value(a), _resolve_scalar_value(b)
        if op_name in ("minimum", "maximum") and (sa is not None or sb is not None):
            if sa is not None and sb is not None:
                raise RuntimeError("GaudiRT FX: minimum/maximum needs a tensor input")
            t, sc = (b, float(sa)) if sa is not None else (a, float(sb))
            lo, hi = (sc, float("inf")) if op_name == "maximum" else (float("-inf"), sc)
            self.append_op(node, _node_ctor("ewise_unary")(self.tensor_id(t), 7, lo, hi),
                           self.shape_of(t), self.dtype_of(t), self.is_dyn_of(t))
            return
        out_sh, out_dt = self._binary_meta(node.args, force_dtype=force_dtype)
        self.append_op(node, Node.binary(_op(op_name), self.tensor_id(a), self.tensor_id(b)), out_sh, out_dt)

    def lower_full_like_scalar(self, ref_node, value, dtype):
        is_int = dtype in (torch.int32, torch.int64)
        fval = 0.0 if is_int else float(value)
        ival = int(value) if is_int else (1 if (dtype == torch.bool and bool(value)) else 0)
        op = _node_ctor("full_like")(
            self.tensor_id(ref_node), fval, ival, _scalar_type_code(dtype), is_int or dtype == torch.bool
        )
        new_id = len(self.tensor_indices) + len(self.operations)
        self.operations.append(op)
        return new_id

    def lower_where(self, node):
        if len(node.args) != 3 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported where arguments")
        cond, x, y = node.args
        sx, sy = _resolve_scalar_value(x), _resolve_scalar_value(y)
        if isinstance(x, torch.fx.Node):
            ref, dtype = x, self.dtype_of(x)
        elif isinstance(y, torch.fx.Node):
            ref, dtype = y, self.dtype_of(y)
        else:
            ref, dtype = cond, torch.float32
        if isinstance(x, torch.fx.Node) and isinstance(y, torch.fx.Node):
            dtype = torch.promote_types(self.dtype_of(x), self.dtype_of(y))
        x_id = self.lower_full_like_scalar(ref, sx, dtype) if sx is not None else self.tensor_id(x)
        y_id = self.lower_full_like_scalar(ref, sy, dtype) if sy is not None else self.tensor_id(y)
        out_sh = _broadcast_shape(
            _broadcast_shape(self.shape_of(cond), self.shape_of(x) if isinstance(x, torch.fx.Node) else ()),
            self.shape_of(y) if isinstance(y, torch.fx.Node) else (),
        )
        self.append_op(node, Node.ternary(_op("where"), self.tensor_id(cond), x_id, y_id), out_sh, dtype)

    def lower_full_like(self, node):
        if len(node.args) < 2:
            raise RuntimeError("GaudiRT FX: unsupported full_like arguments")
        ref = node.args[0]
        fill_val = _resolve_scalar_value(node.args[1])
        if fill_val is None:
            raise RuntimeError("GaudiRT FX: full_like requires a constant scalar fill value")
        dtype = node.kwargs.get("dtype") or self.dtype_of(ref)
        is_int = dtype in (torch.int32, torch.int64)
        fval = 0.0 if is_int else float(fill_val)
        ival = int(fill_val) if is_int else (1 if (dtype == torch.bool and bool(fill_val)) else 0)
        self.append_op(
            node,
            _node_ctor("full_like")(self.tensor_id(ref), fval, ival, _scalar_type_code(dtype),
                                    is_int or dtype == torch.bool),
            self.shape_of(ref), dtype, self.is_dyn_of(ref),
        )

    def lower_unary_ewise(self, node, op_code, alpha=0.0, beta=0.0):
        src = node.args[0]
        self.append_op(node, _node_ctor("ewise_unary")(self.tensor_id(src), op_code, alpha, beta),
                       self.shape_of(src), self.dtype_of(src), self.is_dyn_of(src))

    def lower_gelu(self, node):
        if len(node.args) < 1:
            raise RuntimeError("GaudiRT FX: unsupported gelu arguments")
        approx = node.args[1] if len(node.args) > 1 else node.kwargs.get("approximate", "none")
        if approx not in ("none", "tanh"):
            raise RuntimeError(f"GaudiRT FX: invalid gelu approximate={approx!r}")
        self.lower_unary_ewise(node, 5 if approx == "tanh" else 4)

    def lower_clamp(self, node):
        if not 1 <= len(node.args) <= 3:
            raise RuntimeError("GaudiRT FX: unsupported clamp arguments")
        min_v = node.args[1] if len(node.args) > 1 else node.kwargs.get("min")
        max_v = node.args[2] if len(node.args) > 2 else node.kwargs.get("max")
        lo = float(min_v) if min_v is not None else float("-inf")
        hi = float(max_v) if max_v is not None else float("inf")
        self.lower_unary_ewise(node, 7, min(lo, hi), hi)

    def lower_clamp_min(self, node):
        lo = float(node.args[1] if len(node.args) > 1 else node.kwargs["min"])
        self.lower_unary_ewise(node, 7, lo, float("inf"))

    def lower_clamp_max(self, node):
        hi = float(node.args[1] if len(node.args) > 1 else node.kwargs["max"])
        self.lower_unary_ewise(node, 7, float("-inf"), hi)

    def lower_softcap(self, node):
        self.lower_unary_ewise(node, 6, _finite_float32(float(node.args[1])), 0.0)

    def lower_rms_norm(self, node):
        if len(node.args) not in (3, 4) or set(node.kwargs) - {"weight_offset"} or (
            len(node.args) == 4 and "weight_offset" in node.kwargs
        ):
            raise RuntimeError("GaudiRT FX: unsupported RMSNorm arguments")
        epsilon = node.args[2]
        weight_offset = node.args[3] if len(node.args) == 4 else node.kwargs.get("weight_offset", 0.0)
        if not isinstance(epsilon, (int, float)) or isinstance(epsilon, bool) or not math.isfinite(epsilon) or epsilon <= 0:
            raise RuntimeError("GaudiRT FX: RMSNorm epsilon must be positive and finite")
        if not isinstance(weight_offset, (int, float)) or isinstance(weight_offset, bool) or not math.isfinite(weight_offset):
            raise RuntimeError("GaudiRT FX: RMSNorm weight_offset must be finite")
        self.append_op(
            node,
            Node.norm(self.tensor_id(node.args[0]), self.tensor_id(node.args[1]),
                      _finite_float32(epsilon), _finite_float32(float(weight_offset))),
            self.shape_of(node.args[0]), self.dtype_of(node.args[0]), self.is_dyn_of(node.args[0]),
        )

    def lower_silu_and_mul(self, node):
        if len(node.args) != 1 or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported silu_and_mul arguments")
        sh = self.shape_of(node.args[0])
        out_sh = (*sh[:-1], sh[-1] // 2) if sh else None
        self.append_op(node, Node.unary(Op.silu_and_mul, self.tensor_id(node.args[0])),
                       out_sh, self.dtype_of(node.args[0]), self.is_dyn_of(node.args[0]))

    def lower_gelu_and_mul(self, node):
        if len(node.args) not in (1, 2):
            raise RuntimeError("GaudiRT FX: unsupported gelu_and_mul arguments")
        approx = node.args[1] if len(node.args) == 2 else node.kwargs.get("approximate", "none")
        if approx not in ("none", "tanh"):
            raise RuntimeError(f"GaudiRT FX: invalid gelu_and_mul approximate={approx!r}")
        sh = self.shape_of(node.args[0])
        out_sh = (*sh[:-1], sh[-1] // 2) if sh else None
        self.append_op(node, _node_ctor("gelu_and_mul")(self.tensor_id(node.args[0]), 1 if approx == "tanh" else 0),
                       out_sh, self.dtype_of(node.args[0]), self.is_dyn_of(node.args[0]))

    def lower_reduction(self, node, op_name):
        aten = torch.ops.aten
        src = node.args[0]
        src_shape = self.shape_of(src)
        src_dtype = self.dtype_of(src)
        src_dyn = self.is_dyn_of(src)
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
            self.ids[node] = self.tensor_id(src)
            self.record_derived(node, src_shape, src_dtype, src_dyn)
            return
        if src_dyn and 0 in wdims:
            raise RuntimeError("GaudiRT FX: cannot reduce along dynamic outer token axis")
        cur_id = self.tensor_id(src)
        cur_shape = list(src_shape)
        cur_dtype = dtype or src_dtype
        if op_name == "sum_dim" and cur_dtype == torch.bool:
            cur_dtype = torch.int32
        if cur_dtype != src_dtype:
            cur_id_new = len(self.tensor_indices) + len(self.operations)
            self.operations.append(_node_ctor("cast_dtype")(cur_id, _scalar_type_code(cur_dtype)))
            cur_id = cur_id_new
        reduce_ctor = _node_ctor("reduce")
        op_enum = _op(op_name)
        for d in wdims:
            if keepdim:
                cur_shape[d] = 1
            else:
                cur_shape.pop(d)
            out_id = len(self.tensor_indices) + len(self.operations)
            self.operations.append(reduce_ctor(op_enum, cur_id, d, keepdim))
            cur_id = out_id
        self.ids[node] = cur_id
        self.record_derived(node, tuple(cur_shape), cur_dtype, src_dyn)

    def lower_softmax(self, node, is_log):
        aten = torch.ops.aten
        src = node.args[0]
        src_shape = self.shape_of(src)
        src_dtype = self.dtype_of(src)
        src_dyn = self.is_dyn_of(src)
        if src_shape is None:
            raise RuntimeError("GaudiRT FX: softmax requires known input shape")
        dim = node.args[1] if len(node.args) > 1 else node.kwargs["dim"]
        wdim = int(dim) % len(src_shape)
        if src_dyn and wdim == 0:
            raise RuntimeError("GaudiRT FX: softmax along dynamic outer token axis is unsupported")
        if node.target in (aten._softmax.default, aten._log_softmax.default):
            half_to_float = bool(node.args[2] if len(node.args) > 2 else node.kwargs.get("half_to_float", False))
            target_dtype = torch.float32 if half_to_float else None
        else:
            target_dtype = node.args[2] if len(node.args) > 2 else node.kwargs.get("dtype")
        cur_id = self.tensor_id(src)
        cur_dtype = src_dtype
        if target_dtype is not None and target_dtype != src_dtype:
            cur_id_new = len(self.tensor_indices) + len(self.operations)
            self.operations.append(Node.cast(cur_id, target_dtype == torch.float32))
            cur_id = cur_id_new
            cur_dtype = target_dtype
        op_enum = _op("log_softmax" if is_log else "softmax")
        self.append_op(node, _node_ctor("softmax")(op_enum, cur_id, wdim),
                       src_shape, cur_dtype, src_dyn)

    def lower_layer_norm(self, node, is_native):
        src = node.args[0]
        norm_shape = tuple(int(s) for s in (node.args[1] if len(node.args) > 1 else node.kwargs["normalized_shape"]))
        weight = node.args[2] if len(node.args) > 2 else node.kwargs.get("weight")
        bias = node.args[3] if len(node.args) > 3 else node.kwargs.get("bias")
        eps = float(node.args[4] if len(node.args) > 4 else node.kwargs.get("eps", 1e-5))
        src_shape = self.shape_of(src)
        src_dtype = self.dtype_of(src)
        src_dyn = self.is_dyn_of(src)
        if src_shape is None or len(norm_shape) > len(src_shape):
            raise RuntimeError("GaudiRT FX: invalid layer_norm normalized_shape")
        if len(norm_shape) != 1 or src_shape[-1] != norm_shape[0]:
            raise RuntimeError("GaudiRT FX: layer_norm normalized_shape must match the trailing dimension")
        producer = len(self.tensor_indices) + len(self.operations)
        self.operations.append(_node_ctor("layer_norm")(
            self.tensor_id(src),
            None if weight is None else self.tensor_id(weight),
            None if bias is None else self.tensor_id(bias),
            _finite_float32(eps),
        ))
        if is_native:
            self.layer_norms[node] = (producer, src_shape, src_dtype, src_dyn)
        else:
            self.ids[node] = producer
            self.record_derived(node, src_shape, src_dtype, src_dyn)

    def lower_type_as(self, node):
        src, peer = node.args[0], node.args[1]
        if self.dtype_of(src) == self.dtype_of(peer):
            self.ids[node] = self.tensor_id(src, consume=False)
            self.record_derived(node, self.shape_of(src), self.dtype_of(src), self.is_dyn_of(src))
        else:
            self.append_op(
                node,
                _node_ctor("cast_dtype")(self.tensor_id(src), _scalar_type_code(self.dtype_of(peer))),
                self.shape_of(src), self.dtype_of(peer), self.is_dyn_of(src),
            )

    def _build_target_handlers(self):
        aten = torch.ops.aten
        g = self.groups
        handlers = {
            aten._to_copy.default: self.lower_cast_or_copy,
            aten.to.dtype: self.lower_cast_or_copy,
            aten.to.dtype_layout: self.lower_cast_or_copy,
            aten.slice.Tensor: self.lower_aten_slice,
            _fx_shard_embedding: self.lower_shard_emb,
            _fx_softcap: self.lower_softcap,
            operator.getitem: self.lower_getitem,
        }
        if hasattr(aten, "type_as"):
            handlers[aten.type_as.default] = self.lower_type_as

        direct_map = (
            ("transpose", self.lower_transpose),
            ("unsqueeze", self.lower_unsqueeze),
            ("squeeze", self.lower_squeeze),
            ("expand", self.lower_expand),
            ("split", self.lower_split),
            ("cat", self.lower_cat),
            ("embedding", self.lower_embedding),
            ("rotary", self.lower_rotary),
            ("rotary_pos", self.lower_rotary_pos),
            ("fused_norm", self.lower_fused_norm),
            ("index_select", self.lower_index_select),
            ("linear", self.lower_linear),
            ("bmm", self.lower_bmm),
            ("mm", self.lower_mm),
            ("addmm", self.lower_addmm),
            ("norm", self.lower_rms_norm),
            ("silu_mul", self.lower_silu_and_mul),
            ("gelu_mul", self.lower_gelu_and_mul),
            ("gelu", self.lower_gelu),
            ("clamp", self.lower_clamp),
            ("clamp_min", self.lower_clamp_min),
            ("clamp_max", self.lower_clamp_max),
            ("where", self.lower_where),
            ("full_like", self.lower_full_like),
        )
        for key, fn in direct_map:
            for t in g[key]:
                handlers[t] = fn

        param_map = (
            ("block_fp8", lambda n: self.lower_quant_linear(n, True)),
            ("fp8", lambda n: self.lower_quant_linear(n, False)),
            ("add", lambda n: self.lower_add_or_sub(n, Op.add, Op.scalar_add, "add")),
            ("sub", lambda n: self.lower_add_or_sub(n, Op.sub, Op.scalar_sub, "sub")),
            ("mul", lambda n: self.lower_mul_or_div(n, Op.mul, Op.scalar_mul)),
            ("div", lambda n: self.lower_mul_or_div(n, Op.div, Op.scalar_div, True)),
            ("exp", lambda n: self.lower_unary_ewise(n, 0)),
            ("tanh", lambda n: self.lower_unary_ewise(n, 1)),
            ("sigmoid", lambda n: self.lower_unary_ewise(n, 2)),
            ("silu", lambda n: self.lower_unary_ewise(n, 3)),
            ("rsqrt", lambda n: self.lower_unary_ewise(n, 8)),
            ("sqrt", lambda n: self.lower_unary_ewise(n, 9)),
            ("neg", lambda n: self.lower_unary_ewise(n, 10)),
            ("abs", lambda n: self.lower_unary_ewise(n, 11)),
            ("isnan", lambda n: self.append_op(n, Node.unary(_op("isnan"), self.tensor_id(n.args[0])),
                                               self.shape_of(n.args[0]), torch.bool, self.is_dyn_of(n.args[0]))),
            ("logical_not", lambda n: self.append_op(n, Node.unary(_op("logical_not"), self.tensor_id(n.args[0])),
                                                     self.shape_of(n.args[0]), torch.bool, self.is_dyn_of(n.args[0]))),
            ("min", lambda n: self.lower_binary_op(n, "minimum")),
            ("max", lambda n: self.lower_binary_op(n, "maximum")),
            ("logical_and", lambda n: self.lower_binary_op(n, "logical_and", torch.bool)),
            ("logical_or", lambda n: self.lower_binary_op(n, "logical_or", torch.bool)),
            ("bitwise_and", lambda n: self.lower_binary_op(n, "bitwise_and")),
            ("bitwise_or", lambda n: self.lower_binary_op(n, "bitwise_or")),
            ("eq", lambda n: self.lower_cmp(n, "eq", "scalar_eq")),
            ("ne", lambda n: self.lower_cmp(n, "ne", "scalar_ne")),
            ("lt", lambda n: self.lower_cmp(n, "lt", "scalar_lt")),
            ("le", lambda n: self.lower_cmp(n, "le", "scalar_le")),
            ("gt", lambda n: self.lower_cmp(n, "gt", "scalar_gt")),
            ("ge", lambda n: self.lower_cmp(n, "ge", "scalar_ge")),
            ("sum", lambda n: self.lower_reduction(n, "sum_dim")),
            ("mean", lambda n: self.lower_reduction(n, "mean_dim")),
            ("amax", lambda n: self.lower_reduction(n, "amax_dim")),
            ("softmax", lambda n: self.lower_softmax(n, False)),
            ("log_softmax", lambda n: self.lower_softmax(n, True)),
            ("layer_norm", lambda n: self.lower_layer_norm(n, False)),
            ("native_layer_norm", lambda n: self.lower_layer_norm(n, True)),
        )
        for key, fn in param_map:
            for t in g[key]:
                handlers[t] = fn
        return handlers

    def _bind_placeholders(self):
        placeholders = tuple(node for node in self.nodes if node.op == "placeholder")
        if len(placeholders) != len(self.example_inputs):
            raise RuntimeError("GaudiRT FX: placeholder count mismatch")
        placeholder_tensors = []
        passthrough_placeholders = []
        for index, (node, value) in enumerate(zip(placeholders, self.example_inputs)):
            if not isinstance(value, torch.Tensor):
                continue
            dynamic = getattr(value, "_dynamo_dynamic_indices", set()) or set()
            if dynamic and dynamic != {0}:
                raise RuntimeError("GaudiRT FX: only the outer token dimension may be dynamic")
            role = int(node.meta.get("gaudirt_dynamic", bool(dynamic)))
            if not node.users or (self._active_tensor_args is not None and index not in self._active_tensor_args):
                passthrough_placeholders.append((index, node, role, value))
            else:
                placeholder_tensors.append((index, node, role, value))
        placeholder_tensors.sort(key=lambda item: not item[2])
        for index, node, role, value in placeholder_tensors:
            self.ids[node] = len(self.tensor_indices)
            self.tensor_indices.append(index)
            self.roles.append(role)
            ex = _node_example(node)
            ref = ex if ex is not None else value
            shape = tuple(ref.shape)
            if role:
                if self.dynamic_dim0_token is not None and self.dynamic_dim0_token != str(shape[0]):
                    raise RuntimeError("GaudiRT FX: single BucketProgram segment supports only one dynamic outer extent")
                self.dynamic_dim0_token = str(shape[0])
                self.node_shapes[node] = (-1, *(int(s) for s in shape[1:]))
            else:
                self.node_shapes[node] = tuple(int(s) for s in shape)
            self.node_dtypes[node] = ref.dtype
            self.node_dynamic[node] = bool(role)
        for index, node, role, value in passthrough_placeholders:
            sentinel = -1 - index
            self.ids[node] = sentinel
            self.passthrough_args[sentinel] = index
            ex = _node_example(node)
            ref = ex if ex is not None else value
            shape = tuple(ref.shape)
            if role:
                if self.dynamic_dim0_token is None:
                    self.dynamic_dim0_token = str(shape[0])
                self.node_shapes[node] = (-1, *(int(s) for s in shape[1:]))
            else:
                self.node_shapes[node] = tuple(int(s) for s in shape)
            self.node_dtypes[node] = ref.dtype
            self.node_dynamic[node] = bool(role)
        if self._active_tensor_args is None and not any(self.roles):
            raise RuntimeError("GaudiRT FX: mark each dynamic token input before capture")

    def build(self):
        self._bind_placeholders()
        method_handlers = {
            "transpose": self.lower_transpose,
            "unsqueeze": self.lower_unsqueeze,
            "squeeze": self.lower_squeeze,
            "expand": self.lower_expand,
            "to": self.lower_cast_or_copy,
            "float": self.lower_cast_or_copy,
            "split": self.lower_split,
        }
        output, single_output = None, False
        for node in self.nodes:
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
                self.transposes[node] = node.args[0]
                continue
            if (node.op == "call_function" and node.target is builtins.getattr and len(node.args) == 2 and
                    node.args[1] == "T" and not node.kwargs):
                self.transposes[node] = node.args[0]
                continue
            if node.op == "call_method" and node.target == "long" and len(node.args) == 1 and not node.kwargs:
                self.index_casts[node] = node.args[0]
                continue
            if self.is_reshape_node(node) and not node.kwargs:
                self.lower_reshape(node)
                continue
            if node.op == "call_method":
                fn = method_handlers.get(node.target)
                if fn is None:
                    raise RuntimeError(f"GaudiRT FX: unsupported program node {node.op}: {node.target}")
                fn(node)
                continue
            if node.op != "call_function":
                raise RuntimeError(f"GaudiRT FX: unsupported program node {node.op}: {node.target}")
            handler = self.target_handlers.get(node.target)
            if handler is None:
                raise RuntimeError(f"GaudiRT FX: unsupported program operation {node.target}")
            handler(node)

        if output is None or any(value not in self.ids for value in output):
            raise RuntimeError("GaudiRT FX: unsupported program output")
        output_bases = [self.alias_views[value][0] if value in self.alias_views else self.ids[value] for value in output]
        output_ids, output_views = [], []
        for index, value in enumerate(output):
            base, views = self.alias_views.get(value, (self.ids[value], ()))
            # Only internal values consumed exclusively by attention may lose the
            # producer's stride. Public outputs and shared views retain aliasing.
            compact = (index in self.compact_outputs and base >= len(self.tensor_indices) and
                       output_bases.count(base) == 1 and len(views) == 1 and
                       views[0][0] == "feature" and self.ids[value] != base)
            output_ids.append(self.ids[value] if compact else base)
            output_views.append(() if compact else views)
        computed_ids = list(dict.fromkeys(value for value in output_ids if value >= len(self.tensor_indices)))
        if self.operations and not computed_ids and any(
            operation.op not in (Op.feature_slice, Op.reshape, Op.transpose) for operation in self.operations
        ):
            raise RuntimeError("GaudiRT FX: program operations have no output")
        if self._active_tensor_args is None and computed_ids and len(self.consumed_inputs) < len(self.tensor_indices):
            active = frozenset(self.tensor_indices[i] for i in self.consumed_inputs)
            return _program_backend(
                self.graph_module, self.example_inputs, self.compact_outputs, _active_tensor_args=active
            )
        program = _C.BucketProgram(self.roles, self.operations, computed_ids) if computed_ids else None
        if program is None and (computed_ids or self.validated_token_bindings):
            raise RuntimeError("GaudiRT FX: unsupported identity graph")
        computed_slots = {value: index for index, value in enumerate(computed_ids)}
        result_routes = tuple(
            (True, self.passthrough_args[value], views)
            if value < 0
            else (
                value < len(self.tensor_indices),
                self.tensor_indices[value] if value < len(self.tensor_indices) else computed_slots[value],
                views,
            )
            for value, views in zip(output_ids, output_views)
        )
        dynamic_argument = self.tensor_indices[self.roles.index(1)] if self.validated_row_bindings else None
        tensor_indices = self.tensor_indices
        validated_token_bindings = self.validated_token_bindings
        validated_row_bindings = self.validated_row_bindings

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


def _program_backend(graph_module, example_inputs, compact_outputs=(), _active_tensor_args=None):
    return _ProgramLoweringBuilder(
        graph_module, example_inputs, compact_outputs, _active_tensor_args
    ).build()
