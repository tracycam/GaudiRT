"""Generic Core ATen canonicalization, constant folding, and shape passes."""

import math
import operator

import torch

from .fx_collectives import _fx_collective


def _fx_softcap(source, cap):
    raise RuntimeError("GaudiRT FX: internal _fx_softcap must be lowered")


def _node_shape_and_dtype(node: torch.fx.Node):
    meta = node.meta.get("tensor_meta")
    if meta is not None and hasattr(meta, "shape"):
        return tuple(int(s) if isinstance(s, int) else s for s in meta.shape), getattr(meta, "dtype", None)
    val = node.meta.get("example_value", node.meta.get("val"))
    if isinstance(val, torch.Tensor):
        return tuple(int(s) if isinstance(s, int) else s for s in val.shape), val.dtype
    return None, None


def _resolve_scalar(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, torch.SymInt):
        return int(value)
    if isinstance(value, torch.SymFloat):
        return float(value)
    return None


def _coerce_scalar_dtype(value, dtype: torch.dtype | None):
    if dtype is None:
        return value
    if dtype == torch.bool:
        return bool(value)
    if dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        return int(value)
    if dtype in (torch.bfloat16, torch.float16):
        return float(torch.tensor(float(value), dtype=dtype).item())
    if dtype.is_floating_point:
        return float(value)
    return value


def _get_attr_value(gm: torch.fx.GraphModule, target: str):
    obj = gm
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def _fold_scalar_tensors(gm: torch.fx.GraphModule) -> None:
    graph = gm.graph
    aten = torch.ops.aten
    const_vals: dict[torch.fx.Node, int | float | bool] = {}

    scalar_ctor_targets = {
        torch.tensor,
        torch.scalar_tensor,
        aten.scalar_tensor.default,
        aten.lift_fresh_copy.default,
        aten.lift_fresh.default,
    }
    if hasattr(aten, "tensor"):
        scalar_ctor_targets.add(aten.tensor.default)
    full_targets = {torch.full, aten.full.default}
    passthrough_targets = {
        torch.clone,
        aten.clone.default,
        aten.detach.default,
        aten.alias.default,
        aten._to_copy.default,
        aten.to.dtype,
        aten.to.dtype_layout,
        aten.to.device,
    }
    unary_scalar_fns = {
        operator.neg: lambda x: -x,
        torch.neg: lambda x: -x,
        aten.neg.default: lambda x: -x,
        operator.abs: abs,
        torch.abs: abs,
        aten.abs.default: abs,
        math.sqrt: math.sqrt,
        torch.sqrt: math.sqrt,
        aten.sqrt.default: math.sqrt,
        torch.rsqrt: lambda x: 1.0 / math.sqrt(float(x)),
        aten.rsqrt.default: lambda x: 1.0 / math.sqrt(float(x)),
        math.exp: math.exp,
        torch.exp: math.exp,
        aten.exp.default: math.exp,
        math.tanh: math.tanh,
        torch.tanh: math.tanh,
        aten.tanh.default: math.tanh,
        torch.sigmoid: lambda x: 1.0 / (1.0 + math.exp(-float(x))),
        aten.sigmoid.default: lambda x: 1.0 / (1.0 + math.exp(-float(x))),
        operator.not_: lambda x: not x,
        torch.logical_not: lambda x: not x,
        aten.logical_not.default: lambda x: not x,
    }
    binary_scalar_fns = {
        operator.add: lambda a, b: a + b,
        torch.add: lambda a, b: a + b,
        aten.add.Tensor: lambda a, b: a + b,
        aten.add.Scalar: lambda a, b: a + b,
        operator.sub: lambda a, b: a - b,
        torch.sub: lambda a, b: a - b,
        aten.sub.Tensor: lambda a, b: a - b,
        aten.sub.Scalar: lambda a, b: a - b,
        operator.mul: lambda a, b: a * b,
        torch.mul: lambda a, b: a * b,
        aten.mul.Tensor: lambda a, b: a * b,
        aten.mul.Scalar: lambda a, b: a * b,
        operator.truediv: lambda a, b: a / b,
        torch.div: lambda a, b: a / b,
        aten.div.Tensor: lambda a, b: a / b,
        aten.div.Scalar: lambda a, b: a / b,
        operator.eq: lambda a, b: a == b,
        torch.eq: lambda a, b: a == b,
        aten.eq.Tensor: lambda a, b: a == b,
        aten.eq.Scalar: lambda a, b: a == b,
        operator.ne: lambda a, b: a != b,
        torch.ne: lambda a, b: a != b,
        aten.ne.Tensor: lambda a, b: a != b,
        aten.ne.Scalar: lambda a, b: a != b,
        operator.lt: lambda a, b: a < b,
        torch.lt: lambda a, b: a < b,
        aten.lt.Tensor: lambda a, b: a < b,
        aten.lt.Scalar: lambda a, b: a < b,
        operator.le: lambda a, b: a <= b,
        torch.le: lambda a, b: a <= b,
        aten.le.Tensor: lambda a, b: a <= b,
        aten.le.Scalar: lambda a, b: a <= b,
        operator.gt: lambda a, b: a > b,
        torch.gt: lambda a, b: a > b,
        aten.gt.Tensor: lambda a, b: a > b,
        aten.gt.Scalar: lambda a, b: a > b,
        operator.ge: lambda a, b: a >= b,
        torch.ge: lambda a, b: a >= b,
        aten.ge.Tensor: lambda a, b: a >= b,
        aten.ge.Scalar: lambda a, b: a >= b,
    }
    if hasattr(aten, "item"):
        unary_scalar_fns[aten.item.default] = lambda x: x

    def _folded(arg):
        if isinstance(arg, torch.fx.Node):
            return const_vals.get(arg)
        return _resolve_scalar(arg)

    for node in graph.nodes:
        if node.op == "get_attr":
            try:
                val = _get_attr_value(gm, str(node.target))
            except AttributeError:
                continue
            if isinstance(val, torch.Tensor) and val.ndim == 0:
                const_vals[node] = _coerce_scalar_dtype(val.item(), val.dtype)
            continue

        if node.op == "call_method":
            if node.target in ("to", "detach", "clone", "contiguous") and len(node.args) >= 1:
                base = _folded(node.args[0])
                if base is not None:
                    _, out_dtype = _node_shape_and_dtype(node)
                    for a in node.args[1:]:
                        if isinstance(a, torch.dtype):
                            out_dtype = a
                    if isinstance(node.kwargs.get("dtype"), torch.dtype):
                        out_dtype = node.kwargs["dtype"]
                    const_vals[node] = _coerce_scalar_dtype(base, out_dtype)
            elif node.target == "item" and len(node.args) == 1:
                base = _folded(node.args[0])
                if base is not None:
                    const_vals[node] = base
            continue

        if node.op != "call_function":
            continue
        tgt = node.target

        if tgt in scalar_ctor_targets and len(node.args) >= 1:
            base = _folded(node.args[0])
            if base is not None:
                dtype = node.kwargs.get("dtype")
                const_vals[node] = _coerce_scalar_dtype(base, dtype if isinstance(dtype, torch.dtype) else None)
            continue

        if tgt in full_targets and len(node.args) >= 2 and isinstance(node.args[0], (list, tuple)) and len(node.args[0]) == 0:
            base = _folded(node.args[1])
            if base is not None:
                dtype = node.kwargs.get("dtype")
                const_vals[node] = _coerce_scalar_dtype(base, dtype if isinstance(dtype, torch.dtype) else None)
            continue

        if tgt in passthrough_targets and len(node.args) >= 1:
            base = _folded(node.args[0])
            if base is not None:
                _, out_dtype = _node_shape_and_dtype(node)
                if isinstance(node.kwargs.get("dtype"), torch.dtype):
                    out_dtype = node.kwargs["dtype"]
                for a in node.args[1:]:
                    if isinstance(a, torch.dtype):
                        out_dtype = a
                const_vals[node] = _coerce_scalar_dtype(base, out_dtype)
            continue

        if tgt in unary_scalar_fns and len(node.args) == 1 and not node.kwargs:
            base = _folded(node.args[0])
            if base is not None:
                const_vals[node] = unary_scalar_fns[tgt](base)
            continue

        if tgt in binary_scalar_fns and len(node.args) == 2 and not node.kwargs:
            lhs = _folded(node.args[0])
            rhs = _folded(node.args[1])
            if lhs is not None and rhs is not None:
                if tgt in (operator.truediv, torch.div, aten.div.Tensor, aten.div.Scalar) and rhs == 0:
                    continue
                const_vals[node] = binary_scalar_fns[tgt](lhs, rhs)
            continue

        if tgt is getattr(aten, "type_as", None) or tgt is getattr(getattr(aten, "type_as", None), "default", None):
            if len(node.args) == 2:
                base = _folded(node.args[0])
                if base is not None and isinstance(node.args[1], torch.fx.Node):
                    _, peer_dtype = _node_shape_and_dtype(node.args[1])
                    const_vals[node] = _coerce_scalar_dtype(base, peer_dtype)

    if not const_vals:
        return

    scalar_binary_targets = set(binary_scalar_fns.keys()) | {
        torch.minimum, aten.minimum.default,
        torch.maximum, aten.maximum.default,
    }
    for node in list(graph.nodes):
        if node in const_vals:
            continue
        if node.op == "call_function" and node.target in scalar_binary_targets and len(node.args) == 2:
            new_args = [const_vals[a] if isinstance(a, torch.fx.Node) and a in const_vals else a for a in node.args]
            if new_args != list(node.args):
                node.args = tuple(new_args)
        elif node.op == "call_function" and node.target in (
            torch.clamp, aten.clamp.default, aten.clamp_min.default, aten.clamp_max.default
        ):
            new_args = [const_vals[a] if isinstance(a, torch.fx.Node) and a in const_vals else a for a in node.args]
            new_kwargs = {
                k: (const_vals[v] if isinstance(v, torch.fx.Node) and v in const_vals else v)
                for k, v in node.kwargs.items()
            }
            if new_args != list(node.args) or new_kwargs != dict(node.kwargs):
                node.args = tuple(new_args)
                node.kwargs = new_kwargs
        elif node.op == "call_function" and node.target in (
            aten.masked_fill.Scalar, aten.masked_fill_.Scalar,
            aten.masked_fill.Tensor, aten.masked_fill_.Tensor,
        ):
            if len(node.args) == 3 and isinstance(node.args[2], torch.fx.Node) and node.args[2] in const_vals:
                node.args = (node.args[0], node.args[1], const_vals[node.args[2]])
        elif node.op == "call_method" and node.target in ("masked_fill", "masked_fill_"):
            if len(node.args) == 3 and isinstance(node.args[2], torch.fx.Node) and node.args[2] in const_vals:
                node.args = (node.args[0], node.args[1], const_vals[node.args[2]])
        elif node.op == "call_function" and node.target in (torch.where, aten.where.self) and len(node.args) == 3:
            c, x, y = node.args
            nx = const_vals[x] if isinstance(x, torch.fx.Node) and x in const_vals else x
            ny = const_vals[y] if isinstance(y, torch.fx.Node) and y in const_vals else y
            if nx is not x or ny is not y:
                node.args = (c, nx, ny)


def _functionalize_inplace_arith(graph: torch.fx.Graph) -> None:
    aten = torch.ops.aten
    inplace_fn_map = {
        operator.iadd: operator.add,
        operator.isub: operator.sub,
        operator.imul: operator.mul,
        operator.itruediv: operator.truediv,
        aten.add_.Tensor: operator.add,
        aten.add_.Scalar: operator.add,
        aten.sub_.Tensor: operator.sub,
        aten.sub_.Scalar: operator.sub,
        aten.mul_.Tensor: operator.mul,
        aten.mul_.Scalar: operator.mul,
        aten.div_.Tensor: operator.truediv,
        aten.div_.Scalar: operator.truediv,
    }
    inplace_method_map = {
        "add_": operator.add,
        "sub_": operator.sub,
        "mul_": operator.mul,
        "div_": operator.truediv,
    }
    ordered = list(graph.nodes)
    pos = {n: i for i, n in enumerate(ordered)}
    for node in ordered:
        if node.op == "call_function" and node.target in inplace_fn_map:
            out_op = inplace_fn_map[node.target]
        elif node.op == "call_method" and node.target in inplace_method_map:
            out_op = inplace_method_map[node.target]
        else:
            continue
        if len(node.args) != 2:
            raise RuntimeError("GaudiRT FX: unsupported in-place arithmetic arguments")
        if node.kwargs:
            if set(node.kwargs) != {"alpha"} or node.kwargs.get("alpha") != 1:
                continue
        x, rhs = node.args
        if not isinstance(x, torch.fx.Node):
            raise RuntimeError("GaudiRT FX: in-place arithmetic target must be a tensor node")
        if x.op == "placeholder":
            raise RuntimeError("GaudiRT FX: in-place arithmetic on a graph placeholder input is unsupported")
        pivot = pos[node]
        with graph.inserting_before(node):
            rep = graph.call_function(out_op, args=(x, rhs))
        rep.meta.update(x.meta)
        rep.meta.update(node.meta)
        pos[rep] = pivot
        for user in list(x.users):
            if user is not rep and pos.get(user, -1) > pivot:
                user.replace_input_with(x, rep)
        if node.users:
            node.replace_all_uses_with(rep)
        graph.erase_node(node)


def _outer_dyn_tokens(graph: torch.fx.Graph) -> set[str]:
    tokens: set[str] = set()
    for n in graph.nodes:
        if n.op == "placeholder":
            ex = n.meta.get("example_value", n.meta.get("val"))
            if isinstance(ex, torch.Tensor) and ex.ndim >= 1 and isinstance(ex.shape[0], torch.SymInt):
                tokens.add(str(ex.shape[0]))
    return tokens


def _is_concrete_positive_int(d) -> bool:
    return isinstance(d, int) and not isinstance(d, (bool, torch.SymInt)) and d > 0


def _unfold_head_flattening_reshapes(graph: torch.fx.Graph) -> None:
    aten = torch.ops.aten
    reshape_targets = {torch.reshape, aten.view.default, aten.reshape.default}
    outer_tokens = _outer_dyn_tokens(graph)
    if not outer_tokens:
        return

    norm_targets = set()
    pkt = getattr(torch.ops.gaudirt, "rms_norm", None)
    if pkt is not None:
        norm_targets.add(pkt)
        if hasattr(pkt, "default"):
            norm_targets.add(pkt.default)
    ln_targets = {torch.nn.functional.layer_norm, aten.layer_norm.default}
    unary_fn_targets = {
        _fx_softcap,
        aten.clamp.default, aten.clamp_min.default, aten.clamp_max.default, torch.clamp,
        aten.exp.default, aten.tanh.default, aten.sigmoid.default,
        aten.silu.default, aten.gelu.default, aten.rsqrt.default, aten.sqrt.default,
        aten.neg.default, aten.abs.default, aten._to_copy.default, aten.to.dtype,
        torch.exp, torch.tanh, torch.sigmoid, torch.rsqrt, torch.sqrt, torch.neg, torch.abs,
        torch.nn.functional.silu, torch.nn.functional.gelu,
        operator.neg, operator.abs,
    }
    binary_fn_targets = {
        operator.add, operator.sub, operator.mul, operator.truediv,
        torch.add, torch.sub, torch.mul, torch.div,
        aten.add.Tensor, aten.add.Scalar,
        aten.sub.Tensor, aten.sub.Scalar,
        aten.mul.Tensor, aten.mul.Scalar,
        aten.div.Tensor, aten.div.Scalar,
    }

    def _uses_region(node: torch.fx.Node, region: set[torch.fx.Node]) -> bool:
        for a in node.args:
            if isinstance(a, torch.fx.Node) and a in region:
                return True
            if isinstance(a, (list, tuple)) and any(isinstance(x, torch.fx.Node) and x in region for x in a):
                return True
        for v in node.kwargs.values():
            if isinstance(v, torch.fx.Node) and v in region:
                return True
        return False

    ordered = list(graph.nodes)
    for r_idx, r_in in enumerate(ordered):
        if r_in.graph is None or r_in.op != "call_function" or r_in.target not in reshape_targets or len(r_in.args) != 2:
            continue
        src = r_in.args[0]
        if not isinstance(src, torch.fx.Node):
            continue
        src_ex = src.meta.get("example_value", src.meta.get("val"))
        ex_in = r_in.meta.get("example_value", r_in.meta.get("val"))
        if not (isinstance(src_ex, torch.Tensor) and isinstance(ex_in, torch.Tensor)):
            continue
        if src_ex.ndim < 2 or ex_in.ndim < 2:
            continue
        if str(src_ex.shape[0]) not in outer_tokens or str(ex_in.shape[0]) == str(src_ex.shape[0]):
            continue
        if not all(_is_concrete_positive_int(d) for d in src_ex.shape[1:]):
            continue
        if not all(_is_concrete_positive_int(d) for d in ex_in.shape[1:]):
            continue
        width = math.prod(int(d) for d in src_ex.shape[1:])
        sub_dims = [int(d) for d in ex_in.shape[1:]]
        sub_w = math.prod(sub_dims)
        if not (width > sub_w and width % sub_w == 0):
            continue
        heads = width // sub_w
        if heads <= 1 or 2 + len(sub_dims) > 5 or not ex_in.is_contiguous():
            continue

        region = {r_in}
        interior = []
        valid = True
        has_closing = False
        for node in ordered[r_idx + 1:]:
            if node.graph is None or not _uses_region(node, region):
                continue
            if node.op == "call_function" and node.target in reshape_targets and len(node.args) == 2 and node.args[0] in region:
                ex_u = node.meta.get("example_value", node.meta.get("val"))
                if (isinstance(ex_u, torch.Tensor) and ex_u.ndim >= 2 and
                        str(ex_u.shape[0]) == str(src_ex.shape[0]) and
                        all(_is_concrete_positive_int(d) for d in ex_u.shape[1:]) and
                        math.prod(int(d) for d in ex_u.shape[1:]) == width):
                    has_closing = True
                    continue
                valid = False
                break

            ex_u = node.meta.get("example_value", node.meta.get("val"))
            if not (isinstance(ex_u, torch.Tensor) and tuple(ex_u.shape) == tuple(ex_in.shape) and ex_u.is_contiguous()):
                valid = False
                break

            node_ok = False
            if node.op == "call_function" and node.target in norm_targets and len(node.args) >= 2 and node.args[0] in region:
                w_sh, _ = _node_shape_and_dtype(node.args[1]) if isinstance(node.args[1], torch.fx.Node) else (None, None)
                if w_sh == (sub_dims[-1],) and not any(isinstance(a, torch.fx.Node) and a not in region for a in node.args[2:]):
                    node_ok = True
            elif node.op == "call_function" and node.target in ln_targets and len(node.args) >= 2 and node.args[0] in region:
                norm_shape = node.args[1]
                if (isinstance(norm_shape, (list, tuple)) and len(norm_shape) == 1 and
                        int(norm_shape[0]) == sub_dims[-1]):
                    other_nodes = [a for a in node.args[2:] if isinstance(a, torch.fx.Node)]
                    if all((_node_shape_and_dtype(w)[0] == (sub_dims[-1],)) for w in other_nodes):
                        node_ok = True
            elif ((node.op == "call_function" and node.target in unary_fn_targets) or
                  (node.op == "call_method" and node.target in ("to", "float", "bfloat16"))):
                if len(node.args) >= 1 and node.args[0] in region and not any(
                    isinstance(a, torch.fx.Node) for a in node.args[1:]
                ) and not any(isinstance(v, torch.fx.Node) for v in node.kwargs.values()):
                    node_ok = True
            elif node.op == "call_function" and node.target in binary_fn_targets and len(node.args) == 2 and not node.kwargs:
                arg_ok = True
                for a in node.args:
                    if isinstance(a, torch.fx.Node) and a not in region:
                        ash, _ = _node_shape_and_dtype(a)
                        if ash is None or len(ash) > len(sub_dims) or not all(_is_concrete_positive_int(d) for d in ash):
                            arg_ok = False
                            break
                node_ok = arg_ok

            if not node_ok:
                valid = False
                break
            region.add(node)
            interior.append(node)

        if valid and has_closing:
            r_in.args = (src, [-1, heads, *sub_dims])
            target_shape = (src_ex.shape[0], heads, *sub_dims)
            for n in (r_in, *interior):
                for key in ("example_value", "val"):
                    t = n.meta.get(key)
                    if isinstance(t, torch.Tensor):
                        n.meta[key] = t.view(target_shape)
                n.meta.pop("tensor_meta", None)


def _canonicalize_methods_and_core_ops(gm: torch.fx.GraphModule) -> None:
    graph = gm.graph
    aten = torch.ops.aten

    def _pack_reshape_args(a):
        src = a[0]
        rest = a[1:]
        if len(rest) == 1:
            if isinstance(rest[0], (list, tuple, torch.Size)):
                return (src, list(rest[0]))
            if isinstance(rest[0], torch.fx.Node):
                v = rest[0].meta.get("example_value", rest[0].meta.get("val"))
                if isinstance(v, (list, tuple, torch.Size)):
                    return (src, rest[0])
                return (src, [rest[0]])
            return (src, [rest[0]])
        return (src, list(rest))

    method_to_fn = {
        "view": lambda a: (aten.view.default, _pack_reshape_args(a)),
        "reshape": lambda a: (aten.reshape.default, _pack_reshape_args(a)),
        "transpose": lambda a: (torch.transpose, a),
        "permute": lambda a: (aten.permute.default, a),
        "unsqueeze": lambda a: (aten.unsqueeze.default, a),
        "squeeze": lambda a: (aten.squeeze.dim if len(a) == 2 else aten.squeeze.default, a),
        "expand": lambda a: (aten.expand.default, a),
        "split": lambda a: (torch.split, a),
        "chunk": lambda a: (torch.chunk, a),
        "clone": lambda a: (aten.to.dtype, (a[0],)),
        "tanh": lambda a: (aten.tanh.default, a),
        "sigmoid": lambda a: (aten.sigmoid.default, a),
        "exp": lambda a: (aten.exp.default, a),
        "rsqrt": lambda a: (aten.rsqrt.default, a),
        "sqrt": lambda a: (aten.sqrt.default, a),
        "neg": lambda a: (aten.neg.default, a),
        "abs": lambda a: (aten.abs.default, a),
        "isnan": lambda a: (aten.isnan.default, a),
        "clamp": lambda a: (aten.clamp.default, a),
        "clamp_min": lambda a: (aten.clamp_min.default, a),
        "clamp_max": lambda a: (aten.clamp_max.default, a),
        "softmax": lambda a: (torch.softmax, a),
        "log_softmax": lambda a: (torch.log_softmax, a),
        "sum": lambda a: (torch.sum, a),
        "mean": lambda a: (torch.mean, a),
        "amax": lambda a: (torch.amax, a),
        "logical_not": lambda a: (aten.logical_not.default, a),
        "logical_and": lambda a: (aten.logical_and.default, a),
        "logical_or": lambda a: (aten.logical_or.default, a),
        "bitwise_and": lambda a: (aten.bitwise_and.Tensor, a),
        "bitwise_or": lambda a: (aten.bitwise_or.Tensor, a),
        "masked_fill": lambda a: (aten.masked_fill.Scalar, a),
        "masked_fill_": lambda a: (aten.masked_fill_.Scalar, a),
        "type_as": lambda a: (aten.type_as.default if hasattr(aten, "type_as") else torch.Tensor.type_as, a),
        "pow": lambda a: (operator.pow, a),
    }

    for node in list(graph.nodes):
        if node.op == "call_method" and node.target in method_to_fn:
            new_tgt, new_args = method_to_fn[node.target](node.args)
            kw = dict(node.kwargs)
            if node.target == "clone":
                kw.pop("memory_format", None)
            with graph.inserting_before(node):
                rep = graph.call_function(new_tgt, args=new_args, kwargs=kw)
            rep.meta.update(node.meta)
            node.replace_all_uses_with(rep)
            graph.erase_node(node)

    clone_targets = {torch.clone, aten.clone.default}
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in clone_targets and len(node.args) == 1:
            kw = dict(node.kwargs)
            kw.pop("memory_format", None)
            with graph.inserting_before(node):
                rep = graph.call_function(aten.to.dtype, args=(node.args[0],), kwargs=kw)
            rep.meta.update(node.meta)
            node.replace_all_uses_with(rep)
            graph.erase_node(node)

    sym_size_targets = set()
    if hasattr(aten, "sym_size"):
        sym_size_targets.add(aten.sym_size)
        if hasattr(aten.sym_size, "int"):
            sym_size_targets.add(aten.sym_size.int)
    for node in list(graph.nodes):
        is_size = (node.op == "call_method" and node.target == "size" and len(node.args) == 2 and not node.kwargs) or (
            node.op == "call_function" and node.target in sym_size_targets and len(node.args) == 2 and not node.kwargs
        )
        if is_size and isinstance(node.args[0], torch.fx.Node):
            src_shape, _ = _node_shape_and_dtype(node.args[0])
            dim_arg = node.args[1]
            if src_shape is not None and isinstance(dim_arg, int) and not isinstance(dim_arg, bool):
                wdim = dim_arg % len(src_shape)
                if 0 <= wdim < len(src_shape) and _is_concrete_positive_int(src_shape[wdim]):
                    node.replace_all_uses_with(int(src_shape[wdim]))
                    graph.erase_node(node)

    reshape_targets = {torch.reshape, aten.view.default, aten.reshape.default}
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in reshape_targets and len(node.args) == 2:
            sh_arg = node.args[1]
            has_sym_shape = isinstance(sh_arg, torch.fx.Node) or (
                isinstance(sh_arg, (list, tuple)) and any(isinstance(d, (torch.fx.Node, torch.SymInt)) for d in sh_arg)
            )
            if has_sym_shape:
                ex_out = node.meta.get("example_value", node.meta.get("val"))
                if (isinstance(ex_out, torch.Tensor) and ex_out.ndim >= 1 and
                        all(_is_concrete_positive_int(d) for d in ex_out.shape[1:])):
                    trailing = [int(d) for d in ex_out.shape[1:]]
                    d0 = ex_out.shape[0]
                    if _is_concrete_positive_int(d0):
                        node.args = (node.args[0], [int(d0), *trailing])
                    elif isinstance(d0, torch.SymInt):
                        node.args = (node.args[0], [-1, *trailing])

    noop_targets = {aten.detach.default, aten.alias.default, aten.contiguous.default}
    for node in list(graph.nodes):
        if (node.op == "call_function" and node.target in noop_targets and len(node.args) >= 1) or (
            node.op == "call_method" and node.target in ("detach", "contiguous")
        ):
            src = node.args[0]
            if isinstance(src, torch.fx.Node):
                node.replace_all_uses_with(src)
                graph.erase_node(node)

    relu_targets = {torch.relu, torch.nn.functional.relu, aten.relu.default, aten.relu_.default}
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in relu_targets and len(node.args) >= 1:
            with graph.inserting_before(node):
                clamped = graph.call_function(aten.clamp_min.default, args=(node.args[0], 0.0))
            clamped.meta.update(node.meta)
            node.replace_all_uses_with(clamped)
            graph.erase_node(node)

    pow_targets = {operator.pow, torch.pow, aten.pow.Tensor_Scalar}
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in pow_targets and len(node.args) == 2 and not node.kwargs:
            src, raw_exp = node.args
            exp_val = _resolve_scalar(raw_exp)
            if exp_val is None or isinstance(exp_val, bool):
                continue
            p = float(exp_val)
            with graph.inserting_before(node):
                if p == 2.0:
                    rep = graph.call_function(operator.mul, args=(src, src))
                elif p == 1.0:
                    rep = src
                elif p == 0.5:
                    rep = graph.call_function(aten.sqrt.default, args=(src,))
                elif p == -0.5:
                    rep = graph.call_function(aten.rsqrt.default, args=(src,))
                elif p == -1.0:
                    rep = graph.call_function(operator.truediv, args=(1.0, src))
                else:
                    continue
            if rep is not src:
                rep.meta.update(node.meta)
            node.replace_all_uses_with(rep)
            graph.erase_node(node)

    mf_targets = {
        aten.masked_fill.Scalar,
        aten.masked_fill_.Scalar,
        aten.masked_fill.Tensor,
        aten.masked_fill_.Tensor,
    }
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in mf_targets and len(node.args) == 3:
            src, mask, value = node.args
            with graph.inserting_before(node):
                replaced = graph.call_function(aten.where.self, args=(mask, value, src))
            replaced.meta.update(node.meta)
            node.replace_all_uses_with(replaced)
            graph.erase_node(node)

    like_fill = {
        torch.zeros_like: 0,
        aten.zeros_like.default: 0,
        torch.ones_like: 1,
        aten.ones_like.default: 1,
    }
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in like_fill and len(node.args) >= 1:
            val = like_fill[node.target]
            kwargs = {}
            if "dtype" in node.kwargs and node.kwargs["dtype"] is not None:
                kwargs["dtype"] = node.kwargs["dtype"]
            with graph.inserting_before(node):
                full_node = graph.call_function(aten.full_like.default, args=(node.args[0], val), kwargs=kwargs)
            full_node.meta.update(node.meta)
            node.replace_all_uses_with(full_node)
            graph.erase_node(node)

    permute_targets = {torch.permute, aten.permute.default}
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in permute_targets and len(node.args) == 2:
            src, raw_dims = node.args
            perm = [int(d) for d in raw_dims]
            rank = len(perm)
            perm = [d + rank if d < 0 else d for d in perm]
            if perm == list(range(rank)):
                node.replace_all_uses_with(src)
                graph.erase_node(node)
                continue
            cur_order = list(range(rank))
            cur_node = src
            with graph.inserting_before(node):
                for i in range(rank):
                    if cur_order[i] != perm[i]:
                        j = cur_order.index(perm[i])
                        cur_order[i], cur_order[j] = cur_order[j], cur_order[i]
                        cur_node = graph.call_function(torch.transpose, args=(cur_node, i, j))
            cur_node.meta.update(node.meta)
            node.replace_all_uses_with(cur_node)
            graph.erase_node(node)

    chunk_targets = {torch.chunk, aten.chunk.default}
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in chunk_targets and len(node.args) >= 2:
            src = node.args[0]
            chunks = int(node.args[1])
            dim = int(node.args[2]) if len(node.args) > 2 else int(node.kwargs.get("dim", 0))
            if isinstance(src, torch.fx.Node):
                src_shape, _ = _node_shape_and_dtype(src)
                if src_shape is not None:
                    wdim = dim if dim >= 0 else len(src_shape) + dim
                    if 0 <= wdim < len(src_shape) and isinstance(src_shape[wdim], int):
                        dim_size = src_shape[wdim]
                        split_size = (dim_size + chunks - 1) // chunks
                        sizes = []
                        rem = dim_size
                        while rem > 0:
                            take = min(split_size, rem)
                            sizes.append(take)
                            rem -= take
                        with graph.inserting_before(node):
                            sp = graph.call_function(torch.split, args=(src, sizes, wdim))
                        sp.meta.update(node.meta)
                        node.replace_all_uses_with(sp)
                        graph.erase_node(node)

    mul_targets = {operator.mul, torch.mul, aten.mul.Tensor, aten.mul.Scalar}
    div_targets = {operator.truediv, torch.div, aten.div.Tensor, aten.div.Scalar}
    tanh_targets = {torch.tanh, aten.tanh.default}
    for node in list(graph.nodes):
        if node.op != "call_function" or node.target not in mul_targets or len(node.args) != 2:
            continue
        a, b = node.args
        cap = _resolve_scalar(b)
        tanh_node = a
        if cap is None:
            cap = _resolve_scalar(a)
            tanh_node = b
        if cap is None or isinstance(cap, bool) or float(cap) <= 0.0:
            continue
        cap_f = float(cap)
        if not (isinstance(tanh_node, torch.fx.Node) and tanh_node.op == "call_function" and
                tanh_node.target in tanh_targets and len(tanh_node.args) == 1 and
                len(tanh_node.users) == 1):
            continue
        scaled = tanh_node.args[0]
        if not (isinstance(scaled, torch.fx.Node) and scaled.op == "call_function" and
                len(scaled.args) == 2 and len(scaled.users) == 1):
            continue
        src = None
        if scaled.target in div_targets:
            denom = _resolve_scalar(scaled.args[1])
            if denom is not None and not isinstance(denom, bool) and abs(float(denom) - cap_f) <= 1e-5 * cap_f:
                src = scaled.args[0]
        elif scaled.target in mul_targets:
            inv_l = _resolve_scalar(scaled.args[0])
            inv_r = _resolve_scalar(scaled.args[1])
            if inv_r is not None and not isinstance(inv_r, bool) and abs(float(inv_r) * cap_f - 1.0) <= 1e-5:
                src = scaled.args[0]
            elif inv_l is not None and not isinstance(inv_l, bool) and abs(float(inv_l) * cap_f - 1.0) <= 1e-5:
                src = scaled.args[1]
        if src is None:
            continue
        with graph.inserting_before(node):
            fused = graph.call_function(_fx_softcap, args=(src, cap_f))
        fused.meta.update(node.meta)
        node.replace_all_uses_with(fused)
        graph.erase_node(node)
        graph.erase_node(tanh_node)
        graph.erase_node(scaled)

    softcap_pkt = getattr(torch.ops.gaudirt, "softcap_", None)
    softcap_inplace_targets = tuple(
        t for t in (softcap_pkt, getattr(softcap_pkt, "default", None)) if t is not None
    )
    if softcap_inplace_targets:
        ordered = list(graph.nodes)
        pos = {n: i for i, n in enumerate(ordered)}
        for node in ordered:
            if node.op != "call_function" or node.target not in softcap_inplace_targets:
                continue
            if len(node.args) < 1 or len(node.args) > 2:
                raise RuntimeError("GaudiRT FX: unsupported softcap_ arguments")
            x = node.args[0]
            raw_cap = node.args[1] if len(node.args) == 2 else node.kwargs.get("cap")
            cap_val = _resolve_scalar(raw_cap)
            if not isinstance(x, torch.fx.Node):
                raise RuntimeError("GaudiRT FX: softcap_ target must be a tensor node")
            if x.op == "placeholder":
                raise RuntimeError("GaudiRT FX: in-place softcap_ on a graph placeholder input is unsupported")
            if cap_val is None or isinstance(cap_val, bool) or not math.isfinite(float(cap_val)) or float(cap_val) <= 0.0:
                raise RuntimeError("GaudiRT FX: softcap_ cap must be finite and positive")
            pivot = pos[node]
            with graph.inserting_before(node):
                fused = graph.call_function(_fx_softcap, args=(x, float(cap_val)))
            fused.meta.update(x.meta)
            fused.meta.update(node.meta)
            pos[fused] = pivot
            for user in list(x.users):
                if user is not fused and pos.get(user, -1) > pivot:
                    user.replace_input_with(x, fused)
            if node.users:
                node.replace_all_uses_with(fused)
            graph.erase_node(node)

    _unfold_head_flattening_reshapes(graph)

    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in reshape_targets and len(node.args) == 2 and not node.kwargs:
            src = node.args[0]
            if not isinstance(src, torch.fx.Node):
                continue
            ex_src = src.meta.get("example_value", src.meta.get("val"))
            ex_out = node.meta.get("example_value", node.meta.get("val"))
            if (isinstance(ex_src, torch.Tensor) and isinstance(ex_out, torch.Tensor) and
                    ex_src.dtype == ex_out.dtype and ex_src.is_contiguous() and ex_out.is_contiguous() and
                    len(ex_src.shape) == len(ex_out.shape) and all(
                        (a == b if isinstance(a, int) and isinstance(b, int) else str(a) == str(b))
                        for a, b in zip(ex_src.shape, ex_out.shape)
                    )):
                node.replace_all_uses_with(src)
                graph.erase_node(node)


def decompose_and_canonicalize_fx(
    graph_module: torch.fx.GraphModule, example_inputs=None, extra_impure=()
) -> torch.fx.GraphModule:
    if example_inputs is not None:
        placeholders = [n for n in graph_module.graph.nodes if n.op == "placeholder"]
        for node, ex in zip(placeholders, example_inputs):
            if "example_value" not in node.meta and "val" not in node.meta:
                node.meta["example_value"] = ex
    _functionalize_inplace_arith(graph_module.graph)
    _fold_scalar_tensors(graph_module)
    _canonicalize_methods_and_core_ops(graph_module)
    impure = {
        _fx_collective,
        *extra_impure,
        torch.ops.gaudirt._paged_attn_validated,
        torch.ops.gaudirt._paged_attn_validated.default,
    }
    graph_module.graph.eliminate_dead_code(
        is_impure_node=lambda n: n.op in ("placeholder", "output") or n.target in impure
    )
    graph_module.recompile()
    return graph_module
