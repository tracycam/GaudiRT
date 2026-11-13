"""FX graph normalization passes: VocabParallelEmbedding folding and collective canonicalization."""

import math
import operator
import sys

import torch

from . import _C


def _fx_shard_embedding(tokens, weight, org_start, org_end, added_start, added_end, added_offset):
    raise RuntimeError("GaudiRT FX: internal _fx_shard_embedding must be lowered")


def _fx_collective(source, comm_id, kind):
    raise RuntimeError("GaudiRT FX: internal _fx_collective must be lowered")


def _fx_softcap(source, cap):
    raise RuntimeError("GaudiRT FX: internal _fx_softcap must be lowered")


def _resolve_comm_group(group_name: str):
    if not isinstance(group_name, str) or not group_name:
        raise RuntimeError(f"GaudiRT FX: invalid collective group_name {group_name!r}")
    pg = None
    ps_mod = sys.modules.get("sglang.srt.distributed.parallel_state")
    if ps_mod is not None:
        groups = getattr(ps_mod, "_groups", None)
        if isinstance(groups, dict) and group_name in groups:
            entry = groups[group_name]
            coord = entry() if callable(entry) else entry
            if coord is None:
                raise RuntimeError(f"GaudiRT FX: collective group {group_name!r} was destroyed")
            pg = coord.device_group
    if pg is None:
        from torch.distributed.distributed_c10d import _resolve_process_group

        pg = _resolve_process_group(group_name)
    world_size = int(pg.size())
    if world_size <= 1:
        return -1, 1
    dev = torch.device(f"gaudirt:{_C.current_device()}")
    backend = pg._get_backend(dev)
    if not isinstance(backend, _C.ProcessGroupGaudiRT):
        raise RuntimeError(
            f"GaudiRT FX: group {group_name!r} uses unsupported backend {type(backend).__name__}"
        )
    backend.ensure_initialized()
    return int(backend.comm_id()), world_size


def _fold_vocab_parallel_embedding(graph: torch.fx.Graph) -> None:
    emb_targets = (torch.nn.functional.embedding, torch.ops.aten.embedding.default)
    ge_ops = (operator.ge, torch.ge, torch.ops.aten.ge.Scalar)
    lt_ops = (operator.lt, torch.lt, torch.ops.aten.lt.Scalar)
    and_ops = (operator.and_, torch.bitwise_and, torch.ops.aten.bitwise_and.Tensor)
    or_ops = (operator.or_, torch.bitwise_or, torch.ops.aten.bitwise_or.Tensor)
    inv_ops = (operator.invert, torch.bitwise_not, torch.ops.aten.bitwise_not.default)
    mul_ops = (operator.mul, torch.mul, torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)
    add_ops = (operator.add, torch.add, torch.ops.aten.add.Tensor)
    sub_ops = (operator.sub, torch.sub, torch.ops.aten.sub.Tensor)

    def _const_int(x):
        if isinstance(x, int) and not isinstance(x, bool):
            return x
        if isinstance(x, torch.SymInt):
            return int(x)
        if isinstance(x, torch.fx.Node):
            ex = x.meta.get("example_value", x.meta.get("val"))
            if isinstance(ex, (int, torch.SymInt)) and not isinstance(ex, bool):
                return int(ex)
        return None

    def _is_bin(n, ops):
        return (isinstance(n, torch.fx.Node) and n.op == "call_function" and
                n.target in ops and len(n.args) == 2 and not n.kwargs)

    def _match_range_mask(mask_node):
        if not _is_bin(mask_node, and_ops):
            return None
        a, b = mask_node.args
        if _is_bin(b, ge_ops) and _is_bin(a, lt_ops):
            a, b = b, a
        if not (_is_bin(a, ge_ops) and _is_bin(b, lt_ops)):
            return None
        tok_a, lo_raw = a.args
        tok_b, hi_raw = b.args
        lo = _const_int(lo_raw)
        hi = _const_int(hi_raw)
        if (tok_a is not tok_b or not isinstance(tok_a, torch.fx.Node) or
                lo is None or hi is None):
            return None
        return tok_a, lo, hi, a, b

    def _match_scalar_mask_mul(node, expected_mask):
        if not _is_bin(node, mul_ops):
            return None
        x, y = node.args
        if x is expected_mask:
            return _const_int(y)
        if y is expected_mask:
            return _const_int(x)
        return None

    for emb_node in list(graph.nodes):
        if emb_node.op != "call_function" or emb_node.target not in emb_targets:
            continue
        if emb_node.target is torch.nn.functional.embedding:
            if len(emb_node.args) != 2:
                continue
            idx_arg, weight_arg = emb_node.args
        else:
            if len(emb_node.args) < 2:
                continue
            weight_arg, idx_arg = emb_node.args[:2]
        if not isinstance(idx_arg, torch.fx.Node) or not isinstance(weight_arg, torch.fx.Node):
            continue

        pattern_nodes = set()
        masked_node = idx_arg
        if (idx_arg.op == "call_method" and idx_arg.target == "long" and
                len(idx_arg.args) == 1 and not idx_arg.kwargs and isinstance(idx_arg.args[0], torch.fx.Node)):
            pattern_nodes.add(idx_arg)
            masked_node = idx_arg.args[0]

        if not _is_bin(masked_node, mul_ops):
            continue
        m_left, m_right = masked_node.args
        if _is_bin(m_left, sub_ops):
            sub_node, vocab_mask = m_left, m_right
        elif _is_bin(m_right, sub_ops):
            vocab_mask, sub_node = m_left, m_right
        else:
            continue

        if not (_is_bin(vocab_mask, or_ops) and _is_bin(sub_node, sub_ops)):
            continue
        org_mask, add_mask = vocab_mask.args
        org_info = _match_range_mask(org_mask)
        add_info = _match_range_mask(add_mask)
        if org_info is None or add_info is None:
            continue
        tokens_node, org_start, org_end, ge_org, lt_org = org_info
        tokens_add, add_start, add_end, ge_add, lt_add = add_info
        if tokens_node is not tokens_add:
            continue

        sub_tok, valid_off = sub_node.args
        if sub_tok is not tokens_node or not _is_bin(valid_off, add_ops):
            continue
        mul_a, mul_b = valid_off.args
        off_org = _match_scalar_mask_mul(mul_a, org_mask)
        off_add = _match_scalar_mask_mul(mul_b, add_mask)
        if off_org is None or off_add is None:
            off_org = _match_scalar_mask_mul(mul_b, org_mask)
            off_add = _match_scalar_mask_mul(mul_a, add_mask)
        if off_org != org_start or off_add is None:
            continue

        mf_node = unsq_node = inv_node = None
        for user in list(emb_node.users):
            is_mf = ((user.op == "call_method" and user.target in ("masked_fill_", "masked_fill")) or
                     (user.op == "call_function" and user.target in (
                         torch.ops.aten.masked_fill_.Scalar, torch.ops.aten.masked_fill.Scalar)))
            if not is_mf or len(user.args) != 3 or user.kwargs or user.args[0] is not emb_node or user.args[2] != 0:
                continue
            cand_unsq = user.args[1]
            is_unsq = (isinstance(cand_unsq, torch.fx.Node) and (
                (cand_unsq.op == "call_method" and cand_unsq.target == "unsqueeze") or
                (cand_unsq.op == "call_function" and cand_unsq.target in (
                    torch.unsqueeze, torch.ops.aten.unsqueeze.default))
            ) and len(cand_unsq.args) == 2 and not cand_unsq.kwargs and cand_unsq.args[1] in (-1, 1))
            if not is_unsq:
                continue
            cand_inv = cand_unsq.args[0]
            is_inv = (isinstance(cand_inv, torch.fx.Node) and cand_inv.op == "call_function" and
                      cand_inv.target in inv_ops and len(cand_inv.args) == 1 and
                      not cand_inv.kwargs and cand_inv.args[0] is vocab_mask)
            if is_inv:
                mf_node, unsq_node, inv_node = user, cand_unsq, cand_inv
                break
        if mf_node is None:
            continue

        pattern_nodes.update({
            ge_org, lt_org, org_mask, ge_add, lt_add, add_mask,
            mul_a, mul_b, valid_off, vocab_mask, sub_node, masked_node,
            inv_node, unsq_node,
        })
        full_group = pattern_nodes | {emb_node, mf_node}
        if any(any(u not in full_group for u in n.users) for n in pattern_nodes):
            continue

        with graph.inserting_before(emb_node):
            shard_node = graph.call_function(
                _fx_shard_embedding,
                args=(tokens_node, weight_arg, int(org_start), int(org_end),
                      int(add_start), int(add_end), int(off_add)),
            )
        shard_node.meta.update(emb_node.meta)
        if mf_node.users:
            mf_node.replace_all_uses_with(shard_node)
        graph.erase_node(mf_node)
        emb_node.replace_all_uses_with(shard_node)
        graph.erase_node(emb_node)
        for n in reversed(list(graph.nodes)):
            if n in pattern_nodes and not n.users:
                graph.erase_node(n)


def _normalize_collectives(graph: torch.fx.Graph) -> None:
    c10d_ns = getattr(torch.ops, "_c10d_functional", None)
    sgl_ns = getattr(torch.ops, "sglang", None)

    def _overloads(ns, name):
        if ns is None or not hasattr(ns, name):
            return ()
        pkt = getattr(ns, name)
        default = getattr(pkt, "default", None)
        return (pkt, default) if default is not None else (pkt,)

    wait_targets = _overloads(c10d_ns, "wait_tensor")
    c10d_ar = _overloads(c10d_ns, "all_reduce")
    c10d_ag = _overloads(c10d_ns, "all_gather_into_tensor")
    c10d_rs = _overloads(c10d_ns, "reduce_scatter_tensor")
    c10d_a2a = _overloads(c10d_ns, "all_to_all_single")

    sgl_out_ar = _overloads(sgl_ns, "outplace_all_reduce")
    sgl_in_ar = _overloads(sgl_ns, "inplace_all_reduce")
    sgl_ag = _overloads(sgl_ns, "reg_all_gather_into_tensor")
    sgl_rs = _overloads(sgl_ns, "reg_reduce_scatter_tensor")
    sgl_a2a = _overloads(sgl_ns, "reg_all_to_all_single")
    empty_targets = (torch.empty, torch.ops.aten.empty.memory_format)

    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in wait_targets:
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)

    ordered = list(graph.nodes)
    pos = {n: i for i, n in enumerate(ordered)}

    def _replace_uses_after(target_tensor, replacement, pivot_idx):
        for user in list(target_tensor.users):
            if pos.get(user, -1) > pivot_idx:
                user.replace_input_with(target_tensor, replacement)

    def _emit_outplace(node, src, comm_id, world_size, kind):
        if world_size == 1:
            node.replace_all_uses_with(src)
        else:
            with graph.inserting_before(node):
                coll = graph.call_function(_fx_collective, args=(src, comm_id, int(kind)))
            coll.meta.update(node.meta)
            node.replace_all_uses_with(coll)
        graph.erase_node(node)

    for idx, node in enumerate(ordered):
        if node.op != "call_function":
            continue
        tgt = node.target

        if tgt in c10d_ar or tgt in sgl_out_ar:
            src = node.args[0]
            if tgt in c10d_ar:
                reduce_op = node.args[1] if len(node.args) > 1 else node.kwargs.get("reduceOp", "sum")
                if str(reduce_op).lower() != "sum":
                    raise RuntimeError(f"GaudiRT FX: unsupported all_reduce op {reduce_op!r}")
                group_name = node.args[2] if len(node.args) > 2 else node.kwargs["group_name"]
            else:
                group_name = node.args[1] if len(node.args) > 1 else node.kwargs["group_name"]
            comm_id, world_size = _resolve_comm_group(group_name)
            _emit_outplace(node, src, comm_id, world_size, _C.CollectiveKind.AllReduce)
            continue

        if tgt in c10d_ag:
            src = node.args[0]
            group_name = node.args[2] if len(node.args) > 2 else node.kwargs["group_name"]
            comm_id, world_size = _resolve_comm_group(group_name)
            _emit_outplace(node, src, comm_id, world_size, _C.CollectiveKind.AllGather)
            continue

        if tgt in c10d_rs:
            src = node.args[0]
            reduce_op = node.args[1] if len(node.args) > 1 else node.kwargs.get("reduceOp", "sum")
            if str(reduce_op).lower() != "sum":
                raise RuntimeError(f"GaudiRT FX: unsupported reduce_scatter op {reduce_op!r}")
            group_name = node.args[3] if len(node.args) > 3 else node.kwargs["group_name"]
            comm_id, world_size = _resolve_comm_group(group_name)
            _emit_outplace(node, src, comm_id, world_size, _C.CollectiveKind.ReduceScatter)
            continue

        if tgt in c10d_a2a:
            src = node.args[0]
            out_sp = node.args[1] if len(node.args) > 1 else node.kwargs.get("output_split_sizes", [])
            in_sp = node.args[2] if len(node.args) > 2 else node.kwargs.get("input_split_sizes", [])
            group_name = node.args[3] if len(node.args) > 3 else node.kwargs["group_name"]
            comm_id, world_size = _resolve_comm_group(group_name)
            for sp in (out_sp, in_sp):
                if sp and (len(sp) != world_size or any(s != sp[0] for s in sp)):
                    raise RuntimeError("GaudiRT FX: variable-split all_to_all is unsupported in compiled graphs")
            _emit_outplace(node, src, comm_id, world_size, _C.CollectiveKind.AllToAll)
            continue

        if tgt in sgl_in_ar:
            src = node.args[0]
            group_name = node.args[1] if len(node.args) > 1 else node.kwargs["group_name"]
            comm_id, world_size = _resolve_comm_group(group_name)
            if world_size > 1:
                with graph.inserting_before(node):
                    coll = graph.call_function(_fx_collective, args=(src, comm_id, int(_C.CollectiveKind.AllReduce)))
                coll.meta.update(src.meta)
                pos[coll] = idx
                _replace_uses_after(src, coll, idx)
                if node.users:
                    node.replace_all_uses_with(coll)
            elif node.users:
                node.replace_all_uses_with(src)
            graph.erase_node(node)
            continue

        if tgt in sgl_ag or tgt in sgl_rs or tgt in sgl_a2a:
            out_buf, src = node.args[0], node.args[1]
            group_name = node.args[2] if len(node.args) > 2 else node.kwargs["group_name"]
            comm_id, world_size = _resolve_comm_group(group_name)
            kind = int(_C.CollectiveKind.AllGather if tgt in sgl_ag else
                       _C.CollectiveKind.ReduceScatter if tgt in sgl_rs else
                       _C.CollectiveKind.AllToAll)
            if world_size == 1:
                _replace_uses_after(out_buf, src, idx)
            else:
                with graph.inserting_before(node):
                    coll = graph.call_function(_fx_collective, args=(src, comm_id, kind))
                coll.meta.update(out_buf.meta)
                pos[coll] = idx
                _replace_uses_after(out_buf, coll, idx)
            graph.erase_node(node)
            if (isinstance(out_buf, torch.fx.Node) and not out_buf.users and
                    out_buf.op == "call_function" and out_buf.target in empty_targets):
                graph.erase_node(out_buf)
            continue

    reshape_fns = (torch.reshape, torch.ops.aten.view.default, torch.ops.aten.reshape.default)
    movedim_fns = (torch.movedim, torch.ops.aten.movedim.int, torch.transpose, torch.ops.aten.transpose.int)

    def _is_reshape(n):
        return ((n.op == "call_method" and n.target in ("reshape", "view")) or
                (n.op == "call_function" and n.target in reshape_fns))

    def _is_swap01(n):
        if ((n.op == "call_method" and n.target in ("movedim", "transpose")) or
                (n.op == "call_function" and n.target in movedim_fns)):
            if len(n.args) == 3 and not n.kwargs:
                ex = n.args[0].meta.get("example_value")
                ndim = ex.ndim if isinstance(ex, torch.Tensor) else 3
                d0, d1 = n.args[1] % ndim, n.args[2] % ndim
                return (d0, d1) in ((0, 1), (1, 0))
        if ((n.op == "call_method" and n.target == "permute") or
                (n.op == "call_function" and n.target in (torch.permute, torch.ops.aten.permute.default))):
            dims = n.args[1] if len(n.args) == 2 and isinstance(n.args[1], (tuple, list)) else n.args[1:]
            return tuple(dims) == (1, 0, 2)
        return False

    for coll in list(graph.nodes):
        if (coll.op != "call_function" or coll.target is not _fx_collective or
                coll.args[2] != int(_C.CollectiveKind.AllGather) or len(coll.users) != 1):
            continue
        r1 = next(iter(coll.users))
        if not _is_reshape(r1) or len(r1.users) != 1:
            continue
        m1 = next(iter(r1.users))
        if not _is_swap01(m1) or len(m1.users) != 1:
            continue
        r2 = next(iter(m1.users))
        if not _is_reshape(r2):
            continue
        src_ex = coll.args[0].meta.get("example_value")
        r1_ex = r1.meta.get("example_value")
        r2_ex = r2.meta.get("example_value")
        if not (isinstance(src_ex, torch.Tensor) and isinstance(r1_ex, torch.Tensor) and
                isinstance(r2_ex, torch.Tensor) and src_ex.ndim == 2 and r1_ex.ndim == 3 and r2_ex.ndim == 2):
            continue
        if (str(r1_ex.shape[1]) != str(src_ex.shape[0]) or
                str(r1_ex.shape[2]) != str(src_ex.shape[1]) or
                str(r2_ex.shape[0]) != str(src_ex.shape[0]) or
                not isinstance(r1_ex.shape[0], (int, torch.SymInt)) or
                not isinstance(src_ex.shape[1], (int, torch.SymInt)) or
                not isinstance(r2_ex.shape[1], (int, torch.SymInt)) or
                int(r2_ex.shape[1]) != int(r1_ex.shape[0]) * int(src_ex.shape[1])):
            continue
        with graph.inserting_before(r2):
            folded = graph.call_function(
                _fx_collective, args=(coll.args[0], coll.args[1], int(_C.CollectiveKind.AllGatherDimLast))
            )
        folded.meta.update(r2.meta)
        r2.replace_all_uses_with(folded)
        graph.erase_node(r2)
        graph.erase_node(m1)
        graph.erase_node(r1)
        graph.erase_node(coll)

    softcap_pkt = getattr(torch.ops.gaudirt, "softcap_", None)
    impure = {
        _fx_collective,
        _fx_shard_embedding,
        torch.ops.gaudirt._paged_attn_validated,
        torch.ops.gaudirt._paged_attn_validated.default,
    }
    if softcap_pkt is not None:
        impure.add(softcap_pkt)
        if hasattr(softcap_pkt, "default"):
            impure.add(softcap_pkt.default)
    graph.eliminate_dead_code(
        is_impure_node=lambda n: n.op in ("placeholder", "output")
        or n.target in impure
        or getattr(getattr(n.target, "_schema", None), "is_mutable", False)
    )


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
            elif isinstance(val, torch.Tensor) and val.device.type == "cpu" and val.numel() == 1 and val.ndim <= 1:
                # 1-element CPU scalar buffers like torch.tensor([s]) only fold if 0D or explicitly scalar
                if val.ndim == 0:
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
    for name in ("rms_norm", "gemma_rms_norm"):
        pkt = getattr(torch.ops.gaudirt, name, None)
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

    # Normalize common tensor methods into functional call_function forms
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

    # Fold static 1-dim tensor.size(dim) / aten.sym_size(tensor, dim) and resolve symbolic reshape targets
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

    # Strip identity wrapper ops (detach / alias / contiguous)
    noop_targets = {aten.detach.default, aten.alias.default, aten.contiguous.default}
    for node in list(graph.nodes):
        if (node.op == "call_function" and node.target in noop_targets and len(node.args) >= 1) or (
            node.op == "call_method" and node.target in ("detach", "contiguous")
        ):
            src = node.args[0]
            if isinstance(src, torch.fx.Node):
                node.replace_all_uses_with(src)
                graph.erase_node(node)

    # Canonicalize relu -> clamp_min(..., 0.0)
    relu_targets = {torch.relu, torch.nn.functional.relu, aten.relu.default, aten.relu_.default}
    for node in list(graph.nodes):
        if node.op == "call_function" and node.target in relu_targets and len(node.args) >= 1:
            with graph.inserting_before(node):
                clamped = graph.call_function(aten.clamp_min.default, args=(node.args[0], 0.0))
            clamped.meta.update(node.meta)
            node.replace_all_uses_with(clamped)
            graph.erase_node(node)

    # Canonicalize integer/half scalar powers supported natively without general pow
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

    # Canonicalize masked_fill -> where(mask, val, self)
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

    # Canonicalize zeros_like / ones_like -> full_like
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

    # Canonicalize permute -> transpose swap chain (preserves dim 0 if identity)
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

    # Canonicalize chunk -> split when input shape along dim is statically known
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

    # Fuse Gemma2 logit/attention softcap: cap * tanh(x / cap) or cap * tanh(x * (1 / cap))
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

    # Functionalize in-place torch.ops.gaudirt.softcap_(x, cap)
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

    # Elide contiguous same-shape identity reshape/view nodes
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
    graph_module: torch.fx.GraphModule, example_inputs=None
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
        _fx_shard_embedding,
        torch.ops.gaudirt._paged_attn_validated,
        torch.ops.gaudirt._paged_attn_validated.default,
    }
    graph_module.graph.eliminate_dead_code(
        is_impure_node=lambda n: n.op in ("placeholder", "output") or n.target in impure
    )
    graph_module.recompile()
    return graph_module


def _normalize_fx_graph(graph_module: torch.fx.GraphModule, example_inputs=None) -> None:
    _fold_vocab_parallel_embedding(graph_module.graph)
    _normalize_collectives(graph_module.graph)
    decompose_and_canonicalize_fx(graph_module, example_inputs)
