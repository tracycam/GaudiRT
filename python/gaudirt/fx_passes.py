"""FX graph normalization passes: VocabParallelEmbedding folding and collective canonicalization."""

import operator
import sys

import torch

from . import _C


def _fx_shard_embedding(tokens, weight, org_start, org_end, added_start, added_end, added_offset):
    raise RuntimeError("GaudiRT FX: internal _fx_shard_embedding must be lowered")


def _fx_collective(source, comm_id, kind):
    raise RuntimeError("GaudiRT FX: internal _fx_collective must be lowered")


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

    impure = {
        _fx_collective,
        _fx_shard_embedding,
        torch.ops.gaudirt._paged_attn_validated,
        torch.ops.gaudirt._paged_attn_validated.default,
    }
    graph.eliminate_dead_code(is_impure_node=lambda n: n.op in ("placeholder", "output") or n.target in impure)


def _normalize_fx_graph(graph_module: torch.fx.GraphModule) -> None:
    _fold_vocab_parallel_embedding(graph_module.graph)
    _normalize_collectives(graph_module.graph)
    graph_module.recompile()
