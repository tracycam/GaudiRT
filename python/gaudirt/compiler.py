"""FX entry point for the supported bucketed device subgraphs."""

import builtins
import functools
import math
import operator
import struct
import sys
import threading
import types
import weakref

import torch
import torch.fx.experimental._config

from . import _C


def _fx_shard_embedding(tokens, weight, org_start, org_end, added_start, added_end, added_offset):
    raise RuntimeError("GaudiRT FX: internal _fx_shard_embedding must be lowered")


def _fx_collective(source, comm_id, kind):
    raise RuntimeError("GaudiRT FX: internal _fx_collective must be lowered")


class _CollectiveStartStub:
    def forward(self, *args):
        return None


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
            if world_size == 1:
                node.replace_all_uses_with(src)
            else:
                with graph.inserting_before(node):
                    coll = graph.call_function(_fx_collective, args=(src, comm_id, int(_C.CollectiveKind.AllReduce)))
                coll.meta.update(node.meta)
                node.replace_all_uses_with(coll)
            graph.erase_node(node)
            continue

        if tgt in c10d_ag:
            src = node.args[0]
            group_name = node.args[2] if len(node.args) > 2 else node.kwargs["group_name"]
            comm_id, world_size = _resolve_comm_group(group_name)
            if world_size == 1:
                node.replace_all_uses_with(src)
            else:
                with graph.inserting_before(node):
                    coll = graph.call_function(_fx_collective, args=(src, comm_id, int(_C.CollectiveKind.AllGather)))
                coll.meta.update(node.meta)
                node.replace_all_uses_with(coll)
            graph.erase_node(node)
            continue

        if tgt in c10d_rs:
            src = node.args[0]
            reduce_op = node.args[1] if len(node.args) > 1 else node.kwargs.get("reduceOp", "sum")
            if str(reduce_op).lower() != "sum":
                raise RuntimeError(f"GaudiRT FX: unsupported reduce_scatter op {reduce_op!r}")
            group_name = node.args[3] if len(node.args) > 3 else node.kwargs["group_name"]
            comm_id, world_size = _resolve_comm_group(group_name)
            if world_size == 1:
                node.replace_all_uses_with(src)
            else:
                with graph.inserting_before(node):
                    coll = graph.call_function(_fx_collective, args=(src, comm_id, int(_C.CollectiveKind.ReduceScatter)))
                coll.meta.update(node.meta)
                node.replace_all_uses_with(coll)
            graph.erase_node(node)
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
            if world_size == 1:
                node.replace_all_uses_with(src)
            else:
                with graph.inserting_before(node):
                    coll = graph.call_function(_fx_collective, args=(src, comm_id, int(_C.CollectiveKind.AllToAll)))
                coll.meta.update(node.meta)
                node.replace_all_uses_with(coll)
            graph.erase_node(node)
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


def _program_backend(graph_module, example_inputs, compact_outputs=()):
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
        if node.target is _fx_shard_embedding:
            if len(node.args) != 7 or node.kwargs:
                raise RuntimeError("GaudiRT FX: invalid _fx_shard_embedding arguments")
            indices, weight, org_start, org_end, add_start, add_end, add_offset = node.args
            indices = index_casts.get(indices, indices)
            index_id, weight_id_value = tensor_id(indices), tensor_id(weight)
            if index_id >= len(tensor_indices) or weight_id_value >= len(tensor_indices):
                raise RuntimeError("GaudiRT FX: shard_embedding requires validated external IDs and weights")
            validated_token_bindings.append((index_id, weight_id_value, True))
            ids[node] = len(tensor_indices) + len(operations)
            operations.append(Node.shard_embedding(
                index_id, weight_id_value,
                int(org_start), int(org_end), int(add_start), int(add_end), int(add_offset),
            ))
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
            validated_token_bindings.append((index_id, weight_id_value, False))
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
        elif node.target in block_fp8_targets and len(node.args) in (3, 4):
            if set(node.kwargs) - {"bias"} or (len(node.args) == 4 and "bias" in node.kwargs):
                raise RuntimeError("GaudiRT FX: unsupported block FP8 linear arguments")
            bias = node.args[3] if len(node.args) == 4 else node.kwargs.get("bias")
            operation = Node.block_fp8(*(tensor_id(value) for value in node.args[:3]),
                                       None if bias is None else tensor_id(bias))
        elif node.target in fp8_targets and len(node.args) in (3, 4):
            if set(node.kwargs) - {"bias"} or (len(node.args) == 4 and "bias" in node.kwargs):
                raise RuntimeError("GaudiRT FX: unsupported FP8 linear arguments")
            bias = node.args[3] if len(node.args) == 4 else node.kwargs.get("bias")
            operation = Node.fp8(*(tensor_id(value) for value in node.args[:3]),
                                  None if bias is None else tensor_id(bias))
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


def _restore_segment_output(template, result, args):
    kind, value = template
    if kind == "tensor":
        return result[value]
    if kind == "input":
        return args[value]
    if kind == "tuple":
        return tuple(_restore_segment_output(item, result, args) for item in value)
    if kind == "list":
        return [_restore_segment_output(item, result, args) for item in value]
    return value


class _SegmentExecution:
    def __init__(self, split, runner, input_indices, input_sources, output_template, checks, records):
        self.split = split
        self.runner = runner
        self.input_indices = input_indices
        self.input_sources = input_sources
        self.output_template = output_template
        self.token_checks, self.row_checks = checks
        self.records = records
        self.tracing = False
        self.trace_result = None
        self.trace_deltas = ()
        self.profile_events = None
        self.profile_host_ms = None
        self._first_args = None
        self._first_outputs = None
        self._binding_attempted = False

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

    def consume_trace(self, traces, input_at):
        self.trace_deltas = tuple(trace[2] for trace in traces)
        for i, (segment, host_arguments, tensor_arguments, single) in enumerate(self.records):
            tensor_values, outputs, _ = traces[i]
            actual = [input_at(value) if kind == "input" else value for kind, value in host_arguments]
            for index, tensor in zip(tensor_arguments, tensor_values):
                actual[index] = tensor
            self.trace_result = outputs[0] if single else tuple(outputs)
            segment.forward(*actual)
            traces[i] = None
        self.trace_result = None

    def forward(self, *args):
        for index, weight, is_sharded in self.token_checks:
            vocabulary = getattr(args[index], "_gaudirt_validated_vocab", None)
            if (not isinstance(vocabulary, int) or vocabulary <= 0 or
                    (not is_sharded and vocabulary > args[weight].shape[0])):
                raise RuntimeError("GaudiRT FX: embedding token IDs must pass batch metadata validation")
        for index, dynamic in self.row_checks:
            if getattr(args[index], "_gaudirt_validated_rows", None) != args[dynamic].shape[0]:
                raise RuntimeError("GaudiRT FX: row indices must pass batch metadata validation")
        inputs = [args[index] for index in self.input_indices]
        if self.tracing:
            result, traces = self.runner.run_traced(inputs, list(range(len(self.records))))
            self.consume_trace(traces, lambda index: args[index])
        elif self.profile_events is not None:
            result, self.profile_host_ms = self.runner.run_profiled(inputs, self.profile_events)
        else:
            result = self.runner.run(inputs)

        if not self._binding_attempted and self._first_args is None:
            self._first_args, self._first_outputs = args, result

        return _restore_segment_output(self.output_template, result, args)


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

    def attention_only(value):
        if not value.users:
            return False
        for user in value.users:
            if user.op != "call_module":
                return False
            target = split.get_submodule(user.target)
            if not any(item.op == "call_function" and item.target in paged for item in target.graph.nodes):
                return False
        return True

    def compact_attention_outputs(node, segment):
        returned = next(item for item in segment.graph.nodes if item.op == "output").args[0]
        if isinstance(returned, torch.fx.Node):
            return (0,) if attention_only(node) else ()
        if not isinstance(returned, (tuple, list)):
            return ()
        eligible = set()
        for user in node.users:
            if (user.op == "call_function" and user.target is operator.getitem and
                    len(user.args) == 2 and isinstance(user.args[1], int) and attention_only(user)):
                eligible.add(user.args[1])
        return tuple(sorted(eligible))

    call_modules = []
    getitem_children = {}
    output_node = None
    for node in graph_nodes:
        if node.op == "placeholder":
            continue
        if node.op == "call_function" and node.target is operator.getitem:
            source, index = node.args
            if not isinstance(source, torch.fx.Node) or source.op != "call_module" or not isinstance(index, int):
                raise RuntimeError("GaudiRT FX: unsupported split graph indexing")
            getitem_children.setdefault(source, []).append(node)
            continue
        if node.op == "output":
            output_node = node
            continue
        if node.op != "call_module" or node.kwargs:
            raise RuntimeError("GaudiRT FX: unsupported operation between segments")
        call_modules.append(node)

    if output_node is None:
        raise RuntimeError("GaudiRT FX: split graph has no output")

    topo_index = {node: idx for idx, node in enumerate(call_modules)}
    mod_kind = {}
    mod_deps = {}
    mod_consumers = {node: set() for node in call_modules}
    for node in call_modules:
        segment = split.get_submodule(node.target)
        seg_nodes = tuple(segment.graph.nodes)
        if any(item.op == "call_function" and item.target is _fx_collective for item in seg_nodes):
            mod_kind[node] = "collective"
        elif any(item.op == "call_function" and item.target in paged for item in seg_nodes):
            mod_kind[node] = "attention"
        else:
            mod_kind[node] = "program"
        deps = set()
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                if arg.op == "call_module":
                    deps.add(arg)
                elif arg.op == "call_function" and arg.target is operator.getitem:
                    deps.add(arg.args[0])
        mod_deps[node] = deps
        for dep in deps:
            mod_consumers[dep].add(node)

    available = set()
    started_collectives = {}  # node -> (segment, tensor_arguments)
    in_flight_order = []

    def emit_getitem_children(node):
        for child in getitem_children.get(node, ()):
            source, index = child.args
            if source not in values or not isinstance(values[source], tuple):
                raise RuntimeError("GaudiRT FX: unsupported split graph indexing")
            values[child] = values[source][index]

    def emit_collective_start(node):
        segment = split.get_submodule(node.target)
        segment_nodes = tuple(segment.graph.nodes)
        segment_inputs = tuple(item for item in segment_nodes if item.op == "placeholder")
        if len(segment_inputs) != len(node.args):
            raise RuntimeError("GaudiRT FX: collective segment argument count mismatch")
        colls = [item for item in segment_nodes if item.op == "call_function" and item.target is _fx_collective]
        if len(colls) != 1 or any(item.op not in ("placeholder", "output") and item is not colls[0]
                                  for item in segment_nodes):
            raise RuntimeError("GaudiRT FX: invalid collective partition")
        coll = colls[0]
        src_ph, comm_id, kind = coll.args
        try:
            src_arg_idx = segment_inputs.index(src_ph)
        except ValueError as error:
            raise RuntimeError("GaudiRT FX: collective input must be a segment input") from error
        tensor_arguments = (src_arg_idx,)
        src_slot = tensor_slot(node.args[src_arg_idx])
        dynamic_out = bool(coll.meta.get("gaudirt_dynamic", False))
        coll_id = runner.add_collective_start(
            int(comm_id), _C.CollectiveKind(int(kind)), src_slot, dynamic_out
        )
        records.append((_CollectiveStartStub(), host_arguments(node.args, tensor_arguments),
                        tensor_arguments, True))
        started_collectives[node] = (segment, tensor_arguments, coll_id)
        in_flight_order.append(node)

    def emit_collective_wait(node):
        segment, tensor_arguments, coll_id = started_collectives[node]
        values[node] = runner.add_collective_wait(coll_id)
        records.append((segment, host_arguments(node.args, tensor_arguments), tensor_arguments, True))
        emit_getitem_children(node)
        available.add(node)

    def emit_compute_or_attention(node):
        segment = split.get_submodule(node.target)
        segment_nodes = tuple(segment.graph.nodes)
        segment_inputs = tuple(item for item in segment_nodes if item.op == "placeholder")
        if len(segment_inputs) != len(node.args):
            raise RuntimeError("GaudiRT FX: segment argument count mismatch")
        if mod_kind[node] == "attention":
            stateful = [item for item in segment_nodes if item.op == "call_function" and item.target in paged]
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
            emit_getitem_children(node)
            available.add(node)
            return

        if not any(item.meta.get("gaudirt_dynamic") for item in segment_inputs):
            for idx_in, item in enumerate(segment_inputs):
                ex = item.meta.get("example_value")
                if (node.args[idx_in] not in positions and isinstance(ex, torch.Tensor) and
                        ex.ndim >= 1 and isinstance(ex.shape[0], int) and 0 < ex.shape[0] <= 4096):
                    item.meta["gaudirt_dynamic"] = True
                    break

        examples = [item.meta.get("example_value") for item in segment_inputs]
        program, tensor_arguments, routes, single, token_bindings, row_bindings = \
            _program_backend(segment, examples, compact_attention_outputs(node, segment))._gaudirt_segment
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
        for index, weight, is_sharded in token_bindings:
            token_checks.add((external_index(node.args[index]), external_index(node.args[weight]), is_sharded))
        for index, _ in row_bindings:
            if dynamic_input is None:
                raise RuntimeError("GaudiRT FX: missing dynamic input for validated row indices")
            row_checks.add((external_index(node.args[index]), dynamic_input))
        records.append((segment, host_arguments(node.args, tensor_arguments), tensor_arguments, single))
        emit_getitem_children(node)
        available.add(node)

    def feeds_unstarted_collective(node):
        return any(mod_kind[c] == "collective" and c not in started_collectives
                   for c in mod_consumers[node])

    while len(available) < len(call_modules):
        ready_collectives = [
            n for n in call_modules
            if mod_kind[n] == "collective" and n not in started_collectives and mod_deps[n] <= available
        ]
        if ready_collectives:
            emit_collective_start(ready_collectives[0])
            continue

        ready_compute = [
            n for n in call_modules
            if mod_kind[n] != "collective" and n not in available and mod_deps[n] <= available
        ]
        if ready_compute:
            best = min(ready_compute, key=lambda n: (not feeds_unstarted_collective(n), topo_index[n]))
            emit_compute_or_attention(best)
            continue

        if not in_flight_order:
            raise RuntimeError("GaudiRT FX: cyclic or unresolvable segment schedule")
        coll_to_wait = in_flight_order.pop(0)
        emit_collective_wait(coll_to_wait)

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

    template = output_template(output_node.args[0])
    runner.set_outputs(output_ids)
    sources = tuple(getattr(node.meta.get("gaudirt_source"), "name", None) for node in placeholders)
    return _SegmentExecution(split, runner, input_indices, sources, template,
                             (tuple(sorted(token_checks)), tuple(sorted(row_checks))), records).forward


def _partition_backend(graph_module, example_inputs):
    """Convex DAG partitioner across stateful paged-attention and collective boundaries."""
    from torch.fx.passes.split_module import split_module

    graph = graph_module.graph
    transpose_targets = (torch.transpose, torch.ops.aten.transpose.int)
    split_targets = (torch.split, torch.ops.aten.split_with_sizes.default)
    rotary_targets = (torch.ops.gaudirt._rotary_embedding_validated,
                      torch.ops.gaudirt._rotary_embedding_validated.default)
    fused_norm_targets = (torch.ops.gaudirt.fused_add_rms_norm,
                          torch.ops.gaudirt.fused_add_rms_norm.default)
    paged = (torch.ops.gaudirt._paged_attn_validated, torch.ops.gaudirt._paged_attn_validated.default)
    boundary_targets = (*paged, _fx_collective)

    def is_consumer_helper(n):
        if n.op == "call_method" and n.target in ("t", "long") and len(n.args) == 1 and not n.kwargs:
            return True
        if (n.op == "call_function" and n.target is builtins.getattr and
                len(n.args) == 2 and n.args[1] == "T" and not n.kwargs):
            return True
        if ((n.op == "call_method" and n.target == "transpose") or
                (n.op == "call_function" and n.target in transpose_targets)):
            if len(n.args) == 3 and not n.kwargs:
                ex = n.args[0].meta.get("example_value")
                if isinstance(ex, torch.Tensor) and ex.ndim == 2 and (n.args[1] % 2) != (n.args[2] % 2):
                    return True
        return False

    def is_tuple_producer(n):
        if (n.op == "call_method" and n.target == "split") or (
                n.op == "call_function" and n.target in split_targets):
            return True
        return n.op == "call_function" and (n.target in rotary_targets or n.target in fused_norm_targets)

    # Clone shared consumer helpers (.t(), .T, 2D transpose, .long()) per user so each
    # stays local to its consuming BucketProgram without cross-partition edges.
    for n in list(graph.nodes):
        if is_consumer_helper(n) and len(n.users) > 1:
            users = list(n.users)
            for user in users[1:]:
                with graph.inserting_before(user):
                    clone = graph.node_copy(n)
                clone.meta.update(n.meta)
                user.replace_input_with(n, clone)

    nodes = tuple(graph.nodes)
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
            if node.op == "call_function" and node.target is _fx_collective and not isinstance(example.shape[0], int):
                extents.add(str(example.shape[0]))
            node.meta["gaudirt_dynamic"] = str(example.shape[0]) in extents

    def is_boundary(n):
        return n.op == "call_function" and n.target in boundary_targets

    anc = {}
    for n in nodes:
        if n.op in ("placeholder", "output"):
            anc[n] = frozenset()
            continue
        acc = set()
        for inp in n.all_input_nodes:
            acc |= anc.get(inp, frozenset())
            if is_boundary(inp):
                acc.add(inp)
        anc[n] = frozenset(acc)

    desc = {n: frozenset() for n in nodes}
    for n in reversed(nodes):
        if n.op in ("placeholder", "output"):
            continue
        acc = set()
        for user in n.users:
            if user.op == "output":
                continue
            acc |= desc.get(user, frozenset())
            if is_boundary(user):
                acc.add(user)
        desc[n] = frozenset(acc)

    raw_key = {}
    tuple_getitems = []
    helpers = []
    for n in nodes:
        if n.op in ("placeholder", "output"):
            continue
        if is_boundary(n):
            raw_key[n] = ("boundary", n)
        elif (n.op == "call_function" and n.target is operator.getitem and
              len(n.args) == 2 and isinstance(n.args[1], int) and
              isinstance(n.args[0], torch.fx.Node) and is_tuple_producer(n.args[0])):
            tuple_getitems.append(n)
        elif is_consumer_helper(n):
            helpers.append(n)
        else:
            raw_key[n] = ("compute", anc[n], desc[n])

    for n in tuple_getitems:
        raw_key[n] = raw_key[n.args[0]]
    for n in reversed(helpers):
        if len(n.users) != 1:
            raise RuntimeError("GaudiRT FX: unconsumed or multi-user helper node in partitioner")
        user = next(iter(n.users))
        if user not in raw_key:
            raise RuntimeError("GaudiRT FX: helper node cannot feed graph output directly")
        raw_key[n] = raw_key[user]

    node_order = {n: i for i, n in enumerate(nodes)}
    all_keys = {}
    for n, k in raw_key.items():
        if k not in all_keys or node_order[n] < all_keys[k]:
            all_keys[k] = node_order[n]

    adj = {k: set() for k in all_keys}
    indeg = {k: 0 for k in all_keys}
    for n, k in raw_key.items():
        for inp in n.all_input_nodes:
            pk = raw_key.get(inp)
            if pk is not None and pk != k and k not in adj[pk]:
                adj[pk].add(k)
                indeg[k] += 1

    topo_keys = []
    ready_keys = [k for k, d in indeg.items() if d == 0]
    while ready_keys:
        ready_keys.sort(key=lambda k: all_keys[k])
        cur = ready_keys.pop(0)
        topo_keys.append(cur)
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready_keys.append(nxt)
    if len(topo_keys) != len(all_keys):
        raise RuntimeError("GaudiRT FX: partition quotient graph contains a cycle")

    key_to_id = {k: idx for idx, k in enumerate(topo_keys)}
    partitions = {n: key_to_id[k] for n, k in raw_key.items()}

    split = split_module(graph_module, graph_module, lambda node: partitions[node], keep_original_order=False)
    split_placeholders = [node for node in split.graph.nodes if node.op == "placeholder"]
    ph_by_name = {node.name: node for node in placeholders}
    for target in split_placeholders:
        source = ph_by_name.get(target.name)
        if source is not None:
            target.meta["gaudirt_source"] = getattr(source.meta.get("grapharg"), "source", None)
    return _segment_executor(split, example_inputs, paged)


def gaudirt_backend(graph_module, example_inputs):
    _normalize_fx_graph(graph_module)
    nodes = tuple(graph_module.graph.nodes)
    if any(node.op == "call_function" and node.target in (
        torch.ops.gaudirt._paged_attn_validated,
        torch.ops.gaudirt._paged_attn_validated.default,
        _fx_collective,
    ) for node in nodes):
        return _partition_backend(graph_module, example_inputs)
    return _program_backend(graph_module, example_inputs)


def compile(module, dynamic_arg_indices=(0,), prepare_capture=None):
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
        result = gaudirt_backend(graph, inputs)
        capture.segmented = isinstance(getattr(result, "__self__", None), _SegmentExecution)
        capture.execution = getattr(result, "__self__", None) if capture.segmented else None
        return result

    compiled = torch.compile(module, backend=guarded_backend, dynamic=True, fullgraph=True)
    captured_signatures = set()
    static_signatures = []
    signature_tokens = {}
    fast_paths = []

    def frame_locals(code, cells, bound, args, kwargs):
        result = {name: cell.cell_contents for name, cell in cells}
        if (code.co_argcount == 0 and code.co_kwonlyargcount == 0 and
                code.co_flags & 0x0c == 0x0c and code.co_varnames[:2] == ("args", "kwargs")):
            result.update(args=tuple(args), kwargs=kwargs)
        elif not kwargs and not code.co_kwonlyargcount and not code.co_flags & 0x0c:
            positional = (bound, *args) if bound is not None else tuple(args)
            if len(positional) != code.co_argcount:
                return None
            result.update(zip(code.co_varnames[:code.co_argcount], positional))
        else:
            return None
        return result

    def remember_segmented_path(args, kwargs, model_result):
        # Dynamo's rewritten top-level frame already lifts the graph inputs and
        # reconstructs the Python result. Reuse it only with its original C++
        # guard manager, so changes in module state or ambient PyTorch state
        # still trigger normal capture rather than running a stale graph.
        if not getattr(capture, "segmented", False):
            return
        original = getattr(compiled, "_torchdynamo_orig_callable", None)
        function = getattr(original, "__func__", original)
        bound = getattr(original, "__self__", None)
        code = getattr(function, "__code__", None)
        if code is None:
            return
        cells = tuple(zip(code.co_freevars, function.__closure__ or ()))
        locals_ = frame_locals(code, cells, bound, args, kwargs)
        if locals_ is None:
            return
        for entry in torch._dynamo.eval_frame._debug_get_cache_entry_list(function):
            if (getattr(entry.backend, "compiler_fn", None) is not guarded_backend or
                    entry.code.co_freevars != code.co_freevars or
                    any(candidate[0] is entry for candidate in fast_paths) or
                    not entry.guard_manager.check(locals_)):
                continue
            rewritten = types.FunctionType(entry.code, function.__globals__, function.__name__,
                                           function.__defaults__, function.__closure__)
            if bound is not None:
                rewritten = types.MethodType(rewritten, bound)
            from .segment_binding import bind

            execution = capture.execution
            direct = bind(execution, entry.guard_manager, locals_, function.__globals__, model_result)
            execution._first_args = execution._first_outputs = None
            execution._binding_attempted = True
            fast_paths.insert(0, (entry, rewritten, code, cells, bound, direct, execution))

    def prepare_dynamic_inputs(args):
        prepared_args = list(args)
        staged_inputs, replacements = {}, []
        for index in dynamic_arg_indices:
            original = args[index]
            if id(original) not in staged_inputs:
                if (isinstance(original, torch.Tensor) and original.ndim == 1 and original.stride(0) == 1 and
                        original.device.type == "gaudirt" and original.device.index == _C.current_device() and
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
        return prepared_args, replacements

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
        prepared_args, replacements = prepare_dynamic_inputs(args)
        if (fast_paths and torch._guards.TracingContext.try_get() is None and
                not torch.fx._symbolic_trace.is_fx_symbolic_tracing() and not torch._C._is_tracing()):
            for entry, rewritten, code, cells, bound, direct, execution in fast_paths:
                locals_ = frame_locals(code, cells, bound, prepared_args, kwargs)
                if locals_ is not None and entry.guard_manager.check(locals_):
                    result = direct(locals_) if direct is not None else rewritten(*prepared_args, **kwargs)
                    return _restore_staged_outputs(result, replacements) if replacements else result
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
            if prepared_args[index].ndim > 1:
                torch._dynamo.mark_static(prepared_args[index], list(range(1, prepared_args[index].ndim)))
            torch._dynamo.mark_dynamic(prepared_args[index], 0, min=1, max=4096)
        key = (tuple(static_tokens[index] if static_tokens[index] is not None else
                     signature(value, index in dynamic_arg_indices or index in declared_dynamic)
                     for index, value in enumerate(prepared_args)), signature(kwargs))
        needs_capture = key not in captured_signatures
        while True:
            if needs_capture:
                if prepare_capture is not None:
                    prepare_capture()
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
                execute._gaudirt_last_execution = getattr(capture, "execution", None)
                remember_segmented_path(prepared_args, kwargs, result)
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
