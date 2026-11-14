"""FX graph normalization entry point: domain folds, collectives, and canonicalization."""

import operator

import torch

from .fx_canonicalize import _fx_softcap, decompose_and_canonicalize_fx
from .fx_collectives import (
    _fx_collective,
    _normalize_collectives,
    register_collective_op,
    register_comm_group_resolver,
)


def _fx_shard_embedding(tokens, weight, org_start, org_end, added_start, added_end, added_offset):
    raise RuntimeError("GaudiRT FX: internal _fx_shard_embedding must be lowered")


def _fx_gather_rows(source, indices):
    raise RuntimeError("GaudiRT FX: internal _fx_gather_rows must be lowered")


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


def _match_dynamic_gather_rows(node: torch.fx.Node):
    if node.op == "call_function" and node.target is operator.getitem:
        if len(node.args) != 2 or node.kwargs:
            return None
        src, raw_idx = node.args
        if isinstance(raw_idx, torch.fx.Node):
            idx = raw_idx
        elif (isinstance(raw_idx, (tuple, list)) and
              ((len(raw_idx) == 1 and isinstance(raw_idx[0], torch.fx.Node)) or
               (len(raw_idx) == 2 and isinstance(raw_idx[0], torch.fx.Node) and raw_idx[1] == slice(None)))):
            idx = raw_idx[0]
        else:
            return None
    elif node.op == "call_function" and node.target is torch.ops.aten.index.Tensor:
        if len(node.args) != 2 or node.kwargs or not isinstance(node.args[1], (list, tuple)):
            return None
        if len(node.args[1]) != 1 or not isinstance(node.args[1][0], torch.fx.Node):
            return None
        src, idx = node.args[0], node.args[1][0]
    elif ((node.op == "call_method" and node.target == "index_select") or
          (node.op == "call_function" and node.target in (torch.index_select, torch.ops.aten.index_select.default))):
        if len(node.args) != 3 or node.kwargs or node.args[1] != 0 or not isinstance(node.args[2], torch.fx.Node):
            return None
        src, idx = node.args[0], node.args[2]
    else:
        return None
    if not isinstance(src, torch.fx.Node):
        return None
    src_ex = src.meta.get("example_value")
    idx_ex = idx.meta.get("example_value")
    if (not isinstance(src_ex, torch.Tensor) or src_ex.ndim != 2 or
            not isinstance(src_ex.shape[1], int) or src_ex.shape[1] <= 0):
        return None
    if isinstance(src_ex.shape[0], int) and (src_ex.shape[0] <= 0 or src_ex.shape[0] > 8192):
        return None
    if (not isinstance(idx_ex, torch.Tensor) or idx_ex.ndim != 1 or
            isinstance(idx_ex.shape[0], int) or
            idx_ex.dtype not in (torch.int32, torch.int64)):
        return None
    return src, idx


def _fold_dynamic_gather_rows(graph: torch.fx.Graph) -> None:
    for node in list(graph.nodes):
        matched = _match_dynamic_gather_rows(node)
        if matched is None:
            continue
        src, idx = matched
        with graph.inserting_before(node):
            gathered = graph.call_function(_fx_gather_rows, args=(src, idx))
        gathered.meta.update(node.meta)
        node.replace_all_uses_with(gathered)
        graph.erase_node(node)


def _normalize_fx_graph(graph_module: torch.fx.GraphModule, example_inputs=None) -> None:
    _fold_vocab_parallel_embedding(graph_module.graph)
    _normalize_collectives(graph_module.graph, extra_impure=(_fx_shard_embedding, _fx_gather_rows))
    decompose_and_canonicalize_fx(graph_module, example_inputs, extra_impure=(_fx_shard_embedding, _fx_gather_rows))
    _fold_dynamic_gather_rows(graph_module.graph)
