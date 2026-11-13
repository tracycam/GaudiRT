"""FX collective canonicalization and external collective plugin registration."""

import torch

from . import _C

_COMM_GROUP_RESOLVERS = []
_COLLECTIVE_OPS = {}


def _fx_collective(source, comm_id, kind):
    raise RuntimeError("GaudiRT FX: internal _fx_collective must be lowered")


def register_comm_group_resolver(resolver) -> None:
    """Register a callback `(group_name: str) -> ProcessGroup | None`."""
    if not callable(resolver):
        raise TypeError("GaudiRT FX: comm group resolver must be callable")
    if resolver not in _COMM_GROUP_RESOLVERS:
        _COMM_GROUP_RESOLVERS.append(resolver)


def register_collective_op(
    op_packet,
    kind,
    mode: str = "outplace",
    *,
    src_index: int = 0,
    out_index: int | None = None,
    group_index: int = 1,
) -> None:
    """Register an external custom collective op for FX normalization."""
    if mode not in ("outplace", "inplace_self", "inplace_out"):
        raise ValueError(f"GaudiRT FX: unsupported collective lowering mode {mode!r}")
    if mode == "inplace_out" and out_index is None:
        raise ValueError("GaudiRT FX: inplace_out collective requires out_index")
    spec = (int(kind), mode, src_index, out_index, group_index)
    default = getattr(op_packet, "default", None)
    for target in (op_packet, default):
        if target is not None:
            _COLLECTIVE_OPS[target] = spec


def _resolve_comm_group(group_name: str):
    if not isinstance(group_name, str) or not group_name:
        raise RuntimeError(f"GaudiRT FX: invalid collective group_name {group_name!r}")
    pg = None
    for resolver in _COMM_GROUP_RESOLVERS:
        pg = resolver(group_name)
        if pg is not None:
            break
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


def _overloads(ns, name):
    if ns is None or not hasattr(ns, name):
        return ()
    pkt = getattr(ns, name)
    default = getattr(pkt, "default", None)
    return (pkt, default) if default is not None else (pkt,)


def _normalize_collectives(graph: torch.fx.Graph, extra_impure=()) -> None:
    c10d_ns = getattr(torch.ops, "_c10d_functional", None)
    wait_targets = _overloads(c10d_ns, "wait_tensor")
    c10d_ar = _overloads(c10d_ns, "all_reduce")
    c10d_ag = _overloads(c10d_ns, "all_gather_into_tensor")
    c10d_rs = _overloads(c10d_ns, "reduce_scatter_tensor")
    c10d_a2a = _overloads(c10d_ns, "all_to_all_single")
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

    def _emit_inplace_self(idx, node, src, comm_id, world_size, kind):
        if world_size > 1:
            with graph.inserting_before(node):
                coll = graph.call_function(_fx_collective, args=(src, comm_id, int(kind)))
            coll.meta.update(src.meta)
            pos[coll] = idx
            _replace_uses_after(src, coll, idx)
            if node.users:
                node.replace_all_uses_with(coll)
        elif node.users:
            node.replace_all_uses_with(src)
        graph.erase_node(node)

    def _emit_inplace_out(idx, node, out_buf, src, comm_id, world_size, kind):
        if world_size == 1:
            _replace_uses_after(out_buf, src, idx)
        else:
            with graph.inserting_before(node):
                coll = graph.call_function(_fx_collective, args=(src, comm_id, int(kind)))
            coll.meta.update(out_buf.meta)
            pos[coll] = idx
            _replace_uses_after(out_buf, coll, idx)
        graph.erase_node(node)
        if (isinstance(out_buf, torch.fx.Node) and not out_buf.users and
                out_buf.op == "call_function" and out_buf.target in empty_targets):
            graph.erase_node(out_buf)

    for idx, node in enumerate(ordered):
        if node.op != "call_function":
            continue
        tgt = node.target

        if tgt in c10d_ar:
            src = node.args[0]
            reduce_op = node.args[1] if len(node.args) > 1 else node.kwargs.get("reduceOp", "sum")
            if str(reduce_op).lower() != "sum":
                raise RuntimeError(f"GaudiRT FX: unsupported all_reduce op {reduce_op!r}")
            group_name = node.args[2] if len(node.args) > 2 else node.kwargs["group_name"]
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

        spec = _COLLECTIVE_OPS.get(tgt)
        if spec is None:
            continue
        kind, mode, src_index, out_index, group_index = spec
        group_name = node.args[group_index] if len(node.args) > group_index else node.kwargs["group_name"]
        comm_id, world_size = _resolve_comm_group(group_name)
        src = node.args[src_index]
        if mode == "outplace":
            _emit_outplace(node, src, comm_id, world_size, kind)
        elif mode == "inplace_self":
            _emit_inplace_self(idx, node, src, comm_id, world_size, kind)
        else:
            _emit_inplace_out(idx, node, node.args[out_index], src, comm_id, world_size, kind)

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
        *extra_impure,
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
