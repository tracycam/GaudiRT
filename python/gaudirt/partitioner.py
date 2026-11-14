"""Convex DAG partitioner and ASAP/ALAP segment scheduler across attention and collective boundaries."""

import builtins
import operator

import torch

from . import _C
from .fx_passes import _fx_collective, _fx_gather_rows
from .op_lowering import _program_backend


class _CollectiveStartStub:
    def forward(self, *args):
        return None


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
        self.token_checks, self.row_checks, *extra_checks = checks
        self.static_row_checks = extra_checks[0] if extra_checks else ()
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
        for index, expected_rows in self.static_row_checks:
            if getattr(args[index], "_gaudirt_validated_rows", None) != expected_rows:
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
    token_checks, row_checks, static_row_checks, records = set(), set(), set(), []
    dynamic_inputs = [index for index in input_indices if placeholders[index].meta.get("gaudirt_dynamic")]
    dynamic_input = dynamic_inputs[0] if dynamic_inputs else None
    extent_to_input = {}
    for index in dynamic_inputs:
        ex = placeholders[index].meta.get("example_value")
        if isinstance(ex, torch.Tensor) and ex.ndim:
            extent_to_input.setdefault(str(ex.shape[0]), index)

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
        elif any(item.op == "call_function" and item.target is _fx_gather_rows for item in seg_nodes):
            mod_kind[node] = "gather_rows"
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
    started_collectives = {}
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
            if len(attention.args) not in (12, 13, 14) or attention.kwargs:
                raise RuntimeError("GaudiRT FX: unsupported paged-attention arguments")
            try:
                tensor_arguments = tuple(segment_inputs.index(item) for item in attention.args[:10])
            except ValueError as error:
                raise RuntimeError("GaudiRT FX: attention tensors must be segment inputs") from error
            head_dim, scale = attention.args[10:12]
            softcap = attention.args[12] if len(attention.args) >= 13 else 0.0
            sliding_window = attention.args[13] if len(attention.args) >= 14 else -1
            if (not isinstance(head_dim, int) or isinstance(head_dim, bool) or
                    not isinstance(scale, (int, float)) or isinstance(scale, bool) or
                    not isinstance(softcap, (int, float)) or isinstance(softcap, bool) or
                    not isinstance(sliding_window, int) or isinstance(sliding_window, bool)):
                raise RuntimeError("GaudiRT FX: attention geometry must be static")
            slots = [tensor_slot(node.args[index]) for index in tensor_arguments]
            values[node] = runner.add_attention(slots, head_dim, float(scale), float(softcap), int(sliding_window))
            records.append((segment, host_arguments(node.args, tensor_arguments), tensor_arguments, True))
            emit_getitem_children(node)
            available.add(node)
            return

        if mod_kind[node] == "gather_rows":
            gathers = [item for item in segment_nodes if item.op == "call_function" and item.target is _fx_gather_rows]
            if len(gathers) != 1 or any(item.op not in ("placeholder", "output") and item is not gathers[0]
                                        for item in segment_nodes):
                raise RuntimeError("GaudiRT FX: invalid dynamic gather_rows partition")
            gather = gathers[0]
            if len(gather.args) != 2 or gather.kwargs:
                raise RuntimeError("GaudiRT FX: unsupported dynamic gather_rows arguments")
            try:
                tensor_arguments = tuple(segment_inputs.index(item) for item in gather.args)
            except ValueError as error:
                raise RuntimeError("GaudiRT FX: dynamic gather_rows inputs must be segment inputs") from error
            src_arg, idx_arg = (node.args[index] for index in tensor_arguments)
            idx_ext = external_index(idx_arg)
            src_dim0 = gather.meta.get("gaudirt_gather_src_dim0")
            if src_dim0 is None:
                seg_src_ex = gather.args[0].meta.get("example_value")
                if not isinstance(seg_src_ex, torch.Tensor) or seg_src_ex.ndim != 2:
                    raise RuntimeError("GaudiRT FX: dynamic gather_rows source must be 2D")
                src_dim0 = seg_src_ex.shape[0]
            if src_arg in positions:
                row_checks.add((idx_ext, positions[src_arg]))
            elif isinstance(src_dim0, int):
                static_row_checks.add((idx_ext, int(src_dim0)))
            else:
                src_dim_ext = extent_to_input.get(str(src_dim0))
                if src_dim_ext is None:
                    raise RuntimeError("GaudiRT FX: missing dynamic source input for validated row indices")
                row_checks.add((idx_ext, src_dim_ext))
            slots = [tensor_slot(arg) for arg in (src_arg, idx_arg)]
            values[node] = runner.add_gather_rows(slots[0], slots[1])
            records.append((segment, host_arguments(node.args, tensor_arguments), tensor_arguments, True))
            emit_getitem_children(node)
            available.add(node)
            return

        if not any(item.meta.get("gaudirt_dynamic") for item in segment_inputs):
            max_bucket_rows = int(getattr(_C, "_max_bucket_rows", 8192))
            for idx_in, item in enumerate(segment_inputs):
                ex = item.meta.get("example_value")
                if (node.args[idx_in] not in positions and isinstance(ex, torch.Tensor) and
                        ex.ndim >= 1 and isinstance(ex.shape[0], int) and 0 < ex.shape[0] <= max_bucket_rows):
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
            seg_dyn_ext = next(
                (str(item.meta.get("example_value").shape[0])
                 for pos, item in enumerate(segment_inputs)
                 if pos in tensor_arguments and item.meta.get("gaudirt_dynamic") and
                 isinstance(item.meta.get("example_value"), torch.Tensor) and item.meta["example_value"].ndim > 0),
                None,
            )
            bound_dynamic = extent_to_input.get(seg_dyn_ext, dynamic_input)
            if bound_dynamic is None:
                raise RuntimeError("GaudiRT FX: missing dynamic input for validated row indices")
            row_checks.add((external_index(node.args[index]), bound_dynamic))
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
                             (tuple(sorted(token_checks)), tuple(sorted(row_checks)),
                              tuple(sorted(static_row_checks))), records).forward


def _partition_backend(graph_module, example_inputs):
    """Convex DAG partitioner across stateful paged-attention and collective boundaries."""
    from torch.fx.passes.split_module import split_module

    graph = graph_module.graph
    transpose_targets = (torch.transpose, torch.ops.aten.transpose.int)
    split_targets = (torch.split, torch.ops.aten.split.Tensor, torch.ops.aten.split_with_sizes.default)
    rotary_targets = (torch.ops.gaudirt._rotary_embedding_validated,
                      torch.ops.gaudirt._rotary_embedding_validated.default,
                      torch.ops.gaudirt.rotary_embedding_positions,
                      torch.ops.gaudirt.rotary_embedding_positions.default)
    fused_norm_targets = (torch.ops.gaudirt.fused_add_rms_norm,
                          torch.ops.gaudirt.fused_add_rms_norm.default,
                          torch.ops.aten.native_layer_norm.default)
    paged = (torch.ops.gaudirt._paged_attn_validated, torch.ops.gaudirt._paged_attn_validated.default)
    boundary_targets = (*paged, _fx_collective, _fx_gather_rows)

    def is_boundary(n):
        return n.op == "call_function" and n.target in boundary_targets

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
        node.meta["gaudirt_dynamic"] = bool(dynamic)
        example = node.meta.get("example_value")
        if dynamic:
            if not isinstance(example, torch.Tensor) or example.ndim == 0:
                raise RuntimeError("GaudiRT FX: missing dynamic input shape")
            extents.add(str(example.shape[0]))
    if not extents:
        raise RuntimeError("GaudiRT FX: mark each dynamic token input before capture")

    for node in nodes:
        if node.op == "placeholder":
            continue
        if node.op == "call_function" and node.target is _fx_gather_rows:
            src_ex = node.args[0].meta.get("example_value")
            if not isinstance(src_ex, torch.Tensor) or src_ex.ndim != 2:
                raise RuntimeError("GaudiRT FX: dynamic gather_rows source must be 2D")
            node.meta["gaudirt_gather_src_dim0"] = src_ex.shape[0]
        example = node.meta.get("example_value")
        if isinstance(example, torch.Tensor) and example.ndim:
            if node.op == "call_function" and node.target is _fx_collective and not isinstance(example.shape[0], int):
                extents.add(str(example.shape[0]))
            node.meta["gaudirt_dynamic"] = str(example.shape[0]) in extents

    def is_consumer_helper(n):
        def _non_boundary_users():
            return bool(n.users and all(u.op != "output" and not is_boundary(u) for u in n.users))

        if n.op == "call_method" and n.target in ("t", "long") and len(n.args) == 1 and not n.kwargs:
            return _non_boundary_users()
        if (n.op == "call_function" and n.target is builtins.getattr and
                len(n.args) == 2 and n.args[1] == "T" and not n.kwargs):
            return _non_boundary_users()
        if ((n.op == "call_method" and n.target == "transpose") or
                (n.op == "call_function" and n.target in transpose_targets)):
            if len(n.args) == 3 and not n.kwargs:
                ex = n.args[0].meta.get("example_value")
                if isinstance(ex, torch.Tensor) and ex.ndim == 2 and (n.args[1] % 2) != (n.args[2] % 2):
                    return _non_boundary_users()
        if (not n.meta.get("gaudirt_dynamic", False) and
                ((n.op == "call_method" and n.target in ("to", "float", "bfloat16", "half")) or
                 (n.op == "call_function" and n.target in (
                     torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype))) and
                _non_boundary_users()):
            return True
        return False

    def is_tuple_producer(n):
        if (n.op == "call_method" and n.target == "split") or (
                n.op == "call_function" and n.target in split_targets):
            return True
        return n.op == "call_function" and (n.target in rotary_targets or n.target in fused_norm_targets)

    # Clone shared consumer helpers (.t(), .T, 2D transpose, .long(), static weight casts) per
    # user so each stays local to its consuming BucketProgram without cross-partition edges.
    for n in list(graph.nodes):
        if is_consumer_helper(n) and len(n.users) > 1:
            users = list(n.users)
            for user in users[1:]:
                with graph.inserting_before(user):
                    clone = graph.node_copy(n)
                clone.meta.update(n.meta)
                user.replace_input_with(n, clone)

    nodes = tuple(graph.nodes)

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
