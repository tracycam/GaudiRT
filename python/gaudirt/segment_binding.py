"""Bind stable FX inputs once, retaining Dynamo's C++ guard checks."""

import builtins
import dataclasses
import weakref

import torch


def _result_template(value, outputs):
    if isinstance(value, torch.Tensor):
        for index, output in enumerate(outputs):
            if value is output:
                return ("tensor", index)
        raise ValueError("result tensor is not a segment output")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = dataclasses.fields(value)
        if any(not field.init for field in fields) or hasattr(type(value), "__post_init__"):
            raise ValueError("result dataclass needs custom initialization")
        return ("dataclass", (type(value), tuple((field.name, _result_template(getattr(value, field.name), outputs))
                                                for field in fields)))
    if isinstance(value, tuple):
        return ("tuple", (type(value) if hasattr(value, "_fields") else tuple,
                          tuple(_result_template(item, outputs) for item in value)))
    if isinstance(value, list):
        return ("list", tuple(_result_template(item, outputs) for item in value))
    if value is None:
        return ("literal", value)
    raise ValueError("unsupported result object")


def _restore(template, outputs):
    kind, value = template
    if kind == "tensor":
        return outputs[value]
    if kind == "dataclass":
        cls, fields = value
        return cls(**{name: _restore(item, outputs) for name, item in fields})
    if kind == "tuple":
        cls, items = value
        values = tuple(_restore(item, outputs) for item in items)
        return values if cls is tuple else cls(*values)
    if kind == "list":
        return [_restore(item, outputs) for item in value]
    if kind == "dict":
        return {key: _restore(item, outputs) for key, item in value}
    return value


def bind(execution, guard_manager, frame_locals, globals_, outer_result):
    """Return a guarded runner when all lifted tensor sources are available."""
    args, outputs = execution._first_args, execution._first_outputs
    if args is None:
        return None
    indices = execution.input_indices
    sources = execution.input_sources
    if any(not isinstance(sources[index], str) for index in indices):
        return None
    static = [index for index in indices if
              "._parameters[" in sources[index] or "._buffers[" in sources[index] or
              getattr(args[index], "_gaudirt_static_binding", False)]
    if not static:
        return None
    dynamic = [index for index in indices if index not in static]
    extra = sorted({value for _, arguments, _, _ in execution.records for kind, value in arguments
                    if kind == "input" and value not in indices})
    if any(not isinstance(sources[index], str) for index in extra):
        return None
    try:
        template = _result_template(outer_result, outputs)
        expression = "lambda L, G: (" + ", ".join(sources[index] for index in dynamic) + ",)"
        getter = eval(builtins.compile(expression, "<gaudirt-segment-inputs>", "eval"), {"__builtins__": {}})
        extra_expression = "lambda L, G: (" + ", ".join(sources[index] for index in extra) + ",)" if extra else None
        extra_getter = (eval(builtins.compile(extra_expression, "<gaudirt-segment-host-inputs>", "eval"),
                             {"__builtins__": {}}) if extra else None)
        current = getter(frame_locals, globals_)
        if len(current) != len(dynamic) or any(a is not args[index] for a, index in zip(current, dynamic)):
            return None
    except (ValueError, SyntaxError, KeyError, AttributeError, TypeError, IndexError):
        return None

    managers = {}
    pending = [guard_manager.root]
    while pending:
        manager = pending.pop()
        managers[manager.get_source()] = manager
        pending.extend(manager.get_child_managers())
    if any(sources[index] not in managers for index in static):
        return None
    for index in static:
        managers[sources[index]].add_id_match_guard(id(args[index]), ["GaudiRT static tensor identity"], None)
    if not guard_manager.check(frame_locals):
        return None

    static_values = {index: weakref.ref(args[index]) for index in static}
    dynamic_positions = {index: pos for pos, index in enumerate(dynamic)}
    extra_positions = {index: pos for pos, index in enumerate(extra)}
    if any(index not in static_values and index not in dynamic_positions
           for pair in (execution.token_checks, execution.row_checks) for check in pair for index in check):
        return None
    execution.runner.bind_static([indices.index(index) for index in static], [args[index] for index in static])

    def invoke(locals_):
        values = getter(locals_, globals_)

        def tensor_at(index):
            if index not in static_values:
                return values[dynamic_positions[index]]
            value = static_values[index]()
            if value is None:
                raise RuntimeError("GaudiRT FX: static tensor expired")
            return value

        for index, weight in execution.token_checks:
            vocabulary = getattr(tensor_at(index), "_gaudirt_validated_vocab", None)
            if not isinstance(vocabulary, int) or vocabulary <= 0 or vocabulary > tensor_at(weight).shape[0]:
                raise RuntimeError("GaudiRT FX: embedding token IDs must pass batch metadata validation")
        for index, dynamic_index in execution.row_checks:
            if getattr(tensor_at(index), "_gaudirt_validated_rows", None) != tensor_at(dynamic_index).shape[0]:
                raise RuntimeError("GaudiRT FX: row indices must pass batch metadata validation")
        if execution.tracing:
            result, traces = execution.runner.run_bound_traced(values, list(range(len(execution.records))))
            extras = extra_getter(locals_, globals_) if extra_getter is not None else ()
            execution.consume_trace(traces, lambda index: extras[extra_positions[index]] if index in extra_positions
                                    else tensor_at(index))
        elif execution.profile_events is not None:
            result, execution.profile_host_ms = execution.runner.run_bound_profiled(values, execution.profile_events)
        else:
            result = execution.runner.run_bound(values)
        return _restore(template, result)

    return invoke
