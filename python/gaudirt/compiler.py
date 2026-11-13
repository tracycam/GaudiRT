"""FX entry point, capture guard management, and segmented fast-path dispatch."""

import functools
import threading
import types
import weakref

import torch
import torch.fx.experimental._config

from . import _C
from .fx_passes import _fx_collective, _normalize_fx_graph
from .op_lowering import _program_backend
from .partitioner import _SegmentExecution, _partition_backend

MAX_BUCKET_ROWS = int(getattr(_C, "_max_bucket_rows", 8192))


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


def gaudirt_backend(graph_module, example_inputs):
    _normalize_fx_graph(graph_module, example_inputs)
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
                        0 < original.shape[0] <= MAX_BUCKET_ROWS):
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
            torch._dynamo.mark_dynamic(prepared_args[index], 0)
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
