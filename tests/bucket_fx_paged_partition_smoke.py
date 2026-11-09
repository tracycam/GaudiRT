"""Capture a connected QKV/paged-attention/output graph without Python padding."""

import hashlib
import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F
import gaudirt
import gaudirt.compiler as compiler
from fp64_reference import direct_bf16
import paged_attention_smoke as attention_ref


def connected(x, qkv_weight, out_weight, key_pool, value_pool, locations, requests,
              positions, mapping, header):
    query, key, value = F.linear(x, qkv_weight).split([256, 128, 128], dim=-1)
    attention = torch.ops.gaudirt._paged_attn_validated.default(
        query, key, value, key_pool, value_pool,
        locations, requests, positions, mapping, header, 64, 0.125,
    )
    output = F.linear(attention, out_weight)
    return output.view(-1, 4, 64).reshape(x.shape[0], -1)


def connected_compact(x, qkv_weight, out_weight, key_pool, value_pool, locations, requests,
                      positions, mapping, header):
    query, key, value = F.linear(x, qkv_weight).split([256, 128, 128], dim=-1)
    query, key = query * 1.0, key * 1.0
    attention = torch.ops.gaudirt._paged_attn_validated.default(
        query, key, value, key_pool, value_pool,
        locations, requests, positions, mapping, header, 64, 0.125,
    )
    return F.linear(attention, out_weight).view(-1, 4, 64).reshape(x.shape[0], -1)


def reference(x, qkv_weight, out_weight, key_pool, value_pool, locations, requests,
              positions, mapping, fp64):
    dtype = torch.float64 if fp64 else torch.float32
    rounded = direct_bf16 if fp64 else lambda value: value.to(torch.bfloat16).float()
    qkv = x.to(dtype) @ qkv_weight.to(dtype).t()
    if not fp64:
        # The same-boundary FP32 reference rounds each model-visible BF16
        # operator output, while its arithmetic and softmax remain FP32.
        qkv = rounded(qkv)
    query, key, value = qkv.split([256, 128, 128], dim=-1)
    key_pool = key_pool.to(dtype).clone()
    value_pool = value_pool.to(dtype).clone()
    key_pool[locations] = key.reshape(-1, 2, 64)
    value_pool[locations] = value.reshape(-1, 2, 64)
    attention = attention_ref.expected_unrounded(query, key_pool, value_pool, requests, positions, mapping, dtype)
    if not fp64:
        attention = rounded(attention)
    return rounded(attention @ out_weight.to(dtype).t()).double()


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(41628)
    gaudirt.initialize(hbm_bytes=2 * 1024**3, pinned_bytes=4 * 1024 * 1024)
    original = compiler.gaudirt_backend
    captures, observed, calls, executors, profile_events = [], [], [], [], []
    tracing = os.environ.get("GAUDIRT_TEST_SEGMENT_TRACE") == "1"
    profiling = os.environ.get("GAUDIRT_TEST_SEGMENT_PROFILE") == "1"
    compact = os.environ.get("GAUDIRT_TEST_COMPACT_QKV") == "1"
    compact_reference = os.environ.get("GAUDIRT_TEST_COMPACT_REFERENCE") == "1"
    assert not compact or tracing
    reference_split = os.environ.get("GAUDIRT_TEST_REFERENCE_SPLIT") == "1"
    assert not (tracing and profiling)

    def capture(graph, inputs):
        captures.append(graph)
        assert sum(node.target == torch.ops.gaudirt._paged_attn_validated.default
                   for node in graph.graph.nodes) == 1
        execution = original(graph, inputs)
        executor = execution.__self__
        executors.append(executor)
        if reference_split:
            return execution

        class CountCalls:
            def __init__(self, runner):
                self.runner = runner

            def run(self, values):
                calls.append("run")
                return self.runner.run(values)

            def run_traced(self, values, indices):
                calls.append("trace")
                return self.runner.run_traced(values, indices)

            def run_profiled(self, values, events):
                calls.append("profile")
                return self.runner.run_profiled(values, events)

        executor.runner = CountCalls(executor.runner)
        if tracing:
            executor.enable_trace()
            for index, (_, segment) in enumerate(executor.named_children()):
                previous = segment.forward

                def observe(*args, index=index, previous=previous):
                    output = previous(*args)
                    if compact and not compact_reference and index == 0:
                        values = output if isinstance(output, (tuple, list)) else (output,)
                        kv = [value for value in values if isinstance(value, torch.Tensor) and
                              value.ndim == 2 and value.shape[-1] == 128]
                        assert len(kv) == 2 and all(value.is_contiguous() for value in kv), kv
                    observed.append(index)
                    return output

                segment.forward = observe
        if profiling:
            profile_events.extend((gaudirt.Event(enable_timing=True), gaudirt.Event(enable_timing=True))
                                  for _ in executor.records)
            for start, end in profile_events:
                start.record()
                end.record()
            for start, end in profile_events:
                start.synchronize()
                end.synchronize()
            executor.profile(profile_events)
        return execution

    compiler.gaudirt_backend = capture
    compiled = compiler.compile(connected_compact if compact else connected, dynamic_arg_indices=(0, 5, 6, 7))
    qkv_weight, out_weight = [(torch.randn(*shape) * 0.025).to(torch.bfloat16)
                              for shape in ((512, 256), (256, 256))]
    device_weights = [weight.to("gaudirt:0") for weight in (qkv_weight, out_weight)]
    payload, errors, floors, absolutes = bytearray(), [], [], []
    for rows in (1, 3, 17, 137, 513):
        metadata, locations, requests, positions, mapping = attention_ref.metadata(rows)
        x = torch.randn(rows, 256, dtype=torch.bfloat16)
        base_key, base_value = [torch.randn(attention_ref.SLOTS, 2, 64, dtype=torch.bfloat16)
                                for _ in range(2)]
        combined = gaudirt._C._allocate_paged_kv(1, attention_ref.SLOTS, 2, 64)
        combined[0, 0].copy_(base_key.transpose(0, 1).contiguous())
        combined[0, 1].copy_(base_value.transpose(0, 1).contiguous())
        key_pool, value_pool = combined[0, 0].transpose(0, 1), combined[0, 1].transpose(0, 1)
        plan = metadata.attention_plan(4, 2, 64)
        torch._dynamo.mark_dynamic(plan, 0, min=1, max=536870912)
        result = compiled(x.to("gaudirt:0"), *device_weights, key_pool, value_pool,
                          metadata.row_locations(), metadata.row_requests(), metadata.rotary_positions(),
                          plan, metadata.rotary_header())
        if profiling:
            assert len(executors[0].profile_host_ms) == 3
            assert all(value > 0 for value in executors[0].profile_host_ms)
            device_ms = []
            for start, end in profile_events:
                start.synchronize()
                end.synchronize()
                device_ms.append(start.elapsed_time(end))
            print("segment profile rows", rows, "host_ms", executors[0].profile_host_ms,
                  "device_ms", device_ms, flush=True)
            assert all(value >= 0 for value in device_ms) and any(value > 0 for value in device_ms)
        actual_key, actual_value = key_pool.cpu(), value_pool.cpu()
        untouched = torch.ones(attention_ref.SLOTS, dtype=torch.bool)
        untouched[locations] = False
        assert torch.equal(actual_key[untouched], base_key[untouched])
        assert torch.equal(actual_value[untouched], base_value[untouched])
        actual = result.cpu().double()
        truth = reference(x, qkv_weight, out_weight, base_key, base_value, locations, requests,
                          positions, mapping, True)
        floor = reference(x, qkv_weight, out_weight, base_key, base_value, locations, requests,
                          positions, mapping, False)
        assert torch.isfinite(actual).all() and torch.isfinite(truth).all() and torch.isfinite(floor).all()
        denom = torch.linalg.vector_norm(truth, dim=-1).clamp_min(1e-30)
        error = (torch.linalg.vector_norm(actual - truth, dim=-1) / denom).max().item()
        floor_error = (torch.linalg.vector_norm(floor - truth, dim=-1) / denom).max().item()
        absolute = (actual - truth).abs().max().item()
        assert error <= max(3e-3, 2 * floor_error) and error <= 1e-2, (rows, error, floor_error, absolute)
        errors.append(error); floors.append(floor_error); absolutes.append(absolute)
        for tensor in (result.cpu(), actual_key, actual_value):
            payload.extend(tensor.contiguous().view(torch.int16).numpy().tobytes())
        print("paged FX rows", rows, "cache", gaudirt.recipe_cache_stats(), flush=True)
        del result, key_pool, value_pool, combined, metadata, plan
    assert len(captures) == 1, len(captures)
    if not reference_split:
        assert calls == ["trace" if tracing else "profile" if profiling else "run"] * 5, calls
    if tracing:
        assert observed == [0, 1, 2] * 5, observed
    counts = gaudirt.recipe_cache_stats()
    if os.environ.get("GAUDIRT_TEST_CACHE") == "cold":
        assert counts[0] == 0 and counts[1] > 0, counts
    elif os.environ.get("GAUDIRT_TEST_CACHE") == "warm":
        assert counts[0] > 0 and counts[1] == 0, counts
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    print("paged FX stats", counts, "max row error", max(errors), "FP32 floor", max(floors),
          "max abs", max(absolutes), "sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
