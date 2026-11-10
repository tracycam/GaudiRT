"""Stable parameters bind once; replacements and storage changes stay visible."""

import os
import sys
import gc

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import torch.nn.functional as F
import gaudirt
import gaudirt.compiler as compiler
import paged_attention_smoke as attention


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = torch.nn.Parameter((torch.randn(512, 256) * .025).to(torch.bfloat16).to("gaudirt"), False)
        self.out = torch.nn.Parameter((torch.randn(256, 256) * .025).to(torch.bfloat16).to("gaudirt"), False)

    def forward(self, x, key_pool, value_pool, locations, requests, positions, plan, header):
        q, k, v = F.linear(x, self.qkv).split([256, 128, 128], dim=-1)
        y = torch.ops.gaudirt._paged_attn_validated.default(
            q, k, v, key_pool, value_pool, locations, requests, positions, plan, header, 64, .125)
        return F.linear(y, self.out)


def main():
    torch.manual_seed(641)
    gaudirt.initialize(hbm_bytes=2 * 1024**3, pinned_bytes=8 * 1024**2)
    original = compiler.gaudirt_backend
    captures, bound_calls, bindings = [], [], []

    def capture(graph, inputs):
        forward = original(graph, inputs)
        execution = forward.__self__
        captures.append(execution)

        class CountRunner:
            def __init__(self, runner):
                self.runner = runner

            def bind_static(self, slots, tensors):
                bindings.append((len(slots), len(tensors)))
                return self.runner.bind_static(slots, tensors)

            def run(self, values):
                return self.runner.run(values)

            def run_bound(self, values):
                bound_calls.append(len(values))
                return self.runner.run_bound(values)

            def run_bound_traced(self, values, indices):
                return self.runner.run_bound_traced(values, indices)

        execution.runner = CountRunner(execution.runner)
        return forward

    compiler.gaudirt_backend = capture
    model = Model()
    compiled = compiler.compile(model.forward, dynamic_arg_indices=(0, 3, 4, 5))
    metadata, *_ = attention.metadata(3)
    pool = gaudirt._C._allocate_paged_kv(1, attention.SLOTS, 2, 64)
    key, value = pool[0, 0].transpose(0, 1), pool[0, 1].transpose(0, 1)
    key._gaudirt_static_binding = value._gaudirt_static_binding = True
    plan = metadata.attention_plan(4, 2, 64)
    torch._dynamo.mark_dynamic(plan, 0, min=1, max=536870912)
    x_cpu = torch.randn(3, 256, dtype=torch.bfloat16)
    x = x_cpu.to("gaudirt")
    args = (x, key, value, metadata.row_locations(), metadata.row_requests(),
            metadata.rotary_positions(), plan, metadata.rotary_header())
    first = compiled(*args).cpu()
    second = compiled(*args).cpu()
    assert torch.equal(first, second)
    assert bindings and bindings[0][0] >= 2 and bound_calls, (bindings, bound_calls)
    assert bound_calls[0] < len(captures[0].input_indices), bound_calls

    # Retained trace outputs and the returned output must never be overwritten
    # when the next invocation reuses same-bucket intermediate storage.
    execution = captures[0]
    execution.enable_trace()
    first_segment = next(iter(execution.named_children()))[1]
    previous = first_segment.forward
    observed = []

    def hold(*values):
        output = previous(*values)
        observed.append(tuple(item for item in (output if isinstance(output, tuple) else (output,))
                              if isinstance(item, torch.Tensor)))
        return output

    first_segment.forward = hold
    returned = compiled(*args)
    saved_returned = returned.cpu().clone()
    saved_intermediate = observed[0][0].cpu().clone()
    args2 = ((-x_cpu).to("gaudirt"), *args[1:])
    other = compiled(*args2)
    assert torch.equal(observed[0][0].cpu(), saved_intermediate)
    assert torch.equal(returned.cpu(), saved_returned)
    assert observed[0][0].data_ptr() != observed[1][0].data_ptr()
    assert not torch.equal(other.cpu(), saved_returned)
    del returned, other, observed[:]
    gc.collect()
    execution.tracing = False
    gaudirt.empty_cache()
    assert torch.equal(compiled(*args).cpu(), first)

    # Same Tensor object with new storage must be read through its new pointer.
    model.out.data = torch.zeros((256, 256), dtype=torch.bfloat16).to("gaudirt")
    changed_storage = compiled(*args).cpu()
    assert torch.count_nonzero(changed_storage) == 0 and torch.count_nonzero(first) > 0
    # Replacing the Parameter object must fail the identity guard and recapture.
    model.out = torch.nn.Parameter(torch.ones((256, 256), dtype=torch.bfloat16).to("gaudirt"), False)
    replaced = compiled(*args).cpu()
    assert len(captures) == 2 and not torch.equal(changed_storage, replaced), len(captures)
    print("static bindings, output alias lifetime, cache release and parameter replacement passed", flush=True)


if __name__ == "__main__":
    main()
