"""The segmented fast path must still honor Dynamo's model-state guards."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
import gaudirt.compiler as compiler
import paged_attention_smoke as attention_ref
from bucket_fx_paged_partition_smoke import connected_compact


class Options:
    def __init__(self, offset):
        self.offset = offset


def connected_with_option(x, qkv_weight, out_weight, key_pool, value_pool, locations,
                          requests, positions, mapping, header, options):
    return connected_compact(x, qkv_weight, out_weight, key_pool, value_pool, locations,
                             requests, positions, mapping, header) + options.offset


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(51357)
    gaudirt.initialize(hbm_bytes=2 * 1024**3, pinned_bytes=4 * 1024**2)
    backend = compiler.gaudirt_backend
    captures, marks = [], []

    def capture(graph, inputs):
        captures.append(1)
        return backend(graph, inputs)

    original_mark = torch._dynamo.mark_dynamic

    def mark(*args, **kwargs):
        caller = sys._getframe(1)
        if caller.f_globals is compiler.__dict__ and caller.f_code.co_name == "execute":
            marks.append(1)
        return original_mark(*args, **kwargs)

    compiler.gaudirt_backend = capture
    torch._dynamo.mark_dynamic = mark
    run = compiler.compile(connected_with_option, dynamic_arg_indices=(0, 5, 6, 7))
    rows = 3
    metadata, *_ = attention_ref.metadata(rows)
    plan = metadata.attention_plan(4, 2, 64)
    torch._dynamo.mark_dynamic(plan, 0, min=1, max=536870912)
    combined = gaudirt._C._allocate_paged_kv(1, attention_ref.SLOTS, 2, 64)
    combined.zero_()
    weights = [(torch.randn(*shape) * 0.025).to(torch.bfloat16).to("gaudirt:0")
               for shape in ((512, 256), (256, 256))]
    value = torch.randn(rows, 256, dtype=torch.bfloat16).to("gaudirt:0")
    arguments = (value, *weights, combined[0, 0].transpose(0, 1), combined[0, 1].transpose(0, 1),
                 metadata.row_locations(), metadata.row_requests(), metadata.rotary_positions(),
                 plan, metadata.rotary_header())
    options = Options(0.0)
    first = run(*arguments, options).cpu()
    assert len(captures) == 1 and len(marks) == 4, (captures, marks)
    assert torch.equal(run(*arguments, options).cpu(), first)
    assert len(captures) == 1 and len(marks) == 4, (captures, marks)
    options.offset = 0.125
    changed = run(*arguments, options).cpu()
    expected = (first.float() + 0.125).to(torch.bfloat16)
    assert torch.equal(changed, expected)
    assert len(captures) == 2 and len(marks) == 8, (captures, marks)
    options.offset = 0.0
    assert torch.equal(run(*arguments, options).cpu(), first)
    assert len(captures) == 2 and len(marks) == 8, (captures, marks)
    print("segmented guards: warm hits skipped marking; changed model option recaptured", flush=True)


if __name__ == "__main__":
    main()
