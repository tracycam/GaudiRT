"""Unimplemented FX operations must fail at capture without eager fallback."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt
from gaudirt.compiler import compile


def unsupported(x, weight):
    return torch.mm(x, weight.t()) + 1


gaudirt.initialize(hbm_bytes=32 * 1024 * 1024, pinned_bytes=1024 * 1024)
x = torch.empty((3, 64), dtype=torch.bfloat16, device="gaudirt:0")
weight = torch.empty((64, 64), dtype=torch.bfloat16, device="gaudirt:0")
compiled = compile(unsupported)
try:
    compiled(x, weight)
except Exception as error:
    assert "GaudiRT FX: unsupported" in str(error), str(error)
else:
    raise AssertionError("unsupported FX operation silently fell back")
assert gaudirt.recipe_cache_stats() == [0, 0, 0]
del x, weight
gaudirt.shutdown()
print("unsupported FX operation failed without fallback", flush=True)
