"""A live staged graph-input alias must not break interpreter exit."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch

import gaudirt
gaudirt.initialize(hbm_bytes=128 * 1024**2, pinned_bytes=4 * 1024**2)
original = torch.arange(102, dtype=torch.float32).reshape(3, 34).to("gaudirt:0")
alias = gaudirt._C._prepare_strided_input(original[:, ::2])
assert torch.equal(alias.cpu(), torch.arange(102, dtype=torch.float32).reshape(3, 34)[:, ::2])
print("exiting with a live graph-pool alias", flush=True)
