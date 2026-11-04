"""Exercise interpreter teardown with live device and pinned storages."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt

gaudirt.initialize(hbm_bytes=8 * 1024 * 1024, pinned_bytes=1024 * 1024)
tensor = torch.empty((128,), dtype=torch.float32, device="gaudirt:0")
tensor.fill_(3.5)
view = tensor.narrow(0, 1, 8)
pinned = torch.empty((128,), dtype=torch.float32, pin_memory=True)
pinned.copy_(tensor, non_blocking=True)

if os.environ.get("GAUDIRT_TEST_EXIT") == "exception":
    raise RuntimeError("injected exception with live tensors")

try:
    gaudirt.shutdown()
except RuntimeError as error:
    assert "tensors must be destroyed" in str(error)
else:
    raise AssertionError("shutdown accepted live device tensors")

print("live device and pinned tensors retained for interpreter teardown", flush=True)
