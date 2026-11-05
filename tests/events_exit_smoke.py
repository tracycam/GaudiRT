"""The exit audit releases live tensors, streams, and events before the device."""

import os
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

import torch
import gaudirt

gaudirt.initialize(hbm_bytes=256 * 1024 * 1024, pinned_bytes=4 * 1024 * 1024)
stream = gaudirt.Stream()
event = gaudirt.Event()
with gaudirt.stream(stream):
    value = torch.arange(65536, dtype=torch.int32).to("gaudirt:0", non_blocking=True)
    event.record()
print("leaving live stream, event, and tensor for exit audit", flush=True)
