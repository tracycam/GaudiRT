"""Check fresh Synapse runtime logs after allocator event-reuse tests."""

import argparse
import os
from pathlib import Path
import sys

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"
assert "habana_frameworks" not in sys.modules

parser = argparse.ArgumentParser()
parser.add_argument("logs", nargs="+", type=Path)
args = parser.parse_args()
for path in args.logs:
    content = path.read_text()
    if not content:
        raise AssertionError(f"empty Synapse runtime log: {path}")
    if "using an event that was recorded but not waited upon" in content.lower():
        raise AssertionError(f"unconsumed event record in {path}")
assert "habana_frameworks" not in sys.modules
print(f"Synapse event re-record warning absent from {len(args.logs)} log(s)")
