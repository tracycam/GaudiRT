"""Eight-rank hardware smoke: runner owns its graph and device tensor."""

import argparse
import importlib
import os
from pathlib import Path
import sys
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--uid', required=True)
    p.add_argument('--ready-dir', type=Path, required=True)
    p.add_argument('--runner-path', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, os.environ['GAUDIRT_PRIVATE_BUILD_DIR'])
    sys.path.insert(0, str(args.runner_path))
    ext = importlib.import_module('gaudirt_private_use1')
    from runner import ModelRunner

    torch.utils.rename_privateuse1_backend('gaudirt')
    torch.utils.generate_methods_for_privateuse1_backend(for_packed_sequence=False)

    class Backend:
        current_device = staticmethod(lambda: args.rank)
        device_count = staticmethod(lambda: 8)
        is_available = staticmethod(lambda: True)
        _is_in_bad_fork = staticmethod(lambda: False)

    torch._register_device_module('gaudirt', Backend)
    ext.initialize(args.rank, 8, args.uid, '', 64 * 1024 ** 2)
    compiled = torch.compile(lambda x: x + 1, fullgraph=True,
                             backend=lambda gm, inputs: gm.forward)
    assert compiled(torch.ones(1)).item() == 2.0
    runner = ModelRunner(ext, args.rank, compiled,
                         torch.empty((1,), dtype=torch.int32, device=f'gaudirt:{args.rank}'))
    del compiled
    (args.ready_dir / f'ready-{args.rank}').write_text('ready\n')
    deadline = time.monotonic() + 120
    while not all((args.ready_dir / f'ready-{r}').is_file() for r in range(8)):
        if time.monotonic() > deadline:
            raise TimeoutError('rank initialization timed out')
        time.sleep(0.01)
    if args.rank == 7:
        time.sleep(2)
    before = time.monotonic()
    runner.close()
    elapsed = time.monotonic() - before
    if args.rank == 0 and elapsed < 1:
        raise AssertionError('shutdown returned before delayed rank')
    if 'habana_frameworks' in sys.modules or 'PT_HPU_LAZY_MODE' in os.environ:
        raise RuntimeError('legacy bridge in candidate process')
    print(f'PASS rank={args.rank} shutdown_seconds={elapsed:.3f}', flush=True)


if __name__ == '__main__':
    main()
