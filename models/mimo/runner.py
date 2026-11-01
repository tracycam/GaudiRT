"""Own the compiled decode step and release it before the device runtime."""

import gc
import os
import sys

if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD', '0') != '0':
    raise RuntimeError('device backend autoload must be disabled')
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import torch


class ModelRunner:
    def __init__(self, extension, rank, compiled_step, trigger):
        if ('habana_frameworks' in sys.modules or 'PT_HPU_LAZY_MODE' in os.environ):
            raise RuntimeError('legacy bridge is not allowed in the model runner')
        with open('/proc/self/maps', encoding='utf-8') as maps:
            mapped = maps.read().lower()
        if any(name in mapped for name in ('libhabana_pytorch', 'gaudi-pytorch-bridge',
                                            'habana_frameworks')):
            raise RuntimeError('legacy bridge library is mapped in the model runner')
        self._extension = extension
        self._rank = rank
        self._compiled_step = compiled_step
        self._trigger = trigger

    def step(self):
        if self._compiled_step is None:
            raise RuntimeError('model runner is closed')
        self._compiled_step(self._trigger)
        return self._extension.finish_model_step(self._rank)

    def close(self):
        if self._compiled_step is None:
            raise RuntimeError('model runner is already closed')
        self._compiled_step = None
        self._trigger = None
        torch._dynamo.reset()
        gc.collect()
        self._extension.shutdown(self._rank)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
