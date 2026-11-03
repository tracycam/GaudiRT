# SPDX-License-Identifier: Apache-2.0
"""Check request-to-token mapping at the C++ serving interface."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib.util
import sys

import torch

runner = ModuleType('runner')
runner.ModelRunner = object
sys.modules['runner'] = runner
weights = ModuleType('stream_weights')
weights.iter_dense_weights = weights.iter_layer_weights = weights.load_model_weight = None
sys.modules['stream_weights'] = weights
path = Path(__file__).resolve().parents[1] / 'models/mimo/serving_model.py'
spec = importlib.util.spec_from_file_location('serving_model_batch_tested', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Extension:
    def __init__(self):
        self.released = []

    def prefill_request(self, rank, request_id, tokens, full_rope, swa_rope, length, pages):
        assert rank == 2 and tokens.tolist() == [10, 20] and pages == [4]
        assert full_rope.shape == swa_rope.shape == (4096, 64)
        return 72

    def stage_batch_requests(self, rank, ids, tokens, positions, pages):
        assert (rank, ids, tokens, positions, pages) == (2, ['a', 'b'], [72, 91], [2, 7], [[4], [5]])

    def finish_batch_model_step(self, rank):
        assert rank == 2
        return [81, 92]

    def release_request(self, rank, request_id):
        self.released.append((rank, request_id))


def test_batch_request_tokens_follow_request_ids():
    model = module.ServingModel.__new__(module.ServingModel)
    model.rank = 2
    model.max_num_seqs = 4
    model.extension = Extension()
    model.batch_trigger = object()
    model.batch_compiled = lambda trigger: trigger is model.batch_trigger
    model.request_tokens = {'b': 91}
    model.full_rope = model.swa_rope = torch.zeros((4224, 64), dtype=torch.bfloat16)
    assert model.prefill_request('a', [10, 20], [4], 2) == 72
    steps = [SimpleNamespace(request_id='a', position=2, block_ids=[4]),
             SimpleNamespace(request_id='b', position=7, block_ids=[5])]
    assert model.decode_requests(steps) == [81, 92]
    assert model.request_tokens == {'a': 81, 'b': 92}
    model.release_request('a')
    assert model.request_tokens == {'b': 92} and model.extension.released == [(2, 'a')]
