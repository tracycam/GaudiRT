# SPDX-License-Identifier: Apache-2.0
"""Stream the MiMo TP8 model into GaudiRT for a single vLLM worker."""

import json
from pathlib import Path

import numpy as np
import torch

from runner import ModelRunner
from stream_weights import iter_dense_weights, iter_layer_weights, load_model_weight

_FULL_LAYERS = {0, 7, 15, 23, 31, 39, 47, 55, 62, 69}


class _ModelStep(torch.nn.Module):
    def forward(self, trigger):
        return torch.ops.gaudirt.model_step(trigger)


def _rope(theta):
    inverse = 1.0 / (float(theta) ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    angles = torch.arange(4224, dtype=torch.float64).unsqueeze(1) * inverse
    return torch.cat((angles.cos(), angles.sin()), dim=1).to(torch.bfloat16).contiguous()


class ServingModel:
    def __init__(self, extension, rank, model_path, physical_pages):
        if physical_pages < 64:
            raise ValueError('at least 64 physical KV pages are required')
        self.extension = extension
        self.rank = rank
        self.runner = None
        self.pages = None
        self.finished = False
        model_path = Path(model_path)
        config = json.loads((model_path / 'config.json').read_text())
        self.full_rope = _rope(config['rope_theta'])
        self.swa_rope = _rope(config['swa_rope_theta'])
        extension.enable_model_fusion(rank)
        extension.enable_prefill(rank)
        for layer in range(70):
            for name, tensor in iter_layer_weights(model_path, layer, rank):
                value = tensor.contiguous() if isinstance(tensor, torch.Tensor) else torch.from_numpy(tensor).contiguous()
                extension.upload_weight(rank, layer, False, name, value)
            if layer == 0:
                for name, tensor in iter_dense_weights(model_path, rank):
                    extension.upload_weight(rank, layer, True, name, tensor.contiguous())
            kind = 0 if layer == 0 else (1 if layer in _FULL_LAYERS else 2)
            extension.register_streamed_layer(rank, layer, kind, physical_pages * 128,
                                              64 if kind != 2 else 2)
            extension.register_empty_model_kv(rank, layer)
        for name in ('embedding', 'final_norm', 'lm_head'):
            extension.upload_model_weight(rank, name, load_model_weight(model_path, name))
        trigger = extension.finalize_model(rank, self.full_rope,
                                           self.swa_rope,
                                           torch.arange(64, dtype=torch.int32), 0)

        def capture(graph, inputs):
            calls = [node for node in graph.graph.nodes if node.op == 'call_function']
            if len(calls) != 1 or 'gaudirt.model_step' not in str(calls[0].target):
                raise ValueError('compiled decode graph must contain one model step')
            return graph.forward

        compiled = torch.compile(_ModelStep(), fullgraph=True, backend=capture)
        self.runner = ModelRunner(extension, rank, compiled, trigger)

    def update_pages(self, pages):
        if pages != self.pages:
            self.extension.update_model_pages(self.rank, torch.tensor(pages, dtype=torch.int32))
            self.pages = list(pages)

    def reset(self):
        if not self.finished:
            raise ValueError('reset requires a complete previous prefill')
        self.extension.reset_prefill(self.rank)
        self.finished = False
        self.pages = None

    def prefill(self, tokens, pages, prompt_length):
        if self.finished:
            raise ValueError('reset is required before a new request')
        self.update_pages(pages)
        if not tokens:
            return -1
        result = self.extension.prefill_chunk(self.rank, torch.tensor(tokens, dtype=torch.int32),
                                              self.full_rope[:4096].contiguous(),
                                              self.swa_rope[:4096].contiguous(), prompt_length)
        if result >= 0:
            self.finished = True
        return result

    def decode(self, pages):
        if not self.finished:
            raise ValueError('decode requires a complete prefill')
        self.update_pages(pages)
        return self.runner.step()

    def close(self):
        self.runner.close()
        self.runner = None
