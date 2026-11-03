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


class _BatchModelStep(torch.nn.Module):
    def forward(self, trigger):
        return torch.ops.gaudirt.batch_model_step(trigger)


def _rope(theta):
    inverse = 1.0 / (float(theta) ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
    angles = torch.arange(4224, dtype=torch.float64).unsqueeze(1) * inverse
    return torch.cat((angles.cos(), angles.sin()), dim=1).to(torch.bfloat16).contiguous()


class ServingModel:
    def __init__(self, extension, rank, model_path, physical_pages, max_num_seqs=1):
        if not 1 <= max_num_seqs <= 8 or physical_pages < (256 if max_num_seqs > 1 else 64):
            raise ValueError('invalid request capacity or physical KV page count')
        self.extension = extension
        self.rank = rank
        self.runner = None
        self.batch_compiled = None
        self.batch_trigger = None
        self.pages = None
        self.finished = False
        self.max_num_seqs = max_num_seqs
        self.request_tokens = {}
        model_path = Path(model_path)
        config = json.loads((model_path / 'config.json').read_text())
        self.full_rope = _rope(config['rope_theta'])
        self.swa_rope = _rope(config['swa_rope_theta'])
        extension.enable_model_fusion(rank)
        extension.enable_prefill(rank)
        if max_num_seqs > 1:
            extension.enable_batch_decode(rank)
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
        if max_num_seqs > 1:
            def capture_batch(graph, inputs):
                calls = [node for node in graph.graph.nodes if node.op == 'call_function']
                if len(calls) != 1 or 'gaudirt.batch_model_step' not in str(calls[0].target):
                    raise ValueError('compiled batch graph must contain one model step')
                return graph.forward
            self.batch_compiled = torch.compile(_BatchModelStep(), fullgraph=True, backend=capture_batch)
            self.batch_trigger = trigger

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

    def prefill_request(self, request_id, tokens, pages, prompt_length):
        if self.max_num_seqs <= 1 or not tokens:
            raise ValueError('multi-request prefill requires a nonempty token fragment')
        token = self.extension.prefill_request(
            self.rank, request_id, torch.tensor(tokens, dtype=torch.int32),
            self.full_rope[:4096].contiguous(), self.swa_rope[:4096].contiguous(),
            prompt_length, pages)
        if token >= 0:
            self.request_tokens[request_id] = token
        return token

    def decode_requests(self, steps):
        if self.max_num_seqs <= 1 or not steps:
            raise ValueError('multi-request decode requires scheduled requests')
        ids = [step.request_id for step in steps]
        tokens = [self.request_tokens[request_id] for request_id in ids]
        self.extension.stage_batch_requests(
            self.rank, ids, tokens, [step.position for step in steps],
            [step.block_ids for step in steps])
        self.batch_compiled(self.batch_trigger)
        result = self.extension.finish_batch_model_step(self.rank)
        for request_id, token in zip(ids, result, strict=True):
            self.request_tokens[request_id] = token
        return result

    def release_request(self, request_id):
        self.extension.release_request(self.rank, request_id)
        self.request_tokens.pop(request_id, None)

    def close(self):
        self.batch_compiled = None
        self.batch_trigger = None
        self.runner.close()
        self.runner = None
