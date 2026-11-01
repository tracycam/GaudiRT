# SPDX-License-Identifier: Apache-2.0
"""Stream TP8 checkpoint slices and pack them on CPU, one layer at a time."""
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import safe_open

from gaudi_kernels.block_fp8 import prepare_block_fp8
from gaudi_kernels.serving.executor.packing import pack




def iter_layer_weights(model, layer, rank):
    model = Path(model)
    if layer not in range(70) or rank not in range(8):
        raise ValueError('This loader requires 70-layer TP8 geometry')
    weight_map = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = f'model.layers.{layer}.'
    full_layers = {0, 7, 15, 23, 31, 39, 47, 55, 62, 69}
    nonexpert = ['self_attn.qkv_proj.weight', 'self_attn.qkv_proj.weight_scale_inv',
                 'self_attn.o_proj.weight', 'input_layernorm.weight']
    if layer not in full_layers:
        nonexpert.append('self_attn.attention_sink_bias')
    if layer != 0:
        nonexpert.extend(['post_attention_layernorm.weight', 'mlp.gate.weight',
                          'mlp.gate.e_score_correction_bias'])
    checkpoint_names = sorted({weight_map[prefix + suffix] for suffix in nonexpert})
    if len(checkpoint_names) != 1:
        raise ValueError('Expected one nonexpert checkpoint shard')
    with safe_open(str(model / checkpoint_names[0]), framework='pt', device='cpu') as tensors:
        qkv = tensors.get_slice(prefix + 'self_attn.qkv_proj.weight')[rank * 3392:(rank + 1) * 3392].contiguous()
        scales = tensors.get_slice(prefix + 'self_attn.qkv_proj.weight_scale_inv')[rank * 27:(rank + 1) * 27].contiguous()
        prepared = prepare_block_fp8(qkv, scales)
        yield 'qkv_weight', prepared.weight
        yield 'qkv_scales', prepared.scales
        yield 'qkv_bias', prepared.bias
        yield 'oproj_weight', tensors.get_slice(prefix + 'self_attn.o_proj.weight')[:, rank * 2048:(rank + 1) * 2048]
        if layer not in full_layers:
            yield 'attention_sink', tensors.get_slice(prefix + 'self_attn.attention_sink_bias')[rank * 16:(rank + 1) * 16]
        yield 'input_layernorm', tensors.get_tensor(prefix + 'input_layernorm.weight')
        if layer == 0:
            return
        for name, suffix in [('post_attention_layernorm', 'post_attention_layernorm.weight'),
                             ('router_weight', 'mlp.gate.weight'),
                             ('router_bias', 'mlp.gate.e_score_correction_bias')]:
            yield name, tensors.get_tensor(prefix + suffix)

    gp = np.empty((384 * 6144, 256), np.uint8)
    gs = np.empty((384 * 192, 512), np.uint8)
    dp = np.empty((384 * 12 * 256, 256), np.uint8)
    ds = np.empty((384 * 12 * 8, 512), np.uint8)
    with ExitStack() as stack:
        handles = {}
        for expert in range(384):
            source = model / weight_map[f'{prefix}mlp.experts.{expert}.gate_proj.weight']
            if source not in handles:
                handles[source] = stack.enter_context(safe_open(str(source), framework='np', device='cpu'))
            tensors = handles[source]
            gate = tensors.get_slice(f'{prefix}mlp.experts.{expert}.gate_proj.weight')[rank * 256:(rank + 1) * 256]
            up = tensors.get_slice(f'{prefix}mlp.experts.{expert}.up_proj.weight')[rank * 256:(rank + 1) * 256]
            gate_scale = tensors.get_slice(f'{prefix}mlp.experts.{expert}.gate_proj.weight_scale')[rank * 256:(rank + 1) * 256]
            up_scale = tensors.get_slice(f'{prefix}mlp.experts.{expert}.up_proj.weight_scale')[rank * 256:(rank + 1) * 256]
            w13 = np.concatenate([gate, up], axis=0)
            s13 = np.concatenate([gate_scale, up_scale], axis=0)
            packed, packed_scales = pack(w13, s13)
            gp[expert * 6144:(expert + 1) * 6144] = packed.reshape(6144, 256)
            gs[expert * 192:(expert + 1) * 192] = packed_scales.reshape(192, 512)
            down = tensors.get_slice(f'{prefix}mlp.experts.{expert}.down_proj.weight')[:, rank * 128:(rank + 1) * 128]
            down_scale = tensors.get_slice(f'{prefix}mlp.experts.{expert}.down_proj.weight_scale')[:, rank * 8:(rank + 1) * 8]
            packed, packed_scales = pack(down, down_scale)
            dp[expert * 3072:(expert + 1) * 3072] = packed.reshape(3072, 256)
            ds[expert * 96:(expert + 1) * 96] = packed_scales.reshape(96, 512)
    yield 'moe_gp', gp
    yield 'moe_gs', gs
    yield 'moe_dp', dp
    yield 'moe_ds', ds


def iter_dense_weights(model, rank):
    model = Path(model)
    if rank not in range(8):
        raise ValueError('Dense weights require TP8')
    weight_map = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'model.layers.0.'
    suffixes = ('post_attention_layernorm.weight', 'mlp.gate_proj.weight',
                'mlp.gate_proj.weight_scale_inv', 'mlp.up_proj.weight',
                'mlp.up_proj.weight_scale_inv', 'mlp.down_proj.weight',
                'mlp.down_proj.weight_scale_inv')
    with ExitStack() as stack:
        checkpoints = {name: stack.enter_context(safe_open(
            str(model / name), framework='pt', device='cpu'))
            for name in {weight_map[prefix + suffix] for suffix in suffixes}}

        def tensor(suffix):
            return checkpoints[weight_map[prefix + suffix]].get_slice(prefix + suffix)

        gate_up = torch.cat((tensor('mlp.gate_proj.weight')[rank * 2048:(rank + 1) * 2048],
                             tensor('mlp.up_proj.weight')[rank * 2048:(rank + 1) * 2048]), dim=0).contiguous()
        gate_up_scales = torch.cat((
            tensor('mlp.gate_proj.weight_scale_inv')[rank * 16:(rank + 1) * 16],
            tensor('mlp.up_proj.weight_scale_inv')[rank * 16:(rank + 1) * 16]), dim=0).contiguous()
        if tuple(gate_up.shape) != (4096, 6144) or tuple(gate_up_scales.shape) != (32, 48):
            raise ValueError('layers.0 gate_up TP8 sharding contract changed')
        prepared_gate_up = prepare_block_fp8(gate_up, gate_up_scales)
        down = tensor('mlp.down_proj.weight')[:, rank * 2048:(rank + 1) * 2048].contiguous()
        down_scales = tensor('mlp.down_proj.weight_scale_inv')[:, rank * 16:(rank + 1) * 16].contiguous()
        if tuple(down.shape) != (6144, 2048) or tuple(down_scales.shape) != (48, 16):
            raise ValueError('layers.0 down TP8 sharding contract changed')
        prepared_down = prepare_block_fp8(down, down_scales)
        yield 'post_attention_gamma', checkpoints[weight_map[prefix + 'post_attention_layernorm.weight']].get_tensor(
            prefix + 'post_attention_layernorm.weight')
        yield 'gate_up_weight', prepared_gate_up.weight
        yield 'gate_up_scales', prepared_gate_up.scales
        yield 'gate_up_bias', prepared_gate_up.bias
        yield 'down_weight', prepared_down.weight
        yield 'down_scales', prepared_down.scales
        yield 'down_bias', prepared_down.bias


def load_model_weight(model, name):
    model = Path(model)
    suffixes = {'embedding': 'model.embed_tokens.weight',
                'final_norm': 'model.norm.weight', 'lm_head': 'lm_head.weight'}
    if name not in suffixes:
        raise ValueError('Unknown model weight')
    weight_map = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    suffix = suffixes[name]
    with safe_open(str(model / weight_map[suffix]), framework='pt', device='cpu') as tensors:
        return tensors.get_tensor(suffix)
