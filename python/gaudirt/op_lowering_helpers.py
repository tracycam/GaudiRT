"""Helper functions and op-target tables for FX-to-BucketProgram lowering."""

import math
import operator
import struct

import torch

# c10::ScalarType numeric enum values (verified in torch/headeronly/core/ScalarType.h)
_SCALAR_TYPE_CODES = {
    torch.int32: 3,
    torch.int64: 4,
    torch.float32: 6,
    torch.bool: 11,
    torch.bfloat16: 15,
}


def _scalar_type_code(dtype: torch.dtype) -> int:
    code = _SCALAR_TYPE_CODES.get(dtype)
    if code is None:
        raise RuntimeError(f"GaudiRT FX: unsupported scalar dtype {dtype}")
    return code


def _resolve_scalar_value(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, torch.SymInt):
        return int(value)
    if isinstance(value, torch.SymFloat):
        return float(value)
    return None


def _finite_float32(value):
    try:
        result = struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as error:
        raise RuntimeError("GaudiRT FX: scalar is outside FP32 range") from error
    if not math.isfinite(result):
        raise RuntimeError("GaudiRT FX: scalar must be finite")
    return result


def _node_example(node):
    if not isinstance(node, torch.fx.Node):
        return None
    ex = node.meta.get("example_value", node.meta.get("val"))
    if isinstance(ex, torch.Tensor):
        return ex
    return None


def _broadcast_shape(s1, s2):
    if s1 is None or s2 is None:
        return None
    rank = max(len(s1), len(s2))
    a = (1,) * (rank - len(s1)) + s1
    b = (1,) * (rank - len(s2)) + s2
    out = []
    for x, y in zip(a, b):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            out.append(max(x, y))
    return tuple(out)


def _gaudirt_targets(name: str):
    pkt = getattr(torch.ops.gaudirt, name, None)
    if pkt is None:
        return ()
    default = getattr(pkt, "default", None)
    return (pkt, default) if default is not None else (pkt,)


def build_op_target_groups():
    aten = torch.ops.aten
    return {
        "linear": (torch.nn.functional.linear, aten.linear.default),
        "block_fp8": _gaudirt_targets("block_fp8_linear"),
        "fp8": _gaudirt_targets("fp8_linear"),
        "mm": (torch.mm, aten.mm.default, torch.matmul, aten.matmul.default, operator.matmul),
        "bmm": (torch.bmm, aten.bmm.default),
        "addmm": (torch.addmm, aten.addmm.default),
        "add": (operator.add, torch.add, aten.add.Tensor, aten.add.Scalar),
        "sub": (operator.sub, torch.sub, aten.sub.Tensor, aten.sub.Scalar),
        "mul": (operator.mul, torch.mul, aten.mul.Tensor, aten.mul.Scalar),
        "div": (operator.truediv, torch.div, torch.true_divide, aten.div.Tensor, aten.div.Scalar),
        "norm": _gaudirt_targets("rms_norm"),
        "fused_norm": _gaudirt_targets("fused_add_rms_norm"),
        "silu_mul": _gaudirt_targets("silu_and_mul"),
        "gelu_mul": _gaudirt_targets("gelu_and_mul"),
        "rotary": _gaudirt_targets("_rotary_embedding_validated"),
        "rotary_pos": _gaudirt_targets("rotary_embedding_positions"),
        "split": (torch.split, aten.split_with_sizes.default, aten.split.Tensor),
        "cat": (torch.cat, torch.concat, torch.concatenate, aten.cat.default),
        "embedding": (torch.nn.functional.embedding, aten.embedding.default),
        "index_select": (torch.index_select, aten.index_select.default),
        "reshape": (torch.reshape, aten.view.default, aten.reshape.default),
        "transpose": (torch.transpose, aten.transpose.int),
        "unsqueeze": (torch.unsqueeze, aten.unsqueeze.default),
        "squeeze": (torch.squeeze, aten.squeeze.default, aten.squeeze.dim, aten.squeeze.dims),
        "expand": (aten.expand.default,),
        "exp": (torch.exp, aten.exp.default),
        "tanh": (torch.tanh, aten.tanh.default),
        "sigmoid": (torch.sigmoid, aten.sigmoid.default),
        "silu": (torch.nn.functional.silu, aten.silu.default),
        "gelu": (torch.nn.functional.gelu, aten.gelu.default),
        "clamp": (torch.clamp, aten.clamp.default),
        "clamp_min": (aten.clamp_min.default,),
        "clamp_max": (aten.clamp_max.default,),
        "rsqrt": (torch.rsqrt, aten.rsqrt.default),
        "sqrt": (torch.sqrt, aten.sqrt.default),
        "neg": (operator.neg, torch.neg, aten.neg.default),
        "abs": (operator.abs, torch.abs, aten.abs.default),
        "isnan": (torch.isnan, aten.isnan.default),
        "where": (torch.where, aten.where.self),
        "min": (torch.minimum, aten.minimum.default),
        "max": (torch.maximum, aten.maximum.default),
        "logical_and": (torch.logical_and, aten.logical_and.default),
        "logical_or": (torch.logical_or, aten.logical_or.default),
        "logical_not": (operator.not_, torch.logical_not, aten.logical_not.default),
        "bitwise_and": (operator.and_, torch.bitwise_and, aten.bitwise_and.Tensor),
        "bitwise_or": (operator.or_, torch.bitwise_or, aten.bitwise_or.Tensor),
        "eq": (operator.eq, torch.eq, aten.eq.Tensor, aten.eq.Scalar),
        "ne": (operator.ne, torch.ne, aten.ne.Tensor, aten.ne.Scalar),
        "lt": (operator.lt, torch.lt, aten.lt.Tensor, aten.lt.Scalar),
        "le": (operator.le, torch.le, aten.le.Tensor, aten.le.Scalar),
        "gt": (operator.gt, torch.gt, aten.gt.Tensor, aten.gt.Scalar),
        "ge": (operator.ge, torch.ge, aten.ge.Tensor, aten.ge.Scalar),
        "sum": (torch.sum, aten.sum.default, aten.sum.dim_IntList),
        "mean": (torch.mean, aten.mean.default, aten.mean.dim),
        "amax": (torch.amax, aten.amax.default),
        "softmax": (torch.softmax, torch.nn.functional.softmax, aten.softmax.int, aten._softmax.default),
        "log_softmax": (torch.log_softmax, torch.nn.functional.log_softmax, aten.log_softmax.int, aten._log_softmax.default),
        "layer_norm": (torch.nn.functional.layer_norm, aten.layer_norm.default),
        "native_layer_norm": (aten.native_layer_norm.default,),
        "full_like": (torch.full_like, aten.full_like.default),
    }
