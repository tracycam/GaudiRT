"""GaudiRT PyTorch PrivateUse1 & c10d distributed backend."""

import atexit
import os
import sys

if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
    raise RuntimeError("set TORCH_DEVICE_BACKEND_AUTOLOAD=0 before importing torch")
if "PT_HPU_LAZY_MODE" in os.environ:
    raise RuntimeError("PT_HPU_LAZY_MODE is incompatible with GaudiRT")


def _audit_legacy_bridge():
    if any(name == "habana_frameworks" or name.startswith("habana_frameworks.")
           for name in sys.modules):
        raise RuntimeError("GaudiRT: legacy Python bridge is loaded")
    with open("/proc/self/maps", encoding="utf-8") as maps:
        mapped = maps.read().lower()
    if any(name in mapped for name in ("libhabana_pytorch", "gaudi-pytorch-bridge",
                                       "habana_frameworks")):
        raise RuntimeError("GaudiRT: legacy bridge library is mapped")


_audit_legacy_bridge()
import torch
import torch.distributed as _dist
import _gaudirt_C as _C
_audit_legacy_bridge()


def _resolve_device_index(device=None) -> int:
    count = _C.device_count()
    if device is None:
        idx = _C.current_device()
    elif isinstance(device, bool):
        raise ValueError(f"GaudiRT: invalid device {device!r}")
    elif isinstance(device, int):
        idx = device
    elif isinstance(device, str):
        if device == "gaudirt":
            idx = _C.current_device()
        elif device.startswith("gaudirt:"):
            tail = device.split(":", 1)[1]
            if not tail.isdigit():
                raise ValueError(f"GaudiRT: invalid device {device!r}")
            idx = int(tail)
        else:
            raise ValueError(f"GaudiRT: invalid device {device!r}")
    elif isinstance(device, torch.device):
        if device.type != "gaudirt":
            raise ValueError(f"GaudiRT: invalid device {device!r}")
        idx = _C.current_device() if device.index is None else int(device.index)
    else:
        raise ValueError(f"GaudiRT: invalid device {device!r}")
    if idx < 0 or idx >= count:
        raise ValueError(f"GaudiRT: device index {idx} out of range for visible count {count}")
    return idx


def _check_active_device(device=None) -> int:
    idx = _resolve_device_index(device)
    active = _C.current_device()
    if idx != active:
        raise ValueError(
            f"GaudiRT: process active device is gaudirt:{active}, cannot query gaudirt:{idx}"
        )
    return idx


class Stream(torch.Stream):
    def __new__(cls, device=None, priority=0):
        idx = _resolve_device_index(device)
        return super().__new__(cls, device=f"gaudirt:{idx}", priority=priority)


class Event:
    def __new__(cls, enable_timing=False, blocking=False, interprocess=False):
        if blocking or interprocess:
            raise ValueError("GaudiRT: blocking and interprocess events are unsupported")
        idx = _C.current_device()
        return torch.Event(device=f"gaudirt:{idx}", enable_timing=enable_timing)


def set_device(device):
    idx = _resolve_device_index(device)
    _C.set_device(idx)


def current_stream(device=None):
    idx = _resolve_device_index(device)
    return torch.accelerator.current_stream(idx)


def stream(value):
    if not isinstance(value, torch.Stream) or value.device.type != "gaudirt":
        raise ValueError("GaudiRT: expected a gaudirt stream")
    return value


StreamContext = stream


@torch.library.register_fake("gaudirt::rms_norm")
def _fake_rms_norm(input, weight, epsilon, weight_offset=0.0):
    return torch.empty_like(input, memory_format=torch.contiguous_format)


@torch.library.register_fake("gaudirt::fused_add_rms_norm")
def _fake_fused_add_rms_norm(input, residual, weight, epsilon, weight_offset=0.0):
    return (torch.empty_like(input, memory_format=torch.contiguous_format),
            torch.empty_like(residual, memory_format=torch.contiguous_format))


@torch.library.register_fake("gaudirt::silu_and_mul")
def _fake_silu_and_mul(input):
    return input.new_empty((*input.shape[:-1], input.shape[-1] // 2))


@torch.library.register_fake("gaudirt::gelu_and_mul")
def _fake_gelu_and_mul(input, approximate="none"):
    return input.new_empty((*input.shape[:-1], input.shape[-1] // 2))


@torch.library.register_fake("gaudirt::block_fp8_linear")
def _fake_block_fp8_linear(input, weight, scales, bias=None):
    return input.new_empty((*input.shape[:-1], weight.shape[1]))


@torch.library.register_fake("gaudirt::fp8_linear")
def _fake_fp8_linear(input, weight, scales, bias=None):
    return input.new_empty((*input.shape[:-1], weight.shape[0]))


@torch.library.register_fake("gaudirt::paged_attn")
def _fake_paged_attn(query, key, value, key_pool, value_pool, metadata, head_dim, scale,
                     softcap=0.0, sliding_window=-1):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


@torch.library.register_fake("gaudirt::_paged_attn_validated")
def _fake_paged_attn_validated(query, key, value, key_pool, value_pool, locations,
                               requests, positions, mapping, header, head_dim, scale,
                               softcap=0.0, sliding_window=-1):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


@torch.library.register_fake("gaudirt::rotary_embedding")
def _fake_rotary_embedding(query, key, cache, metadata, head_dim, rotary_dim, is_neox):
    return (torch.empty_like(query, memory_format=torch.contiguous_format),
            torch.empty_like(key, memory_format=torch.contiguous_format))


@torch.library.register_fake("gaudirt::rotary_embedding_positions")
def _fake_rotary_embedding_positions(query, key, cache, positions, head_dim, rotary_dim, is_neox):
    return (torch.empty_like(query, memory_format=torch.contiguous_format),
            torch.empty_like(key, memory_format=torch.contiguous_format))


@torch.library.register_fake("gaudirt::_rotary_embedding_validated")
def _fake_rotary_embedding_validated(query, key, cache, positions, header, context,
                                     head_dim, rotary_dim, is_neox):
    return (torch.empty_like(query, memory_format=torch.contiguous_format),
            torch.empty_like(key, memory_format=torch.contiguous_format))


@torch.library.register_fake("gaudirt::softcap_")
def _fake_softcap_inplace(logits, cap):
    return logits


@torch.library.register_fake("gaudirt::sample_from_logits")
def _fake_sample_from_logits(logits, temperatures, top_ps, top_ks, min_ps,
                             seeds=None, positions=None, nan_to_neg_inf=True):
    return logits.new_empty((logits.shape[0],), dtype=torch.int32)


MAX_BUCKET_ROWS = int(_C._max_bucket_rows)


torch.utils.rename_privateuse1_backend("gaudirt")
torch._register_device_module("gaudirt", sys.modules[__name__])
torch.utils.generate_methods_for_privateuse1_backend(for_packed_sequence=False)


def _create_gaudirt_process_group(dist_opts, pg_opts=None):
    del pg_opts
    group_id = getattr(dist_opts, "group_id", "") or ""
    return _C._create_process_group_gaudirt(
        dist_opts.store,
        int(dist_opts.group_rank),
        int(dist_opts.group_size),
        str(group_id),
    )


if "gaudirt" not in _dist.Backend.backend_list:
    _dist.Backend.register_backend(
        "gaudirt",
        _create_gaudirt_process_group,
        extended_api=True,
        devices=["gaudirt"],
    )


def configure_default_runtime(hbm_bytes=256 * 1024 * 1024, pinned_bytes=8 * 1024 * 1024):
    _C.configure_default_runtime(hbm_bytes, pinned_bytes)


def initialize_current_device(hbm_bytes=256 * 1024 * 1024, pinned_bytes=8 * 1024 * 1024):
    _C.initialize_current_device(hbm_bytes, pinned_bytes)


def initialize(module_id=None, hbm_bytes=256 * 1024 * 1024, pinned_bytes=8 * 1024 * 1024):
    if module_id is None:
        _C.initialize_current_device(hbm_bytes, pinned_bytes)
    else:
        _C.initialize(module_id, hbm_bytes, pinned_bytes)


def _reserve_rotary_cache(cache):
    """Keep the logical lookup extent while reserving static descriptor storage."""
    if (cache.device.type not in ("cpu", "gaudirt") or cache.dtype != torch.bfloat16 or
            cache.ndim != 2 or not cache.is_contiguous() or min(cache.shape) <= 0):
        raise ValueError("GaudiRT: rotary cache must be a contiguous BF16 matrix")
    rows = max(cache.shape[0], _C._rotary_static_rows)
    target_device = (
        cache.device if cache.device.type == "gaudirt"
        else torch.device(f"gaudirt:{_C.current_device()}")
    )
    storage = torch.empty((rows, cache.shape[1]), device=target_device, dtype=cache.dtype)
    visible = storage.narrow(0, 0, cache.shape[0])
    visible.copy_(cache, non_blocking=True)
    return visible


def validated_token_ids(metadata):
    """Return logical token IDs that have passed the batch metadata gate."""
    tokens = metadata.token_ids()
    vocabulary = metadata.vocabulary()
    if vocabulary <= 0:
        raise ValueError("GaudiRT: token metadata has no validated vocabulary")
    tokens._gaudirt_validated_vocab = vocabulary
    return tokens


def validated_last_token_indices(metadata):
    """Return one validated last-row index per request in the current batch."""
    indices = metadata.last_token_indices()
    indices._gaudirt_validated_rows = metadata.rows()
    return indices


def is_initialized():
    return _C.is_initialized()


def mem_get_info(device=None):
    _check_active_device(device)
    return _C.memory_info()


def physical_mem_get_info(device=None):
    _check_active_device(device)
    return _C.physical_memory_info()


def shutdown():
    _C.shutdown()


def synchronize(device=None):
    _check_active_device(device)
    _C.synchronize()


def is_available():
    return True


def device_count():
    return _C.device_count()


def current_device():
    return _C.current_device()


def visible_module_for_device(device=None):
    return _C.visible_module_for_device(_resolve_device_index(device))


def _is_in_bad_fork():
    return False


def memory_allocated(device=None):
    if device is not None:
        _check_active_device(device)
    return _C.memory_allocated()


def max_memory_allocated(device=None):
    if device is not None:
        _check_active_device(device)
    return _C.max_memory_allocated()


def empty_cache():
    _C.empty_cache()


def recipe_cache_stats():
    return _C.recipe_cache_stats()


_unhandled_exception = False
_prev_excepthook = sys.excepthook


def _gaudirt_excepthook(exc_type, exc_value, exc_tb):
    global _unhandled_exception
    _unhandled_exception = True
    _prev_excepthook(exc_type, exc_value, exc_tb)


sys.excepthook = _gaudirt_excepthook


def _exit_audit():
    try:
        _audit_legacy_bridge()
        _C._shutdown_at_exit(_unhandled_exception)
    except Exception as error:
        print(error, file=sys.stderr, flush=True)
        os._exit(1)


from .fx_collectives import register_collective_op, register_comm_group_resolver  # noqa: E402

atexit.register(_exit_audit)
