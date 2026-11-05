"""Single-device GaudiRT PyTorch backend."""

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
import _gaudirt_C as _C
_audit_legacy_bridge()


@torch.library.register_fake("gaudirt::rms_norm")
def _fake_rms_norm(input, weight, epsilon):
    return torch.empty_like(input, memory_format=torch.contiguous_format)


@torch.library.register_fake("gaudirt::fused_add_rms_norm")
def _fake_fused_add_rms_norm(input, residual, weight, epsilon):
    return (torch.empty_like(input, memory_format=torch.contiguous_format),
            torch.empty_like(residual, memory_format=torch.contiguous_format))


@torch.library.register_fake("gaudirt::silu_and_mul")
def _fake_silu_and_mul(input):
    return input.new_empty((*input.shape[:-1], input.shape[-1] // 2))


@torch.library.register_fake("gaudirt::paged_attn")
def _fake_paged_attn(query, key, value, key_pool, value_pool, metadata, head_dim, scale):
    return torch.empty_like(query, memory_format=torch.contiguous_format)


@torch.library.register_fake("gaudirt::rotary_embedding")
def _fake_rotary_embedding(query, key, cache, metadata, head_dim, rotary_dim, is_neox):
    return (torch.empty_like(query, memory_format=torch.contiguous_format),
            torch.empty_like(key, memory_format=torch.contiguous_format))


torch.utils.rename_privateuse1_backend("gaudirt")
torch._register_device_module("gaudirt", sys.modules[__name__])
torch.utils.generate_methods_for_privateuse1_backend(for_packed_sequence=False)


def initialize(module_id=0, hbm_bytes=256 * 1024 * 1024, pinned_bytes=8 * 1024 * 1024):
    _C.initialize(module_id, hbm_bytes, pinned_bytes)


def shutdown():
    _C.shutdown()


def synchronize(device=None):
    if device not in (None, 0, "gaudirt:0", torch.device("gaudirt:0")):
        raise ValueError("GaudiRT: invalid device")
    _C.synchronize()


def is_available():
    return True


def device_count():
    return 1


def current_device():
    return 0


def _is_in_bad_fork():
    return False


def memory_allocated(device=None):
    return _C.memory_allocated()


def max_memory_allocated(device=None):
    return _C.max_memory_allocated()


def empty_cache():
    _C.empty_cache()


def recipe_cache_stats():
    return _C.recipe_cache_stats()


def _exit_audit():
    try:
        _audit_legacy_bridge()
        _C._shutdown_at_exit()
    except Exception as error:
        print(error, file=sys.stderr, flush=True)
        os._exit(1)


atexit.register(_exit_audit)
