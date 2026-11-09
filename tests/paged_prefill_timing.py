"""Compare full prefill attention paths with device-stream timing and KV checks."""

import atexit
import hashlib
import math
import os
import sys
import time

assert os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") == "0"


def no_bridge():
    assert "habana_frameworks" not in sys.modules
    with open("/proc/self/maps", encoding="utf-8") as maps:
        assert not any("habana_pytorch" in line for line in maps)


no_bridge()
atexit.register(no_bridge)

import gaudirt
import torch
from fp64_reference import direct_bf16


def main():
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(45193)
    rows = int(os.environ.get("GAUDIRT_TEST_ROWS", "137"))
    heads = int(os.environ.get("GAUDIRT_TEST_HEADS", "4"))
    kv_heads = int(os.environ.get("GAUDIRT_TEST_KV_HEADS", "2"))
    dim = int(os.environ.get("GAUDIRT_TEST_HEAD_DIM", "128"))
    repetitions = int(os.environ.get("GAUDIRT_TEST_TIMING_REPETITIONS", "3"))
    hbm = int(os.environ.get("GAUDIRT_TEST_HBM_BYTES", str(2 * 1024**3)))
    assert 0 < rows <= 4096 and heads > 0 and kv_heads > 0 and heads % kv_heads == 0
    assert dim in (64, 128) and kv_heads * dim % 128 == 0 and repetitions > 0
    gaudirt.initialize(hbm_bytes=hbm, pinned_bytes=16 * 1024**2)
    slots = rows + 1
    mapping = torch.arange(1, slots, dtype=torch.int32).reshape(1, rows)
    positions = torch.arange(rows, dtype=torch.int32)
    requests = torch.zeros(rows, dtype=torch.int32)
    locations = mapping.flatten().contiguous()
    metadata = torch.classes.gaudirt.PagedMetadata(locations, requests, positions, mapping, slots)
    q, k, v = [torch.randn(rows, width, dtype=torch.bfloat16) for width in
               (heads * dim, kv_heads * dim, kv_heads * dim)]
    combined = gaudirt._C._allocate_paged_kv(1, slots, kv_heads, dim)
    combined.zero_()
    key_pool, value_pool = combined[0, 0].transpose(0, 1), combined[0, 1].transpose(0, 1)
    device_q = q.to("gaudirt:0")
    packed_kv = torch.empty((rows, kv_heads * dim * 2), dtype=torch.bfloat16, device="gaudirt:0")
    device_k, device_v = packed_kv[:, :kv_heads * dim], packed_kv[:, kv_heads * dim:]
    device_k.copy_(k)
    device_v.copy_(v)
    scale = 1 / math.sqrt(dim)

    def launch():
        return torch.ops.gaudirt.paged_attn.default(device_q, device_k, device_v, key_pool, value_pool,
                                                     metadata, dim, scale)

    result = launch()
    gaudirt.synchronize()
    start, end = gaudirt.Event(enable_timing=True), gaudirt.Event(enable_timing=True)
    start.record()
    host_start = time.perf_counter()
    for _ in range(repetitions):
        result = launch()
    host_ms = 1000 * (time.perf_counter() - host_start) / repetitions
    end.record()
    end.synchronize()
    stream_ms = start.elapsed_time(end) / repetitions
    result_cpu = result.cpu()
    actual_k, actual_v = combined[0, 0].cpu().transpose(0, 1), combined[0, 1].cpu().transpose(0, 1)
    assert torch.equal(actual_k[1:], k.reshape(rows, kv_heads, dim))
    assert torch.equal(actual_v[1:], v.reshape(rows, kv_heads, dim))
    assert torch.count_nonzero(actual_k[0]) == 0 and torch.count_nonzero(actual_v[0]) == 0
    head_map = torch.arange(heads) // (heads // kv_heads)
    errors = []
    for row in sorted({0, min(rows - 1, 511), rows // 2, rows - 1}):
        keys = k[:row + 1].reshape(-1, kv_heads, dim)[:, head_map].double()
        values = v[:row + 1].reshape(-1, kv_heads, dim)[:, head_map].double()
        scores = torch.einsum("hd,thd->ht", q[row].reshape(heads, dim).double(), keys) * scale
        truth = direct_bf16(torch.einsum("ht,thd->hd", scores.softmax(-1), values).reshape(-1))
        actual = result_cpu[row].double()
        assert torch.isfinite(actual).all() and torch.isfinite(truth).all()
        errors.append((torch.linalg.vector_norm(actual - truth) /
                       torch.linalg.vector_norm(truth).clamp_min(1e-30)).item())
    assert max(errors) <= 1e-2, max(errors)
    payload = b"".join(t.contiguous().view(torch.int16).numpy().tobytes()
                       for t in (result_cpu, actual_k, actual_v))
    if path := os.environ.get("GAUDIRT_TEST_OUTPUT"):
        with open(path, "wb") as output:
            output.write(payload)
    stats = gaudirt.recipe_cache_stats()
    print("prefill", rows, "heads", heads, "kv_heads", kv_heads, "head_dim", dim,
          "device_stream_ms", stream_ms, "host_submit_ms", host_ms, "samples_max_relative_l2", max(errors),
          "cache", stats, "output_and_kv_sha256", hashlib.sha256(payload).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
