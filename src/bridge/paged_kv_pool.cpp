// SPDX-License-Identifier: Apache-2.0
#include "paged_kv_pool.hpp"

#include "bridge_internal.hpp"

#include <ATen/ATen.h>
#include <c10/util/Exception.h>

#include <array>
#include <cstdint>
#include <limits>

namespace gaudirt::generic {

at::Tensor allocate_paged_kv(int64_t layers, int64_t slots, int64_t heads, int64_t head_dim) {
  TORCH_CHECK(layers > 0 && layers <= 4096 && slots > 0 &&
              static_cast<uint64_t>(slots) <= kPagedKvDescriptorSlots &&
              heads > 0 && heads <= 256 && (head_dim == 64 || head_dim == 128) &&
              (heads * head_dim) % 128 == 0,
              "GaudiRT: unsupported head-major KV pool geometry");
  const uint64_t width = static_cast<uint64_t>(heads) * head_dim;
  const uint64_t layer_bytes = 2 * static_cast<uint64_t>(slots) * width * sizeof(uint16_t);
  const uint64_t descriptor_bytes = 2 * kPagedKvDescriptorSlots * width * sizeof(uint16_t);
  const uint64_t count = static_cast<uint64_t>(layers);
  TORCH_CHECK(layer_bytes <= (std::numeric_limits<uint64_t>::max() - descriptor_bytes) / count,
              "GaudiRT: KV pool size overflow");
  const uint64_t actual_bytes = count * layer_bytes;
  // Each layer's descriptor begins at that layer's K/V data. Reserve one
  // descriptor-length tail for the final layer, rather than one per layer.
  const uint64_t storage_bytes = (count - 1) * layer_bytes + descriptor_bytes;
  auto ctx = current();
  const auto stream = ctx->stream();
  const auto address = ctx->allocator.allocate(storage_bytes, stream);
  auto data = allocation_data_ptr(ctx, address, storage_bytes, stream, AllocationKind::device);
  const std::array<int64_t, 5> shape{layers, 2, heads, slots, head_dim};
  auto result = tensor_from_allocation(std::move(data), storage_bytes, shape,
      at::TensorOptions().device(c10::Device(c10::kPrivateUse1, ctx->device_index)).dtype(at::kBFloat16));
  TORCH_CHECK(synMemsetD32Async(address, 0, actual_bytes / 4, stream) == synSuccess,
              "GaudiRT: failed to initialize KV pool");
  return result;
}

}  // namespace gaudirt::generic
