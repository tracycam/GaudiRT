// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>

namespace gaudirt::generic {

// The fixed Synapse descriptor is backed by this allocation for every layer.
inline constexpr uint64_t kPagedKvDescriptorSlots = 131072;
at::Tensor allocate_paged_kv(int64_t layers, int64_t slots, int64_t heads, int64_t head_dim);

}  // namespace gaudirt::generic
