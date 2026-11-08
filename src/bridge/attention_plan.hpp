// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "bridge_internal.hpp"

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <vector>

namespace gaudirt::generic {

struct AttentionTile {
  uint64_t indices_offset;
  uint64_t header_offset;
  int32_t first_key;
  int32_t valid_keys;
};

struct AttentionBlock {
  int32_t first_row;
  int32_t valid_rows;
  uint32_t query_bucket;
  uint32_t kv_bucket;
  uint64_t positions_offset;
  uint64_t store_header_offset;
  std::vector<AttentionTile> tiles;
};

// Carried by the device tensor itself. This keeps the host schedule alive
// without a global pointer table or any per-layer lookups.
struct AttentionPlan {
  std::weak_ptr<Context> owner;
  uint64_t address;
  uint64_t bytes;
  int64_t rows, slots, query_heads, kv_heads, head_dim;
  std::vector<AttentionBlock> blocks;
};

const AttentionPlan& attention_plan_data(const at::Tensor& tensor, const Context& ctx);

}  // namespace gaudirt::generic
