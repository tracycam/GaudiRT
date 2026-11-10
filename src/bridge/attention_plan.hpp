// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "bridge_internal.hpp"

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace gaudirt::generic {

class AttentionStagingPool {
 public:
  struct Lease { size_t index; uint64_t address; void* host; };
  explicit AttentionStagingPool(Context& context);
  ~AttentionStagingPool();
  void prepare_stream(synStreamHandle stream);
  Lease acquire(uint64_t device_bytes, uint64_t host_bytes);
  void release(size_t index, synStreamHandle compute);
  bool trim();

 private:
  struct Slot {
    uint64_t address = 0, device_bytes = 0, host_bytes = 0;
    void* host = nullptr;
    runtime::EventPool::Event completed;
    bool in_use = false, recorded = false;
  };
  Context& context_;
  runtime::EventPool events_;
  std::mutex mutex_;
  std::vector<Slot> slots_;
  void release_all();
};

struct AttentionTile {
  uint64_t indices_offset;
  uint64_t header_offset;
  int32_t first_key;
  int32_t valid_keys;
};

inline constexpr size_t kMaxFusedAttentionTiles = 8;

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
  uint64_t scratch_offset;
  uint64_t scratch_bytes;
  uint64_t store_header_offset;
  synStreamHandle stream;
  size_t staging_slot = 0;
  int64_t rows, slots, query_heads, kv_heads, head_dim;
  std::vector<AttentionBlock> blocks;
};

const AttentionPlan& attention_plan_data(const at::Tensor& tensor, const Context& ctx);

}  // namespace gaudirt::generic
