// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "attention_plan.hpp"

namespace gaudirt::generic {

struct TiledAttentionInputs {
  uint64_t query = 0;
  uint64_t pool = 0;
  uint64_t output = 0;
  uint64_t direct_key = 0;
  uint64_t direct_value = 0;
  uint64_t tail_key = 0;
  uint64_t tail_value = 0;
};

inline const char* kv_store_guid(int64_t head_dim) {
  TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
              "GaudiRT attention: unsupported head_dim for KV store");
  return "gk_kv_store_head_major_bf16_v1";
}

void launch_layer_kv_store(Context& ctx, const AttentionPlan& plan, uint64_t key, uint64_t value,
                           uint64_t locations, uint64_t pool);
void launch_tiled_attention(Context& ctx, const AttentionPlan& plan, const AttentionBlock& block,
                            const TiledAttentionInputs& input, float scale,
                            float softcap = 0.0f, int32_t sliding_window = -1);

}  // namespace gaudirt::generic
