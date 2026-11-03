// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/prepared_launch.hpp"

#include <synapse_api.h>

#include <cstdint>
#include <memory>

namespace gaudirt::runtime {
class HbmArena;
class WeightManifest;
class RecipeCache;
}

namespace gaudirt::mimo {
using runtime::HbmArena;
using runtime::WeightManifest;
using runtime::RecipeCache;
using runtime::PreparedLaunch;

// BF16 shapes are fastest-first: hidden_states/residual/residual_out and
// output_local [6144,M]. rope_cos/rope_sin are [64,1,M], with each 32-value
// half of the loaded cache row repeated twice.
// key_cache/value_cache are [192,1,cache_slots]/[128,1,cache_slots].
// slot_mapping/position are I32[M]; window_pages/window_groups are I32[2]
// for decode M=1 or I32[64] for the static prefill bucket M=32.
// Full-Attention block list/usage/groups are I32[page_count], page_count <= 64.
struct MimoAttentionIo {
  uint64_t hidden_states = 0;
  uint64_t residual = 0;
  uint64_t residual_out = 0;
  uint64_t rope_cos = 0;
  uint64_t rope_sin = 0;

  uint64_t key_cache = 0;
  uint64_t value_cache = 0;
  uint64_t slot_mapping = 0;
  uint64_t window_pages = 0;
  uint64_t window_groups = 0;
  uint64_t full_block_list = 0;
  uint64_t full_block_usage = 0;
  uint64_t full_block_groups = 0;
  uint64_t position = 0;

  uint64_t output_local = 0;
  uint64_t output_local_fp32 = 0;
  // Required by batched Full Attention; one group id per request row.
  uint64_t full_row_groups = 0;
};

// One decode row of MiMo layer1 on one TP8 rank. Every address is mandatory.
// The graph updates key_cache/value_cache in place and produces TP-local
// output_local. The reference TP contract is BF16 allGather followed by a
// rank-order FP32 sum and BF16 rounding, all outside this graph (no allReduce).
// Required WeightManifest tensors: input_layernorm, qkv_weight, qkv_scales,
// qkv_bias, attention_sink, oproj_weight. All addresses must stay valid until
// stream completion, and cache metadata must reference valid physical pages.
// The model QKV reducer requires zero bias and finite prepared weight scales
// >= 2^-20; its compensation does not support FP32 subnormal terms.
class MimoAttention {
 public:
  static constexpr uint32_t kHiddenSize = 6144;
  static constexpr uint32_t kQkvSize = 3392;
  static constexpr uint32_t kQueryHeads = 16;
  static constexpr uint32_t kHeadDim = 192;
  static constexpr uint32_t kValueHeadDim = 128;
  static constexpr uint32_t kKvBlockSize = 128;

  MimoAttention(HbmArena& arena, const WeightManifest& weights,
                uint32_t cache_slots, RecipeCache* cache = nullptr,
                bool gathered_input = false, uint32_t rows = 1);
  ~MimoAttention();

  MimoAttention(const MimoAttention&) = delete;
  MimoAttention& operator=(const MimoAttention&) = delete;

  void execute(const MimoAttentionIo& io, synStreamHandle stream);
  void rebind_weights(const WeightManifest& weights);
  PreparedLaunch prepare(const MimoAttentionIo& io);

  uint32_t cache_slots() const noexcept;
  uint64_t workspace_bytes() const noexcept;
  uint64_t recipe_identity() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gaudirt::mimo
