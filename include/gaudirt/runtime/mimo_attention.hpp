// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include <cstdint>
#include <memory>

namespace gaudirt::runtime {

class HbmArena;
class WeightManifest;

// BF16 shapes are fastest-first: hidden_states/residual/residual_out/normalized
// and output_local [6144,1]; qkv [3392,1]; query [192,16,1]; new_key
// [192,1,1]; new_value/context [128,1,1] and [128,16,1]. rope_cos/rope_sin
// are [64,1,1], with each 32-value half of the loaded cache row repeated twice.
// key_cache/value_cache are [192,1,cache_slots]/[128,1,cache_slots].
// slot_mapping/position are I32[1]; window_pages/window_groups are I32[2].
struct MimoAttentionIo {
  uint64_t hidden_states = 0;
  uint64_t residual = 0;
  uint64_t residual_out = 0;
  uint64_t normalized = 0;
  uint64_t rope_cos = 0;
  uint64_t rope_sin = 0;

  uint64_t key_cache = 0;
  uint64_t value_cache = 0;
  uint64_t slot_mapping = 0;
  uint64_t window_pages = 0;
  uint64_t window_groups = 0;
  uint64_t position = 0;

  uint64_t qkv = 0;
  uint64_t query = 0;
  uint64_t new_key = 0;
  uint64_t new_value = 0;
  uint64_t context = 0;
  uint64_t output_local = 0;
};

// One decode row of MiMo layer1 on one TP8 rank. Every address is mandatory.
// The graph updates key_cache/value_cache in place and produces TP-local
// output_local. The reference TP contract is BF16 allGather followed by a
// rank-order FP32 sum and BF16 rounding, all outside this graph (no allReduce).
// Required WeightManifest tensors: input_layernorm, qkv_weight, qkv_scales,
// qkv_bias, attention_sink, oproj_weight. All addresses must stay valid until
// stream completion, and cache metadata must reference valid physical pages.
class MimoAttention {
 public:
  static constexpr uint32_t kHiddenSize = 6144;
  static constexpr uint32_t kQkvSize = 3392;
  static constexpr uint32_t kQueryHeads = 16;
  static constexpr uint32_t kHeadDim = 192;
  static constexpr uint32_t kValueHeadDim = 128;
  static constexpr uint32_t kKvBlockSize = 128;

  MimoAttention(HbmArena& arena, const WeightManifest& weights,
                uint32_t cache_slots);
  ~MimoAttention();

  MimoAttention(const MimoAttention&) = delete;
  MimoAttention& operator=(const MimoAttention&) = delete;

  void execute(const MimoAttentionIo& io, synStreamHandle stream);
  void rebind_weights(const WeightManifest& weights);

  uint32_t cache_slots() const noexcept;
  uint64_t workspace_bytes() const noexcept;
  uint64_t recipe_identity() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gaudirt::runtime
