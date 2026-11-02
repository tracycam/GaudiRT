// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "mimo_attention.hpp"

#include <cstdint>
#include <array>
#include <memory>
#include <utility>

namespace gaudirt::mimo {

class MimoFullAttention final {
 public:
  MimoFullAttention(HbmArena& arena, const WeightManifest& weights,
                    uint32_t cache_slots, bool pure_rmsnorm = false,
                    uint32_t page_count = 64, RecipeCache* cache = nullptr,
                    bool gathered_input = false, uint32_t rows = 1);
  ~MimoFullAttention();

  MimoFullAttention(const MimoFullAttention&) = delete;
  MimoFullAttention& operator=(const MimoFullAttention&) = delete;

  void execute(const MimoAttentionIo& io, synStreamHandle stream);
  void rebind_weights(const WeightManifest& weights);
  PreparedLaunch prepare(const MimoAttentionIo& io);

  uint64_t workspace_bytes() const noexcept;
  uint64_t recipe_identity() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gaudirt::mimo
