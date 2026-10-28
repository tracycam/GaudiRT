// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/mimo_attention.hpp"

#include <cstdint>
#include <memory>
#include <utility>

namespace gaudirt::runtime {

class MimoFullAttention final {
 public:
  MimoFullAttention(HbmArena& arena, const WeightManifest& weights,
                    uint32_t cache_slots);
  ~MimoFullAttention();

  MimoFullAttention(const MimoFullAttention&) = delete;
  MimoFullAttention& operator=(const MimoFullAttention&) = delete;

  void execute_prepare(const MimoAttentionIo& io, synStreamHandle stream);
  void execute_finish(const MimoAttentionIo& io, synStreamHandle stream);
  void rebind_weights(const WeightManifest& weights);

  uint64_t workspace_bytes() const noexcept;
  std::pair<uint64_t, uint64_t> recipe_identity() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gaudirt::runtime
