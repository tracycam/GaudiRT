// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/prepared_launch.hpp"

#include <synapse_api.h>

#include <array>
#include <cstdint>
#include <string>

namespace gaudirt::runtime {
class HbmArena;
class RecipeCache;
}

namespace gaudirt::mimo {

// Returns the lowest-index maximum, or -1 if a logit is not finite or the
// embedding status is nonzero. All inputs and the result reside on device.
class ModelArgmax final {
 public:
  explicit ModelArgmax(runtime::HbmArena& arena, runtime::RecipeCache* cache = nullptr,
                       uint32_t rows = 1);
  ~ModelArgmax();
  ModelArgmax(const ModelArgmax&) = delete;
  ModelArgmax& operator=(const ModelArgmax&) = delete;

  void execute(uint64_t logits, uint64_t embedding_status, uint64_t next_token,
               synStreamHandle stream);
  runtime::PreparedLaunch prepare(uint64_t logits, uint64_t embedding_status,
                                  uint64_t next_token);

 private:
  enum Slot : size_t { kLogits, kEmbeddingStatus, kNextToken, kSlotCount };
  runtime::HbmArena& arena_;
  synRecipeHandle recipe_ = nullptr;
  uint64_t workspace_bytes_ = 0;
  std::array<std::string, kSlotCount> tensor_names_{};
  std::array<synLaunchTensorInfo, kSlotCount> launch_tensors_{};
};

}  // namespace gaudirt::mimo
