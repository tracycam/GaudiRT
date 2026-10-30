// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include <array>
#include <cstdint>

namespace gaudirt::runtime {
class HbmArena;
class RecipeCache;
}

namespace gaudirt::mimo {

// Table [152576,6144] BF16 and token I32 stay in HBM. Status is 0 for a
// valid token and 1 for an invalid token; callers must propagate it.
class ModelEmbedding final {
 public:
  ModelEmbedding(runtime::HbmArena& arena, uint64_t table,
                 runtime::RecipeCache* cache = nullptr);
  ~ModelEmbedding();
  ModelEmbedding(const ModelEmbedding&) = delete;
  ModelEmbedding& operator=(const ModelEmbedding&) = delete;

  void execute(uint64_t token, uint64_t embedding, uint64_t status,
               synStreamHandle stream);

 private:
  enum Slot : size_t { kTable, kToken, kEmbedding, kStatus, kSlotCount };
  runtime::HbmArena& arena_;
  synRecipeHandle recipe_ = nullptr;
  uint64_t workspace_bytes_ = 0;
  std::array<synLaunchTensorInfo, kSlotCount> launch_tensors_{};
};

}  // namespace gaudirt::mimo
