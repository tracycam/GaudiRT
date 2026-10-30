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

class ModelHead final {
 public:
  ModelHead(runtime::HbmArena& arena, uint64_t final_norm_gamma,
            uint64_t lm_head_weight, runtime::RecipeCache* cache = nullptr);
  ~ModelHead();

  ModelHead(const ModelHead&) = delete;
  ModelHead& operator=(const ModelHead&) = delete;

  void execute(uint64_t hidden, uint64_t residual, uint64_t normalized,
               uint64_t logits, synStreamHandle stream);
  uint64_t workspace_bytes() const noexcept { return workspace_bytes_; }
  uint64_t recipe_identity() const noexcept {
    return reinterpret_cast<uint64_t>(recipe_);
  }

 private:
  enum Slot : size_t {
    kHidden = 0,
    kResidual,
    kGamma,
    kLmHeadWeight,
    kNormalized,
    kLogits,
    kSlotCount,
  };

  runtime::HbmArena& arena_;
  synRecipeHandle recipe_ = nullptr;
  uint64_t workspace_bytes_ = 0;
  std::array<synLaunchTensorInfo, kSlotCount> launch_tensors_{};
};

}  // namespace gaudirt::mimo
