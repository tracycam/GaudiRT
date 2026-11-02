// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/prepared_launch.hpp"

#include <synapse_api.h>

#include <cstdint>
#include <array>
#include <memory>
#include <utility>

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

class MimoDenseMlp final {
 public:
  struct Inputs {
    uint64_t attention_bf16 = 0;
    uint64_t residual_in_bf16 = 0;
    uint64_t post_attention_gamma_bf16 = 0;
  };

  struct Outputs {
    uint64_t residual_out_bf16 = 0;
    uint64_t activation_fp32 = 0;
    uint64_t output_local_fp32 = 0;
  };

  MimoDenseMlp(HbmArena& arena, const WeightManifest& weights,
               RecipeCache* cache = nullptr, bool gathered_attention = false,
               uint32_t rows = 1);
  ~MimoDenseMlp();

  MimoDenseMlp(const MimoDenseMlp&) = delete;
  MimoDenseMlp& operator=(const MimoDenseMlp&) = delete;

  void execute(const Inputs& inputs, const Outputs& outputs,
               synStreamHandle stream);
  void rebind_weights(const WeightManifest& weights);
  std::array<PreparedLaunch, 2> prepare(const Inputs& inputs,
                                        const Outputs& outputs);

  uint64_t workspace_bytes() const noexcept;
  std::pair<uint64_t, uint64_t> recipe_identity() const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gaudirt::mimo
