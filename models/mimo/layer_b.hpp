// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/device_runtime.hpp"
#include "mimo_full_attention.hpp"
#include "mimo_moe.hpp"
#include "prepared_layer.hpp"
#include "gaudirt/runtime/tp_reduce.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace gaudirt::mimo {

class LayerB final {
 public:
  struct Io {
    runtime::MimoAttentionIo attention;
    runtime::MimoMoeLayer1::Outputs moe_outputs;
    uint64_t attention_output_bf16 = 0;
    uint64_t attention_gathered_fp32 = 0;
    uint64_t moe_gathered_fp32 = 0;
    uint64_t layer_output_bf16 = 0;
  };

  LayerB(runtime::HbmArena& arena, runtime::DeviceRuntime& runtime,
         const runtime::WeightManifest& weights, uint32_t cache_slots,
         uint32_t page_count = 64, runtime::RecipeCache* cache = nullptr,
         bool fused_model = false);

  LayerB(const LayerB&) = delete;
  LayerB& operator=(const LayerB&) = delete;

  void execute(const Io& io, synStreamHandle stream);
  PreparedLayer prepare(const Io& io);
  void rebind_weights(const runtime::WeightManifest& weights);
  std::vector<uint64_t> recipe_identity() const {
    return {attention_.recipe_identity(), moe_.recipe_identity()};
  }

 private:
  runtime::HbmArena& arena_;
  runtime::MimoFullAttention attention_;
  std::unique_ptr<runtime::TpReduce> attention_reduce_;
  runtime::MimoMoeLayer1::Weights moe_weights_{};
  runtime::MimoMoeLayer1 moe_;
  std::unique_ptr<runtime::TpReduce> moe_reduce_;
  hcclComm_t comm_;
  bool fused_model_ = false;
};

}  // namespace gaudirt::mimo
