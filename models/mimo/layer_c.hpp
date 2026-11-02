// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/device_runtime.hpp"
#include "mimo_attention.hpp"
#include "mimo_moe.hpp"
#include "prepared_layer.hpp"
#include "gaudirt/runtime/tp_reduce.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace gaudirt::mimo {

class LayerC final {
 public:
  struct Io {
    MimoAttentionIo attention;
    MimoMoeLayer1::Outputs moe_outputs;
    uint64_t attention_output_bf16 = 0;
    uint64_t attention_gathered_bf16 = 0;
    uint64_t moe_gathered_fp32 = 0;
    uint64_t layer_output_bf16 = 0;
  };

  LayerC(runtime::HbmArena& arena, runtime::DeviceRuntime& runtime,
         const runtime::WeightManifest& weights, uint32_t cache_slots,
         runtime::RecipeCache* cache = nullptr, bool fused_model = false, uint32_t rows = 1,
         uint64_t shared_lut = 0, uint64_t shared_directions = 0);

  LayerC(const LayerC&) = delete;
  LayerC& operator=(const LayerC&) = delete;

  void execute(const Io& io, synStreamHandle stream);
  PreparedLayer prepare(const Io& io);
  void rebind_weights(const runtime::WeightManifest& weights);
  uint64_t moe_lut() const { return moe_weights_.mxfp4_lut_bf16; }
  uint64_t moe_directions() const { return moe_weights_.down_combine_directions_u8; }
  std::vector<uint64_t> recipe_identity() const {
    return {attention_.recipe_identity(), moe_.recipe_identity()};
  }

 private:
  runtime::HbmArena& arena_;
  MimoAttention attention_;
  std::unique_ptr<runtime::TpReduce> attention_reduce_;
  MimoMoeLayer1::Weights moe_weights_{};
  MimoMoeLayer1 moe_;
  std::unique_ptr<runtime::TpReduce> moe_reduce_;
  hcclComm_t comm_;
  bool fused_model_ = false;
  uint32_t rows_ = 1;
};

}  // namespace gaudirt::mimo
