// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/device_runtime.hpp"
#include "mimo_dense_mlp.hpp"
#include "mimo_full_attention.hpp"
#include "prepared_layer.hpp"
#include "gaudirt/runtime/tp_reduce.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace gaudirt::mimo {

class Layer0 final {
 public:
  struct Io {
    runtime::MimoAttentionIo attention;
    runtime::MimoDenseMlp::Outputs dense_outputs;
    uint64_t attention_output_bf16 = 0;
    uint64_t attention_gathered_fp32 = 0;
    uint64_t down_gathered_fp32 = 0;
    uint64_t layer_output_bf16 = 0;
  };

  Layer0(runtime::HbmArena& arena, runtime::DeviceRuntime& runtime,
         const runtime::WeightManifest& attention_weights,
         const runtime::WeightManifest& dense_weights, uint32_t cache_slots,
         uint32_t page_count = 64, runtime::RecipeCache* cache = nullptr,
         bool fused_model = false, uint32_t rows = 1);

  Layer0(const Layer0&) = delete;
  Layer0& operator=(const Layer0&) = delete;

  void execute(const Io& io, synStreamHandle stream);
  PreparedLayer prepare(const Io& io, bool short_prefix = false);
  void rebind_weights(const runtime::WeightManifest& attention_weights,
                      const runtime::WeightManifest& dense_weights);
  std::vector<uint64_t> recipe_identity() const {
    auto dense = dense_.recipe_identity();
    return {attention_.recipe_identity(), dense.first, dense.second};
  }

 private:
  runtime::MimoFullAttention attention_;
  std::unique_ptr<runtime::MimoFullAttention> short_attention_;
  std::unique_ptr<runtime::TpReduce> attention_reduce_;
  runtime::MimoDenseMlp dense_;
  std::unique_ptr<runtime::TpReduce> dense_reduce_;
  hcclComm_t comm_;
  uint64_t post_attention_gamma_ = 0;
  bool fused_model_ = false;
  uint32_t rows_ = 1;
};

}  // namespace gaudirt::mimo
