// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/device_runtime.hpp"
#include "mimo_full_attention.hpp"
#include "mimo_moe.hpp"
#include "gaudirt/runtime/tp_reduce.hpp"

#include <cstdint>

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
         uint32_t page_count = 64);

  LayerB(const LayerB&) = delete;
  LayerB& operator=(const LayerB&) = delete;

  void execute(const Io& io, synStreamHandle stream);

 private:
  runtime::MimoFullAttention attention_;
  runtime::TpReduce attention_reduce_;
  runtime::MimoMoeLayer1 moe_;
  runtime::TpReduce moe_reduce_;
  hcclComm_t comm_;
  uint64_t moe_workspace_ = 0;
};

}  // namespace gaudirt::mimo
