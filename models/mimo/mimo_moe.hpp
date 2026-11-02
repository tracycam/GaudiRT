// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/prepared_launch.hpp"

#include <synapse_api.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace gaudirt::runtime { class RecipeCache; }

namespace gaudirt::mimo {
using runtime::RecipeCache;
using runtime::PreparedLaunch;

// Fixed MiMo layer1 decode MoE: B1/M1, TP8/EP1, H6144, E384, top8.
// All tensors use Synapse FCD-first geometry. HBM addresses refer to contiguous
// payload bytes, not .npy headers. The caller owns every buffer and HCCL.
class MimoMoeLayer1 final {
 public:
  static constexpr uint32_t kTpSize = 8;
  static constexpr uint32_t kHiddenSize = 6144;
  static constexpr uint32_t kExpertCount = 384;
  static constexpr uint32_t kTopK = 8;
  static constexpr uint32_t kLocalGateUpRows = 512;
  static constexpr uint32_t kLocalDownK = 256;

  static constexpr uint64_t kHiddenBf16Bytes = kHiddenSize * 2;
  static constexpr uint64_t kRouterWeightBf16Bytes =
      static_cast<uint64_t>(kExpertCount) * kHiddenSize * 2;
  static constexpr uint64_t kRouterBiasFp32Bytes = kExpertCount * 4;
  static constexpr uint64_t kGateUpPackedWeightBytes = 603979776;
  static constexpr uint64_t kGateUpScaleBytes = 37748736;
  static constexpr uint64_t kDownPackedWeightBytes = 301989888;
  static constexpr uint64_t kDownScaleBytes = 18874368;
  static constexpr uint64_t kMxfp4LutBytes = 1024;
  static constexpr uint64_t kDownCombineDirectionBytes = 512;
  static constexpr uint64_t kTopkIdsBytes = kTopK * 4;
  static constexpr uint64_t kTopkWeightsBytes = kTopK * 4;
  static constexpr uint64_t kMoeLocalFp32Bytes = kHiddenSize * 4;

  // gate_up/down arrays use the historical unpermuted-N512 pack layout and
  // E8M0 scales, matching the existing GP/down/pf_combine kernels (not native-v2).
  struct Weights {
    uint64_t post_attention_gamma_bf16 = 0;
    uint64_t router_weight_bf16 = 0;
    uint64_t router_correction_bias_fp32 = 0;
    uint64_t gate_up_packed_u8 = 0;
    uint64_t gate_up_scales_u8 = 0;
    uint64_t down_packed_u8 = 0;
    uint64_t down_scales_u8 = 0;
    uint64_t mxfp4_lut_bf16 = 0;
    uint64_t down_combine_directions_u8 = 0;
  };

  struct Inputs {
    uint64_t attention_bf16 = 0;
    uint64_t residual_in_bf16 = 0;
  };

  struct Outputs {
    uint64_t residual_out_bf16 = 0;
    uint64_t moe_local_fp32 = 0;
  };

  // Synapse and the required TPC libraries must already be initialized by the
  // driver. This constructor builds and compiles the graph but does not acquire
  // a device, initialize HCCL, upload weights, or launch work.
  MimoMoeLayer1(uint32_t tp_rank, const Weights& weights,
                const char* graph_variant, RecipeCache* cache = nullptr,
                bool gathered_attention = false,
                synDataType gathered_type = syn_type_float, uint32_t rows = 1);
  ~MimoMoeLayer1();

  MimoMoeLayer1(const MimoMoeLayer1&) = delete;
  MimoMoeLayer1& operator=(const MimoMoeLayer1&) = delete;
  MimoMoeLayer1(MimoMoeLayer1&&) = delete;
  MimoMoeLayer1& operator=(MimoMoeLayer1&&) = delete;

  // Enqueues residual post-RMSNorm, router/top8, all-expert routed MXFP4
  // gate/up+SiLU and down, then FP32 routing combine. moe_local_fp32 is the
  // TP-local FP32 partial for the driver's collective; it is not TP-reduced and
  // is not the final BF16 layer output. No synchronization is performed.
  // Concurrent launch calls on one instance are not supported.
  void launch(synStreamHandle stream, const Inputs& inputs,
              const Outputs& outputs, uint64_t workspace_addr);
  void rebind_weights(const Weights& weights);
  PreparedLaunch prepare(const Inputs& inputs, const Outputs& outputs,
                         uint64_t workspace_addr);

  uint32_t tp_rank() const noexcept { return tp_rank_; }
  uint64_t workspace_bytes() const noexcept { return workspace_bytes_; }
  uint64_t recipe_identity() const noexcept {
    return reinterpret_cast<uint64_t>(recipe_);
  }

  // Host bytes for mxfp4_lut_bf16 [512,1] and
  // down_combine_directions_u8 [256,2], respectively.
  static std::array<uint16_t, 512> make_mxfp4_lut_bf16();
  static std::array<uint8_t, 512> make_down_combine_directions();

 private:
  enum Slot : size_t {
    kAttention = 0,
    kResidualIn,
    kGamma,
    kRouterWeight,
    kRouterBias,
    kGateUpWeight,
    kGateUpScale,
    kDownWeight,
    kDownScale,
    kLut,
    kResidualOut,
    kMoeOut,
    kDirections,
    kSlotCount
  };

  static_assert(kSlotCount == 13);

  uint32_t tp_rank_;
  uint32_t rows_;
  bool gathered_attention_ = false;
  synDataType gathered_type_ = syn_type_float;
  synRecipeHandle recipe_ = nullptr;
  uint64_t workspace_bytes_ = 0;
  std::array<synLaunchTensorInfo, kSlotCount> launch_tensors_{};
};

}  // namespace gaudirt::mimo
