// SPDX-License-Identifier: Apache-2.0
#include "layer_c.hpp"

#include "gaudirt/runtime/weight_manifest.hpp"

#include <algorithm>
#include <stdexcept>

namespace gaudirt::mimo {
namespace {

void check(hcclResult_t status, const char* operation) {
  if (status != hcclSuccess) {
    throw std::runtime_error(std::string("MiMo Type C: ") + operation + " failed");
  }
}

}  // namespace

LayerC::LayerC(runtime::HbmArena& arena, runtime::DeviceRuntime& runtime,
               const runtime::WeightManifest& weights, uint32_t cache_slots,
               runtime::RecipeCache* cache, bool fused_model, uint32_t rows,
               uint64_t shared_lut, uint64_t shared_directions)
    : arena_(arena),
      attention_(arena, weights, cache_slots, cache, fused_model, rows),
      attention_reduce_(fused_model ? nullptr :
                        std::make_unique<runtime::TpReduce>(arena, 6144, syn_type_bf16)),
      moe_(runtime.rank(), [&] {
        moe_weights_.post_attention_gamma_bf16 =
            weights.get("post_attention_layernorm").device_address;
        moe_weights_.router_weight_bf16 = weights.get("router_weight").device_address;
        moe_weights_.router_correction_bias_fp32 = weights.get("router_bias").device_address;
        moe_weights_.gate_up_packed_u8 = weights.get("moe_gp").device_address;
        moe_weights_.gate_up_scales_u8 = weights.get("moe_gs").device_address;
        moe_weights_.down_packed_u8 = weights.get("moe_dp").device_address;
        moe_weights_.down_scales_u8 = weights.get("moe_ds").device_address;
        if (bool(shared_lut) != bool(shared_directions)) throw std::invalid_argument("MiMo Type C: incomplete shared LUT");
        if (shared_lut) {
          moe_weights_.mxfp4_lut_bf16 = shared_lut;
          moe_weights_.down_combine_directions_u8 = shared_directions;
        } else {
          const auto lut = MimoMoeLayer1::make_mxfp4_lut_bf16();
          moe_weights_.mxfp4_lut_bf16 = arena.allocate(lut.size() * sizeof(lut[0]));
          arena.copy_h2d_sync(runtime.h2d_stream(), lut.data(), moe_weights_.mxfp4_lut_bf16,
                              lut.size() * sizeof(lut[0]));
          const auto directions = MimoMoeLayer1::make_down_combine_directions();
          moe_weights_.down_combine_directions_u8 = arena.allocate(directions.size());
          arena.copy_h2d_sync(runtime.h2d_stream(), directions.data(),
                              moe_weights_.down_combine_directions_u8, directions.size());
        }
        return moe_weights_;
      }(), "swa", cache, fused_model, syn_type_bf16, rows),
      moe_reduce_(fused_model ? nullptr :
                  std::make_unique<runtime::TpReduce>(arena, 6144, syn_type_float)),
      comm_(runtime.comm()), fused_model_(fused_model), rows_(rows) {
  if (rows_ != 1 && ((rows_ != 32 && rows_ != 512 && rows_ != 1024 && rows_ != 2048) || !fused_model_))
    throw std::invalid_argument("MiMo layer: prefill requires fused execution");
  if (!comm_) throw std::invalid_argument("MiMo Type C: missing HCCL");
  arena.ensure_workspace(std::max(
      {attention_.workspace_bytes(), attention_reduce_ ? attention_reduce_->workspace_bytes() : 0,
       moe_.workspace_bytes(), moe_reduce_ ? moe_reduce_->workspace_bytes() : 0}));
}

PreparedLayer LayerC::prepare(const Io& io) {
  PreparedLayer result;
  result.attention.push_back(attention_.prepare(io.attention));
  if (!fused_model_)
    result.attention_reduce = attention_reduce_->prepare(io.attention_gathered_bf16,
                                                          io.attention_output_bf16);
  result.feed_forward.push_back(moe_.prepare(
      {fused_model_ ? (rows_ >= 512 ? io.attention.output_local : io.attention_gathered_bf16) : io.attention_output_bf16,
       io.attention.residual_out}, io.moe_outputs,
      arena_.workspace_addr()));
  if (!fused_model_)
    result.output_reduce = moe_reduce_->prepare(io.moe_gathered_fp32,
                                                 io.layer_output_bf16);
  result.attention_local = io.attention.output_local;
  result.attention_gathered = rows_ >= 512 ? result.attention_local : io.attention_gathered_bf16;
  result.feed_forward_local = io.moe_outputs.moe_local_fp32;
  result.feed_forward_gathered = rows_ >= 512 ? result.feed_forward_local : io.moe_gathered_fp32;
  result.attention_type = hcclBfloat16;
  result.comm = comm_;
  result.rows = rows_;
  result.validate();
  return result;
}

void LayerC::execute(const Io& io, synStreamHandle stream) {
  if (fused_model_)
    throw std::invalid_argument("MiMo Type C: fused model requires prepared execution");
  if (!stream || !io.attention_output_bf16 || !io.attention_gathered_bf16 ||
      !io.moe_gathered_fp32 || !io.layer_output_bf16) {
    throw std::invalid_argument("MiMo Type C: invalid execution buffers");
  }
  if (arena_.workspace_bytes() < moe_.workspace_bytes()) {
    throw std::runtime_error("MiMo Type C: shared workspace is too small");
  }
  attention_.execute(io.attention, stream);
  check(hcclAllGather(reinterpret_cast<const void*>(io.attention.output_local),
                      reinterpret_cast<void*>(io.attention_gathered_bf16), 6144,
                      hcclBfloat16, comm_, stream),
        "attention AllGather");
  attention_reduce_->execute(io.attention_gathered_bf16, io.attention_output_bf16,
                             stream);
  moe_.launch(stream,
              {io.attention_output_bf16, io.attention.residual_out},
              io.moe_outputs, arena_.workspace_addr());
  check(hcclAllGather(reinterpret_cast<const void*>(io.moe_outputs.moe_local_fp32),
                      reinterpret_cast<void*>(io.moe_gathered_fp32), 6144,
                      hcclFloat32, comm_, stream),
        "MoE AllGather");
  moe_reduce_->execute(io.moe_gathered_fp32, io.layer_output_bf16, stream);
}

void LayerC::rebind_weights(const runtime::WeightManifest& weights) {
  attention_.rebind_weights(weights);
  moe_weights_.post_attention_gamma_bf16 =
      weights.get("post_attention_layernorm").device_address;
  moe_weights_.router_weight_bf16 = weights.get("router_weight").device_address;
  moe_weights_.router_correction_bias_fp32 = weights.get("router_bias").device_address;
  moe_weights_.gate_up_packed_u8 = weights.get("moe_gp").device_address;
  moe_weights_.gate_up_scales_u8 = weights.get("moe_gs").device_address;
  moe_weights_.down_packed_u8 = weights.get("moe_dp").device_address;
  moe_weights_.down_scales_u8 = weights.get("moe_ds").device_address;
  moe_.rebind_weights(moe_weights_);
}

}  // namespace gaudirt::mimo
