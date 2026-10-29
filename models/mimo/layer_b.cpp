// SPDX-License-Identifier: Apache-2.0
#include "layer_b.hpp"

#include "gaudirt/runtime/weight_manifest.hpp"

#include <algorithm>
#include <stdexcept>

namespace gaudirt::mimo {
namespace {

void check(hcclResult_t status, const char* operation) {
  if (status != hcclSuccess) {
    throw std::runtime_error(std::string("MiMo Type B: ") + operation + " failed");
  }
}

}  // namespace

LayerB::LayerB(runtime::HbmArena& arena, runtime::DeviceRuntime& runtime,
               const runtime::WeightManifest& weights, uint32_t cache_slots,
               uint32_t page_count)
    : attention_(arena, weights, cache_slots, false, page_count),
      attention_reduce_(arena, 6144, syn_type_float),
      moe_(runtime.rank(), [&] {
        runtime::MimoMoeLayer1::Weights result{};
        result.post_attention_gamma_bf16 =
            weights.get("post_attention_layernorm").device_address;
        result.router_weight_bf16 = weights.get("router_weight").device_address;
        result.router_correction_bias_fp32 = weights.get("router_bias").device_address;
        result.gate_up_packed_u8 = weights.get("moe_gp").device_address;
        result.gate_up_scales_u8 = weights.get("moe_gs").device_address;
        result.down_packed_u8 = weights.get("moe_dp").device_address;
        result.down_scales_u8 = weights.get("moe_ds").device_address;
        const auto lut = runtime::MimoMoeLayer1::make_mxfp4_lut_bf16();
        result.mxfp4_lut_bf16 = arena.allocate(lut.size() * sizeof(lut[0]));
        arena.copy_h2d_sync(runtime.h2d_stream(), lut.data(), result.mxfp4_lut_bf16,
                            lut.size() * sizeof(lut[0]));
        const auto directions = runtime::MimoMoeLayer1::make_down_combine_directions();
        result.down_combine_directions_u8 = arena.allocate(directions.size());
        arena.copy_h2d_sync(runtime.h2d_stream(), directions.data(),
                            result.down_combine_directions_u8, directions.size());
        return result;
      }()),
      moe_reduce_(arena, 6144, syn_type_float), comm_(runtime.comm()) {
  if (!comm_) throw std::invalid_argument("MiMo Type B: missing HCCL");
  moe_workspace_ = arena.ensure_workspace(std::max(
      {attention_.workspace_bytes(), attention_reduce_.workspace_bytes(),
       moe_.workspace_bytes(), moe_reduce_.workspace_bytes()}));
}

void LayerB::execute(const Io& io, synStreamHandle stream) {
  if (!stream || !io.attention_output_bf16 || !io.attention_gathered_fp32 ||
      !io.moe_gathered_fp32 || !io.layer_output_bf16) {
    throw std::invalid_argument("MiMo Type B: invalid execution buffers");
  }
  attention_.execute(io.attention, stream);
  check(hcclAllGather(reinterpret_cast<const void*>(io.attention.output_local_fp32),
                      reinterpret_cast<void*>(io.attention_gathered_fp32), 6144,
                      hcclFloat32, comm_, stream),
        "attention AllGather");
  attention_reduce_.execute(io.attention_gathered_fp32, io.attention_output_bf16,
                            stream);
  moe_.launch(stream,
              {io.attention_output_bf16, io.attention.residual_out},
              io.moe_outputs, moe_workspace_);
  check(hcclAllGather(reinterpret_cast<const void*>(io.moe_outputs.moe_local_fp32),
                      reinterpret_cast<void*>(io.moe_gathered_fp32), 6144,
                      hcclFloat32, comm_, stream),
        "MoE AllGather");
  moe_reduce_.execute(io.moe_gathered_fp32, io.layer_output_bf16, stream);
}

}  // namespace gaudirt::mimo
