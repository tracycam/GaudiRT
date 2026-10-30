// SPDX-License-Identifier: Apache-2.0
#include "layer0.hpp"

#include "gaudirt/runtime/weight_manifest.hpp"

#include <algorithm>
#include <stdexcept>

namespace gaudirt::mimo {
namespace {

void check(hcclResult_t status, const char* operation) {
  if (status != hcclSuccess) {
    throw std::runtime_error(std::string("MiMo layer0: ") + operation + " failed");
  }
}

}  // namespace

Layer0::Layer0(runtime::HbmArena& arena, runtime::DeviceRuntime& runtime,
               const runtime::WeightManifest& attention_weights,
               const runtime::WeightManifest& dense_weights, uint32_t cache_slots,
               uint32_t page_count, runtime::RecipeCache* cache)
    : attention_(arena, attention_weights, cache_slots, true, page_count, cache),
      attention_reduce_(arena, 6144, syn_type_float),
      dense_(arena, dense_weights, cache),
      dense_reduce_(arena, 6144, syn_type_float), comm_(runtime.comm()),
      post_attention_gamma_(
          dense_weights.get("post_attention_gamma").device_address) {
  if (!comm_ || !post_attention_gamma_) {
    throw std::invalid_argument("MiMo layer0: missing HCCL or post-attention gamma");
  }
  arena.ensure_workspace(std::max({attention_.workspace_bytes(),
                                   attention_reduce_.workspace_bytes(),
                                   dense_.workspace_bytes(),
                                   dense_reduce_.workspace_bytes()}));
}

PreparedLayer Layer0::prepare(const Io& io) {
  auto attention = attention_.prepare(io.attention);
  auto dense = dense_.prepare({io.attention_output_bf16,
                               io.attention.hidden_states,
                               post_attention_gamma_}, io.dense_outputs);
  PreparedLayer result;
  result.attention = {std::move(attention[0]), std::move(attention[1])};
  result.attention_reduce = attention_reduce_.prepare(io.attention_gathered_fp32,
                                                      io.attention_output_bf16);
  result.feed_forward = {std::move(dense[0]), std::move(dense[1])};
  result.output_reduce = dense_reduce_.prepare(io.down_gathered_fp32,
                                                io.layer_output_bf16);
  result.attention_local = io.attention.output_local_fp32;
  result.attention_gathered = io.attention_gathered_fp32;
  result.feed_forward_local = io.dense_outputs.output_local_fp32;
  result.feed_forward_gathered = io.down_gathered_fp32;
  result.attention_type = hcclFloat32;
  result.comm = comm_;
  result.validate();
  return result;
}

void Layer0::execute(const Io& io, synStreamHandle stream) {
  if (!stream || !io.attention_output_bf16 || !io.attention_gathered_fp32 ||
      !io.down_gathered_fp32 || !io.layer_output_bf16) {
    throw std::invalid_argument("MiMo layer0: invalid execution buffers");
  }
  attention_.execute(io.attention, stream);
  check(hcclAllGather(reinterpret_cast<const void*>(io.attention.output_local_fp32),
                      reinterpret_cast<void*>(io.attention_gathered_fp32), 6144,
                      hcclFloat32, comm_, stream),
        "attention AllGather");
  attention_reduce_.execute(io.attention_gathered_fp32, io.attention_output_bf16,
                            stream);
  dense_.execute({io.attention_output_bf16, io.attention.hidden_states,
                  post_attention_gamma_},
                 io.dense_outputs, stream);
  check(hcclAllGather(
            reinterpret_cast<const void*>(io.dense_outputs.output_local_fp32),
            reinterpret_cast<void*>(io.down_gathered_fp32), 6144, hcclFloat32,
            comm_, stream),
        "dense down AllGather");
  dense_reduce_.execute(io.down_gathered_fp32, io.layer_output_bf16, stream);
}

void Layer0::rebind_weights(const runtime::WeightManifest& attention_weights,
                            const runtime::WeightManifest& dense_weights) {
  attention_.rebind_weights(attention_weights);
  dense_.rebind_weights(dense_weights);
  post_attention_gamma_ = dense_weights.get("post_attention_gamma").device_address;
  if (!post_attention_gamma_) {
    throw std::invalid_argument("MiMo layer0: null post-attention gamma");
  }
}

}  // namespace gaudirt::mimo
