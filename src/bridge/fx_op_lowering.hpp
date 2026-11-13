// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fx_node.hpp"
#include "graph_builder.hpp"
#include "tensor_spec.hpp"

#include <c10/core/ScalarType.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gaudirt::generic {

struct EwiseUnaryParameters {
  int32_t op_code;
  float alpha;
  float beta;
};

struct GeluAndMulParameters {
  int32_t approximate;
};

struct RmsNormParameters {
  float eps;
  float weight_offset;
};

std::vector<uint64_t> flatten_2d_syn_dims(const std::vector<uint64_t>& syn_dims);

synTensor emit_cast(SynapseGraphBuilder& builder, synTensor input,
                    const std::vector<uint64_t>& syn_dims,
                    at::ScalarType from, at::ScalarType to,
                    synTensor target_out = nullptr);

synTensor emit_broadcast_to(SynapseGraphBuilder& builder, synTensor input, synDataType dtype,
                            const std::vector<uint64_t>& src_syn_dims,
                            const std::vector<uint64_t>& dst_syn_dims,
                            synTensor target_out = nullptr);

synTensor emit_constant(SynapseGraphBuilder& builder, at::ScalarType dtype,
                        const std::vector<uint64_t>& syn_dims,
                        float float_value, int64_t int_value, bool is_integral,
                        synTensor target_out = nullptr);

void emit_ewise_unary(SynapseGraphBuilder& builder, synTensor input, synTensor output,
                      at::ScalarType dtype, const std::vector<uint64_t>& syn_dims,
                      int32_t op_code, float alpha = 0.f, float beta = 0.f);

void emit_isnan(SynapseGraphBuilder& builder, synTensor input, synTensor output,
                at::ScalarType dtype);

void emit_logical_not(SynapseGraphBuilder& builder, synTensor input, synTensor output);

void emit_binary_pointwise(SynapseGraphBuilder& builder, FxOp op, synTensor left, synTensor right,
                           synTensor output, at::ScalarType in_dtype,
                           const std::vector<uint64_t>& left_dims,
                           const std::vector<uint64_t>& right_dims,
                           const std::vector<uint64_t>& out_dims);

void emit_where(SynapseGraphBuilder& builder, synTensor cond, synTensor x, synTensor y,
                synTensor output, at::ScalarType dtype,
                const std::vector<uint64_t>& cond_dims,
                const std::vector<uint64_t>& x_dims,
                const std::vector<uint64_t>& y_dims,
                const std::vector<uint64_t>& out_dims);

void emit_rms_norm(SynapseGraphBuilder& builder, synTensor input, synTensor residual,
                   synTensor weight, synTensor out_norm, synTensor out_residual,
                   const std::vector<uint64_t>& syn_dims, float epsilon, float weight_offset);

void emit_gated_mul(SynapseGraphBuilder& builder, bool is_gelu, int32_t approximate,
                    synTensor input, synTensor output,
                    const std::vector<uint64_t>& in_syn_dims,
                    const std::vector<uint64_t>& out_syn_dims);

void emit_rotary_pos(SynapseGraphBuilder& builder, synTensor q, synTensor k, synTensor cache,
                     synTensor positions, synTensor q_out, synTensor k_out,
                     const std::vector<uint64_t>& q_syn_dims,
                     const std::vector<uint64_t>& k_syn_dims,
                     at::ScalarType pos_dtype, uint64_t rows,
                     int32_t head_dim, int32_t rotary_dim, bool is_neox);

void emit_reduction(SynapseGraphBuilder& builder, FxOp op, synTensor input, synTensor output,
                    at::ScalarType dtype, const std::vector<uint64_t>& in_syn_dims,
                    const std::vector<uint64_t>& out_syn_dims, int64_t pt_dim, bool keepdim);

void emit_softmax(SynapseGraphBuilder& builder, bool is_log, synTensor input, synTensor output,
                  at::ScalarType dtype, const std::vector<uint64_t>& syn_dims, int64_t pt_dim);

void emit_layer_norm(SynapseGraphBuilder& builder, synTensor input, synTensor weight,
                     synTensor bias, synTensor output, synTensor mean_out, synTensor rstd_out,
                     at::ScalarType dtype, const std::vector<uint64_t>& syn_dims, float eps);

void lower_shard_embedding(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& spec,
                           uint32_t bucket, synTensor row_count,
                           const std::vector<synTensor>& values, synTensor output);

void lower_feature_slice(SynapseGraphBuilder& builder, size_t op_index,
                         const std::vector<int64_t>& roles,
                         const std::vector<FxNode>& operations,
                         const std::vector<int64_t>& outputs,
                         const std::vector<TensorSpec>& specs, uint32_t bucket,
                         std::vector<synTensor>& values);

void lower_slice_dim(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& in_spec,
                     const TensorSpec& out_spec, uint32_t bucket,
                     synTensor input, synTensor output);

void lower_scalar_arith(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& spec,
                        uint32_t bucket, const std::vector<synTensor>& values, synTensor output);

void lower_scalar_cmp(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& in_spec,
                      const TensorSpec& out_spec, uint32_t bucket,
                      const std::vector<synTensor>& values, synTensor output);

void lower_binary_arith(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& spec,
                        const std::vector<TensorSpec>& specs, uint32_t bucket,
                        const std::vector<synTensor>& values, synTensor output);

void lower_block_fp8_linear(SynapseGraphBuilder& builder, synDeviceId device_id, const FxNode& op,
                            const std::vector<TensorSpec>& specs, uint32_t bucket,
                            const std::vector<synTensor>& values, synTensor output);

void lower_fp8_linear(SynapseGraphBuilder& builder, synDeviceId device_id, const FxNode& op,
                      const std::vector<TensorSpec>& specs, uint32_t bucket,
                      const std::vector<synTensor>& values, synTensor output);

}  // namespace gaudirt::generic
