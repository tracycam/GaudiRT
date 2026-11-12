// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fx_node.hpp"
#include "graph_builder.hpp"
#include "tensor_spec.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gaudirt::generic {

void lower_shard_embedding(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& spec,
                           uint32_t bucket, synTensor row_count,
                           const std::vector<synTensor>& values, synTensor output);

void lower_feature_slice(SynapseGraphBuilder& builder, size_t op_index,
                         const std::vector<int64_t>& roles,
                         const std::vector<FxNode>& operations,
                         const std::vector<int64_t>& outputs,
                         const std::vector<TensorSpec>& specs, uint32_t bucket,
                         std::vector<synTensor>& values);

void lower_scalar_arith(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& spec,
                        uint32_t bucket, const std::vector<synTensor>& values, synTensor output);

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
