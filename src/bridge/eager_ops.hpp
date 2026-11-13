// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>
#include <synapse_api.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace gaudirt::generic {

at::Tensor pointwise(std::vector<at::Tensor> inputs, const std::vector<int64_t>& shape,
                     at::ScalarType output_type, const std::string& op,
                     const std::vector<synDataType>& types);
at::Tensor compare(const at::Tensor& a, const at::Tensor& b, const char* op);
at::Tensor compare_scalar(const at::Tensor& a, const at::Scalar& scalar, const char* op);
at::Tensor where(const at::Tensor& condition, const at::Tensor& a, const at::Tensor& b);

at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double epsilon);
std::tuple<at::Tensor, at::Tensor> fused_add_rms_norm(
    const at::Tensor& input, const at::Tensor& residual, const at::Tensor& weight, double epsilon);
at::Tensor gemma_rms_norm(const at::Tensor& input, const at::Tensor& weight, double epsilon);
std::tuple<at::Tensor, at::Tensor> fused_add_gemma_rms_norm(
    const at::Tensor& input, const at::Tensor& residual, const at::Tensor& weight, double epsilon);
at::Tensor gelu_and_mul(const at::Tensor& input, std::string_view approximate = "none");
std::tuple<at::Tensor, at::Tensor> rotary_embedding_positions(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& cos_sin_cache,
    const at::Tensor& positions, int64_t head_size, int64_t rotary_dim, bool is_neox_style);
at::Tensor& softcap_(at::Tensor& logits, double cap);

}  // namespace gaudirt::generic
