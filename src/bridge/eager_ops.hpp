// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>
#include <synapse_api.h>

#include <string>
#include <vector>

namespace gaudirt::generic {

at::Tensor pointwise(std::vector<at::Tensor> inputs, const std::vector<int64_t>& shape,
                     at::ScalarType output_type, const std::string& op,
                     const std::vector<synDataType>& types);
at::Tensor compare(const at::Tensor& a, const at::Tensor& b, const char* op);
at::Tensor compare_scalar(const at::Tensor& a, const at::Scalar& scalar, const char* op);
at::Tensor where(const at::Tensor& condition, const at::Tensor& a, const at::Tensor& b);

}  // namespace gaudirt::generic
