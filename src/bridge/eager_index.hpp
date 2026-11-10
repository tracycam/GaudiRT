// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>

#include <optional>

namespace gaudirt::generic {

at::Tensor index_select(const at::Tensor& input, int64_t axis, const at::Tensor& index);
at::Tensor index_select_validated(const at::Tensor& input, int64_t axis, const at::Tensor& index);
at::Tensor index_tensor(const at::Tensor& input, const c10::List<std::optional<at::Tensor>>& indices);
at::Tensor& index_put(at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices,
                      const at::Tensor& values, bool accumulate, bool unsafe);
at::Tensor& index_put_validated(at::Tensor& self, const at::Tensor& index, const at::Tensor& values);
at::Tensor embedding(const at::Tensor& weight, const at::Tensor& index, int64_t padding_idx,
                     bool scale_grad_by_freq, bool sparse);
at::Tensor argmax(const at::Tensor& input, std::optional<int64_t> dimension, bool keepdim);

}  // namespace gaudirt::generic
