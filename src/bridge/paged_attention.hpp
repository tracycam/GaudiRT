// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>

namespace gaudirt::generic {

at::Tensor paged_attn_validated(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& key_pool, const at::Tensor& value_pool, const at::Tensor& locations,
    const at::Tensor& requests, const at::Tensor& positions, const at::Tensor& plan,
    const at::Tensor& header, int64_t head_dim, double scale,
    double softcap = 0.0, int64_t sliding_window = -1);
at::Tensor paged_attn_internal(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& key_pool, const at::Tensor& value_pool, const at::Tensor& locations,
    const at::Tensor& requests, const at::Tensor& positions, const at::Tensor& plan,
    const at::Tensor& header, int64_t head_dim, double scale,
    bool query_trusted, bool key_trusted, bool value_trusted,
    double softcap = 0.0, int64_t sliding_window = -1);

}  // namespace gaudirt::generic
