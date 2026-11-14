// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>

namespace gaudirt::generic {

at::Tensor sample_from_logits(
    const at::Tensor& logits,
    const at::Tensor& temperatures,
    const at::Tensor& top_ps,
    const at::Tensor& top_ks,
    const at::Tensor& min_ps,
    const c10::optional<at::Tensor>& seeds = c10::nullopt,
    const c10::optional<at::Tensor>& positions = c10::nullopt,
    bool nan_to_neg_inf = true);

at::Tensor& apply_token_bitmask_inplace_(
    at::Tensor& logits,
    const at::Tensor& bitmask,
    const c10::optional<at::Tensor>& indices = c10::nullopt,
    int64_t vocab_size = -1,
    bool mask_tail = false);

}  // namespace gaudirt::generic
