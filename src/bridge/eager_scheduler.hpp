// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>
#include <tuple>

namespace gaudirt::generic {
std::tuple<at::Tensor, at::Tensor> advance_decode_lengths(const at::Tensor& seq_lens,
                                                         const at::Tensor& orig_seq_lens);
}
