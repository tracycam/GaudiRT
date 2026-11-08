// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <tuple>
#include <vector>

namespace gaudirt::generic {

std::tuple<at::Tensor, int64_t> allocate_paged_slots(
    const at::Tensor& last_slots, const at::Tensor& free_pages,
    const std::vector<int64_t>& prefixes, const std::vector<int64_t>& lengths, int64_t page_size);
at::Tensor release_paged_slots(const at::Tensor& slots, int64_t page_size, int64_t max_pages);

}  // namespace gaudirt::generic
