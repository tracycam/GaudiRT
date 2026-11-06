// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <c10/core/ScalarType.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace gaudirt::generic {

struct TensorSpec {
  TensorSpec(c10::ScalarType type, std::vector<int64_t> shape, bool dynamic)
      : dtype(type), sizes(std::move(shape)), token_axis(dynamic ? 0 : -1) {}
  bool dynamic() const { return token_axis >= 0; }
  std::vector<uint64_t> synapse_sizes(uint32_t bucket) const;

  c10::ScalarType dtype;
  std::vector<int64_t> sizes;
  int token_axis;
};

}  // namespace gaudirt::generic
