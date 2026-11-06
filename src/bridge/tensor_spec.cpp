// SPDX-License-Identifier: Apache-2.0
#include "tensor_spec.hpp"

namespace gaudirt::generic {

std::vector<uint64_t> TensorSpec::synapse_sizes(uint32_t bucket) const {
  std::vector<uint64_t> dimensions;
  dimensions.reserve(sizes.size());
  for (auto it = sizes.rbegin(); it != sizes.rend(); ++it)
    dimensions.push_back(*it == -1 ? bucket : static_cast<uint64_t>(*it));
  return dimensions;
}

}  // namespace gaudirt::generic
