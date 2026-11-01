// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "prepared_layer.hpp"

#include <cstdint>
#include <utility>
#include <vector>

namespace gaudirt::mimo {

// Collect the changing metadata slots once, after the layer vectors are final.
// The vectors and their tensor arrays must not be resized while bindings live.
inline std::vector<std::pair<uint64_t*, uint64_t>> prefill_metadata_bindings(
    std::vector<PreparedLayer>& layers, uint64_t base, uint64_t bytes) {
  std::vector<std::pair<uint64_t*, uint64_t>> result;
  for (auto& layer : layers) {
    auto collect = [&](std::vector<runtime::PreparedLaunch>& launches) {
      for (auto& launch : launches) for (auto& tensor : launch.tensors) {
        const uint64_t address = tensor.pTensorAddress;
        if (address >= base && address - base < bytes)
          result.emplace_back(&tensor.pTensorAddress, address - base);
      }
    };
    collect(layer.attention);
    collect(layer.feed_forward);
  }
  return result;
}

}  // namespace gaudirt::mimo
