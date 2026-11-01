// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gaudirt::runtime {

// A recipe and its fixed bindings. The owner keeps the recipe and workspace
// alive until all launches have completed.
struct PreparedLaunch {
  synRecipeHandle recipe = nullptr;
  uint64_t workspace = 0;
  std::vector<synLaunchTensorInfo> tensors;
  uint32_t flags = 0;

  void validate() const {
    if (!recipe || tensors.empty())
      throw std::invalid_argument("PreparedLaunch: missing recipe or bindings");
    for (const auto& tensor : tensors) {
      if (!tensor.pTensorAddress)
        throw std::invalid_argument("PreparedLaunch: null tensor address");
    }
  }

  void execute(synStreamHandle stream) const {
    const auto status = synLaunch(stream, tensors.data(),
                                  static_cast<uint32_t>(tensors.size()),
                                  workspace, recipe, flags);
    if (status != synSuccess)
      throw std::runtime_error("PreparedLaunch: synLaunch failed, status " +
                               std::to_string(static_cast<int>(status)));
  }
};

}  // namespace gaudirt::runtime
