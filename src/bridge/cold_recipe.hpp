// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include <string>
#include <vector>

namespace gaudirt::generic {

struct ColdRecipe {
  explicit ColdRecipe(size_t count = 2) : names(count), tensors(count) {}
  synRecipeHandle recipe = nullptr;
  uint64_t workspace = 0;
  std::vector<std::string> names;
  std::vector<synLaunchTensorInfo> tensors;
  ~ColdRecipe() { if (recipe) synRecipeDestroy(recipe); }
};

}  // namespace gaudirt::generic
