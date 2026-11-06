// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fx_node.hpp"
#include "tensor_spec.hpp"

#include <ATen/core/Tensor.h>

#include <string>
#include <vector>

namespace gaudirt::generic {

struct FxGeometry {
  std::vector<TensorSpec> specs;
  std::string digest;
};

FxGeometry infer_fx_geometry(const std::vector<at::Tensor>& inputs, const std::vector<int64_t>& roles,
                             const std::vector<FxNode>& operations, const std::vector<int64_t>& outputs);

}  // namespace gaudirt::generic
