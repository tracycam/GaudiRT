// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fx_node.hpp"
#include "tensor_spec.hpp"

#include <ATen/core/Tensor.h>

#include <string>
#include <vector>

namespace gaudirt::generic {

// Rotary tables whose validated context fits this range share a static
// descriptor backed by an allocation of at least this many rows.
inline constexpr uint64_t kRotaryStaticRows = 131072;

struct FxGeometry {
  std::vector<TensorSpec> specs;
  std::string digest;
};

std::vector<bool> rotary_cache_inputs(const std::vector<int64_t>& roles,
                                      const std::vector<FxNode>& operations);
FxGeometry infer_fx_geometry(const std::vector<at::Tensor>& inputs, const std::vector<int64_t>& roles,
                             const std::vector<FxNode>& operations, const std::vector<int64_t>& outputs);

}  // namespace gaudirt::generic
