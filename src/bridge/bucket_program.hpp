// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fx_node.hpp"
#include "tensor_spec.hpp"

#include <ATen/core/Tensor.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace gaudirt::generic {

struct Context;
struct ColdRecipe;

// A node's output ID is input_count + its index in the program.
class BucketProgram {
 public:
  BucketProgram(std::vector<int64_t> roles, std::vector<FxNode> operations, std::vector<int64_t> outputs = {});
  ~BucketProgram();
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs);

 private:
  struct CachedContext {
    explicit CachedContext(uint64_t id) : generation(id) {}
    uint64_t generation;
    CachedContext* next = nullptr;
    std::array<std::atomic<ColdRecipe*>, 13> recipes{};
  };

  void initialize_geometry(const std::vector<at::Tensor>& inputs);
  ColdRecipe& recipe(Context& ctx, uint32_t bucket);
  ColdRecipe& cached_recipe(Context& ctx, uint32_t bucket);

  std::vector<int64_t> roles_;
  std::vector<FxNode> operations_;
  std::vector<int64_t> outputs_;
  bool has_embedding_ = false;
  std::vector<TensorSpec> specs_;
  std::string digest_;
  std::once_flag geometry_once_;
  std::atomic<CachedContext*> cached_contexts_{nullptr};
};

}  // namespace gaudirt::generic
