// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "fx_node.hpp"
#include "tensor_spec.hpp"

#include <ATen/core/Tensor.h>
#include <c10/util/intrusive_ptr.h>
#include <synapse_api.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace gaudirt::generic {

inline constexpr int64_t kMaxBucketRows = 8192;

struct Context;
struct ColdRecipe;

// A node's output ID is input_count + its index in the program.
class BucketProgram {
 public:
  struct ReusableOutputs {
    struct Block {
      uint32_t bucket;
      bool dynamic;
      at::Tensor tensor;
    };
    uint32_t bucket = 0;
    std::vector<Block> available;
    std::vector<Block> active;

    void prepare(uint32_t new_bucket);
    void reclaim();
    at::Tensor acquire(const at::Tensor& reference, at::ScalarType dtype,
                       at::IntArrayRef sizes, bool dynamic, int64_t width);
  };
  BucketProgram(std::vector<int64_t> roles, std::vector<FxNode> operations, std::vector<int64_t> outputs = {});
  ~BucketProgram();
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs);
  std::vector<at::Tensor> run_internal(const std::vector<at::Tensor>& inputs, const std::vector<bool>& trusted,
                                       ReusableOutputs* reusable = nullptr,
                                       const std::vector<bool>* reusable_mask = nullptr);

 private:
  struct CachedContext {
    explicit CachedContext(uint64_t id) : generation(id) {}
    uint64_t generation;
    CachedContext* next = nullptr;
    std::array<std::atomic<ColdRecipe*>, 14> recipes{};
  };

  struct StaticInput {
    size_t index;
    c10::weak_intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl> tensor;
    uint64_t address, bytes;
    uint32_t version;
    bool versioned;
    std::vector<int64_t> strides;
  };
  struct StaticBindings {
    uint64_t generation;
    synStreamHandle stream;
    std::vector<StaticInput> inputs;
    StaticBindings* next = nullptr;
  };

  void initialize_geometry(const std::vector<at::Tensor>& inputs);
  ColdRecipe& recipe(Context& ctx, uint32_t bucket);
  ColdRecipe& cached_recipe(Context& ctx, uint32_t bucket);
  const StaticBindings* find_static_bindings(const std::vector<at::Tensor>& inputs,
                                              const Context& ctx, synStreamHandle stream) const;
  const StaticBindings* bind_static_inputs(const std::vector<at::Tensor>& inputs,
                                           Context& ctx, synStreamHandle stream);

  std::vector<int64_t> roles_;
  std::vector<FxNode> operations_;
  std::vector<int64_t> outputs_;
  bool has_embedding_ = false;
  std::vector<bool> rotary_cache_inputs_;
  std::vector<int64_t> rotary_cache_contexts_;
  std::vector<TensorSpec> specs_;
  std::string digest_;
  std::once_flag geometry_once_;
  std::atomic<CachedContext*> cached_contexts_{nullptr};
  std::vector<size_t> static_input_indices_;
  std::atomic<StaticBindings*> static_bindings_{nullptr};
  std::atomic<unsigned> static_binding_count_{0};
};

}  // namespace gaudirt::generic
