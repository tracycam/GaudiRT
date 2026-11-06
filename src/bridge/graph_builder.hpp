// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cold_recipe.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <cstdint>
#include <deque>
#include <string>
#include <variant>
#include <vector>

namespace gaudirt::generic {

struct RotaryParameters {
  int32_t head_dim;
  int32_t rotary_dim;
  int32_t is_neox;
};

using NodeParameters = std::variant<std::monostate, synTransposeParams, synGEMMParams,
                                    ns_GatherKernel::ParamsV2, ns_CastKernel::ParamsV3,
                                    synAxisParams, RotaryParameters, float, int32_t,
                                    ns_ConstantKernel::ParamsV3, ns_BroadcastNonFcd::Params>;

struct NodeSpec {
  std::string guid;
  std::vector<synTensor> inputs;
  std::vector<synTensor> outputs;
  NodeParameters parameters;
};

class SynapseGraphBuilder {
 public:
  SynapseGraphBuilder(std::string prefix, ColdRecipe& recipe);
  ~SynapseGraphBuilder();
  SynapseGraphBuilder(const SynapseGraphBuilder&) = delete;
  SynapseGraphBuilder& operator=(const SynapseGraphBuilder&) = delete;

  synTensor tensor(synDataType type, const std::vector<uint64_t>& dims, int slot);
  void node(const NodeSpec& node);
  void compile(runtime::RecipeCache& cache, const std::string& key);

 private:
  std::string prefix_;
  ColdRecipe& recipe_;
  synGraphHandle graph_ = nullptr;
  std::vector<synSectionHandle> sections_;
  std::vector<synTensor> tensors_;
  std::deque<std::string> tensor_names_;
  size_t next_node_ = 0;
};

}  // namespace gaudirt::generic
