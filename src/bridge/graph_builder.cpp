// SPDX-License-Identifier: Apache-2.0
#include "graph_builder.hpp"

#include <c10/util/Exception.h>

#include <type_traits>
#include <utility>

namespace gaudirt::generic {
namespace {
void check(synStatus status, const char* operation) {
  TORCH_CHECK(status == synSuccess, "GaudiRT FX: ", operation,
              " status=", static_cast<int>(status));
}
}

SynapseGraphBuilder::SynapseGraphBuilder(std::string prefix, ColdRecipe& recipe)
    : prefix_(std::move(prefix)), recipe_(recipe), slot_sections_(recipe.tensors.size(), nullptr) {
  check(synGraphCreate(&graph_, synDeviceGaudi2), "program graph creation failed");
}

SynapseGraphBuilder::~SynapseGraphBuilder() {
  if (graph_) synGraphDestroy(graph_);
  for (auto tensor : tensors_) synTensorDestroy(tensor);
  for (auto section : sections_) synSectionDestroy(section);
}

synTensor SynapseGraphBuilder::tensor(synDataType type, const std::vector<uint64_t>& dims, int slot,
                                     int shared_slot, uint64_t offset, bool dynamic,
                                     const std::vector<uint64_t>& min_dims, bool rmw) {
  TORCH_CHECK(!dims.empty() && dims.size() <= 5, "GaudiRT FX: tensor rank must be 1..5");
  TORCH_CHECK(min_dims.empty() || (dynamic && min_dims.size() == dims.size()),
              "GaudiRT FX: invalid dynamic tensor bounds");
  TORCH_CHECK(!rmw || (slot < 0 && shared_slot < 0 && offset == 0 && !dynamic),
              "GaudiRT FX: invalid RMW tensor");
  TORCH_CHECK(slot < static_cast<int>(slot_sections_.size()) &&
              (shared_slot < 0 || (slot >= 0 && shared_slot < static_cast<int>(slot_sections_.size()) &&
                                   slot_sections_[shared_slot])),
              "GaudiRT FX: invalid persistent tensor section");
  tensor_names_.push_back(prefix_ + "_t" + std::to_string(tensor_names_.size()));
  synSectionHandle section = nullptr;
  if (slot >= 0) {
    if (shared_slot >= 0) {
      section = slot_sections_[shared_slot];
    } else {
      check(synSectionCreate(&section, 0, graph_), "program section creation failed");
      sections_.push_back(section);
      check(synSectionSetPersistent(section, true), "program section setup failed");
    }
    slot_sections_[slot] = section;
    recipe_.names[slot] = tensor_names_.back();
  } else if (rmw) {
    check(synSectionCreate(&section, 0, graph_), "program RMW section creation failed");
    sections_.push_back(section);
    check(synSectionSetPersistent(section, false), "program RMW persistence failed");
    check(synSectionSetRMW(section, true), "program RMW setup failed");
  }
  synTensorDescriptor desc{};
  desc.m_name = slot >= 0 ? recipe_.names[slot].c_str() : tensor_names_.back().c_str();
  desc.m_dataType = type;
  desc.m_tensorType = dynamic ? DATA_TENSOR_DYNAMIC : DATA_TENSOR;
  desc.m_dims = dims.size();
  for (size_t dim = 0; dim < dims.size(); ++dim) {
    desc.m_sizes[dim] = dims[dim];
    desc.m_minSizes[dim] = min_dims.empty() ? (dynamic ? 1 : dims[dim]) : min_dims[dim];
    TORCH_CHECK(desc.m_minSizes[dim] > 0 && desc.m_minSizes[dim] <= dims[dim],
                "GaudiRT FX: invalid dynamic tensor extent");
  }
  synTensor handle = nullptr;
  check(synTensorCreate(&handle, &desc, section, offset), "program tensor creation failed");
  tensors_.push_back(handle);
  if (type == syn_type_fp8_143) {
    synFpQuantParam quant{1.0, 7};
    synFpQuantMetadata metadata{type, &quant, 1};
    check(synTensorSetQuantizationData(handle, SYN_FP_QUANT_METADATA, &metadata, sizeof(metadata)),
          "program FP8 metadata failed");
  }
  if (slot >= 0) {
    auto& binding = recipe_.tensors[slot];
    binding.tensorName = recipe_.names[slot].c_str();
    binding.tensorType = desc.m_tensorType;
    for (size_t dim = 0; dim < dims.size(); ++dim) binding.tensorSize[dim] = dims[dim];
  }
  return handle;
}

void SynapseGraphBuilder::permutation(synTensor tensor, const synTensorPermutation& permutation) {
  check(synTensorSetPermutation(tensor, &permutation), "tensor permutation failed");
}

void SynapseGraphBuilder::node(const NodeSpec& node) {
  TORCH_CHECK(!node.outputs.empty(), "GaudiRT FX: a graph node needs an output");
  const auto name = prefix_ + "_n" + std::to_string(next_node_++);
  std::visit([&](const auto& params) {
    using Parameters = std::decay_t<decltype(params)>;
    const void* data = nullptr;
    size_t size = 0;
    if constexpr (!std::is_same_v<Parameters, std::monostate>) {
      data = &params;
      size = sizeof(params);
    }
    check(synNodeCreate(graph_, node.inputs.data(), node.outputs.data(), node.inputs.size(),
                        node.outputs.size(), data, size, node.guid.c_str(), name.c_str(), nullptr, nullptr),
          "program node creation failed");
  }, node.parameters);
}

void SynapseGraphBuilder::compile(runtime::RecipeCache& cache, const std::string& key) {
  recipe_.recipe = runtime::compile_cached(graph_, prefix_, key, &cache);
  check(synWorkspaceGetSize(&recipe_.workspace, recipe_.recipe), "program workspace query failed");
  runtime::initialize_launch_tensor_ids(recipe_.recipe, recipe_.tensors.data(), recipe_.tensors.size());
}

}  // namespace gaudirt::generic
