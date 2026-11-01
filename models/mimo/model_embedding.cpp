// SPDX-License-Identifier: Apache-2.0
#include "model_embedding.hpp"

#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace gaudirt::mimo {
namespace {
using Sizes = std::initializer_list<uint64_t>;

void check(synStatus status, const char* operation) {
  if (status != synSuccess) {
    throw std::runtime_error(std::string("MiMo embedding: ") + operation +
                             " failed with Synapse status " +
                             std::to_string(static_cast<int>(status)));
  }
}
}  // namespace

ModelEmbedding::ModelEmbedding(runtime::HbmArena& arena, uint64_t table,
                               runtime::RecipeCache* cache, uint32_t rows)
    : arena_(arena) {
  if (rows != 1 && rows != 32 && rows != 256 && rows != 512 && rows != 1024)
    throw std::invalid_argument("MiMo embedding: invalid rows");
  if (!table) throw std::invalid_argument("MiMo embedding: null table");
  synGraphHandle graph = nullptr;
  std::vector<synSectionHandle> sections;
  sections.reserve(kSlotCount);
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "synGraphCreate");
    auto external = [&](Slot slot, const char* name, synDataType dtype,
                        Sizes sizes, uint64_t address) -> synTensor {
      synSectionHandle section = nullptr;
      check(synSectionCreate(&section, 0, graph), name);
      sections.push_back(section);
      check(synSectionSetPersistent(section, true), name);
      synTensorDescriptor descriptor{};
      descriptor.m_name = name;
      descriptor.m_dataType = dtype;
      descriptor.m_dims = static_cast<unsigned>(sizes.size());
      unsigned dim = 0;
      for (uint64_t size : sizes) {
        descriptor.m_sizes[dim] = descriptor.m_minSizes[dim] = size;
        ++dim;
      }
      synTensor tensor = nullptr;
      check(synTensorCreate(&tensor, &descriptor, section, 0), name);
      synLaunchTensorInfo info{};
      info.tensorName = name;
      info.pTensorAddress = address;
      info.tensorType = DATA_TENSOR;
      dim = 0;
      for (uint64_t size : sizes) info.tensorSize[dim++] = static_cast<uint32_t>(size);
      launch_tensors_[slot] = info;
      return tensor;
    };
    synTensor table_tensor = external(kTable, "embedding_table", syn_type_bf16,
                                      {6144, 152576}, table);
    synTensor token_tensor = external(kToken, "embedding_token", syn_type_int32, {rows}, 0);
    synTensor value_tensor = external(kEmbedding, "embedding_value", syn_type_bf16, {6144, rows}, 0);
    synTensor status_tensor = external(kStatus, "embedding_status", syn_type_int32, {rows}, 0);
    synTensor inputs[] = {table_tensor, token_tensor};
    synTensor outputs[] = {value_tensor, status_tensor};
    check(synNodeCreate(graph, inputs, outputs, 2, 2, nullptr, 0,
                        "gk_mimo_embedding_bf16_v1", "mimo_embedding", nullptr, nullptr),
          "synNodeCreate");
    recipe_ = runtime::compile_cached(graph, "mimo_embedding_b1_m" + std::to_string(rows),
                                      rows == 1 ? "model_embedding_v1" : "model_embedding_m" + std::to_string(rows) + "_v1", cache);
    check(synWorkspaceGetSize(&workspace_bytes_, recipe_), "synWorkspaceGetSize");
    arena_.ensure_workspace(workspace_bytes_);
    runtime::initialize_launch_tensor_ids(recipe_, launch_tensors_.data(), launch_tensors_.size());
    synGraphDestroy(graph);
    graph = nullptr;
    for (auto& section : sections) {
      synSectionDestroy(section);
      section = nullptr;
    }
  } catch (...) {
    if (recipe_) synRecipeDestroy(recipe_);
    if (graph) synGraphDestroy(graph);
    for (auto& section : sections) if (section) synSectionDestroy(section);
    throw;
  }
}

ModelEmbedding::~ModelEmbedding() {
  if (recipe_) synRecipeDestroy(recipe_);
}

runtime::PreparedLaunch ModelEmbedding::prepare(uint64_t token,
                                                uint64_t embedding,
                                                uint64_t status) {
  auto tensors = launch_tensors_;
  tensors[kToken].pTensorAddress = token;
  tensors[kEmbedding].pTensorAddress = embedding;
  tensors[kStatus].pTensorAddress = status;
  runtime::PreparedLaunch result{recipe_, arena_.ensure_workspace(workspace_bytes_),
                                 {tensors.begin(), tensors.end()}, 0};
  result.validate();
  return result;
}

void ModelEmbedding::execute(uint64_t token, uint64_t embedding, uint64_t status,
                             synStreamHandle stream) {
  if (!stream || !token || !embedding || !status)
    throw std::invalid_argument("MiMo embedding: invalid execution address");
  launch_tensors_[kToken].pTensorAddress = token;
  launch_tensors_[kEmbedding].pTensorAddress = embedding;
  launch_tensors_[kStatus].pTensorAddress = status;
  const uint64_t workspace = arena_.ensure_workspace(workspace_bytes_);
  check(synLaunch(stream, launch_tensors_.data(), static_cast<uint32_t>(launch_tensors_.size()),
                  workspace, recipe_, 0), "synLaunch");
}

}  // namespace gaudirt::mimo
