// SPDX-License-Identifier: Apache-2.0
#include "model_argmax.hpp"

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
  if (status != synSuccess)
    throw std::runtime_error(std::string("MiMo argmax: ") + operation +
                             " failed with Synapse status " +
                             std::to_string(static_cast<int>(status)));
}
}  // namespace

ModelArgmax::ModelArgmax(runtime::HbmArena& arena, runtime::RecipeCache* cache, uint32_t rows)
    : arena_(arena) {
  if (rows != 1 && rows != 4 && rows != 8)
    throw std::invalid_argument("MiMo argmax: unsupported batch size");
  synGraphHandle graph = nullptr;
  std::vector<synSectionHandle> sections;
  sections.reserve(kSlotCount);
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "synGraphCreate");
    auto tensor = [&](const char* name, synDataType dtype, Sizes sizes,
                      bool persistent, Slot slot) -> synTensor {
      synSectionHandle section = nullptr;
      if (persistent) {
        check(synSectionCreate(&section, 0, graph), name);
        sections.push_back(section);
        check(synSectionSetPersistent(section, true), name);
      }
      synTensorDescriptor descriptor{};
      const std::string scoped_name = rows == 1 ? name : std::string(name) + "_b" + std::to_string(rows);
      if (persistent) tensor_names_[slot] = scoped_name;
      descriptor.m_name = persistent ? tensor_names_[slot].c_str() : scoped_name.c_str();
      descriptor.m_dataType = dtype;
      descriptor.m_dims = static_cast<unsigned>(sizes.size());
      unsigned dim = 0;
      for (uint64_t size : sizes) {
        descriptor.m_sizes[dim] = descriptor.m_minSizes[dim] = size;
        ++dim;
      }
      synTensor value = nullptr;
      check(synTensorCreate(&value, &descriptor, section, 0), name);
      if (persistent) {
        synLaunchTensorInfo info{};
        info.tensorName = tensor_names_[slot].c_str();
        info.tensorType = DATA_TENSOR;
        dim = 0;
        for (uint64_t size : sizes) info.tensorSize[dim++] = static_cast<uint32_t>(size);
        launch_tensors_[slot] = info;
      }
      return value;
    };
    synTensor logits = rows == 1
        ? tensor("argmax_logits", syn_type_bf16, {152576, 1}, true, kLogits)
        : tensor("argmax_logits", syn_type_bf16, {19072, rows, 8}, true, kLogits);
    synTensor status = tensor("argmax_embedding_status", syn_type_int32, {rows}, true, kEmbeddingStatus);
    synTensor output = tensor("argmax_next_token", syn_type_int32, {rows}, true, kNextToken);
    synTensor maxima = rows == 1 ? tensor("argmax_partial_max", syn_type_float, {24}, false, kSlotCount)
                                 : tensor("argmax_partial_max", syn_type_float, {24, rows}, false, kSlotCount);
    synTensor ids = rows == 1 ? tensor("argmax_partial_id", syn_type_int32, {24}, false, kSlotCount)
                              : tensor("argmax_partial_id", syn_type_int32, {24, rows}, false, kSlotCount);
    synTensor bad = rows == 1 ? tensor("argmax_partial_bad", syn_type_int32, {24}, false, kSlotCount)
                              : tensor("argmax_partial_bad", syn_type_int32, {24, rows}, false, kSlotCount);
    synTensor parts_outputs[] = {maxima, ids, bad};
    const std::string parts_name = rows == 1 ? "mimo_argmax_parts" : "mimo_argmax_parts_b" + std::to_string(rows);
    check(synNodeCreate(graph, &logits, parts_outputs, 1, 3, nullptr, 0,
                        rows == 1 ? "gk_mimo_argmax_parts_bf16_v1" : "gk_mimo_argmax_batch_parts_bf16_v1",
                        parts_name.c_str(), nullptr, nullptr),
          "argmax parts");
    synTensor finish_inputs[] = {maxima, ids, bad, status};
    const std::string finish_name = rows == 1 ? "mimo_argmax_finish" : "mimo_argmax_finish_b" + std::to_string(rows);
    check(synNodeCreate(graph, finish_inputs, &output, 4, 1, nullptr, 0,
                        rows == 1 ? "gk_mimo_argmax_finish_v1" : "gk_mimo_argmax_batch_finish_v1",
                        finish_name.c_str(), nullptr, nullptr),
          "argmax finish");
    const std::string recipe_name = rows == 1 ? "mimo_argmax_b1_m1" : "mimo_argmax_b" + std::to_string(rows);
    const std::string cache_key = rows == 1 ? "model_argmax_v1" : "model_argmax_batch_v1_b" + std::to_string(rows);
    recipe_ = runtime::compile_cached(graph, recipe_name, cache_key, cache);
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

ModelArgmax::~ModelArgmax() {
  if (recipe_) synRecipeDestroy(recipe_);
}

runtime::PreparedLaunch ModelArgmax::prepare(uint64_t logits, uint64_t status,
                                             uint64_t next_token) {
  auto tensors = launch_tensors_;
  tensors[kLogits].pTensorAddress = logits;
  tensors[kEmbeddingStatus].pTensorAddress = status;
  tensors[kNextToken].pTensorAddress = next_token;
  runtime::PreparedLaunch result{recipe_, arena_.ensure_workspace(workspace_bytes_),
                                 {tensors.begin(), tensors.end()}, 0};
  result.validate();
  return result;
}

void ModelArgmax::execute(uint64_t logits, uint64_t status, uint64_t next_token,
                          synStreamHandle stream) {
  if (!stream || !logits || !status || !next_token)
    throw std::invalid_argument("MiMo argmax: invalid execution address");
  launch_tensors_[kLogits].pTensorAddress = logits;
  launch_tensors_[kEmbeddingStatus].pTensorAddress = status;
  launch_tensors_[kNextToken].pTensorAddress = next_token;
  const uint64_t workspace = arena_.ensure_workspace(workspace_bytes_);
  check(synLaunch(stream, launch_tensors_.data(), static_cast<uint32_t>(launch_tensors_.size()),
                  workspace, recipe_, 0), "synLaunch");
}

}  // namespace gaudirt::mimo
