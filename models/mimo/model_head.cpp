// SPDX-License-Identifier: Apache-2.0
#include "model_head.hpp"

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
    throw std::runtime_error(std::string("MiMo model head: ") + operation +
                             " failed with Synapse status " +
                             std::to_string(static_cast<int>(status)));
  }
}

}  // namespace

ModelHead::ModelHead(runtime::HbmArena& arena, uint64_t final_norm_gamma,
                     uint64_t lm_head_weight, runtime::RecipeCache* cache)
    : arena_(arena) {
  if (final_norm_gamma == 0 || lm_head_weight == 0) {
    throw std::invalid_argument("MiMo model head: null weight address");
  }
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
      for (uint64_t size : sizes) {
        info.tensorSize[dim++] = static_cast<uint32_t>(size);
      }
      launch_tensors_[slot] = info;
      return tensor;
    };
    auto temporary = [&](const char* name, synDataType dtype,
                         Sizes sizes) -> synTensor {
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
      check(synTensorCreate(&tensor, &descriptor, nullptr, 0), name);
      return tensor;
    };

    synTensor hidden = external(kHidden, "model_final_hidden", syn_type_bf16,
                                {6144, 1}, 0);
    synTensor residual = external(kResidual, "model_final_residual", syn_type_bf16,
                                  {6144, 1}, 0);
    synTensor gamma = external(kGamma, "model_final_gamma", syn_type_bf16,
                               {6144}, final_norm_gamma);
    synTensor weight = external(kLmHeadWeight, "model_lm_head_weight", syn_type_bf16,
                                {6144, 152576}, lm_head_weight);
    synTensor logits = external(kLogits, "model_logits", syn_type_bf16,
                                {152576, 1}, 0);
    synTensor normalized = external(kNormalized, "model_final_normalized",
                                    syn_type_bf16, {6144, 1}, 0);
    synTensor residual_out = temporary("model_final_residual_out", syn_type_bf16,
                                       {6144, 1});
    const float epsilon = 1e-5F;
    synTensor norm_inputs[] = {hidden, residual, gamma};
    synTensor norm_outputs[] = {residual_out, normalized};
    check(synNodeCreate(graph, norm_inputs, norm_outputs, 3, 2, &epsilon,
                        sizeof(epsilon), "gk_residual_rmsnorm_bf16_v1",
                        "model_final_rmsnorm", nullptr, nullptr),
          "final RMSNorm");
    const synGEMMParams gemm_params{false, true};
    synTensor gemm_inputs[] = {normalized, weight};
    check(synNodeCreate(graph, gemm_inputs, &logits, 2, 1, &gemm_params,
                        sizeof(gemm_params), "gemm", "model_lm_head", nullptr,
                        nullptr),
          "lm_head GEMM");
    recipe_ = runtime::compile_cached(graph, "mimo_final_rmsnorm_lm_head_b1_m1",
                                      "model_head_v1", cache);
    check(synWorkspaceGetSize(&workspace_bytes_, recipe_), "synWorkspaceGetSize");
    arena_.ensure_workspace(workspace_bytes_);
    runtime::initialize_launch_tensor_ids(recipe_, launch_tensors_.data(),
                                          launch_tensors_.size());
    synGraphDestroy(graph);
    graph = nullptr;
    for (synSectionHandle& section : sections) {
      synSectionDestroy(section);
      section = nullptr;
    }
  } catch (...) {
    if (recipe_ != nullptr) {
      synRecipeDestroy(recipe_);
      recipe_ = nullptr;
    }
    if (graph != nullptr) synGraphDestroy(graph);
    for (synSectionHandle& section : sections) {
      if (section != nullptr) synSectionDestroy(section);
    }
    throw;
  }
}

ModelHead::~ModelHead() {
  if (recipe_ != nullptr) synRecipeDestroy(recipe_);
}

void ModelHead::execute(uint64_t hidden, uint64_t residual, uint64_t normalized,
                        uint64_t logits, synStreamHandle stream) {
  if (!stream || !hidden || !residual || !normalized || !logits) {
    throw std::invalid_argument("MiMo model head: invalid execution address");
  }
  launch_tensors_[kHidden].pTensorAddress = hidden;
  launch_tensors_[kResidual].pTensorAddress = residual;
  launch_tensors_[kNormalized].pTensorAddress = normalized;
  launch_tensors_[kLogits].pTensorAddress = logits;
  const uint64_t workspace = arena_.ensure_workspace(workspace_bytes_);
  check(synLaunch(stream, launch_tensors_.data(),
                  static_cast<uint32_t>(launch_tensors_.size()), workspace, recipe_, 0),
        "synLaunch");
}

}  // namespace gaudirt::mimo
