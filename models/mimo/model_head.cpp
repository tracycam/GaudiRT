// SPDX-License-Identifier: Apache-2.0
#include "model_head.hpp"
#include "tp_reduce_graph.hpp"

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
                     uint64_t lm_head_weight, runtime::RecipeCache* cache,
                     bool gathered_input, uint32_t vocab_size)
    : arena_(arena), gathered_input_(gathered_input) {
  if (final_norm_gamma == 0 || lm_head_weight == 0) {
    throw std::invalid_argument("MiMo model head: null weight address");
  }
  if (vocab_size != kVocabSize && vocab_size != kLocalVocabSize)
    throw std::invalid_argument("MiMo model head: unsupported vocabulary size");
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

    synTensor hidden = nullptr;
    if (gathered_input_) {
      synTensor gathered = external(kHidden, "model_hidden_gathered", syn_type_float,
                                    {6144, 8}, 0);
      sections.push_back(nullptr);
      hidden = append_tp8_reduce(graph, gathered, syn_type_float, sections.back());
    } else {
      hidden = external(kHidden, "model_final_hidden", syn_type_bf16,
                        {6144, 1}, 0);
    }
    synTensor residual = external(kResidual, "model_final_residual", syn_type_bf16,
                                  {6144, 1}, 0);
    synTensor gamma = external(kGamma, "model_final_gamma", syn_type_bf16,
                               {6144}, final_norm_gamma);
    synTensor weight = external(kLmHeadWeight, "model_lm_head_weight", syn_type_bf16,
                                {6144, vocab_size}, lm_head_weight);
    synTensor logits = external(kLogits, "model_logits", syn_type_bf16,
                                {vocab_size, 1}, 0);
    synTensor normalized = external(kNormalized, "model_final_normalized",
                                    syn_type_bf16, {6144, 1}, 0);
    const char* residual_name = vocab_size == kVocabSize
        ? (gathered_input_ ? "model_final_residual_out_full_gathered" : "model_final_residual_out_full_direct")
        : (gathered_input_ ? "model_final_residual_out_shard_gathered" : "model_final_residual_out_shard_direct");
    synTensor residual_out = temporary(residual_name, syn_type_bf16, {6144, 1});
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
    const std::string cache_key = std::string(gathered_input_ ? "model_head_gathered_tp1_v2_" : "model_head_v1_") +
                                  std::to_string(vocab_size);
    recipe_ = runtime::compile_cached(graph, "mimo_final_rmsnorm_lm_head_b1_m1", cache_key, cache);
    check(synWorkspaceGetSize(&workspace_bytes_, recipe_), "synWorkspaceGetSize");
    arena_.ensure_workspace(workspace_bytes_);
    runtime::initialize_launch_tensor_ids(recipe_, launch_tensors_.data(),
                                          kSlotCount);
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

runtime::PreparedLaunch ModelHead::prepare(uint64_t hidden, uint64_t residual,
                                           uint64_t normalized, uint64_t logits) {
  auto tensors = launch_tensors_;
  tensors[kHidden].pTensorAddress = hidden;
  const size_t count = kSlotCount;
  tensors[kResidual].pTensorAddress = residual;
  tensors[kNormalized].pTensorAddress = normalized;
  tensors[kLogits].pTensorAddress = logits;
  runtime::PreparedLaunch result{recipe_, arena_.ensure_workspace(workspace_bytes_),
                                 {tensors.begin(), tensors.begin() + count}, 0};
  result.validate();
  return result;
}

void ModelHead::execute(uint64_t hidden, uint64_t residual, uint64_t normalized,
                        uint64_t logits, synStreamHandle stream) {
  if (gathered_input_)
    throw std::invalid_argument("MiMo model head: gathered input requires prepared launch");
  if (!stream || !hidden || !residual || !normalized || !logits) {
    throw std::invalid_argument("MiMo model head: invalid execution address");
  }
  launch_tensors_[kHidden].pTensorAddress = hidden;
  launch_tensors_[kResidual].pTensorAddress = residual;
  launch_tensors_[kNormalized].pTensorAddress = normalized;
  launch_tensors_[kLogits].pTensorAddress = logits;
  const uint64_t workspace = arena_.ensure_workspace(workspace_bytes_);
  check(synLaunch(stream, launch_tensors_.data(),
                  static_cast<uint32_t>(kSlotCount), workspace, recipe_, 0),
        "synLaunch");
}

}  // namespace gaudirt::mimo
