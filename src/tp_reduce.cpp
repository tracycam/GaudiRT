// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/tp_reduce.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <stdexcept>
#include <string>
#include <vector>

namespace gaudirt::runtime {
namespace {

void check(synStatus status, const char* operation) {
  if (status != synSuccess) {
    throw std::runtime_error(std::string("TpReduce: ") + operation +
                             " failed, status " + std::to_string(status));
  }
}

synTensor tensor(const char* name, synDataType dtype, uint64_t d0, uint64_t d1,
                 synSectionHandle section = nullptr) {
  synTensorDescriptor descriptor = {};
  descriptor.m_name = name;
  descriptor.m_dataType = dtype;
  descriptor.m_dims = 2;
  descriptor.m_sizes[0] = descriptor.m_minSizes[0] = d0;
  descriptor.m_sizes[1] = descriptor.m_minSizes[1] = d1;
  synTensor result = nullptr;
  check(synTensorCreate(&result, &descriptor, section, 0), name);
  return result;
}

void node(synGraphHandle graph, const std::vector<synTensor>& inputs,
          const std::vector<synTensor>& outputs, const void* params,
          unsigned params_bytes, const char* guid, const char* name) {
  check(synNodeCreate(graph, inputs.data(), outputs.data(), inputs.size(),
                      outputs.size(), params, params_bytes, guid, name, nullptr,
                      nullptr),
        name);
}

}  // namespace

TpReduce::TpReduce(HbmArena& arena, uint32_t elements,
                   synDataType gathered_type, RecipeCache* cache, uint32_t ranks)
    : arena_(arena), elements_(elements), ranks_(ranks) {
  if (elements == 0 ||
      (ranks_ != 1 && ranks_ != 8) || (ranks_ == 1 && gathered_type != syn_type_float) ||
      (gathered_type != syn_type_bf16 && gathered_type != syn_type_float)) {
    throw std::invalid_argument("TpReduce: invalid elements or gathered dtype");
  }
  rank_bytes_ = static_cast<uint64_t>(elements_) *
                (gathered_type == syn_type_bf16 ? 2 : 4);
  try {
    check(synGraphCreate(&graph_, synDeviceGaudi2), "create graph");
    check(synSectionCreate(&output_section_, 0, graph_), "create output section");
    check(synSectionSetPersistent(output_section_, true),
          "persist output section");
    synTensor output = tensor("output", syn_type_bf16, elements, 1,
                              output_section_);

    std::vector<synTensor> ranks(ranks_);
    for (unsigned rank = 0; rank < ranks_; ++rank) {
      check(synSectionCreate(&rank_sections_[rank], 0, graph_),
            "create rank section");
      check(synSectionSetPersistent(rank_sections_[rank], true),
            "persist rank section");
      ranks[rank] = tensor(("rank" + std::to_string(rank)).c_str(),
                           gathered_type, elements, 1, rank_sections_[rank]);
    }

    ns_CastKernel::ParamsV3 cast_params = {};
    cast_params.round_mode = CAST_ROUND_HALF_NE;
    std::vector<synTensor> wide(ranks_);
    for (unsigned rank = 0; rank < ranks_; ++rank) {
      if (gathered_type == syn_type_bf16) {
        wide[rank] = tensor(("wide" + std::to_string(rank)).c_str(),
                            syn_type_float, elements, 1);
        node(graph_, {ranks[rank]}, {wide[rank]}, &cast_params,
             sizeof(cast_params), "cast_bf16_to_f32", "widen_rank");
      } else {
        wide[rank] = ranks[rank];
      }
    }
    synTensor sum = wide[0];
    for (unsigned rank = 1; rank < ranks_; ++rank) {
      synTensor next = tensor(("sum" + std::to_string(rank)).c_str(),
                              syn_type_float, elements, 1);
      node(graph_, {sum, wide[rank]}, {next}, nullptr, 0, "add_fwd_f32",
           "sum_next_rank");
      sum = next;
    }
    node(graph_, {sum}, {output}, &cast_params, sizeof(cast_params),
         "cast_f32_to_bf16", "round_final_bf16");
    const std::string key = (ranks_ == 1 ? "tp_single_cast_v1_" : "tp8_gather_sum_v1_") + std::to_string(elements) + "_" +
                            std::to_string(static_cast<int>(gathered_type));
    recipe_ = compile_cached(graph_, key, key, cache);
    check(synWorkspaceGetSize(&workspace_bytes_, recipe_), "workspace size");
    arena_.ensure_workspace(workspace_bytes_);
    const char* names[] = {"rank0", "rank1", "rank2", "rank3",
                           "rank4", "rank5", "rank6", "rank7"};
    for (unsigned rank = 0; rank < ranks_; ++rank) {
      prepared_template_[rank].tensorName = names[rank];
      prepared_template_[rank].tensorType = DATA_TENSOR;
      prepared_template_[rank].tensorSize[0] = elements_;
      prepared_template_[rank].tensorSize[1] = 1;
    }
    prepared_template_[ranks_].tensorName = "output";
    prepared_template_[ranks_].tensorType = DATA_TENSOR;
    prepared_template_[ranks_].tensorSize[0] = elements_;
    prepared_template_[ranks_].tensorSize[1] = 1;
    initialize_launch_tensor_ids(recipe_, prepared_template_.data(),
                                 ranks_ + 1);
  } catch (...) {
    if (recipe_) synRecipeDestroy(recipe_);
    if (graph_) synGraphDestroy(graph_);
    for (auto& section : rank_sections_) {
      if (section) synSectionDestroy(section);
      section = nullptr;
    }
    if (output_section_) synSectionDestroy(output_section_);
    recipe_ = nullptr;
    graph_ = nullptr;
    output_section_ = nullptr;
    throw;
  }
}

TpReduce::~TpReduce() {
  if (recipe_) synRecipeDestroy(recipe_);
  if (graph_) synGraphDestroy(graph_);
  for (auto& section : rank_sections_) {
    if (section) synSectionDestroy(section);
  }
  if (output_section_) synSectionDestroy(output_section_);
}

PreparedLaunch TpReduce::prepare(uint64_t gathered, uint64_t output) {
  return prepare_strided(gathered, output, rank_bytes_);
}

PreparedLaunch TpReduce::prepare_strided(uint64_t gathered, uint64_t output,
                                         uint64_t rank_stride_bytes) {
  if (rank_stride_bytes < rank_bytes_ || (ranks_ > 1 && rank_stride_bytes > (UINT64_MAX - rank_bytes_) / (ranks_ - 1)))
    throw std::invalid_argument("TpReduce: invalid rank stride");
  const uint64_t gathered_bytes = (ranks_ - 1) * rank_stride_bytes + rank_bytes_;
  const uint64_t output_bytes = static_cast<uint64_t>(elements_) * 2;
  if (!gathered || !output || gathered + gathered_bytes < gathered ||
      output + output_bytes < output ||
      (gathered < output + output_bytes && output < gathered + gathered_bytes))
    throw std::invalid_argument("TpReduce: invalid or overlapping bindings");
  auto tensors = prepared_template_;
  for (unsigned rank = 0; rank < ranks_; ++rank)
    tensors[rank].pTensorAddress = gathered + rank * rank_stride_bytes;
  tensors[ranks_].pTensorAddress = output;
  PreparedLaunch result{recipe_, arena_.ensure_workspace(workspace_bytes_),
                        {tensors.begin(), tensors.begin() + ranks_ + 1}, 0};
  result.validate();
  return result;
}

void TpReduce::execute(uint64_t gathered, uint64_t output,
                       synStreamHandle stream) {
  if (!gathered || !output || !stream || !recipe_) {
    throw std::runtime_error("TpReduce: invalid launch argument");
  }
  prepare(gathered, output).execute(stream);
}

}  // namespace gaudirt::runtime
