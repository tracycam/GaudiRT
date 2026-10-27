// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/tp_reduce.hpp"

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
                   synDataType gathered_type)
    : arena_(arena), elements_(elements) {
  if (elements == 0 ||
      (gathered_type != syn_type_bf16 && gathered_type != syn_type_float)) {
    throw std::invalid_argument("TpReduce: invalid elements or gathered dtype");
  }
  try {
    check(synGraphCreate(&graph_, synDeviceGaudi2), "create graph");
    check(synSectionCreate(&gathered_section_, 0, graph_),
          "create gathered section");
    check(synSectionSetPersistent(gathered_section_, true),
          "persist gathered section");
    check(synSectionCreate(&output_section_, 0, graph_), "create output section");
    check(synSectionSetPersistent(output_section_, true),
          "persist output section");
    synTensor gathered = tensor("gathered", gathered_type, elements, 8,
                                gathered_section_);
    synTensor output = tensor("output", syn_type_bf16, elements, 1,
                              output_section_);

    std::vector<synTensor> ranks(8);
    for (unsigned rank = 0; rank < 8; ++rank) {
      ranks[rank] = tensor(("rank" + std::to_string(rank)).c_str(),
                           gathered_type, elements, 1);
    }
    synSplitParams split{1};
    node(graph_, {gathered}, ranks, &split, sizeof(split), "split",
         "split_gathered_ranks");

    ns_CastKernel::ParamsV3 cast_params = {};
    cast_params.round_mode = CAST_ROUND_HALF_NE;
    std::vector<synTensor> wide(8);
    for (unsigned rank = 0; rank < 8; ++rank) {
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
    for (unsigned rank = 1; rank < 8; ++rank) {
      synTensor next = tensor(("sum" + std::to_string(rank)).c_str(),
                              syn_type_float, elements, 1);
      node(graph_, {sum, wide[rank]}, {next}, nullptr, 0, "add_fwd_f32",
           "sum_next_rank");
      sum = next;
    }
    node(graph_, {sum}, {output}, &cast_params, sizeof(cast_params),
         "cast_f32_to_bf16", "round_final_bf16");
    check(synGraphCompile(&recipe_, graph_, "tp8_gather_fp32_sum_cast", nullptr),
          "compile graph");
    check(synWorkspaceGetSize(&workspace_bytes_, recipe_), "workspace size");
    arena_.ensure_workspace(workspace_bytes_);
  } catch (...) {
    if (recipe_) synRecipeDestroy(recipe_);
    if (graph_) synGraphDestroy(graph_);
    if (gathered_section_) synSectionDestroy(gathered_section_);
    if (output_section_) synSectionDestroy(output_section_);
    recipe_ = nullptr;
    graph_ = nullptr;
    gathered_section_ = nullptr;
    output_section_ = nullptr;
    throw;
  }
}

TpReduce::~TpReduce() {
  if (recipe_) synRecipeDestroy(recipe_);
  if (graph_) synGraphDestroy(graph_);
  if (gathered_section_) synSectionDestroy(gathered_section_);
  if (output_section_) synSectionDestroy(output_section_);
}

void TpReduce::execute(uint64_t gathered, uint64_t output,
                       synStreamHandle stream) {
  if (!gathered || !output || !stream || !recipe_) {
    throw std::runtime_error("TpReduce: invalid launch argument");
  }
  synLaunchTensorInfo tensors[2] = {};
  tensors[0].tensorName = "gathered";
  tensors[0].tensorType = DATA_TENSOR;
  tensors[0].pTensorAddress = gathered;
  tensors[0].tensorSize[0] = elements_;
  tensors[0].tensorSize[1] = 8;
  tensors[1].tensorName = "output";
  tensors[1].tensorType = DATA_TENSOR;
  tensors[1].pTensorAddress = output;
  tensors[1].tensorSize[0] = elements_;
  tensors[1].tensorSize[1] = 1;
  const uint64_t workspace = arena_.ensure_workspace(workspace_bytes_);
  check(synLaunch(stream, tensors, 2, workspace, recipe_, SYN_FLAGS_TENSOR_NAME),
        "launch");
}

}  // namespace gaudirt::runtime
