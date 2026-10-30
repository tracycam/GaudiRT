// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <synapse_common_types.hpp>

#include <array>
#include <stdexcept>
#include <string>

namespace gaudirt::mimo {

// Match TpReduce's rank-ordered FP32 sum and final BF16 rounding inside the
// consumer graph. The eight input tensors are consecutive AllGather ranks.
inline synTensor append_tp8_reduce(synGraphHandle graph,
                                   const std::array<synTensor, 8>& ranks,
                                   synDataType dtype,
                                   synSectionHandle& output_section) {
  if (!graph || (dtype != syn_type_bf16 && dtype != syn_type_float))
    throw std::invalid_argument("MiMo TP reduction: invalid graph or dtype");
  const auto check = [](synStatus status, const char* operation) {
    if (status != synSuccess)
      throw std::runtime_error(std::string("MiMo TP reduction: ") + operation +
                               " failed, status " +
                               std::to_string(static_cast<int>(status)));
  };
  const auto temporary = [&](const std::string& name, synDataType type) {
    synTensorDescriptor descriptor{};
    descriptor.m_name = name.c_str();
    descriptor.m_dataType = type;
    descriptor.m_dims = 2;
    descriptor.m_sizes[0] = descriptor.m_minSizes[0] = 6144;
    descriptor.m_sizes[1] = descriptor.m_minSizes[1] = 1;
    synTensor tensor = nullptr;
    check(synTensorCreate(&tensor, &descriptor, nullptr, 0), "create tensor");
    return tensor;
  };
  ns_CastKernel::ParamsV3 cast_params{};
  cast_params.round_mode = CAST_ROUND_HALF_NE;
  std::array<synTensor, 8> wide{};
  for (unsigned rank = 0; rank < ranks.size(); ++rank) {
    if (!ranks[rank]) throw std::invalid_argument("MiMo TP reduction: null rank");
    if (dtype == syn_type_float) {
      wide[rank] = ranks[rank];
      continue;
    }
    wide[rank] = temporary("tp_reduce_wide_" + std::to_string(rank), syn_type_float);
    check(synNodeCreate(graph, &ranks[rank], &wide[rank], 1, 1, &cast_params,
                        sizeof(cast_params), "cast_bf16_to_f32", "tp_reduce_widen",
                        nullptr, nullptr), "widen rank");
  }
  synTensor sum = wide[0];
  for (unsigned rank = 1; rank < wide.size(); ++rank) {
    synTensor next = temporary("tp_reduce_sum_" + std::to_string(rank), syn_type_float);
    synTensor inputs[] = {sum, wide[rank]};
    check(synNodeCreate(graph, inputs, &next, 2, 1, nullptr, 0, "add_fwd_f32",
                        "tp_reduce_add_rank", nullptr, nullptr), "add rank");
    sum = next;
  }
  check(synSectionCreate(&output_section, 0, graph), "create output section");
  check(synSectionSetPersistent(output_section, false), "set output section");
  synTensorDescriptor descriptor{};
  descriptor.m_name = "tp_reduce_bf16";
  descriptor.m_dataType = syn_type_bf16;
  descriptor.m_dims = 2;
  descriptor.m_sizes[0] = descriptor.m_minSizes[0] = 6144;
  descriptor.m_sizes[1] = descriptor.m_minSizes[1] = 1;
  synTensor output = nullptr;
  check(synTensorCreate(&output, &descriptor, output_section, 0), "create output");
  check(synNodeCreate(graph, &sum, &output, 1, 1, &cast_params, sizeof(cast_params),
                      "cast_f32_to_bf16", "tp_reduce_round", nullptr, nullptr),
        "round result");
  return output;
}

}  // namespace gaudirt::mimo
