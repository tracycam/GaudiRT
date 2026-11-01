// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace gaudirt::mimo {

// Reduce the contiguous TP8 AllGather with FP32 accumulation and BF16 output.
inline synTensor append_tp8_reduce(synGraphHandle graph, synTensor gathered,
                                   synDataType dtype,
                                   synSectionHandle& output_section,
                                   uint32_t rows = 1) {
  if (!graph || !gathered || (rows != 1 && rows != 32 && rows != 512) ||
      (dtype != syn_type_bf16 && dtype != syn_type_float))
    throw std::invalid_argument("MiMo TP reduction: invalid graph, tensor or dtype");
  const auto check = [](synStatus status, const char* operation) {
    if (status != synSuccess)
      throw std::runtime_error(std::string("MiMo TP reduction: ") + operation +
                               " failed, status " +
                               std::to_string(static_cast<int>(status)));
  };
  check(synSectionCreate(&output_section, 0, graph), "create output section");
  check(synSectionSetPersistent(output_section, false), "set output section");
  synTensorDescriptor descriptor{};
  const std::string output_name = rows == 1 ? "tp_reduce_bf16" : "tp_reduce_m" + std::to_string(rows) + "_bf16";
  descriptor.m_name = output_name.c_str();
  descriptor.m_dataType = syn_type_bf16;
  descriptor.m_dims = 2;
  descriptor.m_sizes[0] = descriptor.m_minSizes[0] = 6144;
  descriptor.m_sizes[1] = descriptor.m_minSizes[1] = rows;
  synTensor output = nullptr;
  check(synTensorCreate(&output, &descriptor, output_section, 0), "create output");
  const int32_t ranks = 8;
  const char* guid = dtype == syn_type_float ? "pf_sum_f32_bf16" : "pf_sum_bf16_bf16";
  check(synNodeCreate(graph, &gathered, &output, 1, 1, &ranks, sizeof(ranks),
                      guid, "tp_reduce", nullptr, nullptr), "reduce ranks");
  return output;
}

}  // namespace gaudirt::mimo
