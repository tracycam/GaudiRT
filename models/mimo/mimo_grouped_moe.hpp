// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace gaudirt::runtime {

// This graph consumes the same packed weights as decode. Decoded expert tiles
// are graph temporaries and share the recipe workspace, not persistent weights.
inline void append_grouped_prefill_moe(synGraphHandle graph, uint32_t rows,
                                       synTensor activation, synTensor ids,
                                       synTensor routing, synTensor gate_weight,
                                       synTensor gate_scale, synTensor down_weight,
                                       synTensor down_scale, synTensor lut,
                                       synTensor output) {
  if (rows != 256 && rows != 512) throw std::invalid_argument("grouped MoE rows");
  auto check = [](synStatus status, const char* where) {
    if (status != synSuccess) throw std::runtime_error(std::string(where) + ": " + std::to_string(status));
  };
  auto tensor = [&](const std::string& name, synDataType type,
                    std::initializer_list<uint64_t> shape) {
    synTensorDescriptor d{};
    d.m_name = name.c_str(); d.m_dataType = type; d.m_dims = shape.size();
    unsigned i = 0;
    for (auto size : shape) { d.m_sizes[i] = d.m_minSizes[i] = size; ++i; }
    synTensor result = nullptr;
    check(synTensorCreate(&result, &d, nullptr, 0), name.c_str());
    return result;
  };
  auto node = [&](const std::string& name, const char* guid,
                  const std::vector<synTensor>& in, const std::vector<synTensor>& out,
                  const void* params = nullptr, unsigned bytes = 0) {
    check(synNodeCreate(graph, in.data(), out.data(), in.size(), out.size(),
                        params, bytes, guid, name.c_str(), nullptr, nullptr), name.c_str());
  };
  auto join = [&](const std::string& name, const std::vector<synTensor>& inputs,
                  synTensor result, unsigned axis) {
    if (inputs.size() == 1) node(name, "reshape", inputs, {result});
    else {
      synConcatenateParams p{axis};
      node(name, "concat", inputs, {result}, &p, sizeof(p));
    }
  };
  constexpr uint32_t experts = 384, routes = 8;
  const uint32_t length = rows * routes;
  auto flat = tensor("grouped_ids_flat", syn_type_int32, {length});
  node("grouped_flatten_ids", "reshape", {ids}, {flat});
  auto counts = tensor("grouped_counts", syn_type_int32, {experts});
  auto flags = tensor("grouped_flags", syn_type_int32, {rows});
  auto offsets = tensor("grouped_offsets", syn_type_int32, {length / 64 + 1, experts});
  const int32_t count_params[] = {experts};
  node("grouped_count", "gk_expert_count", {ids, flat}, {counts, flags, offsets}, count_params, sizeof(count_params));
  const uint32_t small = rows / 8;
  const std::array<uint32_t, 2> capacities{small, rows};
  const std::array<uint32_t, 2> slots{experts, std::min(experts, length / (small + 1))};
  std::array<std::array<synTensor, 3>, 2> outputs{};
  for (unsigned band = 0; band < 2; ++band) {
    const uint32_t c = capacities[band], nslots = slots[band];
    const std::string tag = "grouped_b" + std::to_string(band);
    auto prefix = tensor(tag + "_prefix", syn_type_int32, {experts + 1});
    auto selected = tensor(tag + "_experts", syn_type_int32, {nslots});
    auto base = tensor(tag + "_base", syn_type_int32, {nslots});
    auto valid = tensor(tag + "_valid", syn_type_int32, {nslots});
    auto status = tensor(tag + "_status", syn_type_int32, {1});
    const int32_t pp[] = {routes, static_cast<int32_t>(c), static_cast<int32_t>(nslots),
                          static_cast<int32_t>(band ? small + 1 : 1), static_cast<int32_t>(c), 0};
    node(tag + "_prefix_node", "gk_expert_prefix", {counts, flags}, {prefix, selected, base, valid, status}, pp, sizeof(pp));
    auto inverse = tensor(tag + "_inverse", syn_type_int32, {length});
    const int32_t ip[] = {static_cast<int32_t>(c), static_cast<int32_t>(nslots)};
    node(tag + "_inverse_node", "gk_expert_inverse", {ids, flat, prefix, status, offsets}, {inverse}, ip, sizeof(ip));
    auto mapping = tensor(tag + "_mapping", syn_type_int32, {nslots * c});
    const int32_t map_param = c;
    node(tag + "_map_node", "gk_expert_sparse_map", {inverse, valid, status}, {mapping}, &map_param, sizeof(map_param));
    auto expert_view = tensor(tag + "_expert_view", syn_type_int32, {nslots, 1});
    node(tag + "_expert_reshape", "reshape", {selected}, {expert_view});

    std::vector<synTensor> gates;
    for (uint32_t slot = 0; slot < nslots; slot += 8) {
      const uint32_t batch = std::min(8u, nslots - slot);
      const std::string part = tag + "_gp" + std::to_string(slot);
      auto gathered = tensor(part + "_input", syn_type_bf16, {6144, c, batch});
      const int32_t gp[] = {routes, static_cast<int32_t>(c), static_cast<int32_t>(slot), 1};
      node(part + "_gather", "gk_expert_counted_gather", {activation, mapping, valid, status}, {gathered}, gp, sizeof(gp));
      auto decoded = tensor(part + "_weight", syn_type_bf16, {512, 6144, batch});
      const int32_t dp[] = {1, static_cast<int32_t>(slot), 0, 1};
      node(part + "_decode", "gk_expert_decode_k8", {gate_weight, gate_scale, lut, expert_view}, {decoded}, dp, sizeof(dp));
      auto partial = tensor(part + "_partial", syn_type_float, {512, c, batch});
      const synGEMMParams mm{false, false};
      node(part + "_gemm", "batch_gemm", {gathered, decoded}, {partial}, &mm, sizeof(mm));
      auto gate = tensor(part + "_gate", syn_type_bf16, {256, c, batch});
      const int32_t gate_param = slot;
      node(part + "_silu", "gk_expert_gate", {partial, valid, status}, {gate}, &gate_param, sizeof(gate_param));
      gates.push_back(gate);
    }
    std::vector<synTensor> groups;
    for (size_t i = 0; i < gates.size(); i += 2) {
      const uint32_t slot = i * 8, batch = std::min(16u, nslots - slot);
      auto group = tensor(tag + "_group" + std::to_string(slot), syn_type_bf16, {256, c, batch});
      if (i + 1 < gates.size()) join(tag + "_gate_join" + std::to_string(slot), {gates[i], gates[i + 1]}, group, 2);
      else node(tag + "_gate_view" + std::to_string(slot), "reshape", {gates[i]}, {group});
      groups.push_back(group);
    }
    for (uint32_t tile = 0; tile < 3; ++tile) {
      std::vector<synTensor> parts;
      for (size_t gi = 0; gi < groups.size(); ++gi) {
        const uint32_t slot = gi * 16, batch = std::min(16u, nslots - slot);
        const std::string part = tag + "_down" + std::to_string(tile) + "_" + std::to_string(slot);
        auto decoded = tensor(part + "_weight", syn_type_bf16, {2048, 256, batch});
        const int32_t dp[] = {12, static_cast<int32_t>(slot), static_cast<int32_t>(tile * 4), 1};
        node(part + "_decode", "gk_expert_decode_k8", {down_weight, down_scale, lut, expert_view}, {decoded}, dp, sizeof(dp));
        auto product = tensor(part + "_product", syn_type_float, {2048, c, batch});
        const synGEMMParams mm{false, false};
        node(part + "_gemm", "batch_gemm", {groups[gi], decoded}, {product}, &mm, sizeof(mm));
        auto view = tensor(part + "_view", syn_type_float, {2048, c * batch});
        node(part + "_reshape", "reshape", {product}, {view});
        parts.push_back(view);
      }
      auto partial = tensor(tag + "_partial" + std::to_string(tile), syn_type_float, {2048, c * nslots});
      join(tag + "_join" + std::to_string(tile), parts, partial, 1);
      outputs[band][tile] = tensor(tag + "_output" + std::to_string(tile), syn_type_float, {2048, rows});
      node(tag + "_combine" + std::to_string(tile), "gk_expert_combine",
           {partial, routing, inverse, status}, {outputs[band][tile]});
    }
  }
  std::vector<synTensor> tiles;
  for (unsigned tile = 0; tile < 3; ++tile) {
    auto sum = tensor("grouped_sum" + std::to_string(tile), syn_type_float, {2048, rows});
    node("grouped_add" + std::to_string(tile), "add_fwd_f32", {outputs[0][tile], outputs[1][tile]}, {sum});
    tiles.push_back(sum);
  }
  join("grouped_output", tiles, output, 0);
}

}  // namespace gaudirt::runtime
