// SPDX-License-Identifier: Apache-2.0
// Verify a 32-row causal SWA graph and its physical KV writes on an empty cache.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <perf_lib_layer_params.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned kRows = 32, kSlots = 256;
void check(synStatus status) {
  if (status != synSuccess) throw std::runtime_error("prefill SWA graph: Synapse failure " + std::to_string(status));
}
uint16_t bf16(float value) {
  uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>(bits >> 16);
}
float widen(uint16_t value) {
  uint32_t bits = uint32_t{value} << 16;
  float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}
synTensor temp(const char* name, synDataType type, std::initializer_list<uint64_t> shape) {
  synTensorDescriptor d{}; d.m_name = name; d.m_dataType = type; d.m_dims = shape.size();
  unsigned i = 0; for (uint64_t n : shape) { d.m_sizes[i] = d.m_minSizes[i] = n; ++i; }
  synTensor tensor = nullptr; check(synTensorCreate(&tensor, &d, nullptr, 0)); return tensor;
}
void node(synGraphHandle graph, std::initializer_list<synTensor> inputs,
          std::initializer_list<synTensor> outputs, const void* params, unsigned size,
          const char* guid, const char* name) {
  check(synNodeCreate(graph, inputs.begin(), outputs.begin(), inputs.size(), outputs.size(),
                      params, size, guid, name, nullptr, nullptr));
}
}

int main() {
  using namespace gaudirt::runtime;
  DeviceRuntime runtime(DeviceRuntime::Config{});
  HbmArena arena(runtime.device_id(), 128ULL * 1024 * 1024);
  synGraphHandle graph = nullptr; synRecipeHandle recipe = nullptr;
  constexpr unsigned n = 13;
  std::array<synSectionHandle, n> sections{};
  std::array<synTensor, n> tensors{};
  std::array<synLaunchTensorInfo, n> binding{};
  const char* names[] = {"qkv", "cos", "sin", "key", "value", "slot", "pages", "groups",
                         "position", "sink", "key_after", "value_after", "context"};
  const synDataType types[] = {syn_type_bf16, syn_type_bf16, syn_type_bf16, syn_type_bf16,
      syn_type_bf16, syn_type_int32, syn_type_int32, syn_type_int32, syn_type_int32,
      syn_type_bf16, syn_type_bf16, syn_type_bf16, syn_type_bf16};
  const unsigned dims[] = {2,3,3,3,3,1,1,1,1,1,3,3,3};
  const uint64_t shapes[][3] = {{3392,kRows,1},{64,1,kRows},{64,1,kRows},
      {192,1,kSlots},{128,1,kSlots},{kRows,1,1},{kRows,1,1},{kRows,1,1},
      {kRows,1,1},{16,1,1},{192,1,kSlots},{128,1,kSlots},{128,16,kRows}};
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (unsigned i = 0; i < n; ++i) {
    synSectionHandle section = nullptr;
    if (i == 10) section = sections[3];
    else if (i == 11) section = sections[4];
    else { check(synSectionCreate(&section, 0, graph)); check(synSectionSetPersistent(section, true)); sections[i] = section; }
    synTensorDescriptor d{}; d.m_name = names[i]; d.m_dataType = types[i]; d.m_dims = dims[i];
    for (unsigned j = 0; j < dims[i]; ++j) { d.m_sizes[j] = d.m_minSizes[j] = shapes[i][j]; binding[i].tensorSize[j] = shapes[i][j]; }
    check(synTensorCreate(&tensors[i], &d, section, 0));
    binding[i].tensorName = names[i]; binding[i].tensorType = DATA_TENSOR;
  }
  auto query = temp("query", syn_type_bf16, {192,16,kRows});
  auto flat_query = temp("flat_query", syn_type_bf16, {3072,1,kRows});
  auto new_key = temp("new_key", syn_type_bf16, {192,1,kRows});
  auto new_value = temp("new_value", syn_type_bf16, {128,1,kRows});
  const float value_scale = 0.612f;
  node(graph, {tensors[0],tensors[1],tensors[2]}, {query,new_key,new_value}, &value_scale,
       sizeof(value_scale), "gk_qkv_post_f32_f32_v1", "split_rope_value");
  ns_IndexCopy::Params copy{}; copy.axis = 0;
  node(graph, {tensors[3],tensors[5],new_key}, {tensors[10]}, &copy, sizeof(copy), "index_copy_fwd_bf16", "write_keys");
  node(graph, {tensors[4],tensors[5],new_value}, {tensors[11]}, &copy, sizeof(copy), "index_copy_fwd_bf16", "write_values");
  node(graph, {query}, {flat_query}, nullptr, 0, "reshape", "flatten_query");
  const float scale = 1.0f / std::sqrt(192.0f);
  node(graph, {flat_query,tensors[10],tensors[11],tensors[6],tensors[7],tensors[8],tensors[9]},
       {tensors[12]}, &scale, sizeof(scale), "gk_swa128_batch_window_quad_fp32_v2", "causal_swa");
  check(synGraphCompile(&recipe, graph, "mimo_swa_prefill_32_test", nullptr));
  uint64_t workspace = 0; check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, binding.data(), binding.size());
  check(synGraphDestroy(graph)); for (auto s : sections) if (s) check(synSectionDestroy(s));

  std::vector<uint16_t> qkv(3392 * kRows), cosine(64 * kRows, bf16(1)), sine(64 * kRows);
  std::vector<uint16_t> key(192 * kSlots, bf16(-7)), value(128 * kSlots, bf16(-7));
  std::vector<uint16_t> result(128 * 16 * kRows), sink(16, bf16(0));
  std::array<int32_t, kRows> slots{}, pages{}, groups{}, positions{};
  for (unsigned row = 0; row < kRows; ++row) {
    const auto input = bf16(1.0f + 0.25f * (row % 4));
    std::fill_n(qkv.begin() + row * 3392 + 3264, 128, input);
    slots[row] = 128 + row; pages[row] = 1; groups[row] = row; positions[row] = row;
  }
  const void* input[] = {qkv.data(),cosine.data(),sine.data(),key.data(),value.data(),slots.data(),
                         pages.data(),groups.data(),positions.data(),sink.data()};
  const size_t bytes[] = {qkv.size()*2,cosine.size()*2,sine.size()*2,key.size()*2,value.size()*2,
      sizeof(slots),sizeof(pages),sizeof(groups),sizeof(positions),sink.size()*2,key.size()*2,value.size()*2,result.size()*2};
  for (unsigned i = 0; i < n; ++i) {
    if (i == 10) binding[i].pTensorAddress = binding[3].pTensorAddress;
    else if (i == 11) binding[i].pTensorAddress = binding[4].pTensorAddress;
    else binding[i].pTensorAddress = arena.allocate(bytes[i]);
    if (i < 10) arena.copy_h2d_sync(runtime.h2d_stream(), input[i], binding[i].pTensorAddress, bytes[i]);
  }
  PreparedLaunch launch{recipe,arena.ensure_workspace(workspace),{binding.begin(),binding.end()},0};
  launch.validate(); launch.execute(runtime.compute_stream()); runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(), binding[3].pTensorAddress, key.data(), bytes[3]);
  arena.copy_d2h_sync(runtime.d2h_stream(), binding[4].pTensorAddress, value.data(), bytes[4]);
  arena.copy_d2h_sync(runtime.d2h_stream(), binding[12].pTensorAddress, result.data(), bytes[12]);
  double worst = 0, sum = 0;
  for (unsigned row = 0; row < kRows; ++row) {
    const uint16_t expected_v = bf16((1.0f + 0.25f * (row % 4)) * value_scale);
    sum += widen(expected_v);
    const double expected = sum / double(row + 2); // visible tokens plus zero-valued sink
    for (unsigned i = 0; i < 128 * 16; ++i) {
      const double actual = widen(result[row * 128 * 16 + i]);
      if (!std::isfinite(actual)) throw std::runtime_error("non-finite SWA output");
      worst = std::max(worst, std::abs(actual - expected));
    }
  }
  for (unsigned slot = 0; slot < kSlots; ++slot) {
    const bool touched = slot >= 128 && slot < 128 + kRows;
    const uint16_t expected_k = touched ? bf16(0) : bf16(-7);
    const uint16_t expected_v = touched ? bf16((1.0f + 0.25f * ((slot-128) % 4)) * value_scale) : bf16(-7);
    for (unsigned i = 0; i < 192; ++i) if (key[slot*192+i] != expected_k) throw std::runtime_error("key cache slot mismatch");
    for (unsigned i = 0; i < 128; ++i) if (value[slot*128+i] != expected_v) throw std::runtime_error("value cache slot mismatch");
  }
  std::printf("M=32 causal SWA max_abs=%.9g, untouched KV bitwise equal\n",worst);
  check(synRecipeDestroy(recipe));
  if (worst > 0.004) throw std::runtime_error("causal SWA exceeds BF16 tolerance");
}
