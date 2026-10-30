// SPDX-License-Identifier: Apache-2.0
// Exercise the actual SWA graph node at FP32 exponential range boundaries.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace {
void check(synStatus status) {
  if (status != synSuccess) throw std::runtime_error("SWA softmax test: Synapse failure");
}
uint16_t narrow_exact(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0xffffU) != 0) throw std::runtime_error("test value is not exact BF16");
  return static_cast<uint16_t>(bits >> 16);
}
float widen(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
}  // namespace

int main() {
  using namespace gaudirt::runtime;
  DeviceRuntime runtime(DeviceRuntime::Config{});
  HbmArena arena(runtime.device_id(), 64ULL * 1024 * 1024);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synSectionHandle, 8> sections{};
  std::array<synTensor, 8> tensors{};
  std::array<synLaunchTensorInfo, 8> bindings{};
  const char* names[] = {"query", "key", "value", "pages", "groups", "position", "sinks", "output"};
  const synDataType types[] = {syn_type_bf16, syn_type_bf16, syn_type_bf16,
      syn_type_int32, syn_type_int32, syn_type_int32, syn_type_bf16, syn_type_bf16};
  const unsigned dims[] = {3, 3, 3, 1, 1, 1, 1, 3};
  const uint64_t sizes[][3] = {{192,16,1}, {192,1,128}, {128,1,128},
      {1,1,1}, {1,1,1}, {1,1,1}, {16,1,1}, {128,16,1}};
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < tensors.size(); ++i) {
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i]; desc.m_dataType = types[i]; desc.m_dims = dims[i];
    for (unsigned j = 0; j < dims[i]; ++j) {
      desc.m_sizes[j] = desc.m_minSizes[j] = sizes[i][j];
      bindings[i].tensorSize[j] = sizes[i][j];
    }
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    bindings[i].tensorName = names[i]; bindings[i].tensorType = DATA_TENSOR;
  }
  const float scale = std::log(2.0f);
  check(synNodeCreate(graph, tensors.data(), tensors.data() + 7, 7, 1, &scale,
                      sizeof(scale), "gk_swa128_avgroup_group_v0", "swa_boundary", nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, "swa_softmax_test", nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, bindings.data(), bindings.size());
  check(synGraphDestroy(graph));
  for (auto section : sections) check(synSectionDestroy(section));

  std::array<uint16_t, 192 * 16> query{};
  std::array<uint16_t, 192 * 128> key{};
  std::array<uint16_t, 128 * 128> value{};
  std::array<uint16_t, 16> sinks{};
  std::array<uint16_t, 128 * 16> output{};
  for (int head = 0; head < 16; ++head) query[head * 192] = narrow_exact(static_cast<float>(head + 1));
  key[192] = narrow_exact(-1.0f);
  std::fill(value.begin() + 128, value.begin() + 256, narrow_exact(1.0f));
  sinks.fill(narrow_exact(-100.0f));
  const int32_t page = 0, group = 0, position = 1;
  const void* input[] = {query.data(), key.data(), value.data(), &page, &group, &position, sinks.data()};
  const size_t bytes[] = {sizeof(query), sizeof(key), sizeof(value), sizeof(page), sizeof(group),
                          sizeof(position), sizeof(sinks), sizeof(output)};
  for (size_t i = 0; i < tensors.size(); ++i) {
    bindings[i].pTensorAddress = arena.allocate(bytes[i]);
    if (i < 7) arena.copy_h2d_sync(runtime.h2d_stream(), input[i], bindings[i].pTensorAddress, bytes[i]);
  }
  PreparedLaunch launch{recipe, arena.ensure_workspace(workspace), {bindings.begin(), bindings.end()}, 0};
  launch.validate(); launch.execute(runtime.compute_stream()); runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(), bindings[7].pTensorAddress, output.data(), sizeof(output));
  double max_error = 0;
  int worst_head = -1;
  for (int head = 0; head < 16; ++head) {
    const float score = static_cast<float>(head + 1) * scale;
    const double expected = 1.0 / (1.0 + std::exp(static_cast<double>(score)));
    for (int channel = 0; channel < 128; ++channel) {
      const double error = std::abs(widen(output[head * 128 + channel]) - expected);
      if (error > max_error) { max_error = error; worst_head = head; }
    }
  }
  std::printf("SWA softmax max_abs=%.9g worst_head=%d\n", max_error, worst_head);
  std::fflush(stdout);
  check(synRecipeDestroy(recipe));
  if (!(max_error <= 0.002)) throw std::runtime_error("SWA softmax exceeds mathematical tolerance");
}
