// SPDX-License-Identifier: Apache-2.0
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
#include <limits>
#include <stdexcept>

namespace {
void check(synStatus status) {
  if (status != synSuccess) throw std::runtime_error("softmax test: Synapse failure");
}
float widen(uint16_t value) {
  const uint32_t raw = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &raw, sizeof(result));
  return result;
}
}  // namespace

int main() {
  using namespace gaudirt::runtime;
  DeviceRuntime runtime(DeviceRuntime::Config{});
  HbmArena arena(runtime.device_id(), 64ULL * 1024 * 1024);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synSectionHandle, 6> sections{};
  std::array<synTensor, 6> tensors{};
  std::array<synLaunchTensorInfo, 6> bindings{};
  const char* names[] = {"scores", "pages", "usage", "groups", "high", "low"};
  const synDataType types[] = {syn_type_float, syn_type_int32, syn_type_int32,
                               syn_type_int32, syn_type_bf16, syn_type_bf16};
  const uint64_t widths[] = {128, 1, 1, 1, 128, 128};
  const uint64_t rows[] = {16, 1, 1, 1, 16, 16};
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (size_t i = 0; i < tensors.size(); ++i) {
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = names[i]; desc.m_dataType = types[i];
    desc.m_dims = (i == 1 || i == 2 || i == 3) ? 1 : 2;
    desc.m_sizes[0] = desc.m_minSizes[0] = widths[i];
    if (desc.m_dims == 2) desc.m_sizes[1] = desc.m_minSizes[1] = rows[i];
    check(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    bindings[i].tensorName = names[i]; bindings[i].tensorType = DATA_TENSOR;
    bindings[i].tensorSize[0] = widths[i];
    if (desc.m_dims == 2) bindings[i].tensorSize[1] = rows[i];
  }
  struct Params { float scale; int32_t physical_pages; } params{1.0f, 1};
  check(synNodeCreate(graph, tensors.data(), tensors.data() + 4, 4, 2, &params,
                      sizeof(params), "gk_full_mme_softmax_v2", "test_softmax", nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, "full_mme_softmax_test", nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, bindings.data(), bindings.size());
  check(synGraphDestroy(graph));
  for (auto section : sections) check(synSectionDestroy(section));

  std::array<float, 128 * 16> scores{};
  const float ln2 = std::log(2.0f);
  for (int head = 0; head < 16; ++head) {
    float value = static_cast<float>(1 + head / 2) * ln2;
    if (head % 4 == 2) value = std::nextafter(value, -std::numeric_limits<float>::infinity());
    if (head % 4 == 3) value = std::nextafter(value, std::numeric_limits<float>::infinity());
    scores[head * 128 + 1] = -value;
  }
  const std::array<int32_t, 1> pages{0}, usage{2}, groups{0};
  std::array<uint16_t, 128 * 16> high{}, low{};
  const void* input[] = {scores.data(), pages.data(), usage.data(), groups.data()};
  const size_t bytes[] = {sizeof(scores), sizeof(pages), sizeof(usage), sizeof(groups),
                          sizeof(high), sizeof(low)};
  for (size_t i = 0; i < tensors.size(); ++i) {
    bindings[i].pTensorAddress = arena.allocate(bytes[i]);
    if (i < 4)
      arena.copy_h2d_sync(runtime.h2d_stream(), input[i], bindings[i].pTensorAddress, bytes[i]);
  }
  PreparedLaunch launch{recipe, arena.ensure_workspace(workspace),
                        {bindings.begin(), bindings.end()}, 0};
  launch.validate(); launch.execute(runtime.compute_stream()); runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(), bindings[4].pTensorAddress, high.data(), sizeof(high));
  arena.copy_d2h_sync(runtime.d2h_stream(), bindings[5].pTensorAddress, low.data(), sizeof(low));
  double max_error = 0;
  for (int head = 0; head < 16; ++head) {
    const double p1 = 1.0 / (1.0 + std::exp(-static_cast<double>(scores[head * 128 + 1])));
    const double expected[] = {1.0 - p1, p1};
    for (int token = 0; token < 128; ++token) {
      const int index = head * 128 + token;
      const double actual = widen(high[index]) + widen(low[index]);
      const double target = token < 2 ? expected[token] : 0.0;
      max_error = std::max(max_error, std::abs(actual - target));
    }
  }
  std::printf("Full MME softmax max_abs=%.9g\n", max_error);
  check(synRecipeDestroy(recipe));
  if (!(max_error <= 5e-5)) throw std::runtime_error("softmax exceeds mathematical tolerance");
}
