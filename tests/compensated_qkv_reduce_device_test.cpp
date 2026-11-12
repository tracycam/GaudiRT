// SPDX-License-Identifier: Apache-2.0
// Exercise BF16 midpoint rounding of a compensated reduction on Gaudi2.
#include "gaudirt/runtime/device_runtime.hpp"
#include "test_device_slab.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(synStatus status) {
  if (status != synSuccess) throw std::runtime_error("compensated QKV: Synapse failure");
}
float widen(uint16_t value) {
  const uint32_t bits = uint32_t{value} << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
}

int main(int argc, char** argv) {
  using namespace gaudirt::runtime;
  if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--include-subnormal"))
    throw std::invalid_argument("expected optional --include-subnormal");
  const bool include_subnormal = argc == 2;
  constexpr unsigned columns = 1024, rows = 32, groups = 2;
  DeviceRuntime runtime(DeviceRuntime::Config{});
  gaudirt::testing::TestDeviceSlab arena(runtime.device_id(), 64ULL * 1024 * 1024);
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  std::array<synSectionHandle, 5> sections{};
  std::array<synTensor, 5> tensors{};
  std::array<synLaunchTensorInfo, 5> bindings{};
  const char* names[] = {"partial", "activation_scales", "weight_scales", "bias", "output"};
  const synDataType types[] = {syn_type_float, syn_type_float, syn_type_float,
                               syn_type_float, syn_type_bf16};
  const unsigned dims[] = {3, 3, 2, 1, 2};
  const uint64_t sizes[][3] = {{columns, rows, groups}, {1, rows, groups},
                               {groups, columns / 128, 1}, {columns, 1, 1},
                               {columns, rows, 1}};
  check(synGraphCreate(&graph, synDeviceGaudi2));
  for (unsigned i = 0; i < tensors.size(); ++i) {
    check(synSectionCreate(&sections[i], 0, graph));
    check(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor descriptor{};
    descriptor.m_name = names[i];
    descriptor.m_dataType = types[i];
    descriptor.m_dims = dims[i];
    for (unsigned d = 0; d < dims[i]; ++d) {
      descriptor.m_sizes[d] = descriptor.m_minSizes[d] = sizes[i][d];
      bindings[i].tensorSize[d] = sizes[i][d];
    }
    check(synTensorCreate(&tensors[i], &descriptor, sections[i], 0));
    bindings[i].tensorName = names[i];
    bindings[i].tensorType = DATA_TENSOR;
  }
  check(synNodeCreate(graph, tensors.data(), tensors.data() + 4, 4, 1, nullptr, 0,
                      "gk_block128_neumaier_bf16_compensated_v1", "reduce", nullptr, nullptr));
  check(synGraphCompile(&recipe, graph, "compensated_qkv_midpoint_test", nullptr));
  uint64_t workspace = 0;
  check(synWorkspaceGetSize(&workspace, recipe));
  initialize_launch_tensor_ids(recipe, bindings.data(), bindings.size());
  check(synGraphDestroy(graph));
  for (auto section : sections) check(synSectionDestroy(section));

  std::vector<float> partial(columns * rows * groups), activation(rows * groups, 1.f);
  std::vector<float> weights(groups * columns / 128, 1.f), bias(columns);
  std::vector<uint16_t> output(columns * rows), expected(columns * rows);
  for (unsigned col = 0; col < columns; ++col) {
    // Keep the quarter-ULP residual normal in FP32; subnormal TPC behavior
    // needs a separate numerical contract and is not covered by this test.
    const unsigned first = include_subnormal ? 0x0200 : 0x0d00;
    const uint16_t lower = static_cast<uint16_t>(first + (col * 29) % (0x7f7f - first));
    const float lo = widen(lower), hi = widen(static_cast<uint16_t>(lower + 1));
    const bool negative = col % 2 != 0;
    const float sign = negative ? -1.f : 1.f;
    const float middle = static_cast<float>((double(lo) + double(hi)) * 0.5) * sign;
    const float ulp = std::nextafter(std::abs(middle), std::numeric_limits<float>::infinity()) -
                      std::abs(middle);
    bias[col] = col % 3 ? middle * 0.125f : 0.f;
    for (unsigned row = 0; row < rows; ++row) {
      const int direction = static_cast<int>(row % 3) - 1;
      const size_t index = static_cast<size_t>(row) * columns + col;
      partial[index] = middle - bias[col];
      partial[columns * rows + index] = sign * direction * ulp * 0.25f;
      const uint16_t magnitude = static_cast<uint16_t>(
          lower + (direction > 0 || (direction == 0 && (lower & 1))));
      expected[index] = static_cast<uint16_t>(magnitude | (negative ? 0x8000 : 0));
    }
  }
  const void* inputs[] = {partial.data(), activation.data(), weights.data(), bias.data()};
  const size_t bytes[] = {partial.size() * 4, activation.size() * 4, weights.size() * 4,
                          bias.size() * 4, output.size() * 2};
  for (unsigned i = 0; i < bindings.size(); ++i) {
    bindings[i].pTensorAddress = arena.allocate(bytes[i]);
    if (i < 4) arena.copy_h2d_sync(runtime.h2d_stream(), inputs[i], bindings[i].pTensorAddress, bytes[i]);
  }
  check(synLaunch(runtime.compute_stream(), bindings.data(), bindings.size(),
                  arena.ensure_workspace(workspace), recipe, 0));
  runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(), bindings[4].pTensorAddress, output.data(), bytes[4]);
  check(synRecipeDestroy(recipe));
  for (size_t i = 0; i < output.size(); ++i)
    if (output[i] != expected[i]) {
      std::fprintf(stderr, "midpoint mismatch at row=%zu col=%zu: got=%04x expected=%04x\n",
                   i / columns, i % columns, output[i], expected[i]);
      return 1;
    }
  std::printf("%zu compensated BF16 midpoint decisions match exact expectations\n", output.size());
}
