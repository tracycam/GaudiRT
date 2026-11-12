// Verify the single-pass causal softmax, including masked and padded rows.
#include "gaudirt/runtime/device_runtime.hpp"
#include "test_device_slab.hpp"
#include "gaudirt/runtime/recipe_cache.hpp"

#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gaudirt::runtime;

namespace {
void check(synStatus status, const char* stage) {
  if (status != synSuccess)
    throw std::runtime_error(std::string(stage) + " status " + std::to_string(static_cast<int>(status)));
}

float widen(uint16_t value) { return std::bit_cast<float>(static_cast<uint32_t>(value) << 16); }

struct Graph {
  explicit Graph(std::string name) : prefix(std::move(name)) {
    names.reserve(64);
    tensors.reserve(64);
    tensor_sections.reserve(64);
    bindings.reserve(24);
    check(synGraphCreate(&graph, synDeviceGaudi2), "graph creation");
  }
  ~Graph() {
    if (recipe) synRecipeDestroy(recipe);
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) synTensorDestroy(tensor);
    for (auto section : sections) synSectionDestroy(section);
  }
  size_t tensor(std::vector<uint64_t> dims, synDataType type, uint64_t address = 0,
                int shared_section_tensor = -1, uint64_t offset = 0) {
    const auto id = tensors.size();
    names.push_back(prefix + "_t" + std::to_string(id));
    synSectionHandle section = nullptr;
    if (address) {
      if (shared_section_tensor >= 0) {
        section = tensor_sections.at(shared_section_tensor);
        if (!section) throw std::runtime_error("cannot share a nonpersistent section");
      } else {
        check(synSectionCreate(&section, 0, graph), "section creation");
        sections.push_back(section);
        check(synSectionSetPersistent(section, true), "section persistence");
      }
    }
    synTensorDescriptor desc{};
    desc.m_name = names.back().c_str();
    desc.m_dataType = type;
    desc.m_tensorType = DATA_TENSOR;
    desc.m_dims = dims.size();
    for (size_t i = 0; i < dims.size(); ++i) desc.m_sizes[i] = desc.m_minSizes[i] = dims[i];
    synTensor handle = nullptr;
    check(synTensorCreate(&handle, &desc, section, offset), "tensor creation");
    tensors.push_back(handle);
    tensor_sections.push_back(section);
    if (address) {
      synLaunchTensorInfo binding{};
      binding.tensorName = names.back().c_str();
      binding.tensorType = DATA_TENSOR;
      binding.pTensorAddress = address;
      for (size_t i = 0; i < dims.size(); ++i) binding.tensorSize[i] = dims[i];
      bindings.push_back(binding);
    }
    return id;
  }
  void node(const char* guid, std::initializer_list<size_t> inputs, std::initializer_list<size_t> outputs,
            const void* params = nullptr, size_t size = 0) {
    std::vector<synTensor> in, out;
    for (auto id : inputs) in.push_back(tensors.at(id));
    for (auto id : outputs) out.push_back(tensors.at(id));
    const auto name = prefix + "_n" + std::to_string(node_count++);
    check(synNodeCreate(graph, in.data(), out.data(), in.size(), out.size(), const_cast<void*>(params), size,
                        guid, name.c_str(), nullptr, nullptr), "node creation");
  }
  void compile(RecipeCache& cache) {
    recipe = compile_cached(graph, prefix, prefix, &cache);
    check(synWorkspaceGetSize(&workspace, recipe), "workspace query");
    initialize_launch_tensor_ids(recipe, bindings.data(), bindings.size());
  }
  std::string prefix;
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  uint64_t workspace = 0;
  size_t node_count = 0;
  std::vector<std::string> names;
  std::vector<synTensor> tensors;
  std::vector<synSectionHandle> tensor_sections, sections;
  std::vector<synLaunchTensorInfo> bindings;
};

}  // namespace


int main(int argc, char** argv) {
  try {
    if (argc != 6) throw std::invalid_argument("expected cache, keys, rows, visible keys and valid rows");
    const auto keys = std::stoul(argv[2]), rows = std::stoul(argv[3]);
    const auto visible_keys = std::stoul(argv[4]), valid_rows = std::stoul(argv[5]);
    constexpr uint32_t heads = 4;
    if (keys < 128 || keys > 512 || keys % 128 || !rows || rows > 256 ||
        !visible_keys || visible_keys > keys || !valid_rows || valid_rows > rows)
      throw std::invalid_argument("unsupported softmax geometry");
    std::vector<float> scores(keys * rows * heads);
    for (size_t i = 0; i < scores.size(); ++i) scores[i] = float(int(i % 17) - 8) * 0.125f;
    std::vector<int32_t> positions(rows);
    for (size_t i = 0; i < rows; ++i)
      positions[i] = std::max<int>(0, int(visible_keys) - int(rows) + int(i));
    const std::array<int32_t, 3> header{static_cast<int32_t>(valid_rows),
                                        static_cast<int32_t>(visible_keys), 0};
    std::vector<uint16_t> output(scores.size(), 0x7fc0);
    DeviceRuntime runtime({});
    gaudirt::testing::TestDeviceSlab arena(runtime.device_id(), 32ULL * 1024 * 1024);
    RecipeCache cache(argv[1]);
    const auto scores_addr = arena.allocate(scores.size() * 4);
    const auto positions_addr = arena.allocate(positions.size() * 4);
    const auto header_addr = arena.allocate(sizeof(header));
    const auto output_addr = arena.allocate(output.size() * 2);
    const auto h2d = [&](const void* src, uint64_t dst, size_t bytes) {
      arena.copy_h2d_sync(runtime.h2d_stream(), src, dst, bytes);
    };
    h2d(scores.data(), scores_addr, scores.size() * 4);
    h2d(positions.data(), positions_addr, positions.size() * 4);
    h2d(header.data(), header_addr, sizeof(header));
    h2d(output.data(), output_addr, output.size() * 2);
    Graph graph("direct_softmax_k" + std::to_string(keys) + "_r" + std::to_string(rows));
    const auto ts = graph.tensor({keys, rows, heads}, syn_type_float, scores_addr);
    const auto tp = graph.tensor({rows}, syn_type_int32, positions_addr);
    const auto th = graph.tensor({3}, syn_type_int32, header_addr);
    const auto to = graph.tensor({keys, rows, heads}, syn_type_bf16, output_addr);
    const float scale = 1.f;
    graph.node("gk_attn_direct_softmax_bf16_v1", {ts, tp, th}, {to}, &scale, sizeof(scale));
    graph.compile(cache);
    check(synLaunch(runtime.compute_stream(), graph.bindings.data(), graph.bindings.size(),
                    arena.ensure_workspace(graph.workspace), graph.recipe, 0), "launch");
    runtime.synchronize_compute();
    arena.copy_d2h_sync(runtime.d2h_stream(), output_addr, output.data(), output.size() * 2);
    double max_error = 0;
    for (uint32_t head = 0; head < heads; ++head) {
      for (uint32_t row = 0; row < rows; ++row) {
        const auto offset = (head * rows + row) * keys;
        const size_t visible = row < valid_rows ? static_cast<size_t>(positions[row]) + 1 : 0;
        double denominator = 0;
        for (size_t k = 0; k < visible; ++k) denominator += std::exp(double(scores[offset + k]));
        for (size_t k = 0; k < keys; ++k) {
          const float actual = widen(output[offset + k]);
          if (!std::isfinite(actual)) throw std::runtime_error("nonfinite softmax output");
          const double expected = k < visible ? std::exp(double(scores[offset + k])) / denominator : 0.;
          if (k >= visible && actual != 0.f) throw std::runtime_error("masked softmax output is nonzero");
          max_error = std::max(max_error, std::abs(actual - expected));
        }
      }
    }
    if (max_error > 0.0005) throw std::runtime_error("softmax differs from FP64 reference");
    const auto counts = cache.stats();
    std::cout << "keys " << keys << " rows " << rows << " valid_rows " << valid_rows
              << " visible_keys " << visible_keys << " max_abs " << max_error
              << " cache " << counts.first << '/' << counts.second << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
