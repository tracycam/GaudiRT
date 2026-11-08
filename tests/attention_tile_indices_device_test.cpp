// Check dynamic request maps and complete vector writes by the tile-index kernel.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gaudirt::runtime;

namespace {
void check(synStatus status, const char* what) {
  if (status != synSuccess) throw std::runtime_error(std::string(what) + ": " + std::to_string(status));
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 6) throw std::invalid_argument("expected cache mode requests context slots");
    const std::string mode = argv[2];
    const int requests = std::stoi(argv[3]), context = std::stoi(argv[4]), slots = std::stoi(argv[5]);
    if ((mode != "cold" && mode != "warm") || requests < 2 || requests > 256 ||
        context < 73 || context > 131072 || slots < 2 || slots > 131072)
      throw std::invalid_argument("unsupported metadata geometry");
    constexpr int tasks = 8, max_mapping = 256 * 131072;
    std::vector<int32_t> map(static_cast<size_t>(requests) * context);
    for (int r = 0; r < requests; ++r)
      for (int i = 0; i < context; ++i) map[static_cast<size_t>(r) * context + i] = 1 + (r * 11 + i * 7) % (slots - 1);
    const std::array<int32_t, 2> row_requests{0, requests - 1};
    std::array<int32_t, tasks * 4> work{};
    for (int t = 0; t < tasks; ++t) {
      work[t * 4] = t % 2;
      work[t * 4 + 1] = (t % 2) * 64;
      work[t * 4 + 2] = t % 2 ? 9 : 64;
      work[t * 4 + 3] = (t / 2) * slots;
    }
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 32ULL * 1024 * 1024);
    RecipeCache cache(argv[1]);
    const std::array<uint64_t, 5> addresses{arena.allocate(map.size() * 4), arena.allocate(sizeof(row_requests)),
        arena.allocate(sizeof(work)), arena.allocate(4), arena.allocate(tasks * 64 * 4)};
    arena.copy_h2d_sync(runtime.h2d_stream(), map.data(), addresses[0], map.size() * 4);
    arena.copy_h2d_sync(runtime.h2d_stream(), row_requests.data(), addresses[1], sizeof(row_requests));
    arena.copy_h2d_sync(runtime.h2d_stream(), work.data(), addresses[2], sizeof(work));
    arena.copy_h2d_sync(runtime.h2d_stream(), &context, addresses[3], 4);
    std::vector<int32_t> output(tasks * 64, -123);
    arena.copy_h2d_sync(runtime.h2d_stream(), output.data(), addresses[4], output.size() * 4);

    synGraphHandle graph = nullptr;
    std::array<synSectionHandle, 5> sections{};
    std::array<synTensor, 5> tensors{};
    std::array<synLaunchTensorInfo, 5> bindings{};
    std::array<std::string, 5> names{"tile_indices_t0", "tile_indices_t1", "tile_indices_t2",
                                     "tile_indices_t3", "tile_indices_t4"};
    const std::array<std::array<uint64_t, 2>, 5> dims{{{max_mapping, 0}, {2, 0}, {4, tasks}, {1, 0}, {64, tasks}}};
    check(synGraphCreate(&graph, synDeviceGaudi2), "graph");
    for (unsigned i = 0; i < tensors.size(); ++i) {
      check(synSectionCreate(&sections[i], 0, graph), "section");
      check(synSectionSetPersistent(sections[i], true), "persistent");
      synTensorDescriptor desc{};
      desc.m_name = names[i].c_str(); desc.m_dataType = syn_type_int32;
      desc.m_tensorType = i == 0 ? DATA_TENSOR_DYNAMIC : DATA_TENSOR;
      desc.m_dims = dims[i][1] ? 2 : 1;
      for (unsigned d = 0; d < desc.m_dims; ++d) {
        desc.m_sizes[d] = dims[i][d]; desc.m_minSizes[d] = i == 0 ? 1 : dims[i][d];
        bindings[i].tensorSize[d] = i == 0 ? map.size() : dims[i][d];
      }
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "tensor");
      bindings[i].tensorName = names[i].c_str(); bindings[i].tensorType = desc.m_tensorType;
      bindings[i].pTensorAddress = addresses[i];
    }
    check(synNodeCreate(graph, tensors.data(), &tensors[4], 4, 1, nullptr, 0,
                        "gk_attn_tile_indices_i32_v1", "tile_indices_n0", nullptr, nullptr), "node");
    auto recipe = compile_cached(graph, "tile_indices", "tile_indices_v1", &cache);
    const auto counts = cache.stats();
    if ((mode == "cold" && counts != std::pair<uint64_t, uint64_t>{0, 1}) ||
        (mode == "warm" && counts != std::pair<uint64_t, uint64_t>{1, 0}))
      throw std::runtime_error("unexpected recipe cache counts");
    uint64_t workspace = 0;
    check(synWorkspaceGetSize(&workspace, recipe), "workspace");
    initialize_launch_tensor_ids(recipe, bindings.data(), bindings.size());
    check(synLaunch(runtime.compute_stream(), bindings.data(), bindings.size(),
                    arena.ensure_workspace(workspace), recipe, 0), "launch");
    runtime.synchronize_compute();
    arena.copy_d2h_sync(runtime.d2h_stream(), addresses[4], output.data(), output.size() * 4);
    for (int t = 0; t < tasks; ++t)
      for (int lane = 0; lane < 64; ++lane) {
        const int r = row_requests[work[t * 4]], first = work[t * 4 + 1];
        const int expected = work[t * 4 + 3] + (lane < work[t * 4 + 2] ? map[r * context + first + lane] : 0);
        if (output[t * 64 + lane] != expected) throw std::runtime_error("wrong tile index or tail");
      }
    check(synRecipeDestroy(recipe), "recipe destroy");
    check(synGraphDestroy(graph), "graph destroy");
    for (auto tensor : tensors) check(synTensorDestroy(tensor), "tensor destroy");
    for (auto section : sections) check(synSectionDestroy(section), "section destroy");
    std::cout << "tile indices matched; cache " << counts.first << '/' << counts.second << '\n';
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
