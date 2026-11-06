// Verify one dynamic external pool descriptor against two physical pool sizes.
// The optional invalid index bypasses runtime validation solely to test device bounds.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include <perf_lib_layer_params.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gaudirt::runtime;

namespace {
void check(synStatus status, const char* stage) {
  if (status != synSuccess)
    throw std::runtime_error(std::string(stage) + ": " + std::to_string(int(status)));
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 4 && argc != 5) throw std::invalid_argument("expected pool slots, cache directory, cold/warm, optional oob");
    const uint64_t slots = std::stoull(argv[1]);
    const std::string mode = argv[3];
    const bool oob = argc == 5 && std::string(argv[4]) == "oob";
    if (argc == 5 && !oob) throw std::invalid_argument("invalid probe mode");
    constexpr uint64_t width = 1024, max_slots = 131072, tile = 4;
    if (slots < 2 || slots > max_slots || (mode != "cold" && mode != "warm"))
      throw std::invalid_argument("invalid pool size or mode");
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 64ULL * 1024 * 1024);
    RecipeCache cache(argv[2]);
    const auto canary_before = arena.allocate(width * 2);
    const auto pool = arena.allocate(width * slots * 2);
    const auto canary_after = arena.allocate(width * 2);
    const auto indices = arena.allocate(tile * 4);
    const auto output = arena.allocate(width * tile * 2);
    std::vector<uint16_t> host_pool(width * slots);
    for (size_t i = 0; i < host_pool.size(); ++i)
      host_pool[i] = 0x3f00 | static_cast<uint16_t>(i % 128);
    std::vector<uint16_t> guard(width, 0x4110), observed_guard(width), observed(width * tile);
    const std::array<int32_t, tile> host_indices{0, static_cast<int32_t>(slots - 1),
                                                static_cast<int32_t>(slots / 2),
                                                oob ? static_cast<int32_t>(slots) : 1};
    arena.copy_h2d_sync(runtime.h2d_stream(), guard.data(), canary_before, guard.size() * 2);
    arena.copy_h2d_sync(runtime.h2d_stream(), host_pool.data(), pool, host_pool.size() * 2);
    arena.copy_h2d_sync(runtime.h2d_stream(), guard.data(), canary_after, guard.size() * 2);
    arena.copy_h2d_sync(runtime.h2d_stream(), host_indices.data(), indices, sizeof(host_indices));

    synGraphHandle graph = nullptr;
    std::array<synSectionHandle, 3> sections{};
    std::array<synTensor, 3> tensors{};
    std::array<synLaunchTensorInfo, 3> bindings{};
    const std::array<std::string, 3> names{"dynamic_pool_t0", "dynamic_pool_t1", "dynamic_pool_t2"};
    const std::array<uint64_t, 3> addresses{pool, indices, output};
    check(synGraphCreate(&graph, synDeviceGaudi2), "graph");
    for (unsigned i = 0; i < 3; ++i) {
      check(synSectionCreate(&sections[i], 0, graph), "section");
      check(synSectionSetPersistent(sections[i], true), "persistent section");
      synTensorDescriptor desc{};
      desc.m_name = names[i].c_str();
      desc.m_dataType = i == 1 ? syn_type_int32 : syn_type_bf16;
      desc.m_tensorType = i == 0 ? DATA_TENSOR_DYNAMIC : DATA_TENSOR;
      desc.m_dims = i == 1 ? 1 : 2;
      desc.m_sizes[0] = desc.m_minSizes[0] = i == 1 ? tile : width;
      if (i != 1) {
        desc.m_sizes[1] = i == 0 ? max_slots : tile;
        desc.m_minSizes[1] = i == 0 ? 1 : tile;
      }
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "tensor");
      bindings[i].tensorName = names[i].c_str();
      bindings[i].tensorType = i == 0 ? DATA_TENSOR_DYNAMIC : DATA_TENSOR;
      bindings[i].pTensorAddress = addresses[i];
      bindings[i].tensorSize[0] = desc.m_sizes[0];
      if (i != 1) bindings[i].tensorSize[1] = i == 0 ? slots : tile;
    }
    ns_GatherKernel::ParamsV2 params{};
    params.axis = 1;
    params.batchDims = 0;
    check(synNodeCreate(graph, tensors.data(), &tensors[2], 2, 1, &params, sizeof(params),
                        "gather_fwd_bf16", "dynamic_pool_n0", nullptr, nullptr), "node");
    auto recipe = compile_cached(graph, "dynamic_pool", "dynamic_pool", &cache);
    const auto counts = cache.stats();
    if ((mode == "cold" && counts != std::pair<uint64_t, uint64_t>{0, 1}) ||
        (mode == "warm" && counts != std::pair<uint64_t, uint64_t>{1, 0}))
      throw std::runtime_error("unexpected cache counts");
    uint64_t workspace = 0;
    check(synWorkspaceGetSize(&workspace, recipe), "workspace");
    initialize_launch_tensor_ids(recipe, bindings.data(), bindings.size());
    check(synLaunch(runtime.compute_stream(), bindings.data(), bindings.size(),
                    arena.ensure_workspace(workspace), recipe, 0), "launch");
    runtime.synchronize_compute();
    arena.copy_d2h_sync(runtime.d2h_stream(), output, observed.data(), observed.size() * 2);
    for (size_t row = 0; row < tile; ++row)
      for (size_t col = 0; col < width; ++col) {
        if (oob && row == tile - 1) {
          if (observed[row * width + col] != 0)
            throw std::runtime_error("out-of-range read was not zeroed");
        } else if (observed[row * width + col] != host_pool[static_cast<size_t>(host_indices[row]) * width + col])
          throw std::runtime_error("gathered values differ");
      }
    for (auto address : {canary_before, canary_after}) {
      arena.copy_d2h_sync(runtime.d2h_stream(), address, observed_guard.data(), guard.size() * 2);
      if (observed_guard != guard) throw std::runtime_error("guard modified");
    }
    std::cout << "dynamic pool slots " << slots << " loaded " << counts.first
              << " compiled " << counts.second << " guards intact" << (oob ? " oob zero" : "") << "\n";
    check(synRecipeDestroy(recipe), "recipe destroy");
    check(synGraphDestroy(graph), "graph destroy");
    for (auto tensor : tensors) check(synTensorDestroy(tensor), "tensor destroy");
    for (auto section : sections) check(synSectionDestroy(section), "section destroy");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
