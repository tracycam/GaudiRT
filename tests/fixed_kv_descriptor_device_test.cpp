// Probe a fixed maximum-size KV descriptor against smaller physical pools.
// Descriptor spans remain inside the allocated slab and disjoint from every
// other persistent section in this graph. This is a device-layout experiment,
// not a paged-attention implementation or a general allocator guarantee.
#include "gaudirt/runtime/device_runtime.hpp"
#include "test_device_slab.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include <perf_lib_layer_params.h>

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gaudirt::runtime;

namespace {
void check(synStatus status, const char* what) {
  if (status != synSuccess)
    throw std::runtime_error(std::string(what) + ": " + std::to_string(static_cast<int>(status)));
}
}

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::invalid_argument("expected slots, cache path and cold/warm");
    const uint32_t slots = std::stoul(argv[1]);
    const std::string mode = argv[3];
    constexpr uint32_t width = 1024, bucket = 256, rows = 137, max_slots = 131072, tile = 4;
    if (slots < 512 || slots > 8192 || (mode != "cold" && mode != "warm"))
      throw std::invalid_argument("invalid test slots or cache mode");
    const size_t source_count = static_cast<size_t>(width) * bucket;
    const size_t pool_count = static_cast<size_t>(width) * slots;
    constexpr uint16_t key_sentinel = 0x3e80, value_sentinel = 0x3f00, guard_value = 0x4110;
    std::vector<uint16_t> key(source_count), value(source_count), initial_key(pool_count, key_sentinel),
        initial_value(pool_count, value_sentinel), actual_key(pool_count), actual_value(pool_count);
    std::vector<uint16_t> guard(128, guard_value), observed_guard(128);
    // Tail indices deliberately point one row past each logical pool. The
    // valid_rows predicate must prevent those writes from reaching the guard.
    std::vector<int32_t> locations(bucket, slots), source_for_slot(slots, -1);
    for (uint32_t row = 0; row < rows; ++row) {
      locations[row] = row == 0 ? slots - 1 : 1 + 2 * (row - 1);
      source_for_slot[locations[row]] = row;
    }
    for (size_t i = 0; i < source_count; ++i) {
      key[i] = static_cast<uint16_t>(0x3f80 + i % 127);
      value[i] = static_cast<uint16_t>(0x4000 + (i * 7) % 127);
    }
    DeviceRuntime runtime({});
    gaudirt::testing::TestDeviceSlab arena(runtime.device_id(), 832ULL * 1024 * 1024);
    RecipeCache cache(argv[2]);
    std::array<uint64_t, 9> addresses{};
    addresses[0] = arena.allocate(source_count * 2);
    addresses[1] = arena.allocate(source_count * 2);
    addresses[2] = arena.allocate(bucket * 4);
    addresses[3] = arena.allocate(sizeof(int32_t));
    addresses[6] = arena.allocate(tile * sizeof(int32_t));
    addresses[7] = arena.allocate(width * tile * sizeof(uint16_t));
    addresses[8] = arena.allocate(width * tile * sizeof(uint16_t));
    const uint64_t before_k = arena.allocate(guard.size() * 2);
    addresses[4] = arena.allocate(pool_count * 2);
    const uint64_t after_k = arena.allocate(guard.size() * 2);
    const uint64_t pad_bytes = static_cast<uint64_t>(width) * max_slots * 2 - pool_count * 2;
    const uint64_t key_padding = arena.allocate(pad_bytes);
    const uint64_t before_v = arena.allocate(guard.size() * 2);
    addresses[5] = arena.allocate(pool_count * 2);
    const uint64_t after_v = arena.allocate(guard.size() * 2);
    const uint64_t value_padding = arena.allocate(pad_bytes);
    // Model unrelated allocations within each claimed descriptor span. These
    // are not persistent graph inputs and must remain untouched.
    const std::array<uint64_t, 6> neighbors{
        key_padding, key_padding + pad_bytes / 2, key_padding + pad_bytes - guard.size() * 2,
        value_padding, value_padding + pad_bytes / 2, value_padding + pad_bytes - guard.size() * 2};
    arena.copy_h2d_sync(runtime.h2d_stream(), key.data(), addresses[0], source_count * 2);
    arena.copy_h2d_sync(runtime.h2d_stream(), value.data(), addresses[1], source_count * 2);
    arena.copy_h2d_sync(runtime.h2d_stream(), locations.data(), addresses[2], bucket * 4);
    const std::array<int32_t, tile> gather_indices{static_cast<int32_t>(slots - 1), 0, 3, static_cast<int32_t>(slots / 2)};
    arena.copy_h2d_sync(runtime.h2d_stream(), gather_indices.data(), addresses[6], sizeof(gather_indices));
    std::vector<uint16_t> gathered_key(width * tile), gathered_value(width * tile);
    for (auto address : {before_k, after_k, before_v, after_v})
      arena.copy_h2d_sync(runtime.h2d_stream(), guard.data(), address, guard.size() * 2);
    for (auto address : neighbors)
      arena.copy_h2d_sync(runtime.h2d_stream(), guard.data(), address, guard.size() * 2);

    synGraphHandle graph = nullptr;
    std::array<synTensor, 9> tensors{};
    std::array<synSectionHandle, 9> sections{};
    std::array<synLaunchTensorInfo, 9> launch{};
    std::array<std::string, 9> names;
    const std::string prefix = "fixed_kv_disjoint_store_gather_w1024_b256";
    check(synGraphCreate(&graph, synDeviceGaudi2), "graph");
    for (size_t i = 0; i < tensors.size(); ++i) {
      names[i] = prefix + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "section");
      check(synSectionSetPersistent(sections[i], true), "persistent section");
      synTensorDescriptor desc{};
      desc.m_name = names[i].c_str();
      desc.m_dataType = i == 2 || i == 3 || i == 6 ? syn_type_int32 : syn_type_bf16;
      desc.m_tensorType = DATA_TENSOR;
      desc.m_dims = i == 2 || i == 3 || i == 6 ? 1 : 2;
      desc.m_sizes[0] = desc.m_minSizes[0] = i == 2 ? bucket : i == 3 ? 1 : i == 6 ? tile : width;
      if (desc.m_dims == 2) {
        desc.m_sizes[1] = i == 4 || i == 5 ? max_slots : i >= 7 ? tile : bucket;
        desc.m_minSizes[1] = desc.m_sizes[1];
      }
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "tensor");
      launch[i].tensorName = names[i].c_str();
      launch[i].tensorType = desc.m_tensorType;
      launch[i].pTensorAddress = addresses[i];
      launch[i].tensorSize[0] = desc.m_sizes[0];
      if (desc.m_dims == 2) launch[i].tensorSize[1] = desc.m_sizes[1];
    }
    const auto node = prefix + "_n0";
    check(synNodeCreate(graph, tensors.data(), &tensors[4], 4, 2, nullptr, 0,
                        "gk_kv_store_bf16_v1", node.c_str(), nullptr, nullptr), "node");
    ns_GatherKernel::ParamsV2 params{};
    params.axis = 1;
    params.batchDims = 0;
    for (unsigned pool = 0; pool < 2; ++pool) {
      const std::array<synTensor, 2> inputs{tensors[4 + pool], tensors[6]};
      const auto gather_name = prefix + "_n" + std::to_string(pool + 1);
      check(synNodeCreate(graph, inputs.data(), &tensors[7 + pool], 2, 1, &params, sizeof(params),
                          "gather_fwd_bf16", gather_name.c_str(), nullptr, nullptr), "gather");
    }
    auto recipe = compile_cached(graph, prefix, prefix, &cache);
    const auto counts = cache.stats();
    if ((mode == "cold" && counts != std::pair<uint64_t, uint64_t>{0, 1}) ||
        (mode == "warm" && counts != std::pair<uint64_t, uint64_t>{1, 0}))
      throw std::runtime_error("unexpected recipe cache counts");
    uint64_t workspace = 0;
    check(synWorkspaceGetSize(&workspace, recipe), "workspace");
    initialize_launch_tensor_ids(recipe, launch.data(), launch.size());
    for (const int32_t valid : {1, 7, static_cast<int32_t>(rows)}) {
      arena.copy_h2d_sync(runtime.h2d_stream(), &valid, addresses[3], sizeof(valid));
      arena.copy_h2d_sync(runtime.h2d_stream(), initial_key.data(), addresses[4], pool_count * 2);
      arena.copy_h2d_sync(runtime.h2d_stream(), initial_value.data(), addresses[5], pool_count * 2);
      arena.copy_d2h_sync(runtime.d2h_stream(), addresses[4], actual_key.data(), width * 2);
      if (actual_key[0] != key_sentinel) throw std::runtime_error("KV initialization failed");
      check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(),
                      arena.ensure_workspace(workspace), recipe, 0), "launch");
      runtime.synchronize_compute();
      arena.copy_d2h_sync(runtime.d2h_stream(), addresses[4], actual_key.data(), pool_count * 2);
      arena.copy_d2h_sync(runtime.d2h_stream(), addresses[5], actual_value.data(), pool_count * 2);
      arena.copy_d2h_sync(runtime.d2h_stream(), addresses[7], gathered_key.data(), gathered_key.size() * 2);
      arena.copy_d2h_sync(runtime.d2h_stream(), addresses[8], gathered_value.data(), gathered_value.size() * 2);
      for (unsigned row = 0; row < tile; ++row) {
        for (unsigned col = 0; col < width; ++col) {
          const auto src = static_cast<size_t>(gather_indices[row]) * width + col;
          const auto dst = static_cast<size_t>(row) * width + col;
          if (gathered_key[dst] != actual_key[src] || gathered_value[dst] != actual_value[src])
            throw std::runtime_error("post-write gather mismatch");
        }
      }
      size_t wrong_targets = 0, touched_other = 0;
      for (uint32_t slot = 0; slot < slots; ++slot) {
        const int32_t row = source_for_slot[slot];
        const bool targeted = row >= 0 && row < valid;
        for (uint32_t col = 0; col < width; ++col) {
          const size_t dst = static_cast<size_t>(slot) * width + col;
          const size_t src = static_cast<size_t>(targeted ? row : 0) * width + col;
          if (actual_key[dst] != (targeted ? key[src] : key_sentinel) ||
              actual_value[dst] != (targeted ? value[src] : value_sentinel)) {
            auto& count = targeted ? wrong_targets : touched_other;
            if (++count <= 2)
              std::cerr << "valid " << valid << " slot " << slot << " column " << col << " source row " << row
                        << " key " << actual_key[dst] << " expected " << (targeted ? key[src] : key_sentinel)
                        << " value " << actual_value[dst] << " expected " << (targeted ? value[src] : value_sentinel)
                        << '\n';
          }
        }
      }
      for (auto address : {before_k, after_k, before_v, after_v}) {
        arena.copy_d2h_sync(runtime.d2h_stream(), address, observed_guard.data(), guard.size() * 2);
        if (observed_guard != guard) throw std::runtime_error("KV guard modified");
      }
      for (auto address : neighbors) {
        arena.copy_d2h_sync(runtime.d2h_stream(), address, observed_guard.data(), guard.size() * 2);
        if (observed_guard != guard) throw std::runtime_error("neighboring allocation modified");
      }
      if (wrong_targets || touched_other)
        throw std::runtime_error("KV target mismatch or untouched slot modified: " +
                                 std::to_string(wrong_targets) + "/" + std::to_string(touched_other));
    }
    std::cout << "KV actual slots " << slots << " loaded " << counts.first << " compiled " << counts.second
              << " valid rows 1/7/137, target writes, post-write gather and all other bytes correct\n";
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
