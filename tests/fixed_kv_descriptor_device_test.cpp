// Probe a fixed maximum-size head-major KV descriptor against smaller physical pools.
// Descriptor spans remain inside the allocated slab and disjoint from every
// other persistent section in this graph.
#include "gaudirt/runtime/device_runtime.hpp"
#include "test_device_slab.hpp"
#include "gaudirt/runtime/recipe_cache.hpp"
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
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::invalid_argument("expected slots, cache path and cold/warm");
    const uint32_t slots = std::stoul(argv[1]);
    const std::string mode = argv[3];
    constexpr uint32_t head_dim = 128, kv_heads = 8, width = head_dim * kv_heads;
    constexpr uint32_t bucket = 256, rows = 137, max_slots = 131072, tile = 6;
    if (slots < 512 || slots > 8192 || (mode != "cold" && mode != "warm"))
      throw std::invalid_argument("invalid test slots or cache mode");
    const size_t source_count = static_cast<size_t>(width) * bucket;
    const size_t half_pool_count = static_cast<size_t>(width) * slots;
    const size_t pool_count = 2 * half_pool_count;
    constexpr uint16_t key_sentinel = 0x3e80, value_sentinel = 0x3f00, guard_value = 0x4110;
    std::vector<uint16_t> key(source_count), value(source_count), initial_pool(pool_count),
        actual_pool(pool_count);
    std::fill(initial_pool.begin(), initial_pool.begin() + half_pool_count, key_sentinel);
    std::fill(initial_pool.begin() + half_pool_count, initial_pool.end(), value_sentinel);
    std::vector<uint16_t> guard(128, guard_value), observed_guard(128);
    // Tail indices deliberately point one row past each logical pool. The
    // valid_rows predicate must prevent those writes from reaching untouched slots or guards.
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
    // Logical bindings: 0:key, 1:value, 2:locations, 3:header, 4:old_pool, 5:pool, 6:gather_idx, 7:gathered.
    std::array<uint64_t, 8> addresses{};
    addresses[0] = arena.allocate(source_count * 2);
    addresses[1] = arena.allocate(source_count * 2);
    addresses[2] = arena.allocate(bucket * 4);
    addresses[3] = arena.allocate(3 * sizeof(int32_t));
    addresses[6] = arena.allocate(tile * sizeof(int32_t));
    addresses[7] = arena.allocate(head_dim * tile * sizeof(uint16_t));
    const uint64_t before_pool = arena.allocate(guard.size() * 2);
    addresses[4] = arena.allocate(pool_count * 2);
    addresses[5] = addresses[4];
    const uint64_t after_pool = arena.allocate(guard.size() * 2);
    const uint64_t pad_bytes = static_cast<uint64_t>(width) * max_slots * 4 - pool_count * 2;
    const uint64_t pool_padding = arena.allocate(pad_bytes);
    const std::array<uint64_t, 3> neighbors{
        pool_padding, pool_padding + pad_bytes / 2, pool_padding + pad_bytes - guard.size() * 2};
    arena.copy_h2d_sync(runtime.h2d_stream(), key.data(), addresses[0], source_count * 2);
    arena.copy_h2d_sync(runtime.h2d_stream(), value.data(), addresses[1], source_count * 2);
    arena.copy_h2d_sync(runtime.h2d_stream(), locations.data(), addresses[2], bucket * 4);
    const std::array<int32_t, tile> gather_indices{
        static_cast<int32_t>(slots - 1),
        static_cast<int32_t>(3 * slots + 0),
        static_cast<int32_t>(7 * slots + 3),
        static_cast<int32_t>(kv_heads * slots + slots - 1),
        static_cast<int32_t>((kv_heads + 3) * slots + 3),
        static_cast<int32_t>((kv_heads + 7) * slots + 1)};
    arena.copy_h2d_sync(runtime.h2d_stream(), gather_indices.data(), addresses[6], sizeof(gather_indices));
    std::vector<uint16_t> gathered(head_dim * tile);
    for (auto address : {before_pool, after_pool})
      arena.copy_h2d_sync(runtime.h2d_stream(), guard.data(), address, guard.size() * 2);
    for (auto address : neighbors)
      arena.copy_h2d_sync(runtime.h2d_stream(), guard.data(), address, guard.size() * 2);

    synGraphHandle graph = nullptr;
    std::array<synTensor, 8> tensors{};
    std::array<synSectionHandle, 7> sections{};
    std::array<synLaunchTensorInfo, 8> launch{};
    std::array<std::string, 8> names;
    const std::string prefix = "fixed_head_major_kv_w1024_b256";
    check(synGraphCreate(&graph, synDeviceGaudi2), "graph");
    for (auto& section : sections) {
      check(synSectionCreate(&section, 0, graph), "section");
      check(synSectionSetPersistent(section, true), "persistent section");
    }
    for (size_t i = 0; i < tensors.size(); ++i) {
      names[i] = prefix + "_t" + std::to_string(i);
      synTensorDescriptor desc{};
      desc.m_name = names[i].c_str();
      desc.m_dataType = (i == 2 || i == 3 || i == 6) ? syn_type_int32 : syn_type_bf16;
      desc.m_tensorType = DATA_TENSOR;
      desc.m_dims = (i == 2 || i == 3 || i == 6) ? 1 : 2;
      if (i == 2) desc.m_sizes[0] = bucket;
      else if (i == 3) desc.m_sizes[0] = 3;
      else if (i == 6) desc.m_sizes[0] = tile;
      else if (i == 4 || i == 5 || i == 7) desc.m_sizes[0] = head_dim;
      else desc.m_sizes[0] = width;
      desc.m_minSizes[0] = desc.m_sizes[0];
      if (desc.m_dims == 2) {
        desc.m_sizes[1] = (i == 4 || i == 5) ? (2 * max_slots * kv_heads) : (i == 7 ? tile : bucket);
        desc.m_minSizes[1] = desc.m_sizes[1];
      }
      const size_t sec_idx = i >= 5 ? i - 1 : i;
      check(synTensorCreate(&tensors[i], &desc, sections[sec_idx], 0), "tensor");
      launch[i].tensorName = names[i].c_str();
      launch[i].tensorType = desc.m_tensorType;
      launch[i].pTensorAddress = addresses[i];
      launch[i].tensorSize[0] = desc.m_sizes[0];
      if (desc.m_dims == 2) launch[i].tensorSize[1] = desc.m_sizes[1];
    }
    const std::array<synTensor, 5> store_inputs{tensors[0], tensors[1], tensors[2], tensors[3], tensors[4]};
    const auto node = prefix + "_n0";
    check(synNodeCreate(graph, store_inputs.data(), &tensors[5], 5, 1, nullptr, 0,
                        "gk_kv_store_head_major_bf16_v1", node.c_str(), nullptr, nullptr), "node");
    ns_GatherKernel::ParamsV2 params{};
    params.axis = 1;
    params.batchDims = 0;
    const std::array<synTensor, 2> gather_inputs{tensors[5], tensors[6]};
    const auto gather_name = prefix + "_n1";
    check(synNodeCreate(graph, gather_inputs.data(), &tensors[7], 2, 1, &params, sizeof(params),
                        "gather_fwd_bf16", gather_name.c_str(), nullptr, nullptr), "gather");

    auto recipe = compile_cached(graph, prefix, prefix, &cache);
    const auto counts = cache.stats();
    if ((mode == "cold" && counts != std::pair<uint64_t, uint64_t>{0, 1}) ||
        (mode == "warm" && counts != std::pair<uint64_t, uint64_t>{1, 0}))
      throw std::runtime_error("unexpected recipe cache counts");
    uint64_t workspace = 0;
    check(synWorkspaceGetSize(&workspace, recipe), "workspace");
    initialize_launch_tensor_ids(recipe, launch.data(), launch.size());
    for (const int32_t valid : {1, 7, static_cast<int32_t>(rows)}) {
      const std::array<int32_t, 3> header{valid, static_cast<int32_t>(slots), static_cast<int32_t>(kv_heads)};
      arena.copy_h2d_sync(runtime.h2d_stream(), header.data(), addresses[3], sizeof(header));
      arena.copy_h2d_sync(runtime.h2d_stream(), initial_pool.data(), addresses[4], pool_count * 2);
      arena.copy_d2h_sync(runtime.d2h_stream(), addresses[4], actual_pool.data(), head_dim * 2);
      if (actual_pool[0] != key_sentinel) throw std::runtime_error("KV initialization failed");
      check(synLaunch(runtime.compute_stream(), launch.data(), launch.size(),
                      arena.ensure_workspace(workspace), recipe, 0), "launch");
      runtime.synchronize_compute();
      arena.copy_d2h_sync(runtime.d2h_stream(), addresses[4], actual_pool.data(), pool_count * 2);
      arena.copy_d2h_sync(runtime.d2h_stream(), addresses[7], gathered.data(), gathered.size() * 2);
      for (unsigned row = 0; row < tile; ++row) {
        for (unsigned col = 0; col < head_dim; ++col) {
          const auto src = static_cast<size_t>(gather_indices[row]) * head_dim + col;
          const auto dst = static_cast<size_t>(row) * head_dim + col;
          if (gathered[dst] != actual_pool[src]) throw std::runtime_error("post-write gather mismatch");
        }
      }
      size_t wrong_targets = 0, touched_other = 0;
      for (uint32_t h = 0; h < kv_heads; ++h) {
        for (uint32_t slot = 0; slot < slots; ++slot) {
          const int32_t row = source_for_slot[slot];
          const bool targeted = row >= 0 && row < valid;
          const size_t k_dst_base = (static_cast<size_t>(h) * slots + slot) * head_dim;
          const size_t v_dst_base = half_pool_count + k_dst_base;
          const size_t src_base = static_cast<size_t>(targeted ? row : 0) * width + h * head_dim;
          for (uint32_t col = 0; col < head_dim; ++col) {
            const uint16_t exp_k = targeted ? key[src_base + col] : key_sentinel;
            const uint16_t exp_v = targeted ? value[src_base + col] : value_sentinel;
            if (actual_pool[k_dst_base + col] != exp_k || actual_pool[v_dst_base + col] != exp_v) {
              auto& count = targeted ? wrong_targets : touched_other;
              if (++count <= 2) {
                std::cerr << "valid " << valid << " head " << h << " slot " << slot << " col " << col
                          << " k " << actual_pool[k_dst_base + col] << " exp " << exp_k
                          << " v " << actual_pool[v_dst_base + col] << " exp " << exp_v << '\n';
              }
            }
          }
        }
      }
      for (auto address : {before_pool, after_pool}) {
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
              << " valid rows 1/7/137, head-major target writes, post-write gather and all other bytes correct\n";
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
