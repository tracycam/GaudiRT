// SPDX-License-Identifier: Apache-2.0
// Head-major KV write + multi-tile fused MME/TPC KV-chunk attention + online-state finalization.
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
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace gaudirt::runtime;

namespace {
void check(synStatus status, const char* stage) {
  if (status != synSuccess)
    throw std::runtime_error(std::string(stage) + " status " + std::to_string(static_cast<int>(status)));
}

uint16_t bf16(float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  return static_cast<uint16_t>((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
float widen(uint16_t value) { return std::bit_cast<float>(static_cast<uint32_t>(value) << 16); }
uint16_t direct_bf16(double value) {
  if (!std::isfinite(value)) throw std::runtime_error("nonfinite reference");
  const auto bits = std::bit_cast<uint64_t>(value);
  const uint16_t sign = (bits >> 48) & 0x8000;
  const uint64_t exponent = (bits >> 52) & 0x7ff, fraction = bits & ((uint64_t{1} << 52) - 1);
  if (exponent < 897) return sign | static_cast<uint16_t>(std::nearbyint(std::ldexp(std::abs(value), 133)));
  if (exponent > 1150) return sign | 0x7f80;
  uint64_t payload = ((exponent - 896) << 7) | (fraction >> 45);
  const uint64_t tail = fraction & ((uint64_t{1} << 45) - 1), half = uint64_t{1} << 44;
  payload += tail > half || (tail == half && (payload & 1));
  return sign | static_cast<uint16_t>(payload);
}

uint64_t align256(uint64_t bytes) { return (bytes + 255ULL) & ~uint64_t{255}; }

struct Graph {
  explicit Graph(std::string name) : prefix(std::move(name)) {
    names.reserve(512);
    tensors.reserve(512);
    tensor_sections.reserve(512);
    bindings.reserve(64);
    binding_name_idx.reserve(64);
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
      binding.tensorType = DATA_TENSOR;
      binding.pTensorAddress = address;
      for (size_t i = 0; i < dims.size(); ++i) binding.tensorSize[i] = dims[i];
      bindings.push_back(binding);
      binding_name_idx.push_back(id);
    }
    return id;
  }
  void set_perm(size_t id, std::initializer_list<uint8_t> dims) {
    synTensorPermutation perm{};
    perm.dims = dims.size();
    size_t i = 0;
    for (uint8_t d : dims) perm.permutation[i++] = d;
    check(synTensorSetPermutation(tensors.at(id), &perm), "tensor permutation");
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
    for (size_t i = 0; i < bindings.size(); ++i)
      bindings[i].tensorName = names.at(binding_name_idx[i]).c_str();
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
  std::vector<size_t> binding_name_idx;
};

struct State {
  uint64_t maximum = 0, sum = 0, output = 0;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 13 || argc > 15)
      throw std::invalid_argument(
          "expected cache mode output d q_bucket valid_rows kv_tile q_heads kv_heads slots page_size tiles "
          "[query_blocks [max_chunk_tiles]]");
    const std::string mode = argv[2];
    const uint32_t d = std::stoul(argv[4]), rows = std::stoul(argv[5]), valid = std::stoul(argv[6]);
    const uint32_t kv = std::stoul(argv[7]), heads = std::stoul(argv[8]), kvh = std::stoul(argv[9]);
    const uint32_t slots = std::stoul(argv[10]), page = std::stoul(argv[11]), tiles = std::stoul(argv[12]);
    const uint32_t query_blocks = argc >= 14 ? std::stoul(argv[13]) : 1;
    const uint32_t max_chunk_tiles = argc == 15 ? std::stoul(argv[14]) : 8;
    constexpr uint32_t max_slots = 131072;
    if ((mode != "cold" && mode != "warm") || (d != 64 && d != 128) || !rows || rows > 256 ||
        !valid || valid > rows || !kv || kv > 512 || kv % 128 || !kvh || heads % kvh || heads > 256 ||
        uint64_t{kv} * rows * heads > 8ULL * 1024 * 1024 || kvh * d % 128 ||
        slots < 2048 || slots > max_slots || (page != 1 && page != 16 && page != 64) || slots % page ||
        !tiles || uint64_t{tiles} * kv + page >= slots || !query_blocks ||
        uint64_t{query_blocks} * rows > 4096 || !max_chunk_tiles || max_chunk_tiles > 8 ||
        (query_blocks > 1 && (valid != rows || uint64_t{query_blocks} * rows > uint64_t{tiles} * kv)))
      throw std::invalid_argument("unsupported geometry");

    const uint32_t group = heads / kvh, width = kvh * d, total_rows = rows * query_blocks;
    const uint32_t stats_rows = (rows + 63) / 64 * 64;
    const float scale = 1.f / std::sqrt(static_cast<float>(d));
    std::mt19937 generator(20261106);
    std::normal_distribution<float> normal(0.f, 0.6f);
    std::vector<uint16_t> q(uint64_t{d} * total_rows * heads), key_new(uint64_t{width} * total_rows),
        value_new(uint64_t{width} * total_rows);
    for (auto& x : q) x = bf16(normal(generator));
    for (auto& x : key_new) x = bf16(normal(generator));
    for (auto& x : value_new) x = bf16(normal(generator));
    std::vector<std::vector<int32_t>> indices(tiles, std::vector<int32_t>(kv));
    for (uint32_t i = 0; i < tiles * kv; ++i) {
      const uint32_t slot = page == 1 ? 1 + (i * 5) % (slots - 1) :
          (1 + ((i / page) * 5) % (slots / page - 1)) * page + i % page;
      indices[i / kv][i % kv] = static_cast<int32_t>(slot);
    }
    std::vector<uint16_t> key_pool(static_cast<size_t>(slots) * width, 0x3e80);
    std::vector<uint16_t> value_pool(static_cast<size_t>(slots) * width, 0x3f00);
    for (uint32_t i = 0; i < tiles * kv; ++i) {
      const auto slot = indices[i / kv][i % kv];
      for (uint32_t col = 0; col < width; ++col) {
        key_pool[static_cast<size_t>(slot) * width + col] = bf16(normal(generator));
        value_pool[static_cast<size_t>(slot) * width + col] = bf16(normal(generator));
      }
    }
    std::vector<int32_t> positions(total_rows), locations(total_rows, 0);
    const uint32_t last_tile_row_start = rows == 1 ? kv - 1 : 0;
    const uint32_t prefix = query_blocks > 1 ? tiles * kv - total_rows : (tiles - 1) * kv + last_tile_row_start;
    const uint32_t total_valid = query_blocks > 1 ? total_rows : valid;
    for (uint32_t row = 0; row < total_rows; ++row) {
      positions[row] = static_cast<int32_t>(prefix + row);
      if (row < total_valid) locations[row] = indices[positions[row] / kv][positions[row] % kv];
    }

    std::vector<uint16_t> physical_pool(uint64_t{2} * slots * width, 0);
    for (uint32_t head = 0; head < kvh; ++head)
      for (uint32_t slot = 0; slot < slots; ++slot)
        for (uint32_t col = 0; col < d; ++col) {
          physical_pool[(uint64_t{head} * slots + slot) * d + col] =
              key_pool[uint64_t{slot} * width + head * d + col];
          physical_pool[(uint64_t{kvh + head} * slots + slot) * d + col] =
              value_pool[uint64_t{slot} * width + head * d + col];
        }
    auto expected_key = key_pool, expected_value = value_pool;
    auto expected_physical_pool = physical_pool;
    for (uint32_t row = 0; row < total_valid; ++row) {
      std::copy_n(key_new.begin() + static_cast<size_t>(row) * width, width,
                  expected_key.begin() + static_cast<size_t>(locations[row]) * width);
      std::copy_n(value_new.begin() + static_cast<size_t>(row) * width, width,
                  expected_value.begin() + static_cast<size_t>(locations[row]) * width);
      for (uint32_t head = 0; head < kvh; ++head)
        for (uint32_t col = 0; col < d; ++col) {
          const auto physical = (uint64_t{head} * slots + locations[row]) * d + col;
          expected_physical_pool[physical] = key_new[uint64_t{row} * width + head * d + col];
          expected_physical_pool[uint64_t{slots} * width + physical] =
              value_new[uint64_t{row} * width + head * d + col];
        }
    }

    const uint64_t descriptor_bytes = uint64_t{2} * max_slots * width * 2;
    const uint64_t arena_bytes = std::max<uint64_t>(
        832ULL * 1024 * 1024,
        descriptor_bytes + q.size() * 4 + 768ULL * 1024 * 1024);

    DeviceRuntime runtime({});
    gaudirt::testing::TestDeviceSlab arena(runtime.device_id(), arena_bytes);
    RecipeCache cache(argv[1]);
    const auto q_addr = arena.allocate(q.size() * 2);
    const auto key_new_addr = arena.allocate(key_new.size() * 2);
    const auto value_new_addr = arena.allocate(value_new.size() * 2);
    const auto locations_addr = arena.allocate(locations.size() * 4);
    const auto valid_addr = arena.allocate(12);

    const uint64_t kv_idx_bytes = uint64_t{kv} * kvh * 4;
    const uint64_t tile_idx_stride = align256(2 * kv_idx_bytes);
    const uint64_t index_bytes = uint64_t{tiles} * tile_idx_stride;
    const uint64_t pos_stride = align256(uint64_t{rows} * 4);
    const uint64_t hdr_stride = 256;
    const uint64_t block_meta_stride = pos_stride + uint64_t{tiles} * hdr_stride;
    const uint64_t metadata_total_bytes = uint64_t{query_blocks} * block_meta_stride;
    const auto metadata_base = arena.allocate(index_bytes + metadata_total_bytes);

    std::vector<uint64_t> index_addr(tiles), pos_addr(query_blocks), header_addr(query_blocks * tiles);
    for (uint32_t tile = 0; tile < tiles; ++tile)
      index_addr[tile] = metadata_base + uint64_t{tile} * tile_idx_stride;
    for (uint32_t block = 0; block < query_blocks; ++block) {
      pos_addr[block] = metadata_base + index_bytes + uint64_t{block} * block_meta_stride;
      for (uint32_t tile = 0; tile < tiles; ++tile)
        header_addr[block * tiles + tile] = pos_addr[block] + pos_stride + uint64_t{tile} * hdr_stride;
    }

    std::array<State, 2> state{};
    for (auto& s : state) {
      s.maximum = arena.allocate(align256(uint64_t{stats_rows} * heads * 4));
      s.sum = arena.allocate(align256(uint64_t{stats_rows} * heads * 4));
      s.output = arena.allocate(align256(uint64_t{d} * rows * heads * 4));
    }
    const auto output_addr = arena.allocate(uint64_t{d} * total_rows * heads * 2);
    const auto guard_before_pool = arena.allocate(256);
    const auto pool_addr = arena.allocate(physical_pool.size() * 2);
    const auto guard_after_pool = arena.allocate(256);
    if (max_slots > slots) arena.allocate(uint64_t{2} * (max_slots - slots) * width * 2);

    auto h2d = [&](const void* source, uint64_t dest, uint64_t bytes) {
      arena.copy_h2d_sync(runtime.h2d_stream(), source, dest, bytes);
    };
    h2d(q.data(), q_addr, q.size() * 2);
    h2d(key_new.data(), key_new_addr, key_new.size() * 2);
    h2d(value_new.data(), value_new_addr, value_new.size() * 2);
    h2d(locations.data(), locations_addr, locations.size() * 4);
    const int32_t store_hdr[3] = {static_cast<int32_t>(total_valid), static_cast<int32_t>(slots),
                                  static_cast<int32_t>(kvh)};
    h2d(store_hdr, valid_addr, sizeof(store_hdr));
    h2d(physical_pool.data(), pool_addr, physical_pool.size() * 2);
    std::array<uint16_t, 128> guard{};
    guard.fill(0x4110);
    for (auto address : {guard_before_pool, guard_after_pool})
      h2d(guard.data(), address, sizeof(guard));

    const auto tile_valid_keys = [&](uint32_t tile) -> uint32_t {
      return (query_blocks == 1 && tile + 1 == tiles && rows > 1 && valid <= kv - 7) ? (kv - 7) : kv;
    };
    std::vector<int32_t> packed_metadata((index_bytes + metadata_total_bytes) / 4, 0);
    for (uint32_t tile = 0; tile < tiles; ++tile) {
      const size_t base = (uint64_t{tile} * tile_idx_stride) / 4;
      for (uint32_t head = 0; head < kvh; ++head)
        for (uint32_t i = 0; i < kv; ++i) {
          packed_metadata[base + head * kv + i] = static_cast<int32_t>(head * slots + indices[tile][i]);
          packed_metadata[base + kvh * kv + head * kv + i] =
              static_cast<int32_t>((kvh + head) * slots + indices[tile][i]);
        }
    }
    for (uint32_t block = 0; block < query_blocks; ++block) {
      const size_t p_off = (index_bytes + uint64_t{block} * block_meta_stride) / 4;
      std::copy_n(positions.begin() + block * rows, rows, packed_metadata.begin() + p_off);
      for (uint32_t tile = 0; tile < tiles; ++tile) {
        const size_t h_off = p_off + pos_stride / 4 + (uint64_t{tile} * hdr_stride) / 4;
        packed_metadata[h_off + 0] = static_cast<int32_t>(valid);
        packed_metadata[h_off + 1] = static_cast<int32_t>(tile_valid_keys(tile));
        packed_metadata[h_off + 2] = static_cast<int32_t>(tile * kv);
      }
    }
    h2d(packed_metadata.data(), metadata_base, index_bytes + metadata_total_bytes);

    const std::string shape = "_d" + std::to_string(d) + "_q" + std::to_string(rows) + "_k" +
        std::to_string(kv) + "_h" + std::to_string(heads) + "_kh" + std::to_string(kvh);

    Graph store("fixed_tile_hm_store_m" + std::to_string(total_rows) + shape);
    const auto sk = store.tensor({width, total_rows}, syn_type_bf16, key_new_addr);
    const auto sv = store.tensor({width, total_rows}, syn_type_bf16, value_new_addr);
    const auto sl = store.tensor({total_rows}, syn_type_int32, locations_addr);
    const auto sh = store.tensor({3}, syn_type_int32, valid_addr);
    const auto old_pool = store.tensor({d, 2 * max_slots * kvh}, syn_type_bf16, pool_addr);
    const auto new_pool = store.tensor({d, 2 * max_slots * kvh}, syn_type_bf16, pool_addr, old_pool);
    store.node("gk_kv_store_head_major_bf16_v1", {sk, sv, sl, sh, old_pool}, {new_pool});
    store.compile(cache);
    uint64_t max_workspace = store.workspace;
    uint64_t compiled_graphs = 1;

    std::unique_ptr<Graph> direct_graph;
    auto get_direct_graph = [&]() -> Graph& {
      if (direct_graph) return *direct_graph;
      direct_graph = std::make_unique<Graph>("fixed_tile_direct" + shape);
      auto& g = *direct_graph;
      const auto tq = g.tensor({d, rows, group, kvh}, syn_type_bf16, q_addr);
      if (rows > 1) g.set_perm(tq, {0, 2, 3, 1});
      const auto tpool = g.tensor({d, 2 * max_slots * kvh}, syn_type_bf16, pool_addr);
      const auto tp = g.tensor({rows}, syn_type_int32, pos_addr[0]);
      const auto ti = g.tensor({kv * kvh}, syn_type_int32, index_addr[0]);
      const auto vi = g.tensor({kv * kvh}, syn_type_int32, index_addr[0] + kv_idx_bytes, ti, kv_idx_bytes);
      const auto th = g.tensor({3}, syn_type_int32, header_addr[0]);
      const auto tout = rows == 1 ? g.tensor({d, heads, rows}, syn_type_bf16, output_addr)
                                  : g.tensor({d, rows, group, kvh}, syn_type_bf16, output_addr);
      if (rows > 1) g.set_perm(tout, {0, 2, 3, 1});
      ns_GatherKernel::ParamsV2 gather{};
      gather.axis = 1;
      std::array<size_t, 2> kv_tiles{};
      for (unsigned w = 0; w < 2; ++w) {
        const auto gathered = g.tensor({d, kv * kvh}, syn_type_bf16);
        kv_tiles[w] = g.tensor({d, kv, 1, kvh}, syn_type_bf16);
        g.node("gather_fwd_bf16", {tpool, w ? vi : ti}, {gathered}, &gather, sizeof(gather));
        g.node("reshape", {gathered}, {kv_tiles[w]});
      }
      const auto score4 = g.tensor({kv, rows, group, kvh}, syn_type_float);
      const auto score3 = g.tensor({kv, rows, heads}, syn_type_float);
      const auto prob3 = g.tensor({kv, rows, heads}, syn_type_bf16);
      const auto prob4 = g.tensor({kv, rows, group, kvh}, syn_type_bf16);
      const synGEMMParams qk{false, true}, pv{false, false};
      g.node("batch_gemm", {tq, kv_tiles[0]}, {score4}, &qk, sizeof(qk));
      g.node("reshape", {score4}, {score3});
      g.node("gk_attn_direct_softmax_bf16_v1", {score3, tp, th}, {prob3}, &scale, sizeof(scale));
      g.node("reshape", {prob3}, {prob4});
      if (rows == 1) {
        const auto out4 = g.tensor({d, rows, group, kvh}, syn_type_bf16);
        g.node("batch_gemm", {prob4, kv_tiles[1]}, {out4}, &pv, sizeof(pv));
        g.node("reshape", {out4}, {tout});
      } else {
        g.node("batch_gemm", {prob4, kv_tiles[1]}, {tout}, &pv, sizeof(pv));
      }
      g.compile(cache);
      max_workspace = std::max(max_workspace, g.workspace);
      ++compiled_graphs;
      return g;
    };

    using ChunkKey = std::tuple<uint32_t, bool, bool>;
    std::map<ChunkKey, std::unique_ptr<Graph>> chunk_graphs;
    auto get_chunk_graph = [&](uint32_t c_tiles, bool first_c, bool fin_last) -> Graph& {
      const ChunkKey key{c_tiles, first_c, fin_last};
      auto it = chunk_graphs.find(key);
      if (it != chunk_graphs.end()) return *it->second;
      const std::string name = "fixed_tile_chunk_c" + std::to_string(c_tiles) +
          "_f" + std::to_string(first_c ? 1 : 0) + "_e" + std::to_string(fin_last ? 1 : 0) + shape;
      auto ptr = std::make_unique<Graph>(name);
      auto& g = *ptr;
      const auto tq = g.tensor({d, rows, group, kvh}, syn_type_bf16, q_addr);
      if (rows > 1) g.set_perm(tq, {0, 2, 3, 1});
      const auto tpool = g.tensor({d, 2 * max_slots * kvh}, syn_type_bf16, pool_addr);
      const auto tp = g.tensor({rows}, syn_type_int32, pos_addr[0]);
      std::vector<size_t> t_ki(c_tiles), t_vi(c_tiles), t_hdr(c_tiles);
      for (uint32_t t = 0; t < c_tiles; ++t) {
        t_ki[t] = g.tensor({kv * kvh}, syn_type_int32, index_addr[t]);
        t_vi[t] = g.tensor({kv * kvh}, syn_type_int32, index_addr[t] + kv_idx_bytes, t_ki[t], kv_idx_bytes);
        t_hdr[t] = g.tensor({3}, syn_type_int32, header_addr[t]);
      }
      size_t cur_m = 0, cur_l = 0, cur_o = 0;
      if (!first_c) {
        cur_m = g.tensor({stats_rows, heads}, syn_type_float, state[0].maximum);
        cur_l = g.tensor({stats_rows, heads}, syn_type_float, state[0].sum);
        cur_o = g.tensor({d, rows, heads}, syn_type_float, state[0].output);
      }
      size_t tout = 0, out_m = 0, out_l = 0, out_o = 0;
      if (fin_last) {
        tout = g.tensor({d, rows, heads}, syn_type_bf16, output_addr);
        g.set_perm(tout, {0, 2, 1});
      } else {
        out_m = g.tensor({stats_rows, heads}, syn_type_float, state[1].maximum);
        out_l = g.tensor({stats_rows, heads}, syn_type_float, state[1].sum);
        out_o = g.tensor({d, rows, heads}, syn_type_float, state[1].output);
      }
      if (first_c) {
        cur_m = g.tensor({stats_rows, heads}, syn_type_float);
        cur_l = g.tensor({stats_rows, heads}, syn_type_float);
        cur_o = g.tensor({d, rows, heads}, syn_type_float);
        ns_ConstantKernel::ParamsV3 zero{};
        zero.constant.f = 0.f;
        g.node("constant_f32", {}, {cur_m}, &zero, sizeof(zero));
        g.node("constant_f32", {}, {cur_l}, &zero, sizeof(zero));
        g.node("constant_f32", {}, {cur_o}, &zero, sizeof(zero));
      }
      ns_GatherKernel::ParamsV2 gather{};
      gather.axis = 1;
      const synGEMMParams qk{false, true}, pv{false, false};
      for (uint32_t t = 0; t < c_tiles; ++t) {
        std::array<size_t, 2> kv_tiles{};
        for (unsigned w = 0; w < 2; ++w) {
          const auto gathered = g.tensor({d, kv * kvh}, syn_type_bf16);
          kv_tiles[w] = g.tensor({d, kv, 1, kvh}, syn_type_bf16);
          g.node("gather_fwd_bf16", {tpool, w ? t_vi[t] : t_ki[t]}, {gathered}, &gather, sizeof(gather));
          g.node("reshape", {gathered}, {kv_tiles[w]});
        }
        const auto score4 = g.tensor({kv, rows, group, kvh}, syn_type_float);
        const auto score3 = g.tensor({kv, rows, heads}, syn_type_float);
        const auto prob3 = g.tensor({kv, rows, heads}, syn_type_bf16);
        const auto prob4 = g.tensor({kv, rows, group, kvh}, syn_type_bf16);
        const auto tile_m = g.tensor({stats_rows, heads}, syn_type_float);
        const auto tile_l = g.tensor({stats_rows, heads}, syn_type_float);
        const auto tile_o4 = g.tensor({d, rows, group, kvh}, syn_type_float);
        const auto tile_o = g.tensor({d, rows, heads}, syn_type_float);
        g.node("batch_gemm", {tq, kv_tiles[0]}, {score4}, &qk, sizeof(qk));
        g.node("reshape", {score4}, {score3});
        g.node("gk_attn_tile_softmax_bf16_v1", {score3, tp, t_hdr[t]}, {prob3, tile_m, tile_l}, &scale, sizeof(scale));
        g.node("reshape", {prob3}, {prob4});
        g.node("batch_gemm", {prob4, kv_tiles[1]}, {tile_o4}, &pv, sizeof(pv));
        g.node("reshape", {tile_o4}, {tile_o});
        const bool persist_out = (t + 1 == c_tiles) && !fin_last;
        const auto next_m = persist_out ? out_m : g.tensor({stats_rows, heads}, syn_type_float);
        const auto next_l = persist_out ? out_l : g.tensor({stats_rows, heads}, syn_type_float);
        const auto next_o = persist_out ? out_o : g.tensor({d, rows, heads}, syn_type_float);
        g.node("gk_attn_online_merge_f32_v1", {cur_m, cur_l, cur_o, tile_m, tile_l, tile_o},
               {next_m, next_l, next_o});
        cur_m = next_m;
        cur_l = next_l;
        cur_o = next_o;
      }
      if (fin_last) g.node("gk_attn_online_finalize_bf16_v1", {cur_l, cur_o}, {tout});
      g.compile(cache);
      max_workspace = std::max(max_workspace, g.workspace);
      ++compiled_graphs;
      return *chunk_graphs.emplace(key, std::move(ptr)).first->second;
    };

    // Pre-compile all required recipes for the active blocks so compile/cache stats are checked cleanly.
    size_t max_chunk_sections = 0;
    for (uint32_t block = 0; block < query_blocks; ++block) {
      const uint32_t active_tiles =
          std::min(tiles, static_cast<uint32_t>(positions[block * rows + valid - 1]) / kv + 1);
      if (active_tiles == 1) {
        max_chunk_sections = std::max(max_chunk_sections, get_direct_graph().sections.size());
      } else {
        const uint32_t num_chunks = (active_tiles + max_chunk_tiles - 1) / max_chunk_tiles;
        for (uint32_t c = 0; c < num_chunks; ++c) {
          const bool first_c = (c == 0);
          const bool fin_last = (c + 1 == num_chunks);
          const uint32_t c_tiles = fin_last ? (active_tiles - c * max_chunk_tiles) : max_chunk_tiles;
          max_chunk_sections = std::max(max_chunk_sections,
                                        get_chunk_graph(c_tiles, first_c, fin_last).sections.size());
        }
      }
    }

    const auto counts = cache.stats();
    if ((mode == "cold" && counts != std::pair<uint64_t, uint64_t>{0, compiled_graphs}) ||
        (mode == "warm" && counts != std::pair<uint64_t, uint64_t>{compiled_graphs, 0}))
      throw std::runtime_error("unexpected recipe cache counts");
    if (std::getenv("GAUDIRT_COMPILE_ONLY")) {
      std::cout << "compiled graphs " << compiled_graphs << " max_sections " << max_chunk_sections
                << " workspace_bytes " << max_workspace << " cache " << counts.first << '/' << counts.second << '\n';
      return 0;
    }

    const auto workspace = arena.ensure_workspace(max_workspace);
    const auto stream = runtime.compute_stream();
    synEventHandle start_event = nullptr, end_event = nullptr;
    check(synEventCreate(&start_event, runtime.device_id(), EVENT_COLLECT_TIME), "timing start event");
    check(synEventCreate(&end_event, runtime.device_id(), EVENT_COLLECT_TIME), "timing end event");
    uint64_t launches_per_step = 0;
    const auto enqueue = [&] {
      launches_per_step = 1;
      check(synLaunch(stream, store.bindings.data(), store.bindings.size(), workspace, store.recipe, 0), "KV launch");
      for (uint32_t block = 0; block < query_blocks; ++block) {
        const uint64_t block_q = q_addr + uint64_t{block} * rows * heads * d * 2;
        const uint64_t block_out = output_addr + uint64_t{block} * rows * heads * d * 2;
        const uint32_t active_tiles =
            std::min(tiles, static_cast<uint32_t>(positions[block * rows + valid - 1]) / kv + 1);
        if (active_tiles == 1) {
          auto& g = get_direct_graph();
          g.bindings[0].pTensorAddress = block_q;
          g.bindings[1].pTensorAddress = pool_addr;
          g.bindings[2].pTensorAddress = pos_addr[block];
          g.bindings[3].pTensorAddress = index_addr[0];
          g.bindings[4].pTensorAddress = index_addr[0] + kv_idx_bytes;
          g.bindings[5].pTensorAddress = header_addr[block * tiles + 0];
          g.bindings[6].pTensorAddress = block_out;
          check(synLaunch(stream, g.bindings.data(), g.bindings.size(), workspace, g.recipe, 0), "direct launch");
          ++launches_per_step;
          continue;
        }
        const uint32_t num_chunks = (active_tiles + max_chunk_tiles - 1) / max_chunk_tiles;
        for (uint32_t c = 0; c < num_chunks; ++c) {
          const uint32_t tile_off = c * max_chunk_tiles;
          const bool first_c = (c == 0);
          const bool fin_last = (c + 1 == num_chunks);
          const uint32_t c_tiles = fin_last ? (active_tiles - tile_off) : max_chunk_tiles;
          auto& g = get_chunk_graph(c_tiles, first_c, fin_last);
          size_t b = 0;
          g.bindings[b++].pTensorAddress = block_q;
          g.bindings[b++].pTensorAddress = pool_addr;
          g.bindings[b++].pTensorAddress = pos_addr[block];
          for (uint32_t t = 0; t < c_tiles; ++t) {
            const uint32_t tid = tile_off + t;
            g.bindings[b++].pTensorAddress = index_addr[tid];
            g.bindings[b++].pTensorAddress = index_addr[tid] + kv_idx_bytes;
            g.bindings[b++].pTensorAddress = header_addr[block * tiles + tid];
          }
          if (!first_c) {
            const auto& in_s = state[(c - 1) % 2];
            g.bindings[b++].pTensorAddress = in_s.maximum;
            g.bindings[b++].pTensorAddress = in_s.sum;
            g.bindings[b++].pTensorAddress = in_s.output;
          }
          if (fin_last) {
            g.bindings[b++].pTensorAddress = block_out;
          } else {
            const auto& out_s = state[c % 2];
            g.bindings[b++].pTensorAddress = out_s.maximum;
            g.bindings[b++].pTensorAddress = out_s.sum;
            g.bindings[b++].pTensorAddress = out_s.output;
          }
          check(synLaunch(stream, g.bindings.data(), g.bindings.size(), workspace, g.recipe, 0), "chunk launch");
          ++launches_per_step;
        }
      }
    };

    enqueue();
    runtime.synchronize_compute();
    const auto host_start = std::chrono::steady_clock::now();
    check(synEventRecord(start_event, stream), "timing start");
    enqueue();
    check(synEventRecord(end_event, stream), "timing end");
    const auto host_enqueued = std::chrono::steady_clock::now();
    check(synEventSynchronize(end_event), "timing completion");
    const auto host_done = std::chrono::steady_clock::now();
    uint64_t elapsed_ns = 0;
    check(synEventElapsedTime(&elapsed_ns, start_event, end_event), "elapsed time");
    check(synEventDestroy(start_event), "destroy start event");
    check(synEventDestroy(end_event), "destroy end event");

    std::vector<uint16_t> actual_pool(physical_pool.size());
    std::vector<uint16_t> output(uint64_t{d} * total_rows * heads);
    arena.copy_d2h_sync(runtime.d2h_stream(), pool_addr, actual_pool.data(), actual_pool.size() * 2);
    arena.copy_d2h_sync(runtime.d2h_stream(), output_addr, output.data(), output.size() * 2);
    if (std::any_of(output.begin(), output.end(), [](uint16_t value) { return !std::isfinite(widen(value)); }))
      throw std::runtime_error("nonfinite attention output");
    if (actual_pool != expected_physical_pool)
      throw std::runtime_error("KV write corrupted a target or untouched slot");
    std::array<uint16_t, 128> observed{};
    for (auto address : {guard_before_pool, guard_after_pool}) {
      arena.copy_d2h_sync(runtime.d2h_stream(), address, observed.data(), sizeof(observed));
      if (observed != guard) throw std::runtime_error("KV guard changed");
    }

    double max_fp64 = 0, max_tiled = 0;
    std::vector<uint32_t> checked_rows;
    if (query_blocks == 1) {
      for (uint32_t row = 0; row < valid; ++row) checked_rows.push_back(row);
    } else {
      checked_rows = {0, total_rows / 4, total_rows / 2, (3 * total_rows) / 4, total_rows - 1};
    }
    const auto visible = [&](uint32_t row, uint32_t tile) {
      const int64_t count = std::min<int64_t>(tile_valid_keys(tile),
                                              int64_t{positions[row]} - int64_t{tile} * kv + 1);
      return static_cast<uint32_t>(std::clamp<int64_t>(count, 0, kv));
    };
    for (uint32_t row : checked_rows) {
      double err64 = 0, norm64 = 0, err_tiled = 0, norm_tiled = 0;
      for (uint32_t head = 0; head < heads; ++head) {
        std::vector<double> scores64(tiles * kv);
        std::vector<float> scores32(tiles * kv), tile_max(tiles, -std::numeric_limits<float>::infinity());
        double max64 = -std::numeric_limits<double>::infinity();
        for (uint32_t tile = 0; tile < tiles; ++tile) {
          const uint32_t count = visible(row, tile);
          for (uint32_t t = 0; t < count; ++t) {
            double x64 = 0;
            float x32 = 0;
            for (uint32_t col = 0; col < d; ++col) {
              const float x = widen(q[(static_cast<size_t>(row) * heads + head) * d + col]);
              const float y = widen(expected_key[static_cast<size_t>(indices[tile][t]) * width + (head / group) * d + col]);
              x64 += static_cast<double>(x) * y;
              x32 += x * y;
            }
            scores64[tile * kv + t] = x64 * scale;
            scores32[tile * kv + t] = x32 * scale;
            max64 = std::max(max64, scores64[tile * kv + t]);
            tile_max[tile] = std::max(tile_max[tile], scores32[tile * kv + t]);
          }
        }
        double den64 = 0;
        std::vector<double> p64(tiles * kv);
        std::vector<float> p32(tiles * kv), tile_sum(tiles);
        for (uint32_t tile = 0; tile < tiles; ++tile) {
          const uint32_t count = visible(row, tile);
          for (uint32_t t = 0; t < count; ++t) {
            const auto i = tile * kv + t;
            p64[i] = std::exp(scores64[i] - max64);
            p32[i] = std::exp(scores32[i] - tile_max[tile]);
            den64 += p64[i];
            tile_sum[tile] += p32[i];
          }
        }
        for (uint32_t col = 0; col < d; ++col) {
          double out64 = 0;
          float m = 0, sum = 0, acc = 0;
          for (uint32_t tile = 0; tile < tiles; ++tile) {
            const uint32_t count = visible(row, tile);
            if (!count) continue;
            float tile_acc = 0;
            for (uint32_t t = 0; t < count; ++t) {
              const float v = widen(expected_value[static_cast<size_t>(indices[tile][t]) * width + (head / group) * d + col]);
              out64 += p64[tile * kv + t] * v;
              tile_acc += widen(bf16(p32[tile * kv + t] / tile_sum[tile])) * v;
            }
            const float next_m = sum > 0 ? std::max(m, tile_max[tile]) : tile_max[tile];
            const float a = sum > 0 ? std::exp(m - next_m) : 0;
            const float b = std::exp(tile_max[tile] - next_m);
            sum = a * sum + b * tile_sum[tile];
            acc = a * acc + b * tile_sum[tile] * tile_acc;
            m = next_m;
          }
          const double truth = widen(direct_bf16(out64 / den64));
          const double tiled = widen(bf16(acc / sum));
          const double actual = widen(output[(static_cast<size_t>(row) * heads + head) * d + col]);
          if (!std::isfinite(actual)) throw std::runtime_error("nonfinite attention output");
          err64 += (actual - truth) * (actual - truth);
          norm64 += truth * truth;
          err_tiled += (actual - tiled) * (actual - tiled);
          norm_tiled += tiled * tiled;
        }
      }
      max_fp64 = std::max(max_fp64, std::sqrt(err64 / norm64));
      max_tiled = std::max(max_tiled, std::sqrt(err_tiled / norm_tiled));
    }
    if (query_blocks == 1)
      for (uint32_t row = valid; row < rows; ++row)
        for (uint32_t i = 0; i < heads * d; ++i)
          if (output[static_cast<size_t>(row) * heads * d + i] != 0)
            throw std::runtime_error("invalid query row was not zeroed");
    if (max_fp64 > 0.01 || max_tiled > 0.001) throw std::runtime_error("attention numerical gate");
    std::ofstream bytes(argv[3], std::ios::binary);
    bytes.write(reinterpret_cast<const char*>(output.data()), output.size() * 2);
    if (!bytes) throw std::runtime_error("output write failed");
    std::cout << "slots " << slots << " page " << page << " tiles " << tiles
              << " query_blocks " << query_blocks << " max_chunk_tiles " << max_chunk_tiles
              << " launches " << launches_per_step << " reference_rows " << checked_rows.size()
              << " max_sections " << max_chunk_sections << " workspace_bytes " << max_workspace
              << " fp64 " << max_fp64 << " tiled " << max_tiled << " pool_mismatches 0 cache "
              << counts.first << '/' << counts.second << " device_ms " << elapsed_ns / 1e6
              << " host_enqueue_ms " << std::chrono::duration<double, std::milli>(host_enqueued - host_start).count()
              << " host_wall_ms " << std::chrono::duration<double, std::milli>(host_done - host_start).count() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
