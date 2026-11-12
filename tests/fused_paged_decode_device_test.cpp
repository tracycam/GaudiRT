// Fused single-bucket paged attention, with CPU references and whole-pool checks.
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
    if (argc < 10 || argc > 13)
      throw std::invalid_argument("expected cache mode output d kv_bucket q_heads kv_heads slots page_size "
                                  "[effective_kv [rows [valid_rows]]]");
    const std::string mode = argv[2];
    const uint32_t d = std::stoul(argv[4]), kv = std::stoul(argv[5]);
    const uint32_t heads = std::stoul(argv[6]), kvh = std::stoul(argv[7]);
    const uint32_t slots = std::stoul(argv[8]), page = std::stoul(argv[9]);
    const uint32_t effective_kv = argc >= 11 ? std::stoul(argv[10]) : kv;
    const uint32_t rows = argc >= 12 ? std::stoul(argv[11]) : 1;
    const uint32_t valid = argc >= 13 ? std::stoul(argv[12]) : rows;
    constexpr uint32_t tiles = 1, query_blocks = 1;
    // Only the runtime-sized K/V suballocation is populated. Its wider fixed
    // descriptor stays inside the preallocated slab and the other graph I/O
    // is placed before it; this is not a service allocator test.
    constexpr uint32_t max_slots = 131072;
    if (!effective_kv || effective_kv > kv || effective_kv < valid || (rows > 1 && kv > 512) ||
        (mode != "cold" && mode != "warm") || (d != 64 && d != 128) || !rows || rows > 256 ||
        !valid || valid > rows || !kv || kv > 8192 || kv % 128 || !kvh || heads % kvh || heads > 256 ||
        uint64_t{kv} * rows * heads > 4ULL * 1024 * 1024 || kvh * d % 128 ||
        slots < 2048 || slots > max_slots || (page != 1 && page != 16) || slots % page ||
        uint64_t{effective_kv} + page >= slots)
      throw std::invalid_argument("unsupported geometry");
    const uint32_t group = heads / kvh, width = kvh * d, total_rows = rows * query_blocks;
    const float scale = 1.f / std::sqrt(static_cast<float>(d));
    std::mt19937 generator(20261106);
    std::normal_distribution<float> normal(0.f, 0.6f);
    std::vector<uint16_t> q(uint64_t{d} * total_rows * heads), key_new(uint64_t{width} * total_rows),
        value_new(uint64_t{width} * total_rows);
    for (auto& x : q) x = bf16(normal(generator));
    for (auto& x : key_new) x = bf16(normal(generator));
    for (auto& x : value_new) x = bf16(normal(generator));
    std::vector<std::vector<int32_t>> indices(tiles, std::vector<int32_t>(kv));
    for (uint32_t i = 0; i < effective_kv; ++i) {
      const uint32_t slot = page == 1 ? 1 + (i * 5) % (slots - 1) :
          (1 + ((i / page) * 5) % (slots / page - 1)) * page + i % page;
      indices[i / kv][i % kv] = static_cast<int32_t>(slot);
    }
    // Include the final physical page while retaining page-contiguous slots.
    const uint32_t target_page = ((effective_kv - 1) / page) * page;
    auto last_page = std::find(indices[0].begin(), indices[0].end(), static_cast<int32_t>(slots - page));
    if (last_page == indices[0].end()) {
      for (uint32_t i = 0; i < page; ++i) indices[0][target_page + i] = slots - page + i;
    } else {
      const auto old_page = static_cast<uint32_t>(last_page - indices[0].begin());
      for (uint32_t i = 0; i < page; ++i) std::swap(indices[0][target_page + i], indices[0][old_page + i]);
    }
    std::vector<uint16_t> key_pool(static_cast<size_t>(slots) * width, 0x3e80);
    std::vector<uint16_t> value_pool(static_cast<size_t>(slots) * width, 0x3f00);
    for (uint32_t i = 0; i < effective_kv; ++i) {
      const auto slot = indices[i / kv][i % kv];
      for (uint32_t col = 0; col < width; ++col) {
        key_pool[static_cast<size_t>(slot) * width + col] = bf16(normal(generator));
        value_pool[static_cast<size_t>(slot) * width + col] = bf16(normal(generator));
      }
    }
    std::vector<int32_t> positions(total_rows), locations(total_rows, slots);
    const uint32_t prefix = effective_kv - valid;
    const uint32_t total_valid = valid;
    for (uint32_t row = 0; row < total_rows; ++row) {
      positions[row] = static_cast<int32_t>(prefix + row);
      if (row < total_valid) locations[row] = indices[positions[row] / kv][positions[row] % kv];
    }
    std::vector<uint16_t> physical_pool(uint64_t{2} * slots * width, 0);
    for (uint32_t head = 0; head < kvh; ++head)
      for (uint32_t slot = 0; slot < slots; ++slot)
        for (uint32_t col = 0; col < d; ++col) {
          physical_pool[(uint64_t{head} * slots + slot) * d + col] = key_pool[uint64_t{slot} * width + head * d + col];
          physical_pool[(uint64_t{kvh + head} * slots + slot) * d + col] = value_pool[uint64_t{slot} * width + head * d + col];
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
          auto physical = (uint64_t{head} * slots + locations[row]) * d + col;
          expected_physical_pool[physical] = key_new[uint64_t{row} * width + head * d + col];
          expected_physical_pool[uint64_t{slots} * width + physical] = value_new[uint64_t{row} * width + head * d + col];
        }
    }

    DeviceRuntime runtime({});
    gaudirt::testing::TestDeviceSlab arena(runtime.device_id(), 832ULL * 1024 * 1024);
    RecipeCache cache(argv[1]);
    const auto q_addr = arena.allocate(q.size() * 2);
    const auto key_new_addr = arena.allocate(key_new.size() * 2);
    const auto value_new_addr = arena.allocate(value_new.size() * 2);
    const auto locations_addr = arena.allocate(locations.size() * 4);
    const auto valid_addr = arena.allocate(12);
    const uint64_t header_offset = (uint64_t{rows} * 4 + 255) & ~uint64_t{255};
    const uint64_t metadata_bytes = header_offset + 256;
    const uint64_t index_bytes = uint64_t{2} * tiles * kv * kvh * 4;
    const uint64_t metadata_total_bytes = uint64_t{query_blocks} * tiles * metadata_bytes;
    const auto metadata_base = arena.allocate(index_bytes + metadata_total_bytes);
    std::vector<uint64_t> index_addr(tiles), metadata_addr(query_blocks * tiles);
    for (uint32_t tile = 0; tile < tiles; ++tile) index_addr[tile] = metadata_base + uint64_t{2} * tile * kv * kvh * 4;
    for (uint32_t block = 0; block < query_blocks; ++block)
      for (uint32_t tile = 0; tile < tiles; ++tile)
        metadata_addr[block * tiles + tile] = metadata_base + index_bytes +
            uint64_t{block * tiles + tile} * metadata_bytes;
    const auto output_addr = arena.allocate(uint64_t{d} * total_rows * heads * 2);
    const auto guard_before_pool = arena.allocate(256);
    const auto pool_addr = arena.allocate(physical_pool.size() * 2);
    const auto guard_after_pool = arena.allocate(256);
    const uint64_t descriptor_bytes = uint64_t{2} * max_slots * width * 2;
    if (pool_addr < arena.base_addr() ||
        descriptor_bytes > arena.capacity_bytes() - (pool_addr - arena.base_addr()))
      throw std::runtime_error("fixed descriptor exceeds allocated slab");

    auto h2d = [&](const void* source, uint64_t dest, uint64_t bytes) {
      arena.copy_h2d_sync(runtime.h2d_stream(), source, dest, bytes);
    };
    h2d(q.data(), q_addr, q.size() * 2);
    h2d(key_new.data(), key_new_addr, key_new.size() * 2);
    h2d(value_new.data(), value_new_addr, value_new.size() * 2);
    h2d(locations.data(), locations_addr, locations.size() * 4);
    const uint32_t valid_and_slots[] = {total_valid, slots, kvh};
    h2d(valid_and_slots, valid_addr, sizeof(valid_and_slots));
    h2d(physical_pool.data(), pool_addr, physical_pool.size() * 2);
    std::array<uint16_t, 128> guard{};
    guard.fill(0x4110);
    for (auto address : {guard_before_pool, guard_after_pool})
      h2d(guard.data(), address, sizeof(guard));
    std::vector<int32_t> packed_metadata((index_bytes + metadata_total_bytes) / 4);
    for (uint32_t tile = 0; tile < tiles; ++tile)
      for (uint32_t head = 0; head < kvh; ++head)
        for (uint32_t i = 0; i < kv; ++i) {
            packed_metadata[(uint64_t{2 * tile} * kvh + head) * kv + i] = head * slots + indices[tile][i];
            packed_metadata[(uint64_t{2 * tile + 1} * kvh + head) * kv + i] = (kvh + head) * slots + indices[tile][i];
          }
    for (uint32_t block = 0; block < query_blocks; ++block) {
      for (uint32_t tile = 0; tile < tiles; ++tile) {
        const auto offset = (index_bytes + uint64_t{block * tiles + tile} * metadata_bytes) / 4;
        std::copy_n(positions.begin() + block * rows, rows, packed_metadata.begin() + offset);
        packed_metadata[offset + header_offset / 4] = valid;
        packed_metadata[offset + header_offset / 4 + 1] = effective_kv;
        packed_metadata[offset + header_offset / 4 + 2] = tile * kv;
      }
    }
    h2d(packed_metadata.data(), metadata_base, index_bytes + metadata_total_bytes);
    const std::string shape = "_d" + std::to_string(d) + "_q" + std::to_string(rows) + "_k" +
        std::to_string(kv) + "_h" + std::to_string(heads) + "_kh" + std::to_string(kvh);
    Graph fused("fused_decode" + shape);
    const auto tq = fused.tensor({d, rows, group, kvh}, syn_type_bf16, q_addr);
    synTensorPermutation head_major{};
    head_major.dims = 4;
    head_major.permutation[0] = 0;
    head_major.permutation[1] = 2;
    head_major.permutation[2] = 3;
    head_major.permutation[3] = 1;
    if (rows > 1) check(synTensorSetPermutation(fused.tensors[tq], &head_major), "Q memory permutation");
    const auto sk = fused.tensor({width, total_rows}, syn_type_bf16, key_new_addr);
    const auto sv = fused.tensor({width, total_rows}, syn_type_bf16, value_new_addr);
    const auto sl = fused.tensor({total_rows}, syn_type_int32, locations_addr);
    const auto sh = fused.tensor({3}, syn_type_int32, valid_addr);
    const auto old_pool = fused.tensor({d, 2 * max_slots * kvh}, syn_type_bf16, pool_addr);
    const auto pool = fused.tensor({d, 2 * max_slots * kvh}, syn_type_bf16, pool_addr, old_pool);
    const auto ti = fused.tensor({kv * kvh}, syn_type_int32, index_addr[0]);
    const auto vi = fused.tensor({kv * kvh}, syn_type_int32, index_addr[0] + uint64_t{kv} * kvh * 4, ti, uint64_t{kv} * kvh * 4);
    const auto tp = fused.tensor({rows}, syn_type_int32, metadata_addr[0]);
    const auto th = fused.tensor({3}, syn_type_int32, metadata_addr[0] + header_offset, tp, header_offset);
    const auto result = rows == 1 ? fused.tensor({d, heads, rows}, syn_type_bf16, output_addr) :
                                    fused.tensor({d, rows, group, kvh}, syn_type_bf16, output_addr);
    if (rows > 1) check(synTensorSetPermutation(fused.tensors[result], &head_major), "output memory permutation");
    fused.node("gk_kv_store_head_major_bf16_v1", {sk, sv, sl, sh, old_pool}, {pool});
    ns_GatherKernel::ParamsV2 gather{}; gather.axis = 1;
    std::array<size_t,2> cache_tiles{};
    for (unsigned which=0; which<2; ++which) {
      const auto gathered=fused.tensor({d,kv * kvh}, syn_type_bf16);
      cache_tiles[which]=fused.tensor({d,kv,1,kvh},syn_type_bf16);
      fused.node("gather_fwd_bf16", {pool, which ? vi : ti}, {gathered}, &gather, sizeof(gather));
      fused.node("reshape", {gathered}, {cache_tiles[which]});
    }
    const auto score4=fused.tensor({kv,rows,group,kvh},syn_type_float);
    const auto score3=fused.tensor({kv,rows,heads},syn_type_float);
    const auto probability3=fused.tensor({kv,rows,heads},syn_type_bf16);
    const auto probability4=fused.tensor({kv,rows,group,kvh},syn_type_bf16);
    const synGEMMParams qk{false,true}, pv{false,false};
    fused.node("batch_gemm",{tq,cache_tiles[0]},{score4},&qk,sizeof(qk));
    fused.node("reshape",{score4},{score3});
    if (kv <= 512) {
      fused.node("gk_attn_direct_softmax_bf16_v1",{score3,tp,th},{probability3},&scale,sizeof(scale));
    } else {
      const auto exponent=fused.tensor({kv,rows,heads},syn_type_float);
      const auto sum=fused.tensor({64,heads},syn_type_float);
      fused.node("gk_attn_decode_exp_f32_v1",{score3,tp,th},{exponent,sum},&scale,sizeof(scale));
      fused.node("gk_attn_decode_normalize_bf16_v1",{exponent,sum},{probability3});
    }
    fused.node("reshape",{probability3},{probability4});
    if (rows == 1) {
      const auto output4=fused.tensor({d,rows,group,kvh},syn_type_bf16);
      fused.node("batch_gemm",{probability4,cache_tiles[1]},{output4},&pv,sizeof(pv));
      fused.node("reshape",{output4},{result});
    } else {
      fused.node("batch_gemm",{probability4,cache_tiles[1]},{result},&pv,sizeof(pv));
    }
    fused.compile(cache);
    if (fused.sections.size() >= 12) throw std::runtime_error("fused graph exceeds section budget");
    const auto counts=cache.stats();
    if ((mode == "cold" && counts != std::pair<uint64_t,uint64_t>{0,1}) ||
        (mode == "warm" && counts != std::pair<uint64_t,uint64_t>{1,0}))
      throw std::runtime_error("unexpected recipe cache counts");
    if (std::getenv("GAUDIRT_COMPILE_ONLY")) {
      std::cout << "compiled fused sections " << fused.sections.size() << " cache "
                << counts.first << '/' << counts.second << '\n';
      return 0;
    }
    const auto workspace=arena.ensure_workspace(fused.workspace);
    const auto stream = runtime.compute_stream();
    synEventHandle start_event = nullptr, end_event = nullptr;
    check(synEventCreate(&start_event, runtime.device_id(), EVENT_COLLECT_TIME), "timing start event");
    check(synEventCreate(&end_event, runtime.device_id(), EVENT_COLLECT_TIME), "timing end event");
    const auto enqueue = [&] {
      check(synLaunch(stream,fused.bindings.data(),fused.bindings.size(),workspace,fused.recipe,0), "fused launch");
    };
    enqueue();
    runtime.synchronize_compute();
    const auto host_start = std::chrono::steady_clock::now();
    check(synEventRecord(start_event, stream), "timing start");
    constexpr int repetitions = 100;
    for (int iteration=0; iteration<repetitions; ++iteration) enqueue();
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
    if (actual_pool != expected_physical_pool)
      throw std::runtime_error("KV write corrupted a target or untouched slot");
    if (std::any_of(output.begin(), output.end(), [](uint16_t value) { return !std::isfinite(widen(value)); }))
      throw std::runtime_error("nonfinite attention output");
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
      checked_rows = {0, total_rows / 2, total_rows - 1};
    }
    const auto visible = [&](uint32_t row, uint32_t tile) {
      const int64_t count = int64_t{positions[row]} - int64_t{tile} * kv + 1;
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
    std::cout << "rows " << rows << " valid_rows " << valid
              << " logical_slots " << slots << " physical_slots " << slots
              << " page " << page << " effective_kv " << effective_kv << " tiles " << tiles
              << " query_blocks " << query_blocks << " reference_rows " << checked_rows.size() << " sections "
              << fused.sections.size()
              << " fp64 " << max_fp64 << " tiled " << max_tiled << " pool_mismatches 0 cache "
              << counts.first << '/' << counts.second << " device_ms " << elapsed_ns / 1e6 / repetitions
              << " host_enqueue_ms " << std::chrono::duration<double, std::milli>(host_enqueued - host_start).count() / repetitions
              << " host_wall_ms " << std::chrono::duration<double, std::milli>(host_done - host_start).count() / repetitions << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
