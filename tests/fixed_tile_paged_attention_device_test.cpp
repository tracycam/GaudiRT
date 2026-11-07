// KV write, one reusable MME KV-tile graph, and online-state finalization.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

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

struct State {
  uint64_t maximum, sum, output;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 13 && argc != 14)
      throw std::invalid_argument("expected cache mode output d q_bucket valid_rows kv_tile q_heads kv_heads slots page_size tiles [query_blocks]");
    const std::string mode = argv[2];
    const uint32_t d = std::stoul(argv[4]), rows = std::stoul(argv[5]), valid = std::stoul(argv[6]);
    const uint32_t kv = std::stoul(argv[7]), heads = std::stoul(argv[8]), kvh = std::stoul(argv[9]);
    const uint32_t slots = std::stoul(argv[10]), page = std::stoul(argv[11]), tiles = std::stoul(argv[12]);
    const uint32_t query_blocks = argc == 14 ? std::stoul(argv[13]) : 1;
    constexpr uint32_t max_slots = 131072;
    if ((mode != "cold" && mode != "warm") || (d != 64 && d != 128) || !rows || rows > 256 ||
        !valid || valid > rows || !kv || kv > 2048 || kv % 128 || !kvh || heads % kvh || heads > 256 ||
        uint64_t{kv} * rows * heads > 4ULL * 1024 * 1024 || kvh * d % 128 ||
        slots < 2048 || slots > max_slots || (page != 1 && page != 16) || slots % page ||
        !tiles || uint64_t{tiles} * kv + page >= slots || valid > kv - 7 || !query_blocks ||
        uint64_t{query_blocks} * rows > 4096 || (query_blocks > 1 &&
        (valid != rows || uint64_t{query_blocks} * rows > uint64_t{tiles} * kv)))
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
    std::vector<int32_t> positions(total_rows), locations(total_rows, slots);
    const uint32_t last_tile_row_start = rows == 1 ? kv - 1 : 0;
    const uint32_t prefix = query_blocks > 1 ? tiles * kv - total_rows : (tiles - 1) * kv + last_tile_row_start;
    const uint32_t total_valid = query_blocks > 1 ? total_rows : valid;
    for (uint32_t row = 0; row < total_rows; ++row) {
      positions[row] = static_cast<int32_t>(prefix + row);
      if (row < total_valid) locations[row] = indices[positions[row] / kv][positions[row] % kv];
    }
    auto expected_key = key_pool, expected_value = value_pool;
    for (uint32_t row = 0; row < total_valid; ++row) {
      std::copy_n(key_new.begin() + static_cast<size_t>(row) * width, width,
                  expected_key.begin() + static_cast<size_t>(locations[row]) * width);
      std::copy_n(value_new.begin() + static_cast<size_t>(row) * width, width,
                  expected_value.begin() + static_cast<size_t>(locations[row]) * width);
    }

    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 832ULL * 1024 * 1024);
    RecipeCache cache(argv[1]);
    const auto q_addr = arena.allocate(q.size() * 2);
    const auto key_new_addr = arena.allocate(key_new.size() * 2);
    const auto value_new_addr = arena.allocate(value_new.size() * 2);
    const auto locations_addr = arena.allocate(locations.size() * 4);
    const auto valid_addr = arena.allocate(4);
    const uint64_t header_offset = (uint64_t{rows} * 4 + 255) & ~uint64_t{255};
    const uint64_t metadata_bytes = header_offset + 256;
    const uint64_t index_bytes = uint64_t{tiles} * kv * 4;
    const uint64_t metadata_total_bytes = uint64_t{query_blocks} * tiles * metadata_bytes;
    const auto metadata_base = arena.allocate(index_bytes + metadata_total_bytes);
    std::vector<uint64_t> index_addr(tiles), metadata_addr(query_blocks * tiles);
    for (uint32_t tile = 0; tile < tiles; ++tile) index_addr[tile] = metadata_base + uint64_t{tile} * kv * 4;
    for (uint32_t block = 0; block < query_blocks; ++block)
      for (uint32_t tile = 0; tile < tiles; ++tile)
        metadata_addr[block * tiles + tile] = metadata_base + index_bytes +
            uint64_t{block * tiles + tile} * metadata_bytes;
    std::array<State, 2> state{};
    for (auto& s : state) {
      s.maximum = arena.allocate(rows * heads * 4);
      s.sum = arena.allocate(rows * heads * 4);
      s.output = arena.allocate(uint64_t{d} * rows * heads * 4);
    }
    const auto output_addr = arena.allocate(uint64_t{d} * total_rows * heads * 2);
    const auto guard_before_key = arena.allocate(256);
    const auto key_addr = arena.allocate(key_pool.size() * 2);
    const auto guard_after_key = arena.allocate(256);
    arena.allocate(uint64_t{max_slots - slots} * width * 2);
    const auto guard_before_value = arena.allocate(256);
    const auto value_addr = arena.allocate(value_pool.size() * 2);
    const auto guard_after_value = arena.allocate(256);
    arena.allocate(uint64_t{max_slots - slots} * width * 2);

    auto h2d = [&](const void* source, uint64_t dest, uint64_t bytes) {
      arena.copy_h2d_sync(runtime.h2d_stream(), source, dest, bytes);
    };
    h2d(q.data(), q_addr, q.size() * 2);
    h2d(key_new.data(), key_new_addr, key_new.size() * 2);
    h2d(value_new.data(), value_new_addr, value_new.size() * 2);
    h2d(locations.data(), locations_addr, locations.size() * 4);
    h2d(&total_valid, valid_addr, 4);
    h2d(key_pool.data(), key_addr, key_pool.size() * 2);
    h2d(value_pool.data(), value_addr, value_pool.size() * 2);
    std::array<uint16_t, 128> guard{};
    guard.fill(0x4110);
    for (auto address : {guard_before_key, guard_after_key, guard_before_value, guard_after_value})
      h2d(guard.data(), address, sizeof(guard));
    std::vector<int32_t> packed_metadata((index_bytes + metadata_total_bytes) / 4);
    for (uint32_t tile = 0; tile < tiles; ++tile)
      std::copy(indices[tile].begin(), indices[tile].end(), packed_metadata.begin() + uint64_t{tile} * kv);
    for (uint32_t block = 0; block < query_blocks; ++block) {
      for (uint32_t tile = 0; tile < tiles; ++tile) {
        const auto offset = (index_bytes + uint64_t{block * tiles + tile} * metadata_bytes) / 4;
        std::copy_n(positions.begin() + block * rows, rows, packed_metadata.begin() + offset);
        packed_metadata[offset + header_offset / 4] = valid;
        packed_metadata[offset + header_offset / 4 + 1] =
            query_blocks == 1 && tile + 1 == tiles && rows > 1 ? kv - 7 : kv;
        packed_metadata[offset + header_offset / 4 + 2] = tile * kv;
      }
    }
    h2d(packed_metadata.data(), metadata_base, index_bytes + metadata_total_bytes);
    const std::string shape = "_d" + std::to_string(d) + "_q" + std::to_string(rows) + "_k" +
        std::to_string(kv) + "_h" + std::to_string(heads) + "_kh" + std::to_string(kvh);
    Graph store("fixed_tile_store_m" + std::to_string(total_rows) + shape);
    const auto sk = store.tensor({width, total_rows}, syn_type_bf16, key_new_addr);
    const auto sv = store.tensor({width, total_rows}, syn_type_bf16, value_new_addr);
    const auto sl = store.tensor({total_rows}, syn_type_int32, locations_addr);
    const auto sh = store.tensor({1}, syn_type_int32, valid_addr);
    const auto spk = store.tensor({width, max_slots}, syn_type_bf16, key_addr);
    const auto spv = store.tensor({width, max_slots}, syn_type_bf16, value_addr);
    store.node("gk_kv_store_bf16_v1", {sk, sv, sl, sh}, {spk, spv});
    store.compile(cache);
    if (store.sections.size() != 6) throw std::runtime_error("unexpected store section count");

    Graph step("fixed_tile_step" + shape);
    const auto tq_in = step.tensor({d, group, kvh, rows}, syn_type_bf16, q_addr);
    const auto tp = step.tensor({rows}, syn_type_int32, metadata_addr[0]);
    const auto tkpool = step.tensor({width, max_slots}, syn_type_bf16, key_addr);
    const auto tvpool = step.tensor({width, max_slots}, syn_type_bf16, value_addr);
    const auto ti = step.tensor({kv}, syn_type_int32, index_addr[0]);
    const auto th = step.tensor({3}, syn_type_int32, metadata_addr[0] + header_offset, tp, header_offset);
    const auto old_m = step.tensor({rows, heads}, syn_type_float, state[0].maximum);
    const auto old_l = step.tensor({rows, heads}, syn_type_float, state[0].sum);
    const auto old_o = step.tensor({d, rows, heads}, syn_type_float, state[0].output);
    const auto new_m = step.tensor({rows, heads}, syn_type_float, state[1].maximum);
    const auto new_l = step.tensor({rows, heads}, syn_type_float, state[1].sum);
    const auto new_o = step.tensor({d, rows, heads}, syn_type_float, state[1].output);
    const auto tq = step.tensor({d, rows, group, kvh}, syn_type_bf16);
    synTransposeParams q_transpose{};
    q_transpose.tensorDim = 4;
    q_transpose.permutation[0] = static_cast<TransposePermutationDim>(0);
    q_transpose.permutation[1] = static_cast<TransposePermutationDim>(3);
    q_transpose.permutation[2] = static_cast<TransposePermutationDim>(1);
    q_transpose.permutation[3] = static_cast<TransposePermutationDim>(2);
    step.node("transpose", {tq_in}, {tq}, &q_transpose, sizeof(q_transpose));
    ns_GatherKernel::ParamsV2 gather{};
    gather.axis = 1;
    synTransposeParams transpose{};
    transpose.tensorDim = 3;
    transpose.permutation[0] = static_cast<TransposePermutationDim>(0);
    transpose.permutation[1] = static_cast<TransposePermutationDim>(2);
    transpose.permutation[2] = static_cast<TransposePermutationDim>(1);
    std::array<size_t, 2> cache_tiles{};
    for (unsigned which = 0; which < 2; ++which) {
      const auto gathered = step.tensor({width, kv}, syn_type_bf16);
      const auto group_first = step.tensor({d, kvh, kv}, syn_type_bf16);
      const auto row_first = step.tensor({d, kv, kvh}, syn_type_bf16);
      cache_tiles[which] = step.tensor({d, kv, 1, kvh}, syn_type_bf16);
      step.node("gather_fwd_bf16", {which ? tvpool : tkpool, ti}, {gathered}, &gather, sizeof(gather));
      step.node("reshape", {gathered}, {group_first});
      step.node("transpose", {group_first}, {row_first}, &transpose, sizeof(transpose));
      step.node("reshape", {row_first}, {cache_tiles[which]});
    }
    const auto score4 = step.tensor({kv, rows, group, kvh}, syn_type_float);
    const auto score3 = step.tensor({kv, rows, heads}, syn_type_float);
    const auto probability3 = step.tensor({kv, rows, heads}, syn_type_bf16);
    const auto probability4 = step.tensor({kv, rows, group, kvh}, syn_type_bf16);
    const auto tile_m = step.tensor({rows, heads}, syn_type_float);
    const auto tile_l = step.tensor({rows, heads}, syn_type_float);
    const auto tile_o4 = step.tensor({d, rows, group, kvh}, syn_type_float);
    const auto tile_o = step.tensor({d, rows, heads}, syn_type_float);
    const synGEMMParams qk{false, true}, pv{false, false};
    step.node("batch_gemm", {tq, cache_tiles[0]}, {score4}, &qk, sizeof(qk));
    step.node("reshape", {score4}, {score3});
    step.node("gk_attn_tile_softmax_bf16_v1", {score3, tp, th}, {probability3, tile_m, tile_l}, &scale, sizeof(scale));
    step.node("reshape", {probability3}, {probability4});
    step.node("batch_gemm", {probability4, cache_tiles[1]}, {tile_o4}, &pv, sizeof(pv));
    step.node("reshape", {tile_o4}, {tile_o});
    step.node("gk_attn_online_merge_f32_v1", {old_m, old_l, old_o, tile_m, tile_l, tile_o},
              {new_m, new_l, new_o});
    step.compile(cache);
    if (step.sections.size() >= 12) throw std::runtime_error("tile graph exceeds section budget");

    Graph finish("fixed_tile_finish" + shape);
    const auto fl = finish.tensor({rows, heads}, syn_type_float, state[0].sum);
    const auto fo = finish.tensor({d, rows, heads}, syn_type_float, state[0].output);
    const auto result = finish.tensor({d, heads, rows}, syn_type_bf16, output_addr);
    const auto grouped = finish.tensor({d, rows, heads}, syn_type_bf16);
    finish.node("gk_attn_online_finalize_bf16_v1", {fl, fo}, {grouped});
    finish.node("transpose", {grouped}, {result}, &transpose, sizeof(transpose));
    finish.compile(cache);
    const auto counts = cache.stats();
    if ((mode == "cold" && counts != std::pair<uint64_t, uint64_t>{0, 3}) ||
        (mode == "warm" && counts != std::pair<uint64_t, uint64_t>{3, 0}))
      throw std::runtime_error("unexpected recipe cache counts");
    if (std::getenv("GAUDIRT_COMPILE_ONLY")) {
      std::cout << "compiled store/tile/finalize sections " << store.sections.size() << '/'
                << step.sections.size() << '/' << finish.sections.size() << " cache "
                << counts.first << '/' << counts.second << '\n';
      return 0;
    }
    const auto workspace = arena.ensure_workspace(std::max({store.workspace, step.workspace, finish.workspace}));
    const auto stream = runtime.compute_stream();
    synEventHandle start_event = nullptr, end_event = nullptr;
    check(synEventCreate(&start_event, runtime.device_id(), EVENT_COLLECT_TIME), "timing start event");
    check(synEventCreate(&end_event, runtime.device_id(), EVENT_COLLECT_TIME), "timing end event");
    const auto enqueue = [&] {
      check(synLaunch(stream, store.bindings.data(), store.bindings.size(), workspace, store.recipe, 0), "KV launch");
      for (uint32_t block = 0; block < query_blocks; ++block) {
        check(synMemsetD32Async(state[0].maximum, 0, rows * heads, stream), "initial maximum");
        check(synMemsetD32Async(state[0].sum, 0, rows * heads, stream), "initial sum");
        check(synMemsetD32Async(state[0].output, 0, uint64_t{d} * rows * heads, stream), "initial accumulator");
        auto& binding = step.bindings;
        binding[0].pTensorAddress = q_addr + uint64_t{block} * rows * heads * d * 2;
        const auto active_tiles = std::min(tiles, static_cast<uint32_t>(positions[block * rows + valid - 1]) / kv + 1);
        for (uint32_t tile = 0; tile < active_tiles; ++tile) {
          const auto meta = metadata_addr[block * tiles + tile];
          binding[1].pTensorAddress = meta;
          binding[4].pTensorAddress = index_addr[tile];
          binding[5].pTensorAddress = meta + header_offset;
          const auto& in = state[tile % 2];
          const auto& out = state[(tile + 1) % 2];
          binding[6].pTensorAddress = in.maximum;
          binding[7].pTensorAddress = in.sum;
          binding[8].pTensorAddress = in.output;
          binding[9].pTensorAddress = out.maximum;
          binding[10].pTensorAddress = out.sum;
          binding[11].pTensorAddress = out.output;
          check(synLaunch(stream, binding.data(), binding.size(), workspace, step.recipe, 0), "tile launch");
        }
        finish.bindings[0].pTensorAddress = state[active_tiles % 2].sum;
        finish.bindings[1].pTensorAddress = state[active_tiles % 2].output;
        finish.bindings[2].pTensorAddress = output_addr + uint64_t{block} * rows * heads * d * 2;
        check(synLaunch(stream, finish.bindings.data(), finish.bindings.size(), workspace, finish.recipe, 0), "finalize launch");
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
    std::vector<uint16_t> actual_key(key_pool.size()), actual_value(value_pool.size());
    std::vector<uint16_t> output(uint64_t{d} * total_rows * heads);
    arena.copy_d2h_sync(runtime.d2h_stream(), key_addr, actual_key.data(), actual_key.size() * 2);
    arena.copy_d2h_sync(runtime.d2h_stream(), value_addr, actual_value.data(), actual_value.size() * 2);
    arena.copy_d2h_sync(runtime.d2h_stream(), output_addr, output.data(), output.size() * 2);
    if (std::any_of(output.begin(), output.end(), [](uint16_t value) { return !std::isfinite(widen(value)); }))
      throw std::runtime_error("nonfinite attention output");
    if (actual_key != expected_key || actual_value != expected_value)
      throw std::runtime_error("KV write corrupted a target or untouched slot");
    std::array<uint16_t, 128> observed{};
    for (auto address : {guard_before_key, guard_after_key, guard_before_value, guard_after_value}) {
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
              tile_acc += widen(bf16(p32[tile * kv + t])) * v;
            }
            const float next_m = sum > 0 ? std::max(m, tile_max[tile]) : tile_max[tile];
            const float a = sum > 0 ? std::exp(m - next_m) : 0;
            const float b = std::exp(tile_max[tile] - next_m);
            sum = a * sum + b * tile_sum[tile];
            acc = a * acc + b * tile_acc;
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
              << " query_blocks " << query_blocks << " reference_rows " << checked_rows.size() << " sections "
              << store.sections.size() << '/' << step.sections.size() << '/' << finish.sections.size()
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
