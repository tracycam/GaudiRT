// SPDX-License-Identifier: Apache-2.0
// Standalone block-FP8 graph: BF16 dynamic activation quantization, native
// FP8 batch GEMM, and FP32 block-scale/bias reduction to BF16.
#include "gaudirt/runtime/device_runtime.hpp"
#include "test_device_slab.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
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
double ocp(uint8_t code) {
  const unsigned magnitude = code & 0x7f, exponent = magnitude >> 3, mantissa = magnitude & 7;
  const double number = exponent ? std::ldexp(1.0 + mantissa / 8.0, static_cast<int>(exponent) - 7)
                                 : std::ldexp(static_cast<double>(mantissa), -9);
  return code & 0x80 ? -number : number;
}
uint8_t half_rne(uint8_t code) {
  const unsigned magnitude = code & 0x7f;
  if (magnitude == 0x7f) throw std::invalid_argument("OCP NaN");
  const unsigned half = magnitude >= 16 ? magnitude - 8 :
      (magnitude >> 1) + ((magnitude & 1) & ((magnitude >> 1) & 1));
  return (code & 0x80) | half;
}

struct Graph {
  explicit Graph(std::string name) : prefix(std::move(name)) {
    tensors.reserve(16); bindings.reserve(8);
    check(synGraphCreate(&graph, synDeviceGaudi2), "graph creation");
  }
  ~Graph() {
    if (recipe) synRecipeDestroy(recipe);
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) synTensorDestroy(tensor);
    for (auto section : sections) synSectionDestroy(section);
  }
  void enable_rmw() {
    rmw_enabled = true;
  }
  size_t tensor(std::vector<uint64_t> dims, synDataType type, uint64_t address = 0,
                bool rmw = false) {
    const auto id = tensors.size();
    names.push_back(prefix + "_t" + std::to_string(id));
    synSectionHandle section = nullptr;
    if (address) {
      check(synSectionCreate(&section, 0, graph), "section creation");
      sections.push_back(section);
      check(synSectionSetPersistent(section, true), "section persistence");
    } else if (rmw) {
      if (!rmw_enabled) throw std::logic_error("RMW section is not enabled");
      check(synSectionCreate(&section, 0, graph), "RMW section creation");
      sections.push_back(section);
      check(synSectionSetPersistent(section, false), "RMW section persistence");
      check(synSectionSetRMW(section, true), "RMW section enable");
    }
    synTensorDescriptor desc{};
    desc.m_name = names.back().c_str();
    desc.m_dataType = type;
    desc.m_tensorType = DATA_TENSOR;
    desc.m_dims = dims.size();
    for (size_t i = 0; i < dims.size(); ++i) desc.m_sizes[i] = desc.m_minSizes[i] = dims[i];
    synTensor handle = nullptr;
    check(synTensorCreate(&handle, &desc, section, 0), "tensor creation");
    tensors.push_back(handle);
    if (type == syn_type_fp8_143) {
      synFpQuantParam quant{1.0, 7};
      synFpQuantMetadata metadata{type, &quant, 1};
      check(synTensorSetQuantizationData(handle, SYN_FP_QUANT_METADATA, &metadata, sizeof(metadata)),
            "FP8 metadata");
    }
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
            const void* params = nullptr, size_t bytes = 0) {
    node(guid, std::vector<size_t>(inputs), std::vector<size_t>(outputs), params, bytes);
  }
  synNodeId node(const char* guid, const std::vector<size_t>& inputs, const std::vector<size_t>& outputs,
                 const void* params = nullptr, size_t bytes = 0) {
    std::vector<synTensor> in, out;
    for (auto id : inputs) in.push_back(tensors.at(id));
    for (auto id : outputs) out.push_back(tensors.at(id));
    const auto name = prefix + "_n" + std::to_string(node_count++);
    synNodeId id{};
    check(synNodeCreateWithId(graph, in.data(), out.data(), in.size(), out.size(),
                              const_cast<void*>(params), bytes, guid, name.c_str(), &id, nullptr, nullptr),
          "node creation");
    return id;
  }
  void compile(RecipeCache& cache) {
    recipe = compile_cached(graph, prefix, prefix, &cache);
    check(synWorkspaceGetSize(&workspace, recipe), "workspace query");
    initialize_launch_tensor_ids(recipe, bindings.data(), bindings.size());
  }
  std::string prefix;
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  bool rmw_enabled = false;
  uint64_t workspace = 0;
  size_t node_count = 0;
  std::deque<std::string> names;
  std::vector<synTensor> tensors;
  std::vector<synSectionHandle> sections;
  std::vector<synLaunchTensorInfo> bindings;
};
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 10)
      throw std::invalid_argument("expected cache, M, N, K, bias flag, N tile, M tile, input pattern and output path");
    const uint64_t m = std::stoul(argv[2]), n = std::stoul(argv[3]), k = std::stoul(argv[4]);
    const std::string bias_flag = argv[5];
    if (bias_flag != "0" && bias_flag != "1") throw std::invalid_argument("bias flag must be 0 or 1");
    const bool has_bias = bias_flag == "1";
    if (!m || m > 4096 || !n || n > 65536 || n % 128 || !k || k > 32768 || k % 128)
      throw std::invalid_argument("test supports M=1..4096 and aligned N/K within its memory budget");
    const uint64_t groups = k / 128;
    const uint64_t tile_width = std::stoul(argv[6]);
    if (tile_width && (tile_width > n || tile_width % 128))
      throw std::invalid_argument("N tile must be zero or an aligned width no larger than N");
    const uint64_t tile_rows = std::stoul(argv[7]);
    if (tile_rows > m || (m < 512 && !tile_width && tile_rows))
      throw std::invalid_argument("M tile must be zero or no larger than M");
    const uint64_t col_tile = tile_width ? tile_width : std::min<uint64_t>(n, 65536ULL);
    uint64_t default_hybrid_rows = std::min<uint64_t>(m, 4096ULL);
    if (default_hybrid_rows > 2048 && default_hybrid_rows * col_tile > (4096ULL * 16384ULL)) {
      default_hybrid_rows = 2048;
    }
    const uint64_t rows_per_tile = tile_rows ? tile_rows : (m >= 512 ? default_hybrid_rows : m);
    const std::string pattern = argv[8];
    if (pattern != "exact" && pattern != "normal" && pattern != "tiny")
      throw std::invalid_argument("input pattern must be exact, normal or tiny");
    const float values[] = {448, -448, 224, -224, 128, -128, 64, -64, 32, -32,
                            16, -16, 8, -8, 4, -4, 2, -2, 1, -1, 0.5f, -0.5f};
    std::vector<uint16_t> x(m * k), output(m * n, 0x7fc0);
    std::vector<uint8_t> original(n * k), prepared(n * k);
    std::vector<float> scales(n / 128 * groups), original_scales(scales.size()), bias(n);
    std::mt19937 random(12345);
    std::normal_distribution<float> normal;
    for (uint64_t row = 0; row < m; ++row)
      for (uint64_t col = 0; col < k; ++col) {
        const float raw = pattern == "normal" ? normal(random)
                        : pattern == "tiny"   ? normal(random) * (((col / 128) & 1) ? 1e-15f : 1e-12f)
                        : (col % 128 ? values[(row * 19 + col * 7) % std::size(values)] : 448.f);
        x[row * k + col] = bf16(raw);
      }
    for (uint64_t row = 0; row < n; ++row) {
      bias[row] = has_bias ? static_cast<float>(static_cast<int>(row % 19) - 9) * 0.125f : 0.f;
      for (uint64_t col = 0; col < k; ++col) {
        const auto magnitude = static_cast<uint8_t>((row * 13 + col * 17 + 1) % 127);
        const auto code = static_cast<uint8_t>(magnitude | (((row + col / 3) & 1) ? 0x80 : 0));
        original[row * k + col] = code;
        prepared[((col / 128) * n + row) * 128 + col % 128] = half_rne(code);
      }
    }
    for (uint64_t nb = 0; nb < n / 128; ++nb)
      for (uint64_t g = 0; g < groups; ++g) {
        const auto index = nb * groups + g;
        original_scales[index] = static_cast<float>((nb * 7 + g * 11) % 13 + 1) / 1024.f;
        scales[index] = original_scales[index] * 2.f;
      }
    DeviceRuntime runtime({});
    const bool use_hybrid = m >= 512;
    if (!use_hybrid && tile_width) {
      uint64_t limit = 0;
      synDeviceAttribute attribute = DEVICE_ATTRIBUTE_MAX_RMW_SIZE;
      check(synDeviceGetAttribute(&limit, &attribute, 1, runtime.device_id()), "RMW capacity query");
      if (tile_width * rows_per_tile * groups * sizeof(float) > limit)
        throw std::invalid_argument("M/N tile exceeds the device RMW capacity");
    }
    const uint64_t partial_bytes = m * n * groups * sizeof(float);
    if (!use_hybrid && !tile_width && partial_bytes > 256ULL * 1024 * 1024)
      throw std::invalid_argument("untiled partial exceeds the device-test memory budget");
    const uint64_t arena_bytes = prepared.size() + x.size() * sizeof(x[0]) +
        output.size() * sizeof(output[0]) +
        (use_hybrid ? 2560ULL * 1024 * 1024
                    : (tile_width ? 512ULL * 1024 * 1024 : 2 * partial_bytes + 256ULL * 1024 * 1024));
    gaudirt::testing::TestDeviceSlab arena(runtime.device_id(), arena_bytes);
    RecipeCache cache(argv[1]);
    const auto xa = arena.allocate(x.size() * sizeof(x[0]));
    const auto wa = arena.allocate(prepared.size());
    const auto sa = arena.allocate(scales.size() * sizeof(scales[0]));
    const auto ba = (!use_hybrid || has_bias) ? arena.allocate(bias.size() * sizeof(bias[0])) : 0;
    const auto ya = arena.allocate(output.size() * sizeof(output[0]));
    auto upload = [&](const void* src, uint64_t dst, size_t bytes) {
      arena.copy_h2d_sync(runtime.h2d_stream(), src, dst, bytes);
    };
    upload(x.data(), xa, x.size() * sizeof(x[0]));
    upload(prepared.data(), wa, prepared.size());
    upload(scales.data(), sa, scales.size() * sizeof(scales[0]));
    if (has_bias) upload(bias.data(), ba, bias.size() * sizeof(bias[0]));
    upload(output.data(), ya, output.size() * sizeof(output[0]));
    Graph graph("block_fp8_m" + std::to_string(m) + "_n" + std::to_string(n) +
                "_k" + std::to_string(k) + "_bias" + bias_flag + "_tile" + std::to_string(tile_width) +
                "_rows" + std::to_string(rows_per_tile) + (use_hybrid ? "_fp16_3k_fp8" : ""));
    if (!use_hybrid && tile_width) graph.enable_rmw();
    const auto tx = graph.tensor({k, m}, syn_type_bf16, xa);
    const auto tw = graph.tensor({128, n, groups}, syn_type_fp8_143, wa);
    const auto ts = graph.tensor({groups, n / 128}, syn_type_float, sa);
    const auto ty = graph.tensor({n, m}, syn_type_bf16, ya);
    const synGEMMParams gemm{false, true};
    if (use_hybrid) {
      constexpr uint64_t kMaxAct1kBytes = 29ULL << 20;
      constexpr uint64_t kMaxAct3kBytes = 27ULL << 20;
      const uint64_t gc1_max = std::max<uint64_t>(1, kMaxAct1kBytes / (256ULL * rows_per_tile));
      const uint64_t k_chunks_1k = (groups + gc1_max - 1) / gc1_max;
      const uint64_t gc1 = (groups + k_chunks_1k - 1) / k_chunks_1k;
      const uint64_t g_cnt1 = (groups + gc1 - 1) / gc1;

      const uint64_t pairs_max = std::max<uint64_t>(1, kMaxAct3kBytes / (768ULL * rows_per_tile));
      const uint64_t gc2_max = 2 * pairs_max;
      const uint64_t k_chunks_3k = (groups + gc2_max - 1) / gc2_max;
      const uint64_t gc2 = k_chunks_3k == 1
                               ? groups
                               : 2 * ((groups + 2 * k_chunks_3k - 1) / (2 * k_chunks_3k));
      const uint64_t g_cnt2 = (groups + gc2 - 1) / gc2;

      const size_t tb = has_bias ? graph.tensor({n}, syn_type_float, ba) : 0;
      const uint64_t count = (n + col_tile - 1) / col_tile;
      const uint64_t row_count = (m + rows_per_tile - 1) / rows_per_tile;
      const synSplitParams split_bias{0}, split_features{1}, split_groups{2};
      const synConcatenateParams concat_width{0}, concat_rows{1};
      int32_t tag_seq = 0;
      std::vector<size_t> xs, row_scales, q_rows, sn_rows;
      std::vector<size_t> weights, scales_tiled, blk_scales, biases, row_outputs;
      for (uint64_t r = 0; r < row_count; ++r) {
        const auto rows = std::min(rows_per_tile, m - r * rows_per_tile);
        xs.push_back(row_count == 1 ? tx : graph.tensor({k, rows}, syn_type_bf16));
      }
      if (row_count > 1) graph.node("split", std::vector<size_t>{tx}, xs, &split_features, sizeof(split_features));
      for (uint64_t r = 0; r < row_count; ++r) {
        const auto rows = std::min(rows_per_tile, m - r * rows_per_tile);
        const int32_t r_tag = tag_seq++;
        const auto rs = graph.tensor({1, rows}, syn_type_float);
        graph.node("gk_fp8_act_row_scale_f32_v1", {xs[r]}, {rs}, &r_tag, sizeof(r_tag));
        row_scales.push_back(rs);
        const int32_t p_tag = tag_seq++;
        const auto qr = graph.tensor({128, rows, groups}, syn_type_bf16);
        const auto snr = graph.tensor({1, rows, groups}, syn_type_float);
        graph.node("gk_fp8_prequant_act_bf16_v1", {xs[r], rs}, {qr, snr}, &p_tag, sizeof(p_tag));
        q_rows.push_back(qr);
        sn_rows.push_back(snr);
      }
      for (uint64_t i = 0; i < count; ++i) {
        const auto width = std::min(col_tile, n - i * col_tile);
        weights.push_back(count == 1 ? tw : graph.tensor({128, width, groups}, syn_type_fp8_143));
        scales_tiled.push_back(count == 1 ? ts : graph.tensor({groups, width / 128}, syn_type_float));
        if (has_bias) biases.push_back(count == 1 ? tb : graph.tensor({width}, syn_type_float));
      }
      if (count > 1) {
        graph.node("split", std::vector<size_t>{tw}, weights, &split_features, sizeof(split_features));
        graph.node("split", std::vector<size_t>{ts}, scales_tiled, &split_features, sizeof(split_features));
        if (has_bias) graph.node("split", std::vector<size_t>{tb}, biases, &split_bias, sizeof(split_bias));
      }
      for (uint64_t i = 0; i < count; ++i) {
        const auto width = std::min(col_tile, n - i * col_tile);
        const int32_t c_tag = tag_seq++;
        const auto bs = graph.tensor({1, width / 128}, syn_type_float);
        graph.node("gk_fp8_weight_blk_scale_f32_v1", {scales_tiled[i]}, {bs}, &c_tag, sizeof(c_tag));
        blk_scales.push_back(bs);
      }
      std::vector<std::vector<size_t>> w_g1(count), ws_g1(count), w_g2(count), ws_g2(count);
      for (uint64_t i = 0; i < count; ++i) {
        const auto width = std::min(col_tile, n - i * col_tile);
        for (uint64_t g = 0; g < g_cnt1; ++g) {
          const auto gl = std::min(gc1, groups - g * gc1);
          w_g1[i].push_back(g_cnt1 == 1 ? weights[i] : graph.tensor({128, width, gl}, syn_type_fp8_143));
          ws_g1[i].push_back(g_cnt1 == 1 ? scales_tiled[i] : graph.tensor({gl, width / 128}, syn_type_float));
        }
        if (g_cnt1 > 1) {
          graph.node("split", std::vector<size_t>{weights[i]}, w_g1[i], &split_groups, sizeof(split_groups));
          graph.node("split", std::vector<size_t>{scales_tiled[i]}, ws_g1[i], &split_bias, sizeof(split_bias));
        }
        if (gc2 == gc1) {
          w_g2[i] = w_g1[i];
          ws_g2[i] = ws_g1[i];
        } else {
          for (uint64_t g = 0; g < g_cnt2; ++g) {
            const auto gl = std::min(gc2, groups - g * gc2);
            w_g2[i].push_back(g_cnt2 == 1 ? weights[i] : graph.tensor({128, width, gl}, syn_type_fp8_143));
            ws_g2[i].push_back(g_cnt2 == 1 ? scales_tiled[i] : graph.tensor({gl, width / 128}, syn_type_float));
          }
          if (g_cnt2 > 1) {
            graph.node("split", std::vector<size_t>{weights[i]}, w_g2[i], &split_groups, sizeof(split_groups));
            graph.node("split", std::vector<size_t>{scales_tiled[i]}, ws_g2[i], &split_bias, sizeof(split_bias));
          }
        }
      }
      for (uint64_t r = 0; r < row_count; ++r) {
        const auto rows = std::min(rows_per_tile, m - r * rows_per_tile);
        std::vector<size_t> q_g1, sn_g1, q_g2, sn_g2;
        for (uint64_t g = 0; g < g_cnt1; ++g) {
          const auto gl = std::min(gc1, groups - g * gc1);
          q_g1.push_back(g_cnt1 == 1 ? q_rows[r] : graph.tensor({128, rows, gl}, syn_type_bf16));
          sn_g1.push_back(g_cnt1 == 1 ? sn_rows[r] : graph.tensor({1, rows, gl}, syn_type_float));
        }
        if (g_cnt1 > 1) {
          graph.node("split", std::vector<size_t>{q_rows[r]}, q_g1, &split_groups, sizeof(split_groups));
          graph.node("split", std::vector<size_t>{sn_rows[r]}, sn_g1, &split_groups, sizeof(split_groups));
        }
        if (gc2 == gc1) {
          q_g2 = q_g1;
          sn_g2 = sn_g1;
        } else {
          for (uint64_t g = 0; g < g_cnt2; ++g) {
            const auto gl = std::min(gc2, groups - g * gc2);
            q_g2.push_back(g_cnt2 == 1 ? q_rows[r] : graph.tensor({128, rows, gl}, syn_type_bf16));
            sn_g2.push_back(g_cnt2 == 1 ? sn_rows[r] : graph.tensor({1, rows, gl}, syn_type_float));
          }
          if (g_cnt2 > 1) {
            graph.node("split", std::vector<size_t>{q_rows[r]}, q_g2, &split_groups, sizeof(split_groups));
            graph.node("split", std::vector<size_t>{sn_rows[r]}, sn_g2, &split_groups, sizeof(split_groups));
          }
        }

        std::vector<size_t> col_outputs;
        for (uint64_t i = 0; i < count; ++i) {
          const auto width = std::min(col_tile, n - i * col_tile);
          size_t c0 = 0;
          for (uint64_t g = 0; g < g_cnt1; ++g) {
            const auto gl = std::min(gc1, groups - g * gc1);
            const int32_t tile_tag = tag_seq++;
            const auto act_1k = graph.tensor({gl * 128, rows}, syn_type_fp16);
            const auto w_1k = graph.tensor({gl * 128, width}, syn_type_fp16);
            const auto next_c0 = graph.tensor({width, rows}, syn_type_float);
            graph.node("gk_fp8_quant_act_fp16_1k_v1", {q_g1[g], sn_g1[g]}, {act_1k},
                       &tile_tag, sizeof(tile_tag));
            graph.node("gk_fp8_dequant_weight_fp16_1k_v1",
                       {w_g1[i][g], ws_g1[i][g], blk_scales[i]}, {w_1k},
                       &tile_tag, sizeof(tile_tag));
            if (g == 0) {
              graph.node("gemm", {act_1k, w_1k}, {next_c0}, &gemm, sizeof(gemm));
            } else {
              graph.node("gemm", {act_1k, w_1k, c0}, {next_c0}, &gemm, sizeof(gemm));
            }
            c0 = next_c0;
          }

          size_t c1 = 0;
          for (uint64_t g = 0; g < g_cnt2; ++g) {
            const auto gl = std::min(gc2, groups - g * gc2);
            const auto kw = ((gl + 1) / 2) * 768;
            const int32_t tile_tag = tag_seq++;
            const auto act_3k = graph.tensor({kw, rows}, syn_type_fp8_143);
            const auto w_3k = graph.tensor({kw, width}, syn_type_fp8_143);
            const auto next_c1 = graph.tensor({width, rows}, syn_type_float);
            graph.node("gk_fp8_quant_act_fp8_3k_v1", {q_g2[g], sn_g2[g]}, {act_3k},
                       &tile_tag, sizeof(tile_tag));
            graph.node("gk_fp8_dequant_weight_fp8_3k_v1",
                       {w_g2[i][g], ws_g2[i][g], blk_scales[i]}, {w_3k},
                       &tile_tag, sizeof(tile_tag));
            if (g == 0) {
              graph.node("gemm", {act_3k, w_3k}, {next_c1}, &gemm, sizeof(gemm));
            } else {
              graph.node("gemm", {act_3k, w_3k, c1}, {next_c1}, &gemm, sizeof(gemm));
            }
            c1 = next_c1;
          }

          const auto result = (row_count == 1 && count == 1) ? ty : graph.tensor({width, rows}, syn_type_bf16);
          if (!has_bias) {
            graph.node("gk_fp8_hybrid_epilogue_bf16_v1",
                       {c0, c1, row_scales[r], blk_scales[i]}, {result});
          } else {
            graph.node("gk_fp8_hybrid_epilogue_bias_bf16_v1",
                       {c0, c1, row_scales[r], blk_scales[i], biases[i]}, {result});
          }
          col_outputs.push_back(result);
        }
        if (count == 1) {
          row_outputs.push_back(col_outputs[0]);
        } else {
          const auto row_out = row_count == 1 ? ty : graph.tensor({n, rows}, syn_type_bf16);
          graph.node("concat", col_outputs, std::vector<size_t>{row_out}, &concat_width, sizeof(concat_width));
          row_outputs.push_back(row_out);
        }
      }
      if (row_count > 1)
        graph.node("concat", row_outputs, std::vector<size_t>{ty}, &concat_rows, sizeof(concat_rows));
    } else {
      const auto tb = graph.tensor({n}, syn_type_float, ba);
      const auto tq = graph.tensor({128, m, groups}, syn_type_fp8_143);
      const auto tas = graph.tensor({1, m, groups}, syn_type_float);
      if (!has_bias) {
        ns_ConstantKernel::ParamsV3 zero{};
        zero.constant.f = 0.f;
        graph.node("constant_f32", {}, {tb}, &zero, sizeof(zero));
      }
      graph.node("gk_fp8_block_quant_bf16_v1", {tx}, {tq, tas});
      if (!tile_width) {
        const auto tp = graph.tensor({n, m, groups}, syn_type_float);
        graph.node("batch_gemm", {tq, tw}, {tp}, &gemm, sizeof(gemm));
        graph.node("gk_fp8_block_reduce_bf16_v1", {tp, tas, ts, tb}, {ty});
      } else {
      const auto count = (n + tile_width - 1) / tile_width;
      const auto row_count = (m + rows_per_tile - 1) / rows_per_tile;
      std::vector<size_t> weights, scales_tiled, biases, activations, activation_scales, rows;
      for (uint64_t i = 0; i < count; ++i) {
        const auto width = std::min(tile_width, n - i * tile_width);
        weights.push_back(count == 1 ? tw : graph.tensor({128, width, groups}, syn_type_fp8_143));
        scales_tiled.push_back(count == 1 ? ts : graph.tensor({groups, width / 128}, syn_type_float));
        biases.push_back(count == 1 ? tb : graph.tensor({width}, syn_type_float));
      }
      const synSplitParams split_features{1}, split_bias{0};
      if (count > 1) {
        graph.node("split", std::vector<size_t>{tw}, weights, &split_features, sizeof(split_features));
        graph.node("split", std::vector<size_t>{ts}, scales_tiled, &split_features, sizeof(split_features));
        graph.node("split", std::vector<size_t>{tb}, biases, &split_bias, sizeof(split_bias));
      }
      for (uint64_t r = 0; r < row_count; ++r) {
        const auto height = std::min(rows_per_tile, m - r * rows_per_tile);
        activations.push_back(row_count == 1 ? tq : graph.tensor({128, height, groups}, syn_type_fp8_143));
        activation_scales.push_back(row_count == 1 ? tas : graph.tensor({1, height, groups}, syn_type_float));
        rows.push_back(row_count == 1 ? ty : graph.tensor({n, height}, syn_type_bf16));
      }
      if (row_count > 1) {
        graph.node("split", std::vector<size_t>{tq}, activations, &split_features, sizeof(split_features));
        graph.node("split", std::vector<size_t>{tas}, activation_scales, &split_features, sizeof(split_features));
      }
      std::vector<std::vector<size_t>> outputs(row_count);
      for (uint64_t i = 0; i < count; ++i) {
        const auto width = std::min(tile_width, n - i * tile_width);
        for (uint64_t r = 0; r < row_count; ++r) {
          const auto height = std::min(rows_per_tile, m - r * rows_per_tile);
          const auto result = count == 1 ? rows[r] : graph.tensor({width, height}, syn_type_bf16);
          outputs[r].push_back(result);
          const auto partial = graph.tensor({width, height, groups}, syn_type_float, 0, true);
          graph.node("batch_gemm", std::vector<size_t>{activations[r], weights[i]},
                     std::vector<size_t>{partial}, &gemm, sizeof(gemm));
          graph.node("gk_fp8_block_reduce_bf16_v1",
                     std::vector<size_t>{partial, activation_scales[r], scales_tiled[i], biases[i]},
                     std::vector<size_t>{result});
        }
      }
      const synConcatenateParams concat_width{0}, concat_rows{1};
      if (count > 1) {
        for (uint64_t r = 0; r < row_count; ++r)
          graph.node("concat", outputs[r], std::vector<size_t>{rows[r]}, &concat_width, sizeof(concat_width));
      }
      if (row_count > 1)
        graph.node("concat", rows, std::vector<size_t>{ty}, &concat_rows, sizeof(concat_rows));
      }
    }
    graph.compile(cache);
    check(synLaunch(runtime.compute_stream(), graph.bindings.data(), graph.bindings.size(),
                    arena.ensure_workspace(graph.workspace), graph.recipe, 0), "launch");
    runtime.synchronize_compute();
    arena.copy_d2h_sync(runtime.d2h_stream(), ya, output.data(), output.size() * sizeof(output[0]));
    for (uint16_t raw_out : output) {
      if (!std::isfinite(widen(raw_out))) throw std::runtime_error("nonfinite result");
    }
    std::array<double, 127> magnitudes{};
    for (unsigned code = 0; code < magnitudes.size(); ++code) magnitudes[code] = ocp(code);
    std::array<double, 256> decoded{};
    for (unsigned code = 0; code < decoded.size(); ++code)
      if ((code & 0x7f) != 0x7f) decoded[code] = ocp(code);
    std::vector<double> quantized_x(m * k);
    for (uint64_t row = 0; row < m; ++row) {
      for (uint64_t group = 0; group < groups; ++group) {
        float max_abs = 0.0f;
        for (uint64_t i = 0; i < 128; ++i)
          max_abs = std::max(max_abs, std::abs(widen(x[row * k + group * 128 + i])));
        const float activation_scale = max_abs == 0.0f ? 1.0f : max_abs / 448.0f;
        for (uint64_t i = 0; i < 128; ++i) {
          const auto index = row * k + group * 128 + i;
          const float input = widen(x[index]);
          const double scaled = std::min(448.0, std::abs(double(input)) / activation_scale);
          auto upper = std::lower_bound(magnitudes.begin(), magnitudes.end(), scaled);
          unsigned code = upper == magnitudes.end() ? 126 : static_cast<unsigned>(upper - magnitudes.begin());
          if (code) {
            const double down = scaled - magnitudes[code - 1], up = magnitudes[code] - scaled;
            if (down < up || (down == up && ((code - 1) & 1) == 0)) --code;
          }
          quantized_x[index] = std::copysign(magnitudes[code] * double(activation_scale), input);
        }
      }
    }
    std::vector<double> raw_x_f64(m * k);
    for (size_t idx = 0; idx < x.size(); ++idx) raw_x_f64[idx] = double(widen(x[idx]));
    std::vector<double> sq_err(m, 0.0), ref_energy(m, 0.0), id_err(m, 0.0), id_energy(m, 0.0);
    const unsigned hw_threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    constexpr uint64_t kColChunk = 256;
    std::vector<double> decoded_w(kColChunk * k);
    for (uint64_t c0_col = 0; c0_col < n; c0_col += kColChunk) {
      const uint64_t c_width = std::min<uint64_t>(kColChunk, n - c0_col);
      for (uint64_t dc = 0; dc < c_width; ++dc) {
        const uint64_t col = c0_col + dc;
        const float* scale_row = &original_scales[(col / 128) * groups];
        const uint8_t* w_row = &original[col * k];
        double* dst = &decoded_w[dc * k];
        for (uint64_t g = 0; g < groups; ++g) {
          const double ws = double(scale_row[g]);
          for (uint64_t i = 0; i < 128; ++i) dst[g * 128 + i] = decoded[w_row[g * 128 + i]] * ws;
        }
      }
      const unsigned workers = static_cast<unsigned>(std::min<uint64_t>(hw_threads, m));
      std::vector<std::thread> pool;
      pool.reserve(workers);
      std::atomic<bool> nonfinite{false};
      for (unsigned tid = 0; tid < workers; ++tid) {
        const uint64_t r_begin = (m * tid) / workers;
        const uint64_t r_end = (m * (tid + 1)) / workers;
        pool.emplace_back([&, r_begin, r_end]() {
          for (uint64_t row = r_begin; row < r_end; ++row) {
            const double* qx_row = &quantized_x[row * k];
            const double* rx_row = &raw_x_f64[row * k];
            double se = 0, re = 0, ie = 0, ien = 0;
            for (uint64_t dc = 0; dc < c_width; ++dc) {
              const uint64_t col = c0_col + dc;
              const double* dw = &decoded_w[dc * k];
              double reference = bias[col], ideal = bias[col];
              for (uint64_t inner = 0; inner < k; ++inner) {
                reference += qx_row[inner] * dw[inner];
                ideal += rx_row[inner] * dw[inner];
              }
              const double actual = widen(output[row * n + col]);
              if (!std::isfinite(reference) || !std::isfinite(ideal) || !std::isfinite(actual)) {
                nonfinite.store(true, std::memory_order_relaxed);
                return;
              }
              se += (actual - reference) * (actual - reference);
              re += reference * reference;
              ie += (actual - ideal) * (actual - ideal);
              ien += ideal * ideal;
            }
            sq_err[row] += se;
            ref_energy[row] += re;
            id_err[row] += ie;
            id_energy[row] += ien;
          }
        });
      }
      for (auto& th : pool) th.join();
      if (nonfinite.load()) throw std::runtime_error("nonfinite result");
    }
    double max_row_error = 0, max_unquantized_error = 0;
    for (uint64_t row = 0; row < m; ++row) {
      if (!(ref_energy[row] > 0.0) || !(id_energy[row] > 0.0)) throw std::runtime_error("zero reference row");
      max_row_error = std::max(max_row_error, std::sqrt(sq_err[row] / ref_energy[row]));
      max_unquantized_error = std::max(max_unquantized_error, std::sqrt(id_err[row] / id_energy[row]));
    }
    std::ofstream out(argv[9], std::ios::binary);
    out.write(reinterpret_cast<const char*>(output.data()), output.size() * sizeof(output[0]));
    if (!out) throw std::runtime_error("cannot write output");
    const auto counts = cache.stats();
    std::cout << "M=" << m << " N=" << n << " K=" << k << " bias=" << has_bias
              << " pattern=" << pattern << " fp64_ocp_w8a8_l2=" << max_row_error
              << " fp64_unquantized_activation_l2=" << max_unquantized_error
              << " loaded=" << counts.first << " compiled=" << counts.second
              << " N_tile=" << tile_width << " M_tile=" << rows_per_tile
              << " workspace_bytes=" << graph.workspace << '\n';
    if (!(max_row_error <= 0.02))
      throw std::runtime_error("block FP8 exceeds the 0.02 independent FP64 OCP W8A8 error limit");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
