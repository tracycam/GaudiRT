// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/common/synapse_raii.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "mimo_attention.hpp"
#include "mimo_moe.hpp"
#include "gaudirt/runtime/tp_reduce.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr uint32_t kCacheSlots = 17024;
constexpr uint64_t kFirstSlot = 16769;

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  const uint64_t bytes = std::filesystem::file_size(path);
  std::vector<uint8_t> data(bytes);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(data.data()), data.size())) {
    throw std::runtime_error("cannot read " + path.string());
  }
  return data;
}

void write_file(const std::filesystem::path& path,
                const std::vector<uint8_t>& data) {
  std::ofstream output(path, std::ios::binary);
  if (!output.write(reinterpret_cast<const char*>(data.data()), data.size())) {
    throw std::runtime_error("cannot write " + path.string());
  }
}

uint64_t upload(gaudirt::runtime::HbmArena& arena,
                gaudirt::runtime::DeviceRuntime& runtime,
                const std::filesystem::path& path) {
  const auto host = read_file(path);
  const uint64_t address = arena.allocate(host.size());
  arena.copy_h2d_sync(runtime.h2d_stream(), host.data(), address, host.size());
  return address;
}

std::vector<uint8_t> download(gaudirt::runtime::HbmArena& arena,
                              gaudirt::runtime::DeviceRuntime& runtime,
                              uint64_t address, uint64_t bytes) {
  std::vector<uint8_t> host(bytes);
  arena.copy_d2h_sync(runtime.d2h_stream(), address, host.data(), bytes);
  return host;
}

hcclUniqueId exchange_unique_id(uint32_t rank,
                                const std::filesystem::path& path) {
  hcclUniqueId id;
  if (rank == 0) {
    if (hcclGetUniqueId(&id) != hcclSuccess) {
      throw std::runtime_error("hcclGetUniqueId failed");
    }
    auto temporary = path;
    temporary += ".tmp";
    write_file(temporary,
               std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&id),
                                    reinterpret_cast<uint8_t*>(&id) + sizeof(id)));
    std::filesystem::rename(temporary, path);
    return id;
  }
  for (unsigned attempt = 0; attempt < 2400; ++attempt) {
    if (std::filesystem::is_regular_file(path) &&
        std::filesystem::file_size(path) == sizeof(id)) {
      const auto data = read_file(path);
      std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t*>(&id));
      return id;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  throw std::runtime_error("timed out waiting for HCCL unique ID");
}

bool compare(const std::vector<uint8_t>& actual,
             const std::vector<uint8_t>& expected, const char* name,
             uint32_t rank) {
  const bool pass = actual.size() == expected.size() &&
                    std::memcmp(actual.data(), expected.data(), actual.size()) == 0;
  std::cout << "rank=" << rank << " " << name
            << " memcmp=" << (pass ? "PASS" : "FAIL") << '\n';
  return pass;
}

int run(int argc, char** argv) {
  if (argc != 9) {
    std::cerr << "usage: mimo_layer_main ATTENTION_FIXTURES WEIGHTS MOE_FIXTURES "
                 "RESIDUAL RANK WORLD OUTPUT UNIQUE_ID\n";
    return 2;
  }
  const std::filesystem::path attention_fixtures = argv[1];
  const std::filesystem::path weights_root = argv[2];
  const std::filesystem::path moe_fixtures = argv[3];
  const std::filesystem::path residual_root = argv[4];
  const uint32_t rank = static_cast<uint32_t>(std::stoul(argv[5]));
  const uint32_t world = static_cast<uint32_t>(std::stoul(argv[6]));
  const std::filesystem::path output = argv[7];
  const std::filesystem::path unique_id = argv[8];
  const char* oproj_mode_text = std::getenv("MIMO_LAYER_OPROJ_MODE");
  const std::string oproj_mode = oproj_mode_text ? oproj_mode_text : "bf16";
  if (oproj_mode != "bf16") {
    throw std::runtime_error("unsupported MIMO_LAYER_OPROJ_MODE: " + oproj_mode);
  }
  const char* perf_text = std::getenv("MIMO_LAYER_PERF_ITERS");
  const char* warmup_text = std::getenv("MIMO_LAYER_PERF_WARMUP");
  const uint32_t measured =
      perf_text ? static_cast<uint32_t>(std::stoul(perf_text)) : 0;
  const uint32_t warmup =
      measured ? (warmup_text ? static_cast<uint32_t>(std::stoul(warmup_text)) : 2) : 0;
  if (measured > 256 || (measured && (warmup == 0 || warmup > 64)) ||
      (!measured && warmup_text)) {
    throw std::runtime_error("performance requires 1..256 samples and 1..64 warmups");
  }
  if (world != 8 || rank >= world) {
    throw std::runtime_error("mimo_layer1 requires TP8");
  }
  std::filesystem::create_directories(output);

  gaudirt::runtime::DeviceRuntime::Config config;
  config.module_id = rank;
  config.rank = rank;
  config.world_size = world;
  config.pin_cpu = rank * 4;
  gaudirt::runtime::DeviceRuntime runtime(config);
  runtime.init_comm(exchange_unique_id(rank, unique_id));
  gaudirt::runtime::HbmArena arena(runtime.device_id(), 1100000000ULL);

  const auto attention_dir = attention_fixtures / ("rank" + std::to_string(rank));
  const auto weight_dir = weights_root / ("rank" + std::to_string(rank));
  const auto moe_dir = moe_fixtures / ("rank" + std::to_string(rank));
  const auto residual_dir = residual_root / ("rank" + std::to_string(rank));
  gaudirt::runtime::WeightManifest weights(weight_dir, runtime, arena);

  gaudirt::runtime::MimoAttentionIo attention_io;
  attention_io.hidden_states = upload(arena, runtime, attention_dir / "hidden.bin");
  attention_io.residual = upload(arena, runtime, attention_dir / "residual.bin");
  attention_io.residual_out = arena.allocate(12288);
  attention_io.normalized = arena.allocate(12288);
  attention_io.rope_cos = upload(arena, runtime, attention_dir / "rope_cos.bin");
  attention_io.rope_sin = upload(arena, runtime, attention_dir / "rope_sin.bin");
  attention_io.key_cache = upload(arena, runtime, attention_dir / "key_cache.bin");
  attention_io.value_cache = upload(arena, runtime, attention_dir / "value_cache.bin");
  attention_io.slot_mapping = upload(arena, runtime, attention_dir / "slot_mapping.bin");
  attention_io.window_pages = upload(arena, runtime, attention_dir / "window_pages.bin");
  attention_io.window_groups = upload(arena, runtime, attention_dir / "window_groups.bin");
  attention_io.position = upload(arena, runtime, attention_dir / "position.bin");
  attention_io.qkv = arena.allocate(3392 * 2);
  attention_io.query = arena.allocate(16 * 192 * 2);
  attention_io.new_key = arena.allocate(192 * 2);
  attention_io.new_value = arena.allocate(128 * 2);
  attention_io.context = arena.allocate(16 * 128 * 2);
  attention_io.output_local = arena.allocate(12288);

  gaudirt::runtime::MimoAttention attention(arena, weights, kCacheSlots);
  auto attention_reduce =
      std::make_unique<gaudirt::runtime::TpReduce>(arena, 6144, syn_type_bf16);
  auto moe_reduce =
      std::make_unique<gaudirt::runtime::TpReduce>(arena, 6144, syn_type_float);
  const uint64_t attention_output = arena.allocate(12288);
  const uint64_t attention_gathered = arena.allocate(world * 12288);

  gaudirt::runtime::MimoMoeLayer1::Weights moe_weights{};
  moe_weights.post_attention_gamma_bf16 =
      weights.get("post_attention_layernorm").device_address;
  moe_weights.router_weight_bf16 = weights.get("router_weight").device_address;
  moe_weights.router_correction_bias_fp32 = weights.get("router_bias").device_address;
  moe_weights.gate_up_packed_u8 = weights.get("moe_gp").device_address;
  moe_weights.gate_up_scales_u8 = weights.get("moe_gs").device_address;
  moe_weights.down_packed_u8 = weights.get("moe_dp").device_address;
  moe_weights.down_scales_u8 = weights.get("moe_ds").device_address;
  auto lut = gaudirt::runtime::MimoMoeLayer1::make_mxfp4_lut_bf16();
  moe_weights.mxfp4_lut_bf16 = arena.allocate(lut.size() * sizeof(lut[0]));
  arena.copy_h2d_sync(runtime.h2d_stream(), lut.data(), moe_weights.mxfp4_lut_bf16,
                      lut.size() * sizeof(lut[0]));
  auto directions = gaudirt::runtime::MimoMoeLayer1::make_down_combine_directions();
  moe_weights.down_combine_directions_u8 = arena.allocate(directions.size());
  arena.copy_h2d_sync(runtime.h2d_stream(), directions.data(),
                      moe_weights.down_combine_directions_u8, directions.size());

  gaudirt::runtime::MimoMoeLayer1 moe(rank, moe_weights);
  const uint64_t moe_workspace = arena.ensure_workspace(moe.workspace_bytes());
  gaudirt::runtime::MimoMoeLayer1::Inputs moe_inputs{};
  moe_inputs.attention_bf16 = attention_output;
  moe_inputs.residual_in_bf16 = upload(
      arena, runtime, residual_dir / "layer1.post_attention_residual_in.bf16");
  gaudirt::runtime::MimoMoeLayer1::Outputs moe_outputs{};
  moe_outputs.residual_out_bf16 = arena.allocate(12288);
  moe_outputs.normalized_bf16 = arena.allocate(12288);
  moe_outputs.topk_ids_i32 = arena.allocate(32);
  moe_outputs.topk_weights_fp32 = arena.allocate(32);
  moe_outputs.moe_local_fp32 = arena.allocate(24576);
  const uint64_t layer_output = arena.allocate(12288);
  const uint64_t moe_gathered = arena.allocate(world * 24576);
  auto execute_layer = [&]() {
    attention.execute(attention_io, runtime.compute_stream());
    if (hcclAllGather(attention_io.output_local == 0
                          ? nullptr
                          : reinterpret_cast<const void*>(attention_io.output_local),
                      reinterpret_cast<void*>(attention_gathered), 6144,
                      hcclBfloat16, runtime.comm(), runtime.compute_stream()) !=
        hcclSuccess) {
      throw std::runtime_error("attention AllGather failed");
    }
    attention_reduce->execute(attention_gathered, attention_output,
                              runtime.compute_stream());
    moe.launch(runtime.compute_stream(), moe_inputs, moe_outputs, moe_workspace);
    if (hcclAllGather(reinterpret_cast<const void*>(moe_outputs.moe_local_fp32),
                      reinterpret_cast<void*>(moe_gathered), 6144, hcclFloat32,
                      runtime.comm(), runtime.compute_stream()) != hcclSuccess) {
      throw std::runtime_error("MoE AllGather failed");
    }
    moe_reduce->execute(moe_gathered, layer_output, runtime.compute_stream());
  };

  std::vector<double> timings;
  for (uint32_t step = 0; step < warmup + std::max(1U, measured); ++step) {
    std::unique_ptr<gaudirt::common::TimedEventPair> timer;
    if (measured && step >= warmup) {
      timer = std::make_unique<gaudirt::common::TimedEventPair>(runtime.device_id());
      timer->record_start(runtime.compute_stream());
    }
    execute_layer();
    if (timer) timer->record_end(runtime.compute_stream());
    runtime.synchronize_compute();
    if (timer) timings.push_back(timer->elapsed_us());
  }
  if (measured) {
    const auto previous_precision = std::cout.precision(9);
    for (uint32_t sample = 0; sample < timings.size(); ++sample) {
      std::cout << "MIMO_LAYER_PERF_SAMPLE rank=" << rank
                << " oproj=" << oproj_mode << " sample=" << sample
                << " elapsed_us=" << timings[sample] << '\n';
    }
    std::sort(timings.begin(), timings.end());
    std::cout << "MIMO_LAYER_PERF_SUMMARY rank=" << rank
              << " oproj=" << oproj_mode
              << " warmup=" << warmup << " iters=" << measured
              << " p50_us=" << timings[timings.size() / 2]
              << " min_us=" << timings.front()
              << " max_us=" << timings.back() << '\n';
    std::cout.precision(previous_precision);
  }

  bool pass = true;
  pass &= compare(download(arena, runtime, attention_io.residual_out, 12288),
                  read_file(attention_dir / "residual_out_reference.bin"),
                  "attention_residual_out", rank);
  pass &= compare(download(arena, runtime, attention_io.normalized, 12288),
                  read_file(attention_dir / "norm_reference.bin"),
                  "attention_normalized", rank);
  pass &= compare(download(arena, runtime, attention_io.qkv, 3392 * 2),
                  read_file(attention_dir / "qkv_reference.bin"), "qkv", rank);
  pass &= compare(download(arena, runtime, attention_io.query, 16 * 192 * 2),
                  read_file(attention_dir / "query_reference.bin"), "query", rank);
  pass &= compare(download(arena, runtime, attention_io.new_key, 192 * 2),
                  read_file(attention_dir / "new_key_reference.bin"), "new_key", rank);
  pass &= compare(download(arena, runtime, attention_io.new_value, 128 * 2),
                  read_file(attention_dir / "new_value_reference.bin"), "new_value", rank);
  auto expected_key_cache = read_file(attention_dir / "key_cache.bin");
  auto expected_value_cache = read_file(attention_dir / "value_cache.bin");
  const auto expected_new_key = read_file(attention_dir / "new_key_reference.bin");
  const auto expected_new_value = read_file(attention_dir / "new_value_reference.bin");
  const auto slot_mapping = read_file(attention_dir / "slot_mapping.bin");
  uint32_t current_slot = 0;
  if (slot_mapping.size() != sizeof(current_slot) ||
      expected_key_cache.size() != kCacheSlots * 192 * 2 ||
      expected_value_cache.size() != kCacheSlots * 128 * 2 ||
      expected_new_key.size() != 192 * 2 || expected_new_value.size() != 128 * 2) {
    throw std::runtime_error("invalid KV fixture geometry");
  }
  std::memcpy(&current_slot, slot_mapping.data(), sizeof(current_slot));
  if (current_slot >= kCacheSlots) {
    throw std::runtime_error("KV fixture slot is out of range");
  }
  std::memcpy(expected_key_cache.data() + current_slot * 192 * 2,
              expected_new_key.data(), expected_new_key.size());
  std::memcpy(expected_value_cache.data() + current_slot * 128 * 2,
              expected_new_value.data(), expected_new_value.size());
  pass &= compare(download(arena, runtime, attention_io.key_cache,
                           expected_key_cache.size()),
                  expected_key_cache, "key_cache", rank);
  pass &= compare(download(arena, runtime, attention_io.value_cache,
                           expected_value_cache.size()),
                  expected_value_cache, "value_cache", rank);
  pass &= compare(download(arena, runtime, attention_io.context, 16 * 128 * 2),
                  read_file(attention_dir / "context_reference.bin"), "context", rank);
  pass &= compare(download(arena, runtime, attention_io.output_local, 12288),
                  read_file(attention_dir / "oproj_local_reference.bin"),
                  "oproj_local", rank);
  pass &= compare(download(arena, runtime, attention_output, 12288),
                  read_file(attention_dir / "attention_reference.bin"),
                  "attention", rank);
  pass &= compare(download(arena, runtime, moe_outputs.residual_out_bf16, 12288),
                  read_file(residual_dir / "layer1.post_attention_residual_out.bf16"),
                  "post_residual_out", rank);
  pass &= compare(download(arena, runtime, moe_outputs.normalized_bf16, 12288),
                  read_file(moe_dir / "normalized.bin"), "moe_normalized", rank);
  pass &= compare(download(arena, runtime, moe_outputs.topk_ids_i32, 32),
                  read_file(moe_dir / "topk_ids.bin"), "topk_ids", rank);
  pass &= compare(download(arena, runtime, moe_outputs.topk_weights_fp32, 32),
                  read_file(moe_dir / "topk_weights.bin"), "topk_weights", rank);
  pass &= compare(download(arena, runtime, moe_outputs.moe_local_fp32, 24576),
                  read_file(moe_dir / "moe_local.bin"), "moe_local", rank);
  pass &= compare(download(arena, runtime, layer_output, 12288),
                  read_file(moe_dir / "layer_output.bin"), "layer_output", rank);

  write_file(output / "oproj_local.bin",
             download(arena, runtime, attention_io.output_local, 12288));
  write_file(output / "attention.bin",
             download(arena, runtime, attention_output, 12288));
  write_file(output / "moe_normalized.bin",
             download(arena, runtime, moe_outputs.normalized_bf16, 12288));
  write_file(output / "moe_local.bin",
             download(arena, runtime, moe_outputs.moe_local_fp32, 24576));
  write_file(output / "layer_output.bin",
             download(arena, runtime, layer_output, 12288));
  std::cout << "rank=" << rank << " layer_status=" << (pass ? "PASS" : "FAIL")
            << " oproj=" << oproj_mode
            << " attention_workspace=" << attention.workspace_bytes()
            << " moe_workspace=" << moe.workspace_bytes() << '\n';
  return pass ? 0 : 6;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "mimo_layer_main: " << error.what() << '\n';
    return 1;
  }
}
