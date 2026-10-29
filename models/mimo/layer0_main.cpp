// SPDX-License-Identifier: Apache-2.0
#include "layer0.hpp"

#include "gaudirt/common/sha256.hpp"
#include "gaudirt/common/synapse_raii.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

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

float bf16_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool compare_numerics(const std::vector<uint8_t>& actual,
                      const std::vector<uint8_t>& expected, const char* name,
                      uint32_t rank, size_t row_size, bool gate) {
  const size_t count = actual.size() / 2;
  if (actual.size() != expected.size() || actual.size() % 2 != 0 ||
      row_size == 0 || count % row_size != 0) {
    throw std::runtime_error("invalid BF16 comparison geometry");
  }
  std::vector<double> difference_squared(count / row_size, 0);
  std::vector<double> reference_squared(count / row_size, 0);
  double max_abs = 0;
  bool finite = true;
  for (size_t i = 0; i < count; ++i) {
    uint16_t actual_bits;
    uint16_t expected_bits;
    std::memcpy(&actual_bits, actual.data() + 2 * i, sizeof(actual_bits));
    std::memcpy(&expected_bits, expected.data() + 2 * i, sizeof(expected_bits));
    const double a = bf16_to_float(actual_bits);
    const double b = bf16_to_float(expected_bits);
    finite &= std::isfinite(a) && std::isfinite(b);
    const double difference = a - b;
    difference_squared[i / row_size] += difference * difference;
    reference_squared[i / row_size] += b * b;
    max_abs = std::max(max_abs, std::abs(difference));
  }
  double relative = 0;
  for (size_t row = 0; row < difference_squared.size(); ++row) {
    relative = std::max(relative, std::sqrt(difference_squared[row] /
                                            std::max(reference_squared[row], 1e-30)));
  }
  const bool bitwise = std::memcmp(actual.data(), expected.data(), actual.size()) == 0;
  const bool pass = finite && relative <= 1e-3;
  std::cout << std::setprecision(9)
            << "MIMO_LAYER0_METRIC rank=" << rank << " name=" << name
            << " max_abs=" << max_abs << " row_relative_l2=" << relative
            << " bitwise=" << (bitwise ? "true" : "false")
            << " status=" << (gate ? (pass ? "PASS" : "FAIL") : "INFO") << '\n';
  return gate ? pass : finite;
}

bool compare_fp32_numerics(const std::vector<uint8_t>& actual,
                           const std::vector<uint8_t>& expected, const char* name,
                           uint32_t rank, size_t row_size) {
  const size_t count = actual.size() / sizeof(float);
  if (actual.size() % sizeof(float) != 0 || expected.size() != count * sizeof(float) ||
      row_size == 0 || count % row_size != 0) {
    throw std::runtime_error("invalid FP32 comparison geometry");
  }
  std::vector<double> difference_squared(count / row_size, 0);
  std::vector<double> reference_squared(count / row_size, 0);
  double max_abs = 0;
  bool finite = true;
  for (size_t i = 0; i < count; ++i) {
    float a;
    float b;
    std::memcpy(&a, actual.data() + i * sizeof(a), sizeof(a));
    std::memcpy(&b, expected.data() + i * sizeof(b), sizeof(b));
    finite &= std::isfinite(a) && std::isfinite(b);
    const double difference = a - b;
    difference_squared[i / row_size] += difference * difference;
    reference_squared[i / row_size] += b * b;
    max_abs = std::max(max_abs, std::abs(difference));
  }
  double relative = 0;
  for (size_t row = 0; row < difference_squared.size(); ++row) {
    relative = std::max(relative, std::sqrt(difference_squared[row] /
                                            std::max(reference_squared[row], 1e-30)));
  }
  const bool pass = finite && relative <= 1e-3;
  std::cout << std::setprecision(9)
            << "MIMO_LAYER0_METRIC rank=" << rank << " name=" << name
            << " max_abs=" << max_abs << " row_relative_l2=" << relative
            << " bitwise=false status=" << (pass ? "PASS" : "FAIL") << '\n';
  return pass;
}

int run(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "usage: mimo_layer0 FIXTURES ATTENTION_WEIGHTS DENSE_WEIGHTS RANK WORLD OUTPUT UNIQUE_ID\n";
    return 2;
  }
  const std::filesystem::path fixtures = argv[1];
  const std::filesystem::path attention_weights = argv[2];
  const std::filesystem::path dense_weights = argv[3];
  const uint32_t rank = static_cast<uint32_t>(std::stoul(argv[4]));
  const uint32_t world = static_cast<uint32_t>(std::stoul(argv[5]));
  const std::filesystem::path output = argv[6];
  const std::filesystem::path unique_id = argv[7];
  if (world != 8 || rank >= world) {
    throw std::runtime_error("mimo_layer0 requires TP8");
  }
  const auto fixture_dir = fixtures / ("rank" + std::to_string(rank));
  const auto directory = output / ("rank" + std::to_string(rank));
  std::filesystem::create_directories(directory);
  const uint64_t cache_slots =
      std::filesystem::file_size(fixture_dir / "key_cache_before.bin") / (192 * 2);
  const uint64_t page_count =
      std::filesystem::file_size(fixture_dir / "block_list.bin") / sizeof(int32_t);
  if (cache_slots < 256 || cache_slots % 128 != 0 || page_count == 0 ||
      page_count > 64 ||
      std::filesystem::file_size(fixture_dir / "value_cache_before.bin") !=
          cache_slots * 128 * 2) {
    throw std::runtime_error("invalid layer0 fixture geometry");
  }

  gaudirt::runtime::DeviceRuntime::Config config;
  config.module_id = rank;
  config.rank = rank;
  config.world_size = world;
  config.pin_cpu = rank * 4;
  gaudirt::runtime::DeviceRuntime runtime(config);
  runtime.init_comm(exchange_unique_id(rank, unique_id));
  gaudirt::runtime::HbmArena arena(runtime.device_id(), 256 * 1024 * 1024ULL);
  const gaudirt::runtime::WeightManifest attention_weight_manifest(
      attention_weights / ("rank" + std::to_string(rank)), runtime, arena);
  const gaudirt::runtime::WeightManifest dense_weight_manifest(
      dense_weights / ("rank" + std::to_string(rank)), runtime, arena);

  gaudirt::mimo::Layer0::Io io{};
  io.attention.hidden_states = upload(arena, runtime, fixture_dir / "residual_in.bin");
  io.attention.residual = io.attention.hidden_states;
  io.attention.normalized = arena.allocate(12288);
  io.attention.rope_cos = upload(arena, runtime, fixture_dir / "rope_cos.bin");
  io.attention.rope_sin = upload(arena, runtime, fixture_dir / "rope_sin.bin");
  io.attention.key_cache = upload(arena, runtime, fixture_dir / "key_cache_before.bin");
  io.attention.value_cache = upload(arena, runtime, fixture_dir / "value_cache_before.bin");
  io.attention.slot_mapping = upload(arena, runtime, fixture_dir / "slot_mapping.bin");
  io.attention.full_block_list = upload(arena, runtime, fixture_dir / "block_list.bin");
  io.attention.full_block_usage = upload(arena, runtime, fixture_dir / "block_usage.bin");
  io.attention.full_block_groups = upload(arena, runtime, fixture_dir / "block_groups.bin");
  io.attention.qkv = arena.allocate(3392 * 2);
  io.attention.query = arena.allocate(16 * 192 * 2);
  io.attention.new_key = arena.allocate(192 * 2);
  io.attention.new_value = arena.allocate(128 * 2);
  io.attention.context = arena.allocate(16 * 128 * 2);
  io.attention.output_local_fp32 = arena.allocate(6144 * 4);
  io.dense_outputs.residual_out_bf16 = arena.allocate(12288);
  io.dense_outputs.activation_fp32 = arena.allocate(2048 * 4);
  io.dense_outputs.output_local_fp32 = arena.allocate(6144 * 4);
  io.attention_output_bf16 = arena.allocate(12288);
  io.attention_gathered_fp32 = arena.allocate(world * 6144 * 4);
  io.down_gathered_fp32 = arena.allocate(world * 6144 * 4);
  io.layer_output_bf16 = arena.allocate(12288);

  gaudirt::mimo::Layer0 layer(arena, runtime, attention_weight_manifest,
                              dense_weight_manifest, cache_slots, page_count);
  const char* perf_text = std::getenv("MIMO_LAYER_PERF_ITERS");
  const char* warmup_text = std::getenv("MIMO_LAYER_PERF_WARMUP");
  const uint32_t measured = perf_text ? static_cast<uint32_t>(std::stoul(perf_text)) : 0;
  const uint32_t warmup = measured ? (warmup_text ? std::stoul(warmup_text) : 2) : 0;
  if (measured > 64 || (measured && (warmup == 0 || warmup > 16)) ||
      (!measured && warmup_text)) {
    throw std::runtime_error("invalid layer0 performance iteration configuration");
  }
  std::vector<double> timings;
  for (uint32_t step = 0; step < warmup + std::max(1U, measured); ++step) {
    std::unique_ptr<gaudirt::common::TimedEventPair> timer;
    if (measured && step >= warmup) {
      timer = std::make_unique<gaudirt::common::TimedEventPair>(runtime.device_id());
      timer->record_start(runtime.compute_stream());
    }
    layer.execute(io, runtime.compute_stream());
    if (timer) timer->record_end(runtime.compute_stream());
    runtime.synchronize_compute();
    if (timer) timings.push_back(timer->elapsed_us());
  }
  for (uint32_t sample = 0; sample < timings.size(); ++sample) {
    std::cout << "MIMO_LAYER0_PERF_SAMPLE rank=" << rank << " sample=" << sample
              << " elapsed_us=" << timings[sample] << '\n';
  }
  if (!timings.empty()) {
    std::sort(timings.begin(), timings.end());
    std::cout << "MIMO_LAYER0_PERF_SUMMARY rank=" << rank
              << " p50_us=" << timings[timings.size() / 2]
              << " min_us=" << timings.front() << " max_us=" << timings.back() << '\n';
  }
  write_file(directory / "down_local.actual.bin",
             download(arena, runtime, io.dense_outputs.output_local_fp32, 6144 * 4));

  bool pass = true;
  const auto check_bytes = [&](const char* name, const char* reference,
                               uint64_t address, uint64_t bytes) {
    const auto actual = download(arena, runtime, address, bytes);
    write_file(directory / (std::string(name) + ".actual.bin"), actual);
    pass &= compare(actual, read_file(fixture_dir / reference), name, rank);
  };
  const auto check_large = [&](const char* name, const char* reference,
                               uint64_t address, uint64_t bytes) {
    const auto actual = download(arena, runtime, address, bytes);
    const auto expected = read_file(fixture_dir / reference);
    const bool equal = actual.size() == expected.size() &&
                       std::memcmp(actual.data(), expected.data(), actual.size()) == 0;
    const std::string actual_sha = gaudirt::common::sha256_bytes(actual.data(), actual.size());
    const std::string expected_sha = gaudirt::common::sha256_bytes(expected.data(), expected.size());
    std::cout << "rank=" << rank << " " << name
              << " memcmp=" << (equal ? "PASS" : "FAIL")
              << " actual_sha256=" << actual_sha
              << " reference_sha256=" << expected_sha << '\n';
    pass &= equal && actual_sha == expected_sha;
  };
  const auto math_attention = fixtures.parent_path() / "fp64-truth-v1" /
                              ("rank" + std::to_string(rank));
  const auto math_layer = fixtures.parent_path() / "fp64-layer0-truth-v4" /
                          ("rank" + std::to_string(rank));
  const auto check_numeric = [&](const char* name, const char* reference,
                                 uint64_t address, uint64_t bytes, bool attention) {
    const auto actual = download(arena, runtime, address, bytes);
    write_file(directory / (std::string(name) + ".actual.bin"), actual);
    const auto& root = attention ? math_attention : math_layer;
    const size_t row_size = std::string(name) == "attention_context" ? 128 : bytes / 2;
    pass &= compare_numerics(actual, read_file(root / reference), name, rank,
                             row_size, true);
  };
  const auto check_fp32 = [&](const char* name, const char* reference,
                              uint64_t address, uint64_t bytes) {
    const auto actual = download(arena, runtime, address, bytes);
    write_file(directory / (std::string(name) + ".actual.bin"), actual);
    pass &= compare_fp32_numerics(actual, read_file(math_layer / reference), name,
                                  rank, bytes / sizeof(float));
  };
  check_bytes("query", "query.bin", io.attention.query, 16 * 192 * 2);
  check_bytes("new_key", "new_key.bin", io.attention.new_key, 192 * 2);
  check_bytes("new_value", "new_value.bin", io.attention.new_value, 128 * 2);
  check_large("key_cache_after", "key_cache_after.bin", io.attention.key_cache,
              cache_slots * 192 * 2);
  check_large("value_cache_after", "value_cache_after.bin", io.attention.value_cache,
              cache_slots * 128 * 2);
  check_bytes("block_list", "block_list.bin", io.attention.full_block_list,
              page_count * sizeof(int32_t));
  check_bytes("block_usage", "block_usage.bin", io.attention.full_block_usage,
              page_count * sizeof(int32_t));
  check_bytes("block_groups", "block_groups.bin", io.attention.full_block_groups,
              page_count * sizeof(int32_t));
  check_numeric("attention_context", "fp64_attention_context.bin", io.attention.context,
                4096, true);
  check_numeric("attention", "fp64_attention.bin", io.attention_output_bf16, 12288, true);
  check_numeric("residual_out", "fp64_residual_out.bin",
                io.dense_outputs.residual_out_bf16, 12288, false);
  check_fp32("activation", "fp64_activation.f32", io.dense_outputs.activation_fp32,
             2048 * 4);
  check_numeric("layer_output", "fp64_layer_output.bin", io.layer_output_bf16, 12288, false);
  const auto actual_legacy = download(arena, runtime, io.layer_output_bf16, 12288);
  pass &= compare_numerics(actual_legacy, read_file(fixture_dir / "layer_output.bin"),
                           "layer_output_legacy_flat_pa", rank, 6144, false);
  std::cout << "MIMO_LAYER0_RESULT rank=" << rank
            << " status=" << (pass ? "PASS" : "FAIL") << '\n';
  return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "mimo_layer0: " << error.what() << '\n';
    return 2;
  }
}
