// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/common/sha256.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "mimo_full_attention.hpp"
#include "gaudirt/runtime/tp_reduce.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
            << "MIMO_FULL_METRIC rank=" << rank << " name=" << name
            << " max_abs=" << max_abs << " row_relative_l2=" << relative
            << " bitwise=" << (bitwise ? "true" : "false")
            << " status=" << (gate ? (pass ? "PASS" : "FAIL") : "INFO") << '\n';
  return gate ? pass : finite;
}

int run(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: mimo_full_attention FIXTURES WEIGHTS RANK WORLD OUTPUT UNIQUE_ID\n";
    return 2;
  }
  const std::filesystem::path fixtures = argv[1];
  const std::filesystem::path weights_root = argv[2];
  const uint32_t rank = static_cast<uint32_t>(std::stoul(argv[3]));
  const uint32_t world = static_cast<uint32_t>(std::stoul(argv[4]));
  const std::filesystem::path output = argv[5];
  const std::filesystem::path unique_id = argv[6];
  if (world != 8 || rank >= world) {
    throw std::runtime_error("mimo_full_attention requires TP8");
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
          cache_slots * 128 * 2 ||
      std::filesystem::file_size(fixture_dir / "block_usage.bin") !=
          page_count * sizeof(int32_t) ||
      std::filesystem::file_size(fixture_dir / "block_groups.bin") !=
          page_count * sizeof(int32_t)) {
    throw std::runtime_error("invalid full-attention fixture geometry");
  }

  gaudirt::runtime::DeviceRuntime::Config config;
  config.module_id = rank;
  config.rank = rank;
  config.world_size = world;
  config.pin_cpu = rank * 4;
  gaudirt::runtime::DeviceRuntime runtime(config);
  runtime.init_comm(exchange_unique_id(rank, unique_id));
  gaudirt::runtime::HbmArena arena(runtime.device_id(), 256 * 1024 * 1024ULL);
  const gaudirt::runtime::WeightManifest weights(
      weights_root / ("rank" + std::to_string(rank)), runtime, arena);
  gaudirt::runtime::MimoAttentionIo io{};
  io.hidden_states = upload(arena, runtime, fixture_dir / "residual_in.bin");
  io.residual = io.hidden_states;
  io.residual_out = arena.allocate(12288);
  io.normalized = arena.allocate(12288);
  io.rope_cos = upload(arena, runtime, fixture_dir / "rope_cos.bin");
  io.rope_sin = upload(arena, runtime, fixture_dir / "rope_sin.bin");
  io.key_cache = upload(arena, runtime, fixture_dir / "key_cache_before.bin");
  io.value_cache = upload(arena, runtime, fixture_dir / "value_cache_before.bin");
  io.slot_mapping = upload(arena, runtime, fixture_dir / "slot_mapping.bin");
  io.full_block_list = upload(arena, runtime, fixture_dir / "block_list.bin");
  io.full_block_usage = upload(arena, runtime, fixture_dir / "block_usage.bin");
  io.full_block_groups = upload(arena, runtime, fixture_dir / "block_groups.bin");
  io.qkv = arena.allocate(3392 * 2);
  io.query = arena.allocate(16 * 192 * 2);
  io.new_key = arena.allocate(192 * 2);
  io.new_value = arena.allocate(128 * 2);
  io.context = arena.allocate(16 * 128 * 2);
  io.output_local_fp32 = arena.allocate(6144 * 4);

  gaudirt::runtime::MimoFullAttention attention(arena, weights, cache_slots, true,
                                                page_count);
  gaudirt::runtime::TpReduce reduce(arena, 6144, syn_type_float);
  const uint64_t gathered = arena.allocate(world * 6144 * 4);
  const uint64_t attention_output = arena.allocate(12288);
  attention.execute(io, runtime.compute_stream());
  if (hcclAllGather(reinterpret_cast<const void*>(io.output_local_fp32),
                    reinterpret_cast<void*>(gathered), 6144, hcclFloat32,
                    runtime.comm(), runtime.compute_stream()) != hcclSuccess) {
    throw std::runtime_error("attention AllGather failed");
  }
  reduce.execute(gathered, attention_output, runtime.compute_stream());
  runtime.synchronize_compute();
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
  const auto math_dir = fixtures.parent_path() / "fp64-truth-v1" /
                        ("rank" + std::to_string(rank));
  const auto check_numeric = [&](const char* name, const char* reference,
                                 uint64_t address, uint64_t bytes) {
    const auto actual = download(arena, runtime, address, bytes);
    write_file(directory / (std::string(name) + ".actual.bin"), actual);
    const size_t row_size = std::string(name) == "attention_context" ? 128 : bytes / 2;
    pass &= compare_numerics(actual, read_file(math_dir / reference), name, rank,
                             row_size, true);
  };
  const auto check_legacy = [&](const char* name, const char* reference,
                                uint64_t address, uint64_t bytes) {
    const auto actual = download(arena, runtime, address, bytes);
    const size_t row_size = bytes == 4096 ? 128 : bytes / 2;
    pass &= compare_numerics(actual, read_file(fixture_dir / reference), name, rank,
                             row_size, false);
  };
  check_bytes("query", "query.bin", io.query, 16 * 192 * 2);
  check_bytes("new_key", "new_key.bin", io.new_key, 192 * 2);
  check_bytes("new_value", "new_value.bin", io.new_value, 128 * 2);
  check_large("key_cache_after", "key_cache_after.bin", io.key_cache,
              cache_slots * 192 * 2);
  check_large("value_cache_after", "value_cache_after.bin", io.value_cache,
              cache_slots * 128 * 2);
  check_bytes("block_list", "block_list.bin", io.full_block_list,
              page_count * sizeof(int32_t));
  check_bytes("block_usage", "block_usage.bin", io.full_block_usage,
              page_count * sizeof(int32_t));
  check_bytes("block_groups", "block_groups.bin", io.full_block_groups,
              page_count * sizeof(int32_t));
  check_numeric("attention_context", "fp64_attention_context.bin", io.context,
                16 * 128 * 2);
  check_numeric("attention", "fp64_attention.bin", attention_output, 12288);
  check_legacy("attention_context_legacy_flat_pa", "attention_context.bin", io.context,
               16 * 128 * 2);
  check_legacy("attention_legacy_flat_pa", "attention.bin", attention_output, 12288);
  std::cout << "MIMO_FULL_RESULT rank=" << rank
            << " status=" << (pass ? "PASS" : "FAIL") << '\n';
  return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "mimo_full_attention: " << error.what() << '\n';
    return 2;
  }
}
