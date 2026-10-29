// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "mimo_dense_mlp.hpp"
#include "gaudirt/runtime/tp_reduce.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
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

float bf16_to_float(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

bool compare_numeric(const std::vector<uint8_t>& actual,
                     const std::vector<uint8_t>& expected, const char* name,
                     uint32_t rank, bool fp32) {
  const size_t width = fp32 ? sizeof(float) : sizeof(uint16_t);
  const size_t count = actual.size() / width;
  if (actual.size() % width != 0 || expected.size() != actual.size()) {
    throw std::runtime_error("invalid Dense comparison geometry");
  }
  double difference_squared = 0;
  double reference_squared = 0;
  double max_abs = 0;
  bool finite = true;
  for (size_t i = 0; i < count; ++i) {
    float a, b;
    if (fp32) {
      std::memcpy(&a, actual.data() + i * width, width);
      std::memcpy(&b, expected.data() + i * width, width);
    } else {
      uint16_t a_bits, b_bits;
      std::memcpy(&a_bits, actual.data() + i * width, width);
      std::memcpy(&b_bits, expected.data() + i * width, width);
      a = bf16_to_float(a_bits);
      b = bf16_to_float(b_bits);
    }
    finite &= std::isfinite(a) && std::isfinite(b);
    const double difference = a - b;
    difference_squared += difference * difference;
    reference_squared += static_cast<double>(b) * b;
    max_abs = std::max(max_abs, std::abs(difference));
  }
  const double relative = std::sqrt(difference_squared / std::max(reference_squared, 1e-30));
  const bool pass = finite && relative <= 1e-3;
  std::cout << "MIMO_DENSE_METRIC rank=" << rank << " name=" << name
            << " max_abs=" << max_abs << " row_relative_l2=" << relative
            << " status=" << (pass ? "PASS" : "FAIL") << '\n';
  return pass;
}

int run(int argc, char** argv) {
  if (argc != 7) {
    std::cerr << "usage: mimo_dense FIXTURES WEIGHTS RANK WORLD OUTPUT UNIQUE_ID\n";
    return 2;
  }
  const std::filesystem::path fixtures = argv[1];
  const std::filesystem::path weights_root = argv[2];
  const uint32_t rank = static_cast<uint32_t>(std::stoul(argv[3]));
  const uint32_t world = static_cast<uint32_t>(std::stoul(argv[4]));
  const std::filesystem::path output = argv[5];
  const std::filesystem::path unique_id = argv[6];
  if (world != 8 || rank >= world) {
    throw std::runtime_error("mimo_dense requires TP8");
  }
  const auto directory = output / ("rank" + std::to_string(rank));
  std::filesystem::create_directories(directory);

  gaudirt::runtime::DeviceRuntime::Config config;
  config.module_id = rank;
  config.rank = rank;
  config.world_size = world;
  config.pin_cpu = rank * 4;
  gaudirt::runtime::DeviceRuntime runtime(config);
  runtime.init_comm(exchange_unique_id(rank, unique_id));
  gaudirt::runtime::HbmArena arena(runtime.device_id(), 256 * 1024 * 1024ULL);
  const auto fixture_dir = fixtures / ("rank" + std::to_string(rank));
  const gaudirt::runtime::WeightManifest weights(
      weights_root / ("rank" + std::to_string(rank)), runtime, arena);

  gaudirt::runtime::MimoDenseMlp::Inputs inputs{};
  inputs.attention_bf16 = upload(arena, runtime, fixture_dir / "attention.bin");
  inputs.residual_in_bf16 = upload(arena, runtime, fixture_dir / "residual_in.bin");
  inputs.post_attention_gamma_bf16 =
      weights.get("post_attention_gamma").device_address;
  gaudirt::runtime::MimoDenseMlp::Outputs outputs{};
  outputs.residual_out_bf16 = arena.allocate(12288);
  outputs.activation_fp32 = arena.allocate(2048 * 4);
  outputs.output_local_fp32 = arena.allocate(6144 * 4);

  gaudirt::runtime::MimoDenseMlp dense(arena, weights);
  gaudirt::runtime::TpReduce reduce(arena, 6144, syn_type_float);
  const uint64_t gathered = arena.allocate(world * 6144 * 4);
  const uint64_t layer_output = arena.allocate(12288);
  dense.execute(inputs, outputs, runtime.compute_stream());
  if (hcclAllGather(reinterpret_cast<const void*>(outputs.output_local_fp32),
                    reinterpret_cast<void*>(gathered), 6144, hcclFloat32,
                    runtime.comm(), runtime.compute_stream()) != hcclSuccess) {
    throw std::runtime_error("dense down AllGather failed");
  }
  reduce.execute(gathered, layer_output, runtime.compute_stream());
  runtime.synchronize_compute();

  bool pass = true;
  const auto math_dir = fixtures.parent_path() / "fp64-dense-truth-v1" /
                        ("rank" + std::to_string(rank));
  const auto check = [&](const char* name, const char* reference, uint64_t address,
                         uint64_t bytes, bool fp32) {
    const auto actual = download(arena, runtime, address, bytes);
    write_file(directory / (std::string(name) + ".actual.bin"), actual);
    pass &= compare_numeric(actual, read_file(math_dir / reference), name, rank, fp32);
  };
  check("residual_out", "fp64_residual_out.bin", outputs.residual_out_bf16, 12288, false);
  check("activation", "fp64_activation.f32", outputs.activation_fp32, 2048 * 4, true);
  const auto actual_output = download(arena, runtime, layer_output, 12288);
  write_file(directory / "layer_output.actual.bin", actual_output);
  pass &= compare_numeric(actual_output, read_file(math_dir / "fp64_layer_output.bin"),
                          "layer_output", rank, false);
  std::cout << "MIMO_DENSE_RESULT rank=" << rank
            << " status=" << (pass ? "PASS" : "FAIL") << '\n';
  return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "mimo_dense: " << error.what() << '\n';
    return 2;
  }
}
