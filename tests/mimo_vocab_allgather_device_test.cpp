// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"

#include <hccl.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
hcclUniqueId exchange_id(uint32_t rank, const std::filesystem::path& path) {
  hcclUniqueId id{};
  if (rank == 0) {
    if (hcclGetUniqueId(&id) != hcclSuccess) throw std::runtime_error("HCCL unique ID failed");
    auto temporary = path;
    temporary += ".tmp";
    {
      std::ofstream output(temporary, std::ios::binary);
      if (!output.write(reinterpret_cast<const char*>(&id), sizeof(id)))
        throw std::runtime_error("HCCL unique ID write failed");
    }
    std::filesystem::rename(temporary, path);
    return id;
  }
  for (int attempt = 0; attempt < 2400; ++attempt) {
    if (std::filesystem::is_regular_file(path)) {
      std::ifstream input(path, std::ios::binary);
      if (!input.read(reinterpret_cast<char*>(&id), sizeof(id)))
        throw std::runtime_error("HCCL unique ID read failed");
      return id;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  throw std::runtime_error("HCCL unique ID timed out");
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) throw std::invalid_argument("expected rank and HCCL unique ID path");
  const auto rank = static_cast<uint32_t>(std::stoul(argv[1]));
  if (rank >= 8) throw std::invalid_argument("rank must be less than 8");
  constexpr uint64_t kLocalVocab = 19072;
  gaudirt::runtime::DeviceRuntime::Config config{};
  config.module_id = rank;
  config.rank = rank;
  config.world_size = 8;
  gaudirt::runtime::DeviceRuntime runtime(config);
  gaudirt::runtime::HbmArena arena(runtime.device_id(), 16 * 1024 * 1024);
  runtime.init_comm(exchange_id(rank, argv[2]));
  const uint64_t local = arena.allocate(kLocalVocab * sizeof(uint16_t));
  const uint64_t gathered = arena.allocate(8 * kLocalVocab * sizeof(uint16_t));
  std::vector<uint16_t> input(kLocalVocab), output(8 * kLocalVocab);
  for (uint64_t i = 0; i < kLocalVocab; ++i)
    input[i] = static_cast<uint16_t>(rank * kLocalVocab + i);
  arena.copy_h2d_sync(runtime.h2d_stream(), input.data(), local, input.size() * sizeof(uint16_t));
  if (hcclAllGather(reinterpret_cast<const void*>(local), reinterpret_cast<void*>(gathered),
                    kLocalVocab, hcclBfloat16, runtime.comm(), runtime.compute_stream()) != hcclSuccess)
    throw std::runtime_error("vocabulary HCCL AllGather failed");
  runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(), gathered, output.data(), output.size() * sizeof(uint16_t));
  uint64_t mismatches = 0;
  for (uint64_t i = 0; i < output.size(); ++i)
    mismatches += output[i] != static_cast<uint16_t>(i);
  std::printf("{\"rank\":%u,\"mismatches\":%llu}\n", rank,
              static_cast<unsigned long long>(mismatches));
  return mismatches == 0 ? 0 : 1;
}
