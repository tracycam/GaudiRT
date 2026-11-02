// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/bridge/rank_lifecycle.hpp"

#include "gaudirt/runtime/device_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace gaudirt::bridge {
namespace {

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  const uint64_t bytes = std::filesystem::file_size(path);
  std::vector<uint8_t> data(bytes);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(data.data()), bytes))
    throw std::runtime_error("PrivateUse1: cannot read HCCL unique ID");
  return data;
}

void write_file(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
  std::ofstream output(path, std::ios::binary);
  if (!output.write(reinterpret_cast<const char*>(data.data()), data.size()))
    throw std::runtime_error("PrivateUse1: cannot write HCCL unique ID");
}

}  // namespace

hcclUniqueId exchange_unique_id(uint32_t rank, const std::filesystem::path& path) {
  hcclUniqueId id;
  if (rank == 0) {
    if (hcclGetUniqueId(&id) != hcclSuccess)
      throw std::runtime_error("PrivateUse1: hcclGetUniqueId failed");
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
  throw std::runtime_error("PrivateUse1: HCCL unique ID timeout");
}

void synchronize_shutdown(runtime::DeviceRuntime& device, uint64_t barrier_buffer) {
  device.synchronize_all();
  auto* buffer = reinterpret_cast<void*>(barrier_buffer);
  if (hcclAllReduce(buffer, buffer, 1, hcclFloat32, hcclSum,
                    device.comm(), device.compute_stream()) != hcclSuccess)
    throw std::runtime_error("PrivateUse1: shutdown rank barrier failed");
  device.synchronize_compute();
}

}  // namespace gaudirt::bridge
