// SPDX-License-Identifier: Apache-2.0
// Merged pinned blocks wait for the latest fence on each distinct stream.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/pinned_host_pool.hpp"

#include <array>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>

using namespace gaudirt::runtime;

int main() {
  try {
    const auto synchronizes = reinterpret_cast<uint64_t (*)()>(
        dlsym(RTLD_DEFAULT, "gaudirt_test_event_synchronize_count"));
    if (!synchronizes) throw std::runtime_error("event counter interposer is required");
    constexpr uint64_t bytes = 256;
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 1024 * 1024);
    PinnedHostPool pool(runtime.device_id(), 3 * bytes);
    std::array<uint8_t, 3 * bytes> input{};
    for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<uint8_t>((i * 37 + 19) & 255);
    const auto device = arena.allocate(input.size());
    arena.copy_h2d_sync(runtime.h2d_stream(), input.data(), device, input.size());

    for (const bool mixed_streams : {false, true}) {
      std::array<void*, 3> host{};
      for (auto& address : host) address = pool.allocate(bytes);
      const std::array streams{runtime.d2h_stream(),
                               mixed_streams ? runtime.compute_stream() : runtime.d2h_stream(),
                               runtime.d2h_stream()};
      for (size_t i = 0; i < host.size(); ++i)
        pool.copy_d2h(host[i], device + i * bytes, bytes, streams[i], true);
      pool.free(host[0]);
      pool.free(host[2]);
      pool.free(host[1]);
      const auto before = synchronizes();
      auto* merged = static_cast<uint8_t*>(pool.allocate(3 * bytes));
      const auto waits = synchronizes() - before;
      if (waits != (mixed_streams ? 2 : 1))
        throw std::runtime_error("fence waits=" + std::to_string(waits) +
                                 ", expected=" + std::to_string(mixed_streams ? 2 : 1));
      if (merged != host[0]) throw std::runtime_error("pinned blocks did not coalesce");
      for (size_t i = 0; i < input.size(); ++i)
        if (merged[i] != input[i]) throw std::runtime_error("pinned data reused before a transfer completed");
      pool.free(merged);
    }
    std::cout << "pinned coalescing waits: one shared stream = 1, two distinct streams = 2; data correct\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
