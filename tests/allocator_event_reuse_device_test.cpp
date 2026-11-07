// Re-record pooled allocator events across three streams without host waits.
#include "gaudirt/runtime/block_caching_allocator.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"

#include <array>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace gaudirt::runtime;

namespace {
void check(synStatus status, const char* stage) {
  if (status != synSuccess) throw std::runtime_error(std::string(stage) + ": " + std::to_string(int(status)));
}
}

int main() {
  try {
    const auto creates = reinterpret_cast<uint64_t (*)()>(dlsym(RTLD_DEFAULT, "gaudirt_test_event_create_count"));
    const auto destroys = reinterpret_cast<uint64_t (*)()>(dlsym(RTLD_DEFAULT, "gaudirt_test_event_destroy_count"));
    if (!creates || !destroys) throw std::runtime_error("event counter interposer is required");
    constexpr uint64_t steps = 512, bytes = 256;
    DeviceRuntime runtime({});
    HbmArena arena(runtime.device_id(), 4ULL * 1024 * 1024);
    BlockCachingAllocator pool(runtime.device_id(), bytes);
    void* memory = nullptr;
    check(synHostMalloc(runtime.device_id(), steps * bytes, 0, &memory), "host allocation");
    auto* input = static_cast<uint8_t*>(memory);
    for (uint64_t step = 0; step < steps; ++step)
      for (uint64_t byte = 0; byte < bytes; ++byte)
        input[step * bytes + byte] = static_cast<uint8_t>((step * 37 + byte * 19) & 255);
    const auto output = arena.allocate(steps * bytes);
    const std::array streams{runtime.compute_stream(), runtime.h2d_stream(), runtime.d2h_stream()};
    auto step = [&](uint64_t i) {
      const auto producer = streams[i % streams.size()];
      const auto consumer = streams[(i + 1) % streams.size()];
      const auto target = pool.allocate(bytes, producer);
      check(synMemCopyAsync(producer, reinterpret_cast<uint64_t>(input + i * bytes), bytes,
                            target, HOST_TO_DRAM), "H2D");
      pool.free(target);
      const auto reused = pool.allocate(bytes, consumer);
      if (reused != target) throw std::runtime_error("allocation not reused");
      check(synMemCopyAsync(consumer, reused, bytes, output + i * bytes, DRAM_TO_DRAM), "D2D");
      pool.free(reused);
    };
    // Prime all three producer streams before checking steady-state events.
    for (uint64_t i = 0; i < streams.size(); ++i) step(i);
    const auto before_create = creates(), before_destroy = destroys();
    for (uint64_t i = streams.size(); i < steps; ++i) step(i);
    if (creates() != before_create || destroys() != before_destroy)
      throw std::runtime_error("event created or destroyed after allocator warmup");
    for (const auto stream : streams) check(synStreamSynchronize(stream), "stream wait");
    std::vector<uint8_t> result(steps * bytes);
    arena.copy_d2h_sync(runtime.d2h_stream(), output, result.data(), result.size());
    for (uint64_t i = 0; i < result.size(); ++i)
      if (result[i] != input[i]) throw std::runtime_error("pooled event ordering failed");
    check(synHostFree(runtime.device_id(), memory, 0), "host free");
    std::cout << "allocator pooled event steps " << steps
              << " warm_creates 0 warm_destroys 0 mismatches 0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
