// Stress repeated cross-stream record/wait on two persistent Synapse events.
#include "gaudirt/runtime/device_runtime.hpp"
#include "test_device_slab.hpp"
#include "gaudirt/runtime/pinned_host_pool.hpp"

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
    auto event_count = reinterpret_cast<uint64_t (*)()>(dlsym(RTLD_DEFAULT, "gaudirt_test_event_create_count"));
    if (!event_count) throw std::runtime_error("event counter interposer is required");
    constexpr uint64_t steps = 512, bytes = 256;
    DeviceRuntime runtime({});
    gaudirt::testing::TestDeviceSlab arena(runtime.device_id(), 4ULL * 1024 * 1024);
    void* memory = nullptr;
    check(synHostMalloc(runtime.device_id(), steps * bytes, 0, &memory), "host allocation");
    auto* input = static_cast<uint8_t*>(memory);
    for (uint64_t step = 0; step < steps; ++step)
      for (uint64_t byte = 0; byte < bytes; ++byte)
        input[step * bytes + byte] = static_cast<uint8_t>((step * 37 + byte * 19) & 255);
    const auto device_input = arena.allocate(bytes);
    const auto device_output = arena.allocate(steps * bytes);
    StreamEvent reusable(runtime.device_id()), ready(runtime.device_id());
    const auto before = event_count();
    for (uint64_t step = 0; step < steps; ++step) {
      reusable.record_and_wait(runtime.compute_stream(), runtime.h2d_stream());
      check(synMemCopyAsync(runtime.h2d_stream(), reinterpret_cast<uint64_t>(input + step * bytes), bytes,
                            device_input, HOST_TO_DRAM), "H2D");
      ready.record_and_wait(runtime.h2d_stream(), runtime.compute_stream());
      check(synMemCopyAsync(runtime.compute_stream(), device_input, bytes,
                            device_output + step * bytes, DRAM_TO_DRAM), "D2D");
    }
    runtime.synchronize_compute();
    if (event_count() != before) throw std::runtime_error("event created during steady steps");
    std::vector<uint8_t> output(steps * bytes);
    arena.copy_d2h_sync(runtime.d2h_stream(), device_output, output.data(), output.size());
    for (uint64_t i = 0; i < output.size(); ++i)
      if (output[i] != input[i]) throw std::runtime_error("cross-stream event ordering failed");
    check(synHostFree(runtime.device_id(), memory, 0), "host free");
    std::cout << "persistent event steps " << steps << " event_creates_in_steps 0 mismatches 0\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
