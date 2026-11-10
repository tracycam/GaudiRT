// Count driver calls and coordinate the concurrent-launch regression test.
#include <synapse_api.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>

namespace {
std::atomic<uint64_t> split_nodes{0};
std::atomic<uint64_t> d2d_copies{0};
std::atomic<uint64_t> h2d_copies{0};
std::atomic<synStreamHandle> last_h2d_stream{nullptr};
std::atomic<uint64_t> h2d_stream_waits{0};
std::atomic<uint64_t> d2h_copies{0};
std::atomic<uint64_t> event_creates{0};
std::atomic<uint64_t> event_destroys{0};
std::atomic<uint64_t> event_synchronizes{0};
std::atomic<uint64_t> stream_event_creates{0};
std::atomic<uint64_t> launches{0};
std::atomic<bool> launch_gate_enabled{false};
std::mutex launch_gate_mutex;
std::condition_variable launch_gate_cv;
unsigned launch_gate_waiting = 0;
bool launch_gate_timed_out = false;
}

extern "C" uint64_t gaudirt_test_split_node_count() { return split_nodes.load(); }
extern "C" uint64_t gaudirt_test_d2d_copy_count() { return d2d_copies.load(); }
extern "C" uint64_t gaudirt_test_h2d_copy_count() { return h2d_copies.load(); }
extern "C" uint64_t gaudirt_test_h2d_stream_wait_count() { return h2d_stream_waits.load(); }
extern "C" uint64_t gaudirt_test_d2h_copy_count() { return d2h_copies.load(); }
extern "C" uint64_t gaudirt_test_event_create_count() { return event_creates.load(); }
extern "C" uint64_t gaudirt_test_event_destroy_count() { return event_destroys.load(); }
extern "C" uint64_t gaudirt_test_event_synchronize_count() { return event_synchronizes.load(); }
extern "C" uint64_t gaudirt_test_stream_event_create_count() { return stream_event_creates.load(); }
extern "C" uint64_t gaudirt_test_launch_count() { return launches.load(); }
extern "C" void gaudirt_test_launch_gate_arm() {
  std::lock_guard lock(launch_gate_mutex);
  launch_gate_waiting = 0;
  launch_gate_timed_out = false;
  launch_gate_enabled.store(true);
}
extern "C" bool gaudirt_test_launch_gate_passed() {
  std::lock_guard lock(launch_gate_mutex);
  return launch_gate_waiting == 2 && !launch_gate_timed_out;
}

extern "C" synStatus SYN_API_CALL synLaunch(
    synStreamHandle stream, const synLaunchTensorInfo* tensors, uint32_t count,
    uint64_t workspace, synRecipeHandle recipe, uint32_t flags) {
  ++launches;
  using Function = decltype(&synLaunch);
  static const auto real = [] {
    void* library = dlopen("libSynapse.so", RTLD_NOW | RTLD_NOLOAD);
    if (!library) std::abort();
    auto function = reinterpret_cast<Function>(dlsym(library, "synLaunch"));
    if (!function) std::abort();
    return function;
  }();
  if (launch_gate_enabled.load()) {
    std::unique_lock lock(launch_gate_mutex);
    if (launch_gate_enabled.load()) {
      ++launch_gate_waiting;
      if (launch_gate_waiting == 2) {
        launch_gate_enabled.store(false);
        launch_gate_cv.notify_all();
      } else if (!launch_gate_cv.wait_for(lock, std::chrono::seconds(10),
                                          [] { return launch_gate_waiting == 2; })) {
        launch_gate_timed_out = true;
        launch_gate_enabled.store(false);
      }
    }
  }
  return real(stream, tensors, count, workspace, recipe, flags);
}

extern "C" synStatus SYN_API_CALL synMemCopyAsync(
    synStreamHandle stream, uint64_t source, uint64_t bytes, uint64_t destination, synDmaDir direction) {
  using Function = decltype(&synMemCopyAsync);
  static const auto real = [] {
    void* library = dlopen("libSynapse.so", RTLD_NOW | RTLD_NOLOAD);
    if (!library) std::abort();
    auto function = reinterpret_cast<Function>(dlsym(library, "synMemCopyAsync"));
    if (!function) std::abort();
    return function;
  }();
  if (direction == DRAM_TO_DRAM) ++d2d_copies;
  if (direction == HOST_TO_DRAM) { ++h2d_copies; last_h2d_stream.store(stream); }
  if (direction == DRAM_TO_HOST) ++d2h_copies;
  return real(stream, source, bytes, destination, direction);
}

extern "C" synStatus SYN_API_CALL synStreamWaitEvent(
    synStreamHandle stream, synEventHandle event, uint32_t flags) {
  using Function = decltype(&synStreamWaitEvent);
  static const auto real = [] {
    void* library = dlopen("libSynapse.so", RTLD_NOW | RTLD_NOLOAD);
    if (!library) std::abort();
    auto function = reinterpret_cast<Function>(dlsym(library, "synStreamWaitEvent"));
    if (!function) std::abort();
    return function;
  }();
  if (stream == last_h2d_stream.load()) ++h2d_stream_waits;
  return real(stream, event, flags);
}

extern "C" synStatus SYN_API_CALL synEventCreate(synEventHandle* event, synDeviceId device, uint32_t flags) {
  using Function = decltype(&synEventCreate);
  static const auto real = [] {
    void* library = dlopen("libSynapse.so", RTLD_NOW | RTLD_NOLOAD);
    if (!library) std::abort();
    auto function = reinterpret_cast<Function>(dlsym(library, "synEventCreate"));
    if (!function) std::abort();
    return function;
  }();
  Dl_info caller{};
  if (dladdr(__builtin_return_address(0), &caller) && caller.dli_sname &&
      std::strstr(caller.dli_sname, "11StreamEventC")) ++stream_event_creates;
  ++event_creates;
  return real(event, device, flags);
}

extern "C" synStatus SYN_API_CALL synEventDestroy(synEventHandle event) {
  using Function = decltype(&synEventDestroy);
  static const auto real = [] {
    void* library = dlopen("libSynapse.so", RTLD_NOW | RTLD_NOLOAD);
    if (!library) std::abort();
    auto function = reinterpret_cast<Function>(dlsym(library, "synEventDestroy"));
    if (!function) std::abort();
    return function;
  }();
  ++event_destroys;
  return real(event);
}

extern "C" synStatus SYN_API_CALL synEventSynchronize(synEventHandle event) {
  using Function = decltype(&synEventSynchronize);
  static const auto real = [] {
    void* library = dlopen("libSynapse.so", RTLD_NOW | RTLD_NOLOAD);
    if (!library) std::abort();
    auto function = reinterpret_cast<Function>(dlsym(library, "synEventSynchronize"));
    if (!function) std::abort();
    return function;
  }();
  ++event_synchronizes;
  return real(event);
}

extern "C" synStatus SYN_API_CALL synNodeCreate(
    synGraphHandle graph, const synTensor* inputs, const synTensor* outputs,
    uint32_t input_count, uint32_t output_count, const void* params, unsigned params_size,
    const char* guid, const char* name, const char** input_layouts, const char** output_layouts) {
  using Function = decltype(&synNodeCreate);
  static const auto real = [] {
    void* library = dlopen("libSynapse.so", RTLD_NOW | RTLD_NOLOAD);
    if (!library) std::abort();
    auto function = reinterpret_cast<Function>(dlsym(library, "synNodeCreate"));
    if (!function) std::abort();
    return function;
  }();
  if (guid && std::strcmp(guid, "split") == 0) ++split_nodes;
  return real(graph, inputs, outputs, input_count, output_count, params, params_size,
              guid, name, input_layouts, output_layouts);
}
