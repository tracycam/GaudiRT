// Count allocator stream registrations while preserving the real allocator.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>

namespace {
std::atomic<uint64_t> calls{0};
using Record = void (*)(void*, uint64_t, void*);
constexpr const char* symbol = "_ZN7gaudirt7runtime21BlockCachingAllocator13record_streamEmP20InternalStreamHandle";
}

extern "C" void counted_record(void*, uint64_t, void*) __asm__(
    "_ZN7gaudirt7runtime21BlockCachingAllocator13record_streamEmP20InternalStreamHandle");
extern "C" void counted_record(void* self, uint64_t address, void* stream) {
  static const auto real = [] {
    const char* path = std::getenv("GAUDIRT_TEST_EXTENSION");
    void* module = path ? dlopen(path, RTLD_NOW | RTLD_NOLOAD) : nullptr;
    auto function = module ? reinterpret_cast<Record>(dlsym(module, symbol)) : nullptr;
    if (!function) std::abort();
    return function;
  }();
  calls.fetch_add(1, std::memory_order_relaxed);
  real(self, address, stream);
}

extern "C" void gaudirt_test_reset_records() { calls.store(0, std::memory_order_relaxed); }
extern "C" uint64_t gaudirt_test_records() { return calls.load(std::memory_order_relaxed); }
