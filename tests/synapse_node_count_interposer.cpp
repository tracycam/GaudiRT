// Count graph-construction calls in the split regression test.
#include <synapse_api.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {
std::atomic<uint64_t> split_nodes{0};
}

extern "C" uint64_t gaudirt_test_split_node_count() { return split_nodes.load(); }

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
