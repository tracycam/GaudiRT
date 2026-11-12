// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/common/sha256.hpp"
#include "gaudirt/runtime/block_caching_allocator.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/pinned_host_pool.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace gaudirt::runtime;

namespace {

void test_sha256_vectors() {
  assert(gaudirt::common::sha256_bytes(nullptr, 0) ==
         "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  const uint8_t abc[] = {'a', 'b', 'c'};
  assert(gaudirt::common::sha256_bytes(abc, sizeof(abc)) ==
         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  const std::vector<uint8_t> million_a(1000000, 'a');
  assert(gaudirt::common::sha256_bytes(million_a.data(), million_a.size()) ==
         "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

void test_recipe_cache_contract() {
  RecipeCache disabled;
  assert(!disabled.enabled());
  assert(disabled.load("invalid/key") == nullptr);
  disabled.save(nullptr, "invalid/key");
  assert(disabled.stats().first == 0 && disabled.stats().second == 0);

  setenv("GC_KERNEL_PATH", "/usr/lib/habanalabs/libtpc_kernels.so", 0);
  RecipeCache enabled(std::filesystem::temp_directory_path() / "gaudirt_cache_test");
  bool rejected = false;
  try {
    static_cast<void>(enabled.load("invalid/key"));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  assert(enabled.stats().first == 0 && enabled.stats().second == 0);
}

synRecipeHandle compile_test_add_recipe(const std::string& recipe_name,
                                        TSize elements,
                                        RecipeCache* cache) {
  synGraphHandle graph = nullptr;
  assert(synGraphCreate(&graph, synDeviceGaudi2) == synSuccess);

  auto make_persistent_tensor = [&](const char* name, synSectionHandle* out_sec) {
    assert(synSectionCreate(out_sec, 0, graph) == synSuccess);
    assert(synSectionSetPersistent(*out_sec, true) == synSuccess);
    synTensorDescriptor desc{};
    desc.m_dataType = syn_type_float;
    desc.m_dims = 1;
    desc.m_sizes[0] = elements;
    desc.m_name = name;
    synTensor tensor = nullptr;
    assert(synTensorCreate(&tensor, &desc, *out_sec, 0) == synSuccess);
    return tensor;
  };

  synSectionHandle sec_a = nullptr;
  synSectionHandle sec_b = nullptr;
  synSectionHandle sec_out = nullptr;
  synTensor t_a = make_persistent_tensor("in_a", &sec_a);
  synTensor t_b = make_persistent_tensor("in_b", &sec_b);
  synTensor t_out = make_persistent_tensor("out_c", &sec_out);

  synTensor inputs[2] = {t_a, t_b};
  synTensor outputs[1] = {t_out};
  assert(synNodeCreate(graph, inputs, outputs, 2, 1, nullptr, 0, "add_fwd_f32",
                       "add_node", nullptr, nullptr) == synSuccess);

  synRecipeHandle recipe = compile_cached(graph, recipe_name, "add_f32_1024", cache);

  synGraphDestroy(graph);
  synTensorDestroy(t_a);
  synTensorDestroy(t_b);
  synTensorDestroy(t_out);
  synSectionDestroy(sec_a);
  synSectionDestroy(sec_b);
  synSectionDestroy(sec_out);
  return recipe;
}

void test_foundation_on_hardware() {
  setenv("GC_KERNEL_PATH", "/usr/lib/habanalabs/libtpc_kernels.so", 0);
  DeviceRuntime::Config cfg{};
  cfg.module_id = 0;
  cfg.uclamp_min = 1024;

  DeviceRuntime rt(cfg);
  BlockCachingAllocator allocator(rt.device_id(), 16ULL * 1024 * 1024);
  allocator.prepare_stream(rt.compute_stream());
  PinnedHostPool pinned(rt.device_id(), 1024 * 1024);

  constexpr TSize kElems = 1024;
  constexpr uint64_t kBytes = kElems * sizeof(float);

  const auto recipe_root = std::filesystem::temp_directory_path() /
                           ("gaudirt_test_add_" + std::to_string(getpid()));
  RecipeCache disk_cache(recipe_root);
  synRecipeHandle cold = compile_test_add_recipe("gaudirt_test_add", kElems, &disk_cache);
  assert(disk_cache.stats().first == 0 && disk_cache.stats().second == 1);
  assert(synRecipeDestroy(cold) == synSuccess);

  synRecipeHandle warm = disk_cache.load("add_f32_1024");
  assert(warm != nullptr && disk_cache.stats().first == 1);
  std::filesystem::remove_all(recipe_root);

  uint64_t ws_bytes = 0;
  assert(synWorkspaceGetSize(&ws_bytes, warm) == synSuccess);

  uint64_t addr_a = allocator.allocate(kBytes, rt.compute_stream());
  uint64_t addr_b = allocator.allocate(kBytes, rt.compute_stream());
  uint64_t addr_out = allocator.allocate(kBytes, rt.compute_stream());
  uint64_t addr_ws = ws_bytes > 0 ? allocator.allocate(ws_bytes, rt.compute_stream()) : 0;

  auto* host_a = static_cast<float*>(pinned.allocate(kBytes));
  auto* host_b = static_cast<float*>(pinned.allocate(kBytes));
  auto* host_out = static_cast<float*>(pinned.allocate(kBytes));
  std::fill_n(host_a, kElems, 2.5f);
  std::fill_n(host_b, kElems, 1.5f);
  std::fill_n(host_out, kElems, 0.0f);

  pinned.copy_h2d(addr_a, host_a, kBytes, rt.h2d_stream(), false);
  pinned.copy_h2d(addr_b, host_b, kBytes, rt.h2d_stream(), false);
  pinned.copy_d2h(host_out, addr_a, kBytes, rt.d2h_stream(), false);
  for (size_t i = 0; i < kElems; ++i) assert(host_out[i] == 2.5f);

  std::array<synLaunchTensorInfo, 3> launch = {{
      {"in_a", addr_a, DATA_TENSOR, {kElems}, 0},
      {"in_b", addr_b, DATA_TENSOR, {kElems}, 0},
      {"out_c", addr_out, DATA_TENSOR, {kElems}, 0},
  }};
  initialize_launch_tensor_ids(warm, launch.data(), launch.size());
  assert(synLaunch(rt.compute_stream(), launch.data(), launch.size(), addr_ws, warm, 0) ==
         synSuccess);
  rt.synchronize_compute();

  std::fill_n(host_out, kElems, 0.0f);
  pinned.copy_d2h(host_out, addr_out, kBytes, rt.d2h_stream(), false);
  for (size_t i = 0; i < kElems; ++i) assert(host_out[i] == 4.0f);

  assert(synRecipeDestroy(warm) == synSuccess);
  pinned.free(host_out);
  pinned.free(host_b);
  pinned.free(host_a);
  if (addr_ws) allocator.free(addr_ws);
  allocator.free(addr_out);
  allocator.free(addr_b);
  allocator.free(addr_a);
}

void test_invalid_module_fails() {
  DeviceRuntime::Config cfg{};
  cfg.module_id = UINT32_MAX;
  bool rejected = false;
  try {
    DeviceRuntime invalid(cfg);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
}

}  // namespace

int main(int argc, char** argv) {
  test_sha256_vectors();
  test_recipe_cache_contract();
  if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) {
    std::printf("[GaudiRT] Host-only runtime_unit_tests PASSED.\n");
    return 0;
  }
  test_foundation_on_hardware();
  test_invalid_module_fails();
  std::printf("[GaudiRT] Host tests and 1-card foundation PASSED.\n");
  return 0;
}
