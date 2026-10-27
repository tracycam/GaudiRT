// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/inflight_credit_manager.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include "gaudirt/scal/long_so.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

using namespace gaudirt::runtime;

namespace {

void test_long_so_encoding_and_monitor_expansion() {
  using namespace gaudirt::scal;
  const uint64_t targets[] = {0,
                              1,
                              0x7FFF,
                              0x8000,
                              0x8001,
                              (1ULL << 30) - 1,
                              1ULL << 30,
                              1ULL << 45,
                              (1ULL << 60) - 1};
  for (const uint64_t target : targets) {
    assert(long_so_target_valid(target));
    assert(long_so_target(long_so_chunks(target)) == target);
  }
  assert(!long_so_target_valid(1ULL << 60));

  const std::array<uint8_t, 16> native_wait = {
      0x0a, 0xb4, 0x4c, 0x00, 0x02, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03};
  sched_arc_cmd_wait_for_ext_signal_t wait{};
  std::memcpy(&wait, native_wait.data(), sizeof(wait));
  uint32_t first_dword = 0;
  std::memcpy(&first_dword, &wait, sizeof(first_dword));
  for (const uint64_t target : targets) {
    const uint8_t reserved = wait.reserved0;
    assert(set_wait_for_ext_signal_target(wait, target));
    assert(wait_for_ext_signal_target(wait) == target);
    uint32_t current_dword = 0;
    std::memcpy(&current_dword, &wait, sizeof(current_dword));
    assert(current_dword == first_dword);
    assert(wait.reserved0 == reserved);
    assert(std::memcmp(wait.engine_group_type, native_wait.data() + 12, 4) == 0);
  }
  const uint64_t old_target = wait_for_ext_signal_target(wait);
  assert(!set_wait_for_ext_signal_target(wait, 1ULL << 60));
  assert(wait_for_ext_signal_target(wait) == old_target);

  constexpr uint32_t base = 0xFC70F080;
  LongMonitorState state;
  for (unsigned chunk = kLongSoChunks; chunk-- > 0;) {
    const uint32_t data = monitor_arm_data(0x200B8, chunk, 1);
    state.observe(base + 4 * chunk, data);
  }
  std::vector<sched_arc_lbw_write_t> writes = {{base, 0x200B8}};
  assert(state.rewrite(writes, 0x7FFF) && writes.size() == 1);
  assert(((writes[0].data >> 17) & 0x7FFF) == 0x7FFF);

  const bool crossed = state.rewrite(writes, 0x8000);
  assert(crossed && writes.size() == 2);
  assert(writes[0].addr == base + 4 && writes[1].addr == base);
  assert(((writes[0].data >> 17) & 0x7FFF) == 1);
  assert((writes[0].data & 0xFF00) == 0 && (writes[0].data & 0xFF) == 0);
  assert(state.chunk1_writes() == 1 && state.high_chunk_writes() == 1);
  writes.erase(writes.begin());
  assert(state.rewrite(writes, 0x8001) && writes.size() == 1);
  assert(state.chunk1_writes() == 1);
  assert(state.target_chunk(base, 0) == 1 && state.target_chunk(base, 1) == 1);
  assert(state.target_chunk(base, 2) == 0 && state.target_chunk(base, 3) == 0);
  assert(state.rewrite(writes, 0x8003) && writes.size() == 1);
  assert(state.target_chunk(base, 0) == 3 && state.target_chunk(base, 1) == 1);

  assert(state.rewrite(writes, 1ULL << 30) && writes.size() == 3);
  assert(writes[0].addr == base + 8 && writes[1].addr == base + 4 &&
         writes[2].addr == base);
  assert(state.chunk1_writes() == 2 && state.high_chunk_writes() == 3);
  writes = {{0xFC300100, 0x12345678}};
  assert(state.rewrite(writes, 0x8001));
  assert(writes.size() == 1 && writes[0].data == 0x12345678);
  assert(!state.rewrite(writes, 1ULL << 60));
}

void test_credit_formula() {
  assert(InFlightCreditManager::max_launches_for_distinct(1, 575) == 573);
  assert(InFlightCreditManager::max_launches_for_distinct(12, 575) == 551);
  assert(InFlightCreditManager::max_launches_for_distinct(36, 575) == 503);
  assert(InFlightCreditManager::max_launches_for_distinct(72, 575) == 431);
  assert(InFlightCreditManager::max_launches_for_distinct(140, 575) == 295);
  assert(InFlightCreditManager::max_launches_for_distinct(144, 575) == 287);
}

void test_binding_table_value_semantics_and_dtype() {
  assert(tensor_byte_size(syn_type_int4, 3) == 2);
  assert(tensor_byte_size(syn_type_uint4, 1) == 1);
  assert(tensor_byte_size(syn_type_uint4, 9) == 5);
  assert(tensor_byte_size(syn_type_float, 1024) == 4096);
  bool rejected = false;
  try {
    tensor_byte_size(syn_type_na, 1);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  assert(tensor_byte_size(syn_type_int4, UINT64_MAX) == UINT64_MAX / 2 + 1);
  bool overflow = false;
  try {
    tensor_byte_size(syn_type_float, UINT64_MAX);
  } catch (const std::overflow_error&) {
    overflow = true;
  }
  assert(overflow);

  std::vector<TensorSlotMeta> slots(2);
  slots[0].name = "x";
  slots[0].tensor_id = 1;
  slots[1].name = "out";
  slots[1].tensor_id = 2;
  BindingTable original(slots);
  original.bind_name("x", 0x1000);
  original.bind_name("out", 0x2000);

  auto verify = [](const BindingTable& table) {
    const auto& infos = table.launch_infos();
    assert(infos.size() == 2);
    assert(std::strcmp(infos[0].tensorName, "x") == 0);
    assert(std::strcmp(infos[1].tensorName, "out") == 0);
    assert(infos[0].pTensorAddress == 0x1000);
    assert(infos[1].pTensorAddress == 0x2000);
  };

  std::vector<BindingTable> tables;
  for (size_t i = 0; i < 8; ++i) tables.push_back(original);
  BindingTable copied = tables.front();
  copied = tables.back();
  BindingTable moved = std::move(copied);
  tables.push_back(std::move(moved));
  for (const auto& table : tables) verify(table);

  std::vector<TensorSlotMeta> one_slot(1);
  one_slot[0].name = "a_much_longer_tensor_name_for_heap_storage";
  BindingTable assigned(original);
  assigned = BindingTable(one_slot);
  assert(assigned.size() == 1);
  assert(std::strcmp(assigned.launch_infos()[0].tensorName,
                     one_slot[0].name.c_str()) == 0);
  assigned = std::move(original);
  verify(assigned);
  auto move_assign = [](BindingTable& to, BindingTable&& from) {
    to = std::move(from);
  };
  move_assign(assigned, std::move(assigned));
  verify(assigned);
}

synRecipeHandle compile_test_add_recipe(const std::string& recipe_name,
                                        TSize elements) {
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

  synRecipeHandle recipe = nullptr;
  assert(synGraphCompile(&recipe, graph, recipe_name.c_str(), nullptr) ==
         synSuccess);

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
  DeviceRuntime::Config cfg{};
  cfg.module_id = 0;
  cfg.rank = 0;
  cfg.world_size = 1;
  cfg.uclamp_min = 1024;
  cfg.uclamp_max = 1024;

  DeviceRuntime rt(cfg);
  HbmArena arena(rt.device_id(), 64ULL * 1024 * 1024);
  RecipeRegistry registry;

  constexpr TSize kElems = 1024;
  constexpr uint64_t kBytes = kElems * sizeof(float);

  synRecipeHandle raw_recipe = compile_test_add_recipe("gaudirt_test_add", kElems);
  auto id_mem = registry.register_compiled("add_mem", raw_recipe);

  const std::string recipe_path =
      "/tmp/gaudirt_test_add_" + std::to_string(getpid()) + ".recipe";
  registry.save_to_file(id_mem, recipe_path);
  auto id_disk = registry.load_from_file("add_disk", recipe_path);
  unlink(recipe_path.c_str());

  const auto& disk_entry = registry.get(id_disk);
  assert(disk_entry.slots.size() == 3);
  for (const auto& slot : disk_entry.slots) assert(slot.byte_size == kBytes);

  uint64_t addr_a = arena.allocate(kBytes);
  uint64_t addr_b = arena.allocate(kBytes);
  uint64_t addr_out = arena.allocate(kBytes);
  std::vector<float> host_a(kElems, 2.5f);
  std::vector<float> host_b(kElems, 1.5f);
  std::vector<float> host_out(kElems, 0.0f);
  arena.copy_h2d_sync(rt.h2d_stream(), host_a.data(), addr_a, kBytes);
  arena.copy_h2d_sync(rt.h2d_stream(), host_b.data(), addr_b, kBytes);
  arena.copy_d2h_sync(rt.d2h_stream(), addr_a, host_out.data(), kBytes);
  for (const float value : host_out) assert(value == 2.5f);

  BindingTable bindings = registry.create_binding_table(id_disk);
  bindings.bind_name("in_a", addr_a);
  bindings.bind_name("in_b", addr_b);
  bindings.bind_name("out_c", addr_out);
  assert(bindings.address_at(bindings.find_slot("in_a")) == addr_a);
  assert(bindings.address_at(bindings.find_slot("in_b")) == addr_b);
  assert(bindings.address_at(bindings.find_slot("out_c")) == addr_out);
  const uint64_t generation = bindings.generation();
  bindings.bind_slot(bindings.find_slot("in_b"), addr_a);
  assert(bindings.address_at(bindings.find_slot("in_b")) == addr_a);
  assert(bindings.generation() == generation + 1);
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
  test_long_so_encoding_and_monitor_expansion();
  test_credit_formula();
  test_binding_table_value_semantics_and_dtype();
  if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) {
    std::printf("[GaudiRT] Host-only runtime_unit_tests PASSED.\n");
    return 0;
  }
  test_foundation_on_hardware();
  test_invalid_module_fails();
  std::printf("[GaudiRT] Host tests and 1-card foundation PASSED.\n");
  return 0;
}
