// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/common/sha256.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/inflight_credit_manager.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include "gaudirt/runtime/tp_reduce.hpp"
#include "gaudirt/scal/long_so.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
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

void test_weight_manifest() {
  const std::vector<uint8_t> million_a(1000000, 'a');
  assert(gaudirt::common::sha256_bytes(million_a.data(), million_a.size()) ==
         "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("gaudirt_weight_manifest_" + std::to_string(getpid()));
  std::filesystem::create_directories(directory);
  const std::string digest =
      "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
  {
    std::ofstream raw(directory / "x.bin", std::ios::binary);
    raw << "abc";
  }
  {
    std::ofstream manifest(directory / "weights.manifest");
    manifest << "GAUDIRT_WEIGHTS_V1\n"
             << "T x uint8 3 1 3 x.bin " << digest << '\n'
             << "T e4m3 fp8_e4m3 3 1 3 x.bin " << digest << '\n'
             << "T e5m2 fp8_e5m2 3 1 3 x.bin " << digest << '\n';
  }
  const auto tensors = WeightManifest::read_manifest(directory);
  assert(tensors.size() == 3);
  assert(tensors[0].name == "x" && tensors[0].dtype == syn_type_uint8);
  assert(tensors[0].elements == 3 && tensors[0].bytes == 3);
  assert(tensors[1].dtype == syn_type_fp8_143);
  assert(tensors[2].dtype == syn_type_fp8_152);
  {
    std::ofstream raw(directory / "x.bin", std::ios::binary);
    raw << "abd";
  }
  bool rejected = false;
  try {
    WeightManifest::read_manifest(directory);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  {
    std::ofstream manifest(directory / "weights.manifest");
    manifest << "GAUDIRT_WEIGHTS_V1\n"
             << "T x fp32 1 1 1 ../x.bin "
             << std::string(64, '0') << '\n';
  }
  rejected = false;
  try {
    WeightManifest::read_manifest(directory);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);

  WeightTensor borrowed;
  borrowed.name = "borrowed";
  borrowed.dtype = syn_type_bf16;
  borrowed.shape = {2, 4};
  borrowed.elements = 8;
  borrowed.bytes = 16;
  borrowed.device_address = 0x1000;
  WeightManifest external(std::vector<WeightTensor>{borrowed});
  assert(external.get("borrowed").device_address == 0x1000);
  auto rejects = [](WeightTensor tensor) {
    try {
      WeightManifest invalid(std::vector<WeightTensor>{tensor});
    } catch (const std::invalid_argument&) {
      return true;
    }
    return false;
  };
  auto invalid = borrowed;
  invalid.device_address = 0;
  assert(rejects(invalid));
  invalid = borrowed;
  invalid.bytes = 1;
  assert(rejects(invalid));
  rejected = false;
  try {
    WeightManifest duplicate(std::vector<WeightTensor>{borrowed, borrowed});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  std::filesystem::remove_all(directory);
}

void test_borrowed_arena() {
  gaudirt::runtime::HbmArena arena(0, 0x1000, 4096,
                                   gaudirt::runtime::HbmArena::BorrowedSlab{});
  assert(arena.base_addr() == 0x1000);
  assert(arena.capacity_bytes() == 4096);
  const uint64_t workspace = arena.ensure_workspace(128);
  assert(workspace == 0x1000);
  assert(arena.workspace_bytes() == 128);
  assert(arena.allocate(128) == 0x1100);
  assert(arena.ensure_workspace(128) == workspace);

  gaudirt::runtime::HbmArena external(0, 0x1000, 4096,
                                      gaudirt::runtime::HbmArena::BorrowedSlab{});
  external.set_borrowed_workspace(0x8000, 512);
  assert(external.ensure_workspace(128) == 0x8000);
  assert(external.ensure_workspace(512) == 0x8000);
  assert(external.allocate(128) == 0x1000);
  bool rejected = false;
  try {
    external.ensure_workspace(513);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  assert(external.used_bytes() == 128);
  rejected = false;
  try {
    external.set_borrowed_workspace(0x9000, 512);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);
  rejected = false;
  try {
    gaudirt::runtime::HbmArena invalid(0, 0x1001, 4096,
                                       gaudirt::runtime::HbmArena::BorrowedSlab{});
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
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

  const std::filesystem::path weight_dir =
      std::filesystem::temp_directory_path() /
      ("gaudirt_weight_upload_" + std::to_string(getpid()));
  std::filesystem::create_directories(weight_dir);
  {
    std::ofstream raw(weight_dir / "x.bin", std::ios::binary);
    raw << "abc";
  }
  {
    std::ofstream manifest(weight_dir / "weights.manifest");
    manifest << "GAUDIRT_WEIGHTS_V1\n"
             << "T x uint8 3 1 3 x.bin "
             << "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\n";
  }
  WeightManifest uploaded(weight_dir, rt, arena);
  std::vector<uint8_t> roundtrip(3);
  arena.copy_d2h_sync(rt.d2h_stream(), uploaded.get("x").device_address,
                      roundtrip.data(), roundtrip.size());
  assert(std::string(roundtrip.begin(), roundtrip.end()) == "abc");
  std::filesystem::remove_all(weight_dir);

  std::vector<float> gathered(8 * kElems);
  for (size_t rank = 0; rank < 8; ++rank) {
    std::fill_n(gathered.begin() + rank * kElems, kElems,
                static_cast<float>(rank + 1));
  }
  const uint64_t gathered_addr = arena.allocate(gathered.size() * sizeof(float));
  arena.copy_h2d_sync(rt.h2d_stream(), gathered.data(), gathered_addr,
                      gathered.size() * sizeof(float));
  const uint64_t reduced_addr = arena.allocate(kElems * sizeof(uint16_t));
  TpReduce reduce(arena, kElems, syn_type_float);
  reduce.execute(gathered_addr, reduced_addr, rt.compute_stream());
  assert(synStreamSynchronize(rt.compute_stream()) == synSuccess);
  std::vector<uint16_t> reduced(kElems);
  arena.copy_d2h_sync(rt.d2h_stream(), reduced_addr, reduced.data(),
                      reduced.size() * sizeof(uint16_t));
  for (const uint16_t value : reduced) assert(value == 0x4210);

  const uint64_t prepared_addr = arena.allocate(kElems * sizeof(uint16_t));
  auto prepared = reduce.prepare(gathered_addr, prepared_addr);
  prepared.execute(rt.compute_stream());
  assert(synStreamSynchronize(rt.compute_stream()) == synSuccess);
  std::vector<uint16_t> prepared_result(kElems);
  arena.copy_d2h_sync(rt.d2h_stream(), prepared_addr, prepared_result.data(),
                      prepared_result.size() * sizeof(uint16_t));
  assert(prepared_result == reduced);
  bool overlap_rejected = false;
  try {
    static_cast<void>(reduce.prepare(gathered_addr, gathered_addr));
  } catch (const std::invalid_argument&) {
    overlap_rejected = true;
  }
  assert(overlap_rejected);

  const std::array<uint16_t, 8> rank_bf16 = {
      0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100};
  std::vector<uint16_t> gathered_bf16(8 * kElems);
  for (size_t rank = 0; rank < 8; ++rank)
    std::fill_n(gathered_bf16.begin() + rank * kElems, kElems, rank_bf16[rank]);
  const uint64_t gathered_bf16_addr = arena.allocate(gathered_bf16.size() * sizeof(uint16_t));
  arena.copy_h2d_sync(rt.h2d_stream(), gathered_bf16.data(), gathered_bf16_addr,
                      gathered_bf16.size() * sizeof(uint16_t));
  TpReduce reduce_bf16(arena, kElems, syn_type_bf16);
  auto prepared_bf16 = reduce_bf16.prepare(gathered_bf16_addr, prepared_addr);
  prepared_bf16.execute(rt.compute_stream());
  assert(synStreamSynchronize(rt.compute_stream()) == synSuccess);
  arena.copy_d2h_sync(rt.d2h_stream(), prepared_addr, prepared_result.data(),
                      prepared_result.size() * sizeof(uint16_t));
  assert(prepared_result == reduced);
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

void test_recipe_cache_contract() {
  RecipeCache disabled;
  assert(!disabled.enabled());
  assert(disabled.load("invalid/key") == nullptr);
  disabled.save(nullptr, "invalid/key");
  assert(disabled.stats().first == 0 && disabled.stats().second == 0);

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

int main(int argc, char** argv) {
  test_long_so_encoding_and_monitor_expansion();
  test_credit_formula();
  test_recipe_cache_contract();
  test_binding_table_value_semantics_and_dtype();
  test_weight_manifest();
  test_borrowed_arena();
  if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) {
    std::printf("[GaudiRT] Host-only runtime_unit_tests PASSED.\n");
    return 0;
  }
  test_foundation_on_hardware();
  test_invalid_module_fails();
  std::printf("[GaudiRT] Host tests and 1-card foundation PASSED.\n");
  return 0;
}
