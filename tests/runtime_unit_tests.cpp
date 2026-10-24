// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/compiled_region_executor.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/execution_engine.hpp"
#include "gaudirt/runtime/generation_epoch_tracker.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/inflight_credit_manager.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include "gaudirt/runtime/unified_region_lowerer.hpp"
#include "gaudirt/scal/long_so.hpp"

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <new>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
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
  if (!crossed || writes.size() != 2) {
    std::fprintf(stderr, "long-SO cross failed: ok=%d writes=%zu\n", crossed,
                 writes.size());
    for (const auto& write : writes) {
      std::fprintf(stderr, "  addr=0x%x data=0x%x\n", write.addr, write.data);
    }
  }
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

void test_unified_region_lowerer_production70() {
  std::vector<CapturedCommandDesc> cmds(286);
  const uint64_t comm_id = 0x7777;
  uint64_t next_alloc = 100;
  uint64_t cur = next_alloc++;

  for (size_t i = 0; i < 286; ++i) {
    auto& c = cmds[i];
    c.original_index = i;
    c.complete = true;
    c.external_events = 0;
    if (i == 5) {
      c.compute = true;
      c.collectives = 1;
      c.communicator = comm_id;
      uint64_t mid = next_alloc++;
      uint64_t out = next_alloc++;
      c.compute_reads = {cur, next_alloc++};
      c.compute_writes = {mid};
      c.collective_reads = {mid};
      c.collective_writes = {out};
      cur = out;
    } else if (i == 0 || i == 2 || i == 4 || i == 282 || i == 283 || i == 285 ||
               (i >= 6 && i < 282 && (i % 2 == 0))) {
      c.compute = true;
      c.collectives = 0;
      uint64_t out = next_alloc++;
      c.compute_reads = {cur, next_alloc++};
      c.compute_writes = {out};
      cur = out;
    } else {
      c.compute = false;
      c.collectives = 1;
      c.communicator = comm_id;
      uint64_t out = next_alloc++;
      c.collective_reads = {cur};
      c.collective_writes = {out};
      cur = out;
    }
  }

  auto legacy = UnifiedRegionLowerer::lower(cmds, false);
  assert(legacy.original_commands == 286);
  assert(legacy.regions.size() == 2);
  assert(legacy.coalesced_collectives == 141);
  assert(legacy.regions[0].first_original_command == 0);
  assert(legacy.regions[0].last_original_command == 4);
  assert(legacy.regions[1].first_original_command == 6);
  assert(legacy.regions[1].last_original_command == 285);

  auto unified = UnifiedRegionLowerer::lower(cmds, true);
  assert(unified.original_commands == 286);
  assert(unified.decomposed_mixed_commands == 1);
  assert(unified.subcalls.size() == 287);
  assert(unified.total_compute_subcalls == 145);
  assert(unified.total_collective_subcalls == 142);
  assert(unified.regions.size() == 1);
  assert(unified.coalesced_collectives == 142);
  assert(unified.regions[0].first_original_command == 0);
  assert(unified.regions[0].last_original_command == 285);
}

void test_mixed_command_without_internal_raw_stays_barrier() {
  std::vector<CapturedCommandDesc> cmds(5);
  const uint64_t comm = 1;
  cmds[0] = {0, true, 0, 0, true, 0, {1}, {10}, {}, {}};
  cmds[1] = {1, false, 1, 0, true, comm, {}, {}, {10}, {11}};
  cmds[2] = {2, true, 1, 0, true, comm, {11}, {12}, {999}, {13}};
  cmds[3] = {3, false, 1, 0, true, comm, {}, {}, {13}, {14}};
  cmds[4] = {4, true, 0, 0, true, 0, {14}, {15}, {}, {}};

  auto rep = UnifiedRegionLowerer::lower(cmds, true);
  assert(rep.decomposed_mixed_commands == 0);
}

void test_generation_epoch_tracker() {
  std::vector<BindingSnapshot> snap = {
      {reinterpret_cast<void*>(1), reinterpret_cast<void*>(2), 0x1000, 64, 0, 1,
       11},
      {reinterpret_cast<void*>(3), reinterpret_cast<void*>(4), 0x2000, 64, 0, 1,
       22},
  };
  GenerationEpochTracker tracker;
  tracker.initialize(snap, {0x1000, 0x2000}, 7);

  assert(tracker.verify_bindings(snap));
  assert(tracker.full_scans() == 1);
  assert(tracker.fast_hits() == 0);
  assert(tracker.needs_entry_wait() == true);

  assert(tracker.verify_bindings(snap));
  assert(tracker.full_scans() == 1);
  assert(tracker.fast_hits() == 1);
  assert(tracker.needs_entry_wait() == false);

  tracker.notify_external_producer(7, 0x1000);
  assert(tracker.needs_entry_wait() == false);

  tracker.notify_external_producer(99, 0x2000);
  assert(tracker.needs_entry_wait() == true);
  assert(tracker.needs_entry_wait() == false);

  auto mutated = snap;
  mutated[1].device_address = 0x3000;
  tracker.notify_binding_mutation();
  assert(!tracker.verify_bindings(mutated));
}

void test_credit_formula() {
  assert(InFlightCreditManager::max_launches_for_distinct(1, 575) == 573);
  assert(InFlightCreditManager::max_launches_for_distinct(12, 575) == 551);
  assert(InFlightCreditManager::max_launches_for_distinct(36, 575) == 503);
  assert(InFlightCreditManager::max_launches_for_distinct(72, 575) == 431);
  assert(InFlightCreditManager::max_launches_for_distinct(140, 575) == 295);
  assert(InFlightCreditManager::max_launches_for_distinct(144, 575) == 287);
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

  // Destroy graph BEFORE tensors/sections (verified Synapse 1.24.2 lifecycle rule)
  synGraphDestroy(graph);
  synTensorDestroy(t_a);
  synTensorDestroy(t_b);
  synTensorDestroy(t_out);
  synSectionDestroy(sec_a);
  synSectionDestroy(sec_b);
  synSectionDestroy(sec_out);
  return recipe;
}

void test_libgaudirt_foundation_on_hardware() {
  DeviceRuntime::Config cfg{};
  cfg.module_id = 0;
  cfg.rank = 0;
  cfg.world_size = 1;
  cfg.uclamp_min = 1024;
  cfg.uclamp_max = 1024;

  DeviceRuntime rt(cfg);
  HbmArena arena(rt.device_id(), 64ULL * 1024 * 1024);  // 64 MiB slab
  RecipeRegistry registry;

  constexpr TSize kElems = 1024;
  constexpr uint64_t kBytes = kElems * sizeof(float);

  // 1. Compile recipe, register, serialize to .recipe file, and deserialize back
  synRecipeHandle raw_recipe = compile_test_add_recipe("gaudirt_test_add", kElems);
  auto id_mem = registry.register_compiled("add_mem", raw_recipe);

  const std::string recipe_path =
      "/tmp/gaudirt_test_add_" + std::to_string(getpid()) + ".recipe";
  registry.save_to_file(id_mem, recipe_path);
  auto id_disk = registry.load_from_file("add_disk", recipe_path);
  unlink(recipe_path.c_str());

  const auto& disk_entry = registry.get(id_disk);
  assert(disk_entry.slots.size() == 3);
  for (const auto& slot : disk_entry.slots) {
    assert(slot.byte_size == kBytes);
  }

  // 2. Allocate HBM tensors from HbmArena and transfer inputs via H2D DMA
  uint64_t addr_a = arena.allocate(kBytes);
  uint64_t addr_b1 = arena.allocate(kBytes);
  uint64_t addr_b2 = arena.allocate(kBytes);
  uint64_t addr_mid = arena.allocate(kBytes);
  uint64_t addr_out = arena.allocate(kBytes);

  std::vector<float> host_a(kElems, 2.5f);
  std::vector<float> host_b1(kElems, 1.5f);
  std::vector<float> host_b2(kElems, 10.0f);
  std::vector<float> host_out(kElems, 0.0f);

  arena.copy_h2d_sync(rt.h2d_stream(), host_a.data(), addr_a, kBytes);
  arena.copy_h2d_sync(rt.h2d_stream(), host_b1.data(), addr_b1, kBytes);
  arena.copy_h2d_sync(rt.h2d_stream(), host_b2.data(), addr_b2, kBytes);

  // 3. Build BindingTables using name lookup and O(1) slot rebinding
  BindingTable bt1 = registry.create_binding_table(id_disk);
  bt1.bind_name("in_a", addr_a);
  bt1.bind_name("in_b", addr_b1);
  bt1.bind_name("out_c", addr_mid);

  BindingTable bt2 = registry.create_binding_table(id_disk);
  bt2.bind_name("in_a", addr_mid);
  bt2.bind_name("in_b", addr_b1);
  bt2.bind_name("out_c", addr_out);

  // 4. Execute a 2-op step program (out = (a + b1) + b1 = 2.5 + 1.5 + 1.5 = 5.5)
  ExecutionEngine engine(rt, arena, registry);
  engine.add_compute(id_disk, &bt1);
  engine.add_compute(id_disk, &bt2);
  engine.finalize();

  engine.step(4);  // 4 steps in-flight with credit manager
  engine.drain();

  arena.copy_d2h_sync(rt.d2h_stream(), addr_out, host_out.data(), kBytes);
  for (size_t i = 0; i < kElems; ++i) {
    assert(std::fabs(host_out[i] - 5.5f) < 1e-5f);
  }

  auto st1 = engine.stats();
  assert(st1.steps_submitted == 4);
  assert(st1.compute_launches == 8);
  assert(st1.distinct_recipes == 1);
  assert(st1.generation_fast_hits == 1);
  assert(st1.generation_rebuilds == 0);

  // 5. Dynamically rebind slot in bt2 (in_b -> addr_b2 = 10.0f)
  // Expected: out = (2.5 + 1.5) + 10.0 = 14.0
  size_t slot_in_b = bt2.find_slot("in_b");
  bt2.bind_slot(slot_in_b, addr_b2);

  engine.step(2);
  engine.drain();

  arena.copy_d2h_sync(rt.d2h_stream(), addr_out, host_out.data(), kBytes);
  for (size_t i = 0; i < kElems; ++i) {
    assert(std::fabs(host_out[i] - 14.0f) < 1e-5f);
  }

  auto st2 = engine.stats();
  assert(st2.steps_submitted == 6);
  assert(st2.compute_launches == 12);
  assert(st2.generation_rebuilds == 1);
}

struct SharedRankCoord {
  hcclUniqueId uid{};
  std::atomic<int> uid_ready{0};
};

void test_libgaudirt_8card_compute_and_collective() {
  constexpr uint32_t kWorldSize = 8;
  void* raw_shm = mmap(nullptr, sizeof(SharedRankCoord), PROT_READ | PROT_WRITE,
                       MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  assert(raw_shm != MAP_FAILED);
  auto* shm = new (raw_shm) SharedRankCoord();

  std::vector<pid_t> pids(kWorldSize, -1);
  for (uint32_t r = 0; r < kWorldSize; ++r) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
      {
        DeviceRuntime::Config cfg{};
        cfg.module_id = r;
        cfg.rank = r;
        cfg.world_size = kWorldSize;
        cfg.uclamp_min = 1024;
        cfg.uclamp_max = 1024;

        DeviceRuntime rt(cfg);
        if (r == 0) {
          if (hcclGetUniqueId(&shm->uid) != hcclSuccess) _exit(1);
          shm->uid_ready.store(1, std::memory_order_release);
        } else {
          while (shm->uid_ready.load(std::memory_order_acquire) == 0) {
          }
        }
        rt.init_comm(shm->uid);

        HbmArena arena(rt.device_id(), 64ULL * 1024 * 1024);
        RecipeRegistry registry;

        constexpr TSize kElems = 256;
        constexpr uint64_t kBytes = kElems * sizeof(float);
        synRecipeHandle raw =
            compile_test_add_recipe("gaudirt_8card_add", kElems);
        auto rid = registry.register_compiled("add_f32", raw);

        uint64_t addr_a = arena.allocate(kBytes);
        uint64_t addr_b = arena.allocate(kBytes);
        uint64_t addr_local_sum = arena.allocate(kBytes);
        uint64_t addr_allreduce = arena.allocate(kBytes);

        // Rank r computes local_sum = (r + 1) + 0.5 = r + 1.5
        // Sum across 8 ranks (r = 0..7): sum(r + 1.5) = 28 + 12 = 40.0f
        std::vector<float> h_a(kElems, static_cast<float>(r + 1));
        std::vector<float> h_b(kElems, 0.5f);
        std::vector<float> h_out(kElems, 0.0f);

        arena.copy_h2d_sync(rt.h2d_stream(), h_a.data(), addr_a, kBytes);
        arena.copy_h2d_sync(rt.h2d_stream(), h_b.data(), addr_b, kBytes);

        BindingTable bt = registry.create_binding_table(rid);
        bt.bind_name("in_a", addr_a);
        bt.bind_name("in_b", addr_b);
        bt.bind_name("out_c", addr_local_sum);

        ExecutionEngine engine(rt, arena, registry);
        engine.add_compute(rid, &bt);
        engine.add_collective(CollectiveKind::AllReduce, addr_local_sum,
                              addr_allreduce, kElems, hcclFloat32, hcclSum);
        engine.finalize();

        engine.step(3);
        engine.drain();

        arena.copy_d2h_sync(rt.d2h_stream(), addr_allreduce, h_out.data(),
                            kBytes);
        for (size_t i = 0; i < kElems; ++i) {
          if (std::fabs(h_out[i] - 40.0f) > 1e-4f) {
            std::fprintf(stderr, "Rank %u mismatch: got %f expected 40.0\n", r,
                         h_out[i]);
            _exit(2);
          }
        }
        auto st = engine.stats();
        if (st.steps_submitted != 3 || st.compute_launches != 3 ||
            st.collective_launches != 3) {
          _exit(3);
        }
      }
      _exit(0);
    }
    pids[r] = pid;
  }

  for (uint32_t r = 0; r < kWorldSize; ++r) {
    int status = 0;
    assert(waitpid(pids[r], &status, 0) == pids[r]);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  }
  munmap(raw_shm, sizeof(SharedRankCoord));
}

}  // namespace

int main(int argc, char** argv) {
  test_long_so_encoding_and_monitor_expansion();
  test_unified_region_lowerer_production70();
  test_mixed_command_without_internal_raw_stays_barrier();
  test_generation_epoch_tracker();
  test_credit_formula();
  if (argc == 2 && std::strcmp(argv[1], "--host-only") == 0) {
    std::printf("[GaudiRT] Host-only runtime_unit_tests PASSED.\n");
    return 0;
  }
  test_libgaudirt_8card_compute_and_collective();
  test_libgaudirt_foundation_on_hardware();
  std::printf("[GaudiRT] All runtime_unit_tests (including 1-card & 8-card hardware foundation) PASSED.\n");
  return 0;
}
