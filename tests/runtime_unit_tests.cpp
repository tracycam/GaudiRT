// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/compiled_region_executor.hpp"
#include "gaudirt/runtime/generation_epoch_tracker.hpp"
#include "gaudirt/runtime/inflight_credit_manager.hpp"
#include "gaudirt/runtime/unified_region_lowerer.hpp"

#include <cassert>
#include <cstdio>
#include <vector>

using namespace gaudirt::runtime;

namespace {

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
  // Cmd 0: compute writes 10
  cmds[0] = {0, true, 0, 0, true, 0, {1}, {10}, {}, {}};
  // Cmd 1: collective reads 10, writes 11
  cmds[1] = {1, false, 1, 0, true, comm, {}, {}, {10}, {11}};
  // Cmd 2: mixed compute+collective, but collective reads 999 (NOT written by
  // Cmd 2's compute!)
  cmds[2] = {2, true, 1, 0, true, comm, {11}, {12}, {999}, {13}};
  // Cmd 3: collective reads 13, writes 14
  cmds[3] = {3, false, 1, 0, true, comm, {}, {}, {13}, {14}};
  // Cmd 4: compute reads 14, writes 15
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

  // First call performs full scan and needs entry wait
  assert(tracker.verify_bindings(snap));
  assert(tracker.full_scans() == 1);
  assert(tracker.fast_hits() == 0);
  assert(tracker.needs_entry_wait() == true);

  // Second call hits O(1) fast path and skips entry wait
  assert(tracker.verify_bindings(snap));
  assert(tracker.full_scans() == 1);
  assert(tracker.fast_hits() == 1);
  assert(tracker.needs_entry_wait() == false);

  // Same-stream producer does not trigger entry wait
  tracker.notify_external_producer(7, 0x1000);
  assert(tracker.needs_entry_wait() == false);

  // External stream producer on footprint address triggers entry wait
  tracker.notify_external_producer(99, 0x2000);
  assert(tracker.needs_entry_wait() == true);
  assert(tracker.needs_entry_wait() == false);

  // Mutation triggers full scan; mismatch rejects
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

}  // namespace

int main() {
  test_unified_region_lowerer_production70();
  test_mixed_command_without_internal_raw_stays_barrier();
  test_generation_epoch_tracker();
  test_credit_formula();
  std::printf("[GaudiRT] All runtime_unit_tests PASSED.\n");
  return 0;
}
