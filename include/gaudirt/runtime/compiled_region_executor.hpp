// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/generation_epoch_tracker.hpp"
#include "gaudirt/runtime/inflight_credit_manager.hpp"
#include "gaudirt/runtime/unified_region_lowerer.hpp"

#include <hccl.h>
#include <synapse_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace gaudirt::runtime {

// General-purpose GaudiRT execution engine for compiled Synapse recipes and
// HCL collectives. Combines the three verified runtime capabilities:
//   - RT-1: Unified sub-call lowering of mixed compute+collective commands into
//           a single continuous compute-stream execution region.
//   - RT-2: O(1) generation-guarded binding verification and O(1) epoch fence
//           tracking instead of per-step O(N_addresses) hash-table scans.
//   - RT-3: Recipe-reuse-aware MappedMemMgr credit & backpressure management
//           (2 * N_distinct + N_in_flight <= 575).
class CompiledRegionExecutor {
 public:
  struct ComputeCallSpec {
    synRecipeHandle recipe = nullptr;
    uint64_t workspace_addr = 0;
    std::vector<synLaunchTensorInfoExt> bindings;
  };

  struct CollectiveCallSpec {
    hcclComm_t comm = nullptr;
    uint64_t send_addr = 0;
    uint64_t recv_addr = 0;
    size_t count = 0;
    hcclDataType_t dtype = hcclFloat32;
  };

  struct ExecutableSubCall {
    SubCallKind kind = SubCallKind::Compute;
    ComputeCallSpec compute;
    CollectiveCallSpec collective;
  };

  struct ExecutionStats {
    uint64_t steps_submitted = 0;
    uint64_t compute_launches = 0;
    uint64_t collective_launches = 0;
    uint64_t fast_binding_hits = 0;
    uint64_t full_binding_scans = 0;
    size_t distinct_recipes = 0;
    size_t max_safe_in_flight_launches = 0;
    InFlightCreditManager::Stats credit_stats{};
  };

  CompiledRegionExecutor(synDeviceId device_id, synStreamHandle compute_stream,
                         synStreamHandle comm_stream,
                         size_t total_chunk_budget = 575,
                         size_t safety_margin_chunks = 16)
      : device_id_(device_id),
        compute_stream_(compute_stream),
        comm_stream_(comm_stream),
        credit_mgr_(device_id, total_chunk_budget, safety_margin_chunks) {
    if (synEventCreate(&entry_bridge_event_, device_id_, 0) != synSuccess ||
        synEventCreate(&exit_epoch_event_, device_id_, 0) != synSuccess) {
      throw std::runtime_error(
          "Failed to create events in CompiledRegionExecutor");
    }
  }

  ~CompiledRegionExecutor() {
    if (entry_bridge_event_) synEventDestroy(entry_bridge_event_);
    if (exit_epoch_event_) synEventDestroy(exit_epoch_event_);
  }

  CompiledRegionExecutor(const CompiledRegionExecutor&) = delete;
  CompiledRegionExecutor& operator=(const CompiledRegionExecutor&) = delete;

  void compile(std::vector<ExecutableSubCall> subcalls,
               std::vector<BindingSnapshot> captured_bindings,
               std::vector<uint64_t> footprint_addresses,
               uint64_t region_stream_id = 1) {
    subcalls_ = std::move(subcalls);
    distinct_recipes_.clear();
    compute_per_step_ = 0;
    collective_per_step_ = 0;

    std::unordered_set<synRecipeHandle> seen;
    for (const auto& sc : subcalls_) {
      if (sc.kind == SubCallKind::Compute) {
        ++compute_per_step_;
        if (sc.compute.recipe && seen.insert(sc.compute.recipe).second) {
          distinct_recipes_.push_back(sc.compute.recipe);
        }
      } else {
        ++collective_per_step_;
      }
    }

    epoch_tracker_.initialize(std::move(captured_bindings),
                              std::move(footprint_addresses), region_stream_id);
  }

  void notify_binding_mutation() noexcept {
    epoch_tracker_.notify_binding_mutation();
  }

  void notify_external_producer(uint64_t stream_id,
                                uint64_t address) noexcept {
    epoch_tracker_.notify_external_producer(stream_id, address);
  }

  // Submit `repetitions` steps of the compiled region with O(1) generation
  // guard and automatic MappedMemMgr credit management.
  bool replay(const std::vector<BindingSnapshot>& live_bindings,
              size_t repetitions = 1) {
    if (!epoch_tracker_.verify_bindings(live_bindings)) {
      return false;
    }

    if (epoch_tracker_.needs_entry_wait() && comm_stream_ != nullptr) {
      if (synEventRecord(entry_bridge_event_, comm_stream_) != synSuccess ||
          synStreamWaitEvent(compute_stream_, entry_bridge_event_, 0) !=
              synSuccess) {
        throw std::runtime_error("Entry stream bridge wait failed");
      }
    }

    for (size_t rep = 0; rep < repetitions; ++rep) {
      credit_mgr_.acquire_step_budget(compute_per_step_, distinct_recipes_);
      for (const auto& sc : subcalls_) {
        if (sc.kind == SubCallKind::Compute) {
          synStatus st = synLaunchWithExternalEventsExt(
              compute_stream_, sc.compute.bindings.data(),
              sc.compute.bindings.size(), sc.compute.workspace_addr,
              sc.compute.recipe, nullptr, 0, 0);
          if (st != synSuccess) {
            throw std::runtime_error(
                "synLaunchWithExternalEventsExt failed in replay");
          }
          ++steps_compute_launches_;
        } else {
          hcclResult_t hr = hcclAllGather(
              reinterpret_cast<const void*>(sc.collective.send_addr),
              reinterpret_cast<void*>(sc.collective.recv_addr),
              sc.collective.count, sc.collective.dtype, sc.collective.comm,
              compute_stream_);
          if (hr != hcclSuccess) {
            throw std::runtime_error("hcclAllGather failed in replay");
          }
          ++steps_collective_launches_;
        }
      }
      ++steps_submitted_;
      credit_mgr_.commit_step(steps_submitted_, compute_stream_,
                              compute_per_step_, distinct_recipes_);
    }

    if (synEventRecord(exit_epoch_event_, compute_stream_) != synSuccess) {
      throw std::runtime_error("Exit epoch event record failed");
    }
    epoch_tracker_.publish_region_epoch(steps_submitted_);
    return true;
  }

  void drain() { credit_mgr_.drain_all(); }

  ExecutionStats stats() const noexcept {
    ExecutionStats s{};
    s.steps_submitted = steps_submitted_;
    s.compute_launches = steps_compute_launches_;
    s.collective_launches = steps_collective_launches_;
    s.fast_binding_hits = epoch_tracker_.fast_hits();
    s.full_binding_scans = epoch_tracker_.full_scans();
    s.distinct_recipes = distinct_recipes_.size();
    s.max_safe_in_flight_launches =
        InFlightCreditManager::max_launches_for_distinct(
            distinct_recipes_.size());
    s.credit_stats = credit_mgr_.stats();
    return s;
  }

 private:
  synDeviceId device_id_;
  synStreamHandle compute_stream_;
  synStreamHandle comm_stream_;
  synEventHandle entry_bridge_event_ = nullptr;
  synEventHandle exit_epoch_event_ = nullptr;
  std::vector<ExecutableSubCall> subcalls_;
  std::vector<synRecipeHandle> distinct_recipes_;
  size_t compute_per_step_ = 0;
  size_t collective_per_step_ = 0;
  uint64_t steps_submitted_ = 0;
  uint64_t steps_compute_launches_ = 0;
  uint64_t steps_collective_launches_ = 0;
  GenerationEpochTracker epoch_tracker_;
  InFlightCreditManager credit_mgr_;
};

}  // namespace gaudirt::runtime
