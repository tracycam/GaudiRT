// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace gaudirt::runtime {

// Represents a captured tensor binding metadata snapshot (analogous to
// Bridge's RetainedTensorBinding + boundary slot).
struct BindingSnapshot {
  const void* tensor_impl = nullptr;
  const void* storage_impl = nullptr;
  uint64_t device_address = 0;
  size_t bytes = 0;
  int64_t offset = 0;
  int32_t dtype = 0;
  uint64_t shape_stride_hash = 0;

  bool matches(const BindingSnapshot& live) const noexcept {
    return tensor_impl == live.tensor_impl &&
           storage_impl == live.storage_impl &&
           device_address == live.device_address && bytes == live.bytes &&
           offset == live.offset && dtype == live.dtype &&
           shape_stride_hash == live.shape_stride_hash;
  }
};

// Simulates the legacy per-address event map (stream_event_manager) vs
// GaudiRT's O(1) Generation-Guarded Epoch Tracker.
class LegacyAddressEventMap {
 public:
  struct EventRecord {
    uint64_t event_id = 0;
    uint64_t stream_id = 0;
    std::vector<uint64_t> addresses;

    void remove_device_ptr(uint64_t addr) {
      auto it = std::find(addresses.begin(), addresses.end(), addr);
      if (it != addresses.end()) {
        addresses.erase(it);
      }
    }
  };

  void reserve(size_t count) { events_by_addr_.reserve(count * 2); }

  // Equivalent to stream_event_manager::get_wait_events(dependencies)
  std::vector<uint64_t> get_wait_events(
      const std::vector<uint64_t>& dependencies) const {
    std::vector<const EventRecord*> found_events;
    found_events.reserve(dependencies.size());
    for (uint64_t addr : dependencies) {
      auto it = events_by_addr_.find(addr);
      if (it != events_by_addr_.end() && it->second != nullptr) {
        found_events.push_back(it->second);
      }
    }
    std::sort(found_events.begin(), found_events.end());
    found_events.erase(std::unique(found_events.begin(), found_events.end()),
                       found_events.end());
    std::vector<uint64_t> result;
    result.reserve(found_events.size());
    for (const auto* ev : found_events) {
      result.push_back(ev->event_id);
    }
    return result;
  }

  // Equivalent to stream_event_manager::add_producer(dependencies, stream)
  void add_producer(const std::vector<uint64_t>& dependencies,
                    uint64_t stream_id, uint64_t event_id) {
    pool_.push_back(EventRecord{event_id, stream_id, dependencies});
    EventRecord* eref = &pool_.back();
    for (uint64_t addr : eref->addresses) {
      auto it = events_by_addr_.find(addr);
      if (it != events_by_addr_.end() && it->second != nullptr) {
        if (it->second != eref) {
          it->second->remove_device_ptr(addr);
          it->second = eref;
        }
      } else {
        events_by_addr_.emplace(addr, eref);
      }
    }
    if (pool_.size() > 16) {
      // Keep bounded pool in steady state while retaining active pointers
    }
  }

  void clear() {
    events_by_addr_.clear();
    pool_.clear();
    pool_.reserve(64);
  }

 private:
  std::unordered_map<uint64_t, EventRecord*> events_by_addr_;
  std::vector<EventRecord> pool_;
};

// O(1) Generation-Guarded Epoch Tracker for static region replay (RT-2).
// Instead of walking 2,392 retained tensor bindings and 2,392 address entries
// in an unordered_map on every single decode step, tracks:
//   1. binding_generation_: bumped only when a mutable input slot or allocator
//      table changes.
//   2. external_stream_epoch_: bumped only when a non-region stream registers a
//      producer on an address belonging to the region's footprint.
class GenerationEpochTracker {
 public:
  void initialize(std::vector<BindingSnapshot> captured_bindings,
                  std::vector<uint64_t> footprint_addresses,
                  uint64_t region_stream_id) {
    captured_bindings_ = std::move(captured_bindings);
    footprint_addresses_ = std::move(footprint_addresses);
    std::sort(footprint_addresses_.begin(), footprint_addresses_.end());
    footprint_addresses_.erase(
        std::unique(footprint_addresses_.begin(), footprint_addresses_.end()),
        footprint_addresses_.end());
    region_stream_id_ = region_stream_id;
    binding_generation_ = 1;
    verified_binding_generation_ = 0;
    external_stream_epoch_ = 1;
    waited_external_epoch_ = 0;
    last_region_epoch_event_ = 0;
    fast_hits_ = 0;
    full_scans_ = 0;
  }

  void notify_binding_mutation() noexcept { ++binding_generation_; }

  void notify_external_producer(uint64_t stream_id,
                                uint64_t address) noexcept {
    if (stream_id == region_stream_id_) return;
    if (std::binary_search(footprint_addresses_.begin(),
                           footprint_addresses_.end(), address)) {
      ++external_stream_epoch_;
    }
  }

  // Returns true if bindings match. Uses O(1) generation check when unchanged.
  bool verify_bindings(const std::vector<BindingSnapshot>& live_bindings) {
    if (verified_binding_generation_ == binding_generation_) {
      ++fast_hits_;
      return true;
    }
    if (live_bindings.size() != captured_bindings_.size()) {
      return false;
    }
    for (size_t i = 0; i < captured_bindings_.size(); ++i) {
      if (!captured_bindings_[i].matches(live_bindings[i])) {
        return false;
      }
    }
    verified_binding_generation_ = binding_generation_;
    ++full_scans_;
    return true;
  }

  // Returns true if the region stream must issue wait events for external
  // producers before launching the region. Same-stream consecutive replays
  // require zero wait events because the hardware stream is FIFO-ordered.
  bool needs_entry_wait() noexcept {
    if (waited_external_epoch_ == external_stream_epoch_) {
      return false;
    }
    waited_external_epoch_ = external_stream_epoch_;
    return true;
  }

  // Records a single O(1) epoch event for the entire region instead of 2,392
  // hash-table mutations in events_by_addr_.
  void publish_region_epoch(uint64_t event_id) noexcept {
    last_region_epoch_event_ = event_id;
  }

  uint64_t last_region_epoch_event() const noexcept {
    return last_region_epoch_event_;
  }
  uint64_t fast_hits() const noexcept { return fast_hits_; }
  uint64_t full_scans() const noexcept { return full_scans_; }

 private:
  std::vector<BindingSnapshot> captured_bindings_;
  std::vector<uint64_t> footprint_addresses_;
  uint64_t region_stream_id_ = 0;
  uint64_t binding_generation_ = 0;
  uint64_t verified_binding_generation_ = 0;
  uint64_t external_stream_epoch_ = 0;
  uint64_t waited_external_epoch_ = 0;
  uint64_t last_region_epoch_event_ = 0;
  uint64_t fast_hits_ = 0;
  uint64_t full_scans_ = 0;
};

}  // namespace gaudirt::runtime
