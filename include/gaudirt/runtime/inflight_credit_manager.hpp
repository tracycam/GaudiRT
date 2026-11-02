// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace gaudirt::runtime {

// Enforces the exact Synapse MappedMemMgr in-flight chunk conservation law
// discovered and verified in Gate 0 (G0-2 & G0-7):
//
//   2 * N_distinct_in_flight + N_launches_in_flight <= kMaxMappedChunks (575)
//
// On Gaudi2 (Synapse 1.24.2-481), MappedMemMgr holds 576 DataChunks per queue
// (575 available when 1 is held by an active gate/prior launch). Each distinct
// synRecipeHandle in flight pins 2 non-patchable chunks, while each launch
// consumes 1 patchable chunk until its completion copy finishes.
//
// For production70 (145 compute launches/step, 12 distinct recipes reused
// across model layers):
//   L_max(12) = 575 - 2 * 12 = 551 launches (~3.80 steps).
class InFlightCreditManager {
 public:
  static constexpr size_t kDefaultTotalChunks = 575;
  static constexpr size_t kChunksPerDistinctRecipe = 2;

  struct StepLease {
    uint64_t step_id = 0;
    size_t launch_count = 0;
    std::vector<synRecipeHandle> distinct_recipes;
    synEventHandle completion_event = nullptr;
  };

  struct Stats {
    size_t total_chunk_budget = kDefaultTotalChunks;
    size_t safety_margin_chunks = 0;
    size_t current_in_flight_steps = 0;
    size_t current_in_flight_launches = 0;
    size_t current_distinct_recipes = 0;
    size_t current_used_chunks = 0;
    size_t peak_in_flight_steps = 0;
    size_t peak_in_flight_launches = 0;
    size_t peak_used_chunks = 0;
    size_t nonblocking_reclaims = 0;
    size_t backpressure_waits = 0;
    double total_backpressure_wait_us = 0.0;
  };

  explicit InFlightCreditManager(
      synDeviceId device_id,
      size_t total_chunk_budget = kDefaultTotalChunks,
      size_t safety_margin_chunks = 16)
      : device_id_(device_id),
        total_chunk_budget_(total_chunk_budget),
        safety_margin_chunks_(safety_margin_chunks) {
    stats_.total_chunk_budget = total_chunk_budget_;
    stats_.safety_margin_chunks = safety_margin_chunks_;
  }

  ~InFlightCreditManager() {
    for (auto ev : event_pool_) {
      if (ev) synEventDestroy(ev);
    }
    for (auto& lease : active_steps_) {
      if (lease.completion_event) synEventDestroy(lease.completion_event);
    }
  }

  InFlightCreditManager(const InFlightCreditManager&) = delete;
  InFlightCreditManager& operator=(const InFlightCreditManager&) = delete;

  // Theoretical maximum in-flight launches before waitForCompletionCopy blocks
  // for a given number of distinct in-flight recipes.
  static constexpr size_t max_launches_for_distinct(
      size_t num_distinct_recipes,
      size_t total_chunks = kDefaultTotalChunks) noexcept {
    const size_t pinned = kChunksPerDistinctRecipe * num_distinct_recipes;
    return (total_chunks > pinned) ? (total_chunks - pinned) : 0;
  }

  size_t effective_budget() const noexcept {
    return (total_chunk_budget_ > safety_margin_chunks_)
               ? (total_chunk_budget_ - safety_margin_chunks_)
               : total_chunk_budget_;
  }

  // Poll completed step events without blocking.
  size_t poll_reclaim() {
    size_t reclaimed = 0;
    while (!active_steps_.empty()) {
      auto& oldest = active_steps_.front();
      synStatus st = synEventQuery(oldest.completion_event);
      if (st == synSuccess) {
        retire_oldest();
        ++reclaimed;
        ++stats_.nonblocking_reclaims;
      } else {
        break;
      }
    }
    return reclaimed;
  }

  // Ensure enough chunk credits exist to submit a step with `launch_count`
  // compute launches and `step_distinct_recipes`. If credits are insufficient,
  // first polls non-blockingly, then waits on the oldest in-flight step event
  // (preventing a blind mid-step stall inside MappedMemMgr).
  void acquire_step_budget(
      size_t launch_count,
      const std::vector<synRecipeHandle>& step_distinct_recipes,
      double* wait_us_out = nullptr) {
    poll_reclaim();
    while (!can_accommodate(launch_count, step_distinct_recipes)) {
      if (active_steps_.empty()) {
        throw std::runtime_error(
            "Single step exceeds total MappedMemMgr chunk budget");
      }
      auto t0 = std::chrono::steady_clock::now();
      synStatus st =
          synEventSynchronize(active_steps_.front().completion_event);
      auto t1 = std::chrono::steady_clock::now();
      if (st != synSuccess) {
        throw std::runtime_error("synEventSynchronize failed in backpressure");
      }
      double w_us =
          std::chrono::duration<double, std::micro>(t1 - t0).count();
      stats_.total_backpressure_wait_us += w_us;
      if (wait_us_out) *wait_us_out += w_us;
      ++stats_.backpressure_waits;
      retire_oldest();
      poll_reclaim();
    }
  }

  // Record the step completion fence on `stream` and commit the step's chunk
  // usage to the in-flight window.
  void commit_step(uint64_t step_id, synStreamHandle stream,
                   size_t launch_count,
                   std::vector<synRecipeHandle> step_distinct_recipes) {
    synEventHandle ev = allocate_event();
    if (synEventRecord(ev, stream) != synSuccess) {
      throw std::runtime_error("synEventRecord failed in commit_step");
    }
    for (auto h : step_distinct_recipes) {
      ++recipe_ref_counts_[h];
    }
    in_flight_launches_ += launch_count;
    active_steps_.push_back(StepLease{
        step_id, launch_count, std::move(step_distinct_recipes), ev});
    update_stats();
  }

  void drain_all() {
    while (!active_steps_.empty()) {
      synEventSynchronize(active_steps_.front().completion_event);
      retire_oldest();
    }
  }

  Stats stats() const noexcept { return stats_; }

 private:
  bool can_accommodate(
      size_t add_launches,
      const std::vector<synRecipeHandle>& add_recipes) const noexcept {
    size_t projected_distinct = recipe_ref_counts_.size();
    for (auto h : add_recipes) {
      if (recipe_ref_counts_.find(h) == recipe_ref_counts_.end()) {
        ++projected_distinct;
      }
    }
    const size_t projected_launches = in_flight_launches_ + add_launches;
    const size_t projected_chunks =
        kChunksPerDistinctRecipe * projected_distinct + projected_launches;
    return projected_chunks <= effective_budget();
  }

  void retire_oldest() {
    auto oldest = std::move(active_steps_.front());
    active_steps_.pop_front();
    in_flight_launches_ -= oldest.launch_count;
    for (auto h : oldest.distinct_recipes) {
      auto it = recipe_ref_counts_.find(h);
      if (it != recipe_ref_counts_.end()) {
        if (--(it->second) == 0) {
          recipe_ref_counts_.erase(it);
        }
      }
    }
    event_pool_.push_back(oldest.completion_event);
    update_stats();
  }

  void update_stats() noexcept {
    stats_.current_in_flight_steps = active_steps_.size();
    stats_.current_in_flight_launches = in_flight_launches_;
    stats_.current_distinct_recipes = recipe_ref_counts_.size();
    stats_.current_used_chunks =
        kChunksPerDistinctRecipe * recipe_ref_counts_.size() +
        in_flight_launches_;
    stats_.peak_in_flight_steps =
        std::max(stats_.peak_in_flight_steps, stats_.current_in_flight_steps);
    stats_.peak_in_flight_launches = std::max(
        stats_.peak_in_flight_launches, stats_.current_in_flight_launches);
    stats_.peak_used_chunks =
        std::max(stats_.peak_used_chunks, stats_.current_used_chunks);
  }

  synEventHandle allocate_event() {
    if (!event_pool_.empty()) {
      synEventHandle ev = event_pool_.back();
      event_pool_.pop_back();
      return ev;
    }
    synEventHandle ev = nullptr;
    if (synEventCreate(&ev, device_id_, 0) != synSuccess) {
      throw std::runtime_error("synEventCreate failed in InFlightCreditManager");
    }
    return ev;
  }

  synDeviceId device_id_;
  size_t total_chunk_budget_;
  size_t safety_margin_chunks_;
  size_t in_flight_launches_ = 0;
  std::unordered_map<synRecipeHandle, size_t> recipe_ref_counts_;
  std::deque<StepLease> active_steps_;
  std::vector<synEventHandle> event_pool_;
  Stats stats_{};
};

}  // namespace gaudirt::runtime
