// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/inflight_credit_manager.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <hccl.h>
#include <synapse_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gaudirt::runtime {

enum class CollectiveKind : uint8_t {
  AllGather = 0,
  AllReduce = 1,
  ReduceScatter = 2,
};

// High-level step execution engine built on DeviceRuntime, HbmArena,
// RecipeRegistry, and InFlightCreditManager. Executes mixed compute recipes and
// HCL collectives on a single continuous compute stream with automatic
// workspace sizing, O(1) binding-generation checks, and MappedMemMgr credit
// backpressure (575 - 2 * N_distinct).
class ExecutionEngine {
 public:
  struct Stats {
    uint64_t steps_submitted = 0;
    uint64_t compute_launches = 0;
    uint64_t collective_launches = 0;
    uint64_t generation_fast_hits = 0;
    uint64_t generation_rebuilds = 0;
    size_t distinct_recipes = 0;
    size_t max_safe_in_flight_launches = 0;
    InFlightCreditManager::Stats credit_stats{};
  };

  ExecutionEngine(DeviceRuntime& runtime, HbmArena& arena,
                  const RecipeRegistry& registry,
                  size_t total_chunk_budget = 575,
                  size_t safety_margin_chunks = 16);
  ~ExecutionEngine() = default;

  ExecutionEngine(const ExecutionEngine&) = delete;
  ExecutionEngine& operator=(const ExecutionEngine&) = delete;

  // Append a compute recipe launch bound to a caller-managed BindingTable.
  void add_compute(RecipeRegistry::RecipeId recipe_id,
                   const BindingTable* bindings);

  // Append an HCL collective operation on the unified compute stream.
  void add_collective(CollectiveKind kind, uint64_t send_addr,
                      uint64_t recv_addr, size_t count,
                      hcclDataType_t dtype = hcclFloat32,
                      hcclRedOp_t reduce_op = hcclSum);

  // Finalize the step program: sizes the shared workspace in HbmArena,
  // deduplicates distinct recipes, and snapshots initial binding generations.
  void finalize();

  // Submit `steps` iterations of the finalized program.
  // Automatically checks arena + binding table generations in O(N_tables)
  // instead of O(N_tensors) and enforces MappedMemMgr credit backpressure.
  void step(size_t steps = 1);

  // Wait for all in-flight steps to complete on the device.
  void drain();

  Stats stats() const noexcept;

 private:
  struct Op {
    bool is_collective = false;
    RecipeRegistry::RecipeId recipe_id = 0;
    synRecipeHandle recipe_handle = nullptr;
    const BindingTable* bindings = nullptr;

    CollectiveKind coll_kind = CollectiveKind::AllGather;
    uint64_t send_addr = 0;
    uint64_t recv_addr = 0;
    size_t count = 0;
    hcclDataType_t dtype = hcclFloat32;
    hcclRedOp_t reduce_op = hcclSum;
  };

  bool check_generations();

  DeviceRuntime& runtime_;
  HbmArena& arena_;
  const RecipeRegistry& registry_;
  InFlightCreditManager credit_mgr_;

  std::vector<Op> ops_;
  std::vector<synRecipeHandle> distinct_recipes_;
  std::vector<const BindingTable*> unique_tables_;
  std::vector<uint64_t> captured_table_gens_;
  uint64_t captured_arena_gen_ = 0;
  uint64_t required_workspace_bytes_ = 0;
  size_t compute_ops_per_step_ = 0;
  bool finalized_ = false;

  uint64_t steps_submitted_ = 0;
  uint64_t compute_launches_ = 0;
  uint64_t collective_launches_ = 0;
  uint64_t generation_fast_hits_ = 0;
  uint64_t generation_rebuilds_ = 0;
};

}  // namespace gaudirt::runtime
