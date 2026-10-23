// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/execution_engine.hpp"

#include <stdexcept>
#include <unordered_set>

namespace gaudirt::runtime {

ExecutionEngine::ExecutionEngine(DeviceRuntime& runtime, HbmArena& arena,
                                 const RecipeRegistry& registry,
                                 size_t total_chunk_budget,
                                 size_t safety_margin_chunks)
    : runtime_(runtime),
      arena_(arena),
      registry_(registry),
      credit_mgr_(runtime.device_id(), total_chunk_budget,
                  safety_margin_chunks) {}

void ExecutionEngine::add_compute(RecipeRegistry::RecipeId recipe_id,
                                  const BindingTable* bindings) {
  if (finalized_) {
    throw std::logic_error(
        "ExecutionEngine::add_compute called after finalize()");
  }
  if (bindings == nullptr) {
    throw std::invalid_argument("ExecutionEngine::add_compute: null bindings");
  }
  const auto& entry = registry_.get(recipe_id);
  if (bindings->size() != entry.slots.size()) {
    throw std::invalid_argument(
        "ExecutionEngine::add_compute: binding slot count mismatch");
  }

  Op op{};
  op.is_collective = false;
  op.recipe_id = recipe_id;
  op.recipe_handle = entry.handle;
  op.bindings = bindings;
  ops_.push_back(op);
}

void ExecutionEngine::add_collective(CollectiveKind kind, uint64_t send_addr,
                                     uint64_t recv_addr, size_t count,
                                     hcclDataType_t dtype,
                                     hcclRedOp_t reduce_op) {
  if (finalized_) {
    throw std::logic_error(
        "ExecutionEngine::add_collective called after finalize()");
  }
  Op op{};
  op.is_collective = true;
  op.coll_kind = kind;
  op.send_addr = send_addr;
  op.recv_addr = recv_addr;
  op.count = count;
  op.dtype = dtype;
  op.reduce_op = reduce_op;
  ops_.push_back(op);
}

void ExecutionEngine::finalize() {
  distinct_recipes_.clear();
  unique_tables_.clear();
  captured_table_gens_.clear();
  required_workspace_bytes_ = 0;
  compute_ops_per_step_ = 0;

  std::unordered_set<synRecipeHandle> seen_recipes;
  std::unordered_set<const BindingTable*> seen_tables;

  for (const auto& op : ops_) {
    if (!op.is_collective) {
      ++compute_ops_per_step_;
      const auto& entry = registry_.get(op.recipe_id);
      if (entry.workspace_bytes > required_workspace_bytes_) {
        required_workspace_bytes_ = entry.workspace_bytes;
      }
      if (seen_recipes.insert(op.recipe_handle).second) {
        distinct_recipes_.push_back(op.recipe_handle);
      }
      if (seen_tables.insert(op.bindings).second) {
        unique_tables_.push_back(op.bindings);
        captured_table_gens_.push_back(op.bindings->generation());
      }
    }
  }

  if (required_workspace_bytes_ > 0) {
    arena_.ensure_workspace(required_workspace_bytes_);
  }
  captured_arena_gen_ = arena_.generation();
  finalized_ = true;
}

bool ExecutionEngine::check_generations() {
  bool unchanged = (arena_.generation() == captured_arena_gen_);
  for (size_t i = 0; i < unique_tables_.size(); ++i) {
    if (unique_tables_[i]->generation() != captured_table_gens_[i]) {
      unchanged = false;
    }
  }
  if (unchanged) {
    ++generation_fast_hits_;
    return true;
  }

  // Validate that all bound slots have non-zero HBM addresses and update
  // captured generations.
  for (size_t i = 0; i < unique_tables_.size(); ++i) {
    const auto& infos = unique_tables_[i]->launch_infos();
    for (const auto& info : infos) {
      if (info.pTensorAddress == 0) {
        throw std::runtime_error(
            "ExecutionEngine: unbound tensor slot (address == 0)");
      }
    }
    captured_table_gens_[i] = unique_tables_[i]->generation();
  }
  captured_arena_gen_ = arena_.generation();
  ++generation_rebuilds_;
  return false;
}

void ExecutionEngine::step(size_t steps) {
  if (!finalized_) {
    finalize();
  }

  check_generations();
  const uint64_t ws_addr = arena_.workspace_addr();
  const synStreamHandle stream = runtime_.compute_stream();

  for (size_t s = 0; s < steps; ++s) {
    if (compute_ops_per_step_ > 0) {
      credit_mgr_.acquire_step_budget(compute_ops_per_step_, distinct_recipes_);
    }

    for (const auto& op : ops_) {
      if (!op.is_collective) {
        synStatus st = synLaunchWithExternalEventsExt(
            stream, op.bindings->data(), op.bindings->size(), ws_addr,
            op.recipe_handle, nullptr, 0, 0);
        if (st != synSuccess) {
          throw std::runtime_error(
              "ExecutionEngine: synLaunchWithExternalEventsExt failed");
        }
        ++compute_launches_;
      } else {
        if (runtime_.comm() == nullptr) {
          throw std::runtime_error(
              "ExecutionEngine: collective op requested without initialized "
              "hcclComm_t");
        }
        hcclResult_t hr = hcclSuccess;
        switch (op.coll_kind) {
          case CollectiveKind::AllGather:
            hr = hcclAllGather(reinterpret_cast<const void*>(op.send_addr),
                               reinterpret_cast<void*>(op.recv_addr), op.count,
                               op.dtype, runtime_.comm(), stream);
            break;
          case CollectiveKind::AllReduce:
            hr = hcclAllReduce(reinterpret_cast<const void*>(op.send_addr),
                               reinterpret_cast<void*>(op.recv_addr), op.count,
                               op.dtype, op.reduce_op, runtime_.comm(), stream);
            break;
          case CollectiveKind::ReduceScatter:
            hr = hcclReduceScatter(reinterpret_cast<const void*>(op.send_addr),
                                   reinterpret_cast<void*>(op.recv_addr),
                                   op.count, op.dtype, op.reduce_op,
                                   runtime_.comm(), stream);
            break;
        }
        if (hr != hcclSuccess) {
          throw std::runtime_error("ExecutionEngine: HCCL collective failed");
        }
        ++collective_launches_;
      }
    }

    ++steps_submitted_;
    if (compute_ops_per_step_ > 0) {
      credit_mgr_.commit_step(steps_submitted_, stream, compute_ops_per_step_,
                              distinct_recipes_);
    }
  }
}

void ExecutionEngine::drain() {
  credit_mgr_.drain_all();
  runtime_.synchronize_compute();
}

ExecutionEngine::Stats ExecutionEngine::stats() const noexcept {
  Stats s{};
  s.steps_submitted = steps_submitted_;
  s.compute_launches = compute_launches_;
  s.collective_launches = collective_launches_;
  s.generation_fast_hits = generation_fast_hits_;
  s.generation_rebuilds = generation_rebuilds_;
  s.distinct_recipes = distinct_recipes_.size();
  s.max_safe_in_flight_launches =
      InFlightCreditManager::max_launches_for_distinct(
          distinct_recipes_.size());
  s.credit_stats = credit_mgr_.stats();
  return s;
}

}  // namespace gaudirt::runtime
