// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "cold_recipe.hpp"
#include "gaudirt/runtime/block_caching_allocator.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/pinned_host_pool.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <c10/core/Stream.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace gaudirt::generic {

struct Context;
class AttentionStagingPool;
struct EventState {
  Context* owner;
  synEventHandle handle;
  bool timing;
};
extern std::mutex event_mutex;
extern thread_local c10::StreamId active_stream_id;

class StreamRegistry {
 public:
  struct StreamState {
    StreamState(synDeviceId device, synStreamHandle stream);
    synStreamHandle handle;
    runtime::StreamEvent metadata_reusable, metadata_ready, validation_ready, validation_completed;
    runtime::StreamEvent h2d_reusable, h2d_ready, d2h_reusable, d2h_ready;
  };

  StreamRegistry(synDeviceId device, synStreamHandle primary);
  ~StreamRegistry();
  c10::StreamId create();
  synStreamHandle get(c10::StreamId id);
  StreamState& state(c10::StreamId id);
  void synchronize();

 private:
  synDeviceId device_;
  StreamState* primary_ = nullptr;
  std::mutex mutex_;
  std::vector<std::unique_ptr<StreamState>> streams_;
};

struct Context {
  Context(uint32_t module, uint64_t hbm_bytes, uint64_t pinned_bytes);
  ~Context();
  synStreamHandle stream();
  void synchronize_stream();
  void synchronize_all();
  AttentionStagingPool& attention_staging();
  uint64_t row_count_address(uint32_t rows);
  static constexpr uint64_t kRowCountTableBytes = 4096 * 256;
  static std::filesystem::path cache_path();
  static runtime::DeviceRuntime::Config config(uint32_t module);

  const uint64_t generation;
  runtime::DeviceRuntime device;
  StreamRegistry streams;
  runtime::BlockCachingAllocator allocator;
  std::unique_ptr<runtime::GraphPool> graph_pool;
  runtime::PinnedHostPool pinned;
  std::once_flag attention_staging_once;
  std::unique_ptr<AttentionStagingPool> attention_staging_pool;
  std::once_flag row_count_once;
  uint64_t row_count_table = 0;
  runtime::RecipeCache cache;
  std::mutex cold_mutex;
  std::unordered_map<std::string, std::unique_ptr<ColdRecipe>> cold_recipes;
  std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> attention_zero_states;
  std::unordered_set<EventState*> events;
};

}  // namespace gaudirt::generic
