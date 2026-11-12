// SPDX-License-Identifier: Apache-2.0
#include "context.hpp"
#include "attention_plan.hpp"
#include "segmented_runner.hpp"

#include <c10/util/Exception.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <exception>

namespace gaudirt::generic {
namespace {
std::atomic<uint64_t> next_context_generation{1};
}

std::mutex event_mutex;
thread_local c10::StreamId active_stream_id = 0;

StreamRegistry::StreamState::StreamState(synDeviceId device, synStreamHandle stream)
    : handle(stream), metadata_reusable(device), metadata_ready(device),
      validation_ready(device), validation_completed(device),
      h2d_reusable(device), h2d_ready(device), d2h_reusable(device), d2h_ready(device) {}

StreamRegistry::StreamRegistry(synDeviceId device, synStreamHandle primary) : device_(device) {
  streams_.push_back(std::make_unique<StreamState>(device_, primary));
  primary_ = streams_.front().get();
}

StreamRegistry::~StreamRegistry() {
  for (size_t i = 1; i < streams_.size(); ++i) {
    synStreamSynchronize(streams_[i]->handle);
    synStreamDestroy(streams_[i]->handle);
  }
}

c10::StreamId StreamRegistry::create() {
  synStreamHandle stream = nullptr;
  TORCH_CHECK(synStreamCreateGeneric(&stream, device_, 0) == synSuccess, "GaudiRT: stream creation failed");
  std::lock_guard lock(mutex_);
  try { streams_.push_back(std::make_unique<StreamState>(device_, stream)); }
  catch (...) { synStreamDestroy(stream); throw; }
  return static_cast<c10::StreamId>(streams_.size() - 1);
}

synStreamHandle StreamRegistry::get(c10::StreamId id) { return state(id).handle; }

std::vector<synStreamHandle> StreamRegistry::handles() {
  std::lock_guard lock(mutex_);
  std::vector<synStreamHandle> result;
  result.reserve(streams_.size());
  for (const auto& stream : streams_) result.push_back(stream->handle);
  return result;
}

StreamRegistry::StreamState& StreamRegistry::state(c10::StreamId id) {
  if (id == 0) return *primary_;
  std::lock_guard lock(mutex_);
  TORCH_CHECK(id >= 0 && static_cast<size_t>(id) < streams_.size(), "GaudiRT: invalid stream");
  return *streams_[static_cast<size_t>(id)];
}

void StreamRegistry::synchronize() {
  std::lock_guard lock(mutex_);
  for (const auto& stream : streams_)
    TORCH_CHECK(synStreamSynchronize(stream->handle) == synSuccess, "GaudiRT: stream synchronization failed");
}

Context::Context(uint32_t module, uint64_t hbm_bytes, uint64_t pinned_bytes)
    : generation(next_context_generation.fetch_add(1, std::memory_order_relaxed)),
      device(config(module)), streams(device.device_id(), device.compute_stream()),
      allocator(device.device_id(), hbm_bytes), pinned(device.device_id(), pinned_bytes),
      attention_staging_pool(std::make_unique<AttentionStagingPool>(*this)), cache(cache_path()) {
  for (auto stream : {device.compute_stream(), device.h2d_stream(), device.d2h_stream()})
    prepare_stream(stream);
}

Context::~Context() {
  const bool unwinding = std::uncaught_exceptions() > 0;
  if (!unwinding) {
    try { streams.synchronize(); } catch (...) {}
    synStreamSynchronize(device.h2d_stream());
    synStreamSynchronize(device.d2h_stream());
  }
  trim_segment_outputs();
  if (comm_teardown) {
    try { comm_teardown(unwinding); } catch (...) {}
    comm_teardown = nullptr;
    comm_trim = nullptr;
    comm_state.reset();
  }
  std::lock_guard lock(event_mutex);
  for (auto* event : events) {
    if (event->handle) synEventDestroy(event->handle);
    event->handle = nullptr;
    event->owner = nullptr;
  }
  events.clear();
}

synStreamHandle Context::stream() {
  return active_stream_id == 0 ? device.compute_stream() : streams.get(active_stream_id);
}

void Context::synchronize_stream() {
  TORCH_CHECK(synStreamSynchronize(stream()) == synSuccess, "GaudiRT: stream synchronization failed");
}

void Context::synchronize_all() { streams.synchronize(); device.synchronize_all(); }

void Context::prepare_stream(synStreamHandle stream) {
  allocator.prepare_stream(stream);
  pinned.prepare_stream(stream);
  attention_staging_pool->prepare_stream(stream);
  std::lock_guard lock(cold_mutex);
  if (graph_pool) graph_pool->prepare_stream(stream);
}

AttentionStagingPool& Context::attention_staging() { return *attention_staging_pool; }

runtime::GraphPool& Context::ensure_graph_pool(uint64_t budget, synStreamHandle stream) {
  if (!graph_pool) {
    auto pool = std::make_unique<runtime::GraphPool>(allocator, budget, stream);
    for (auto registered : streams.handles()) pool->prepare_stream(registered);
    pool->prepare_stream(device.h2d_stream());
    pool->prepare_stream(device.d2h_stream());
    graph_pool = std::move(pool);
  }
  return *graph_pool;
}

void Context::register_segment_outputs(const std::shared_ptr<SegmentOutputPool>& pool) {
  std::lock_guard lock(segment_outputs_mutex);
  segment_outputs.emplace_back(pool);
}

bool Context::trim_segment_outputs() {
  std::lock_guard lock(segment_outputs_mutex);
  bool idle = true;
  for (auto it = segment_outputs.begin(); it != segment_outputs.end();) {
    if (auto pool = it->lock()) {
      idle = pool->trim() && idle;
      ++it;
    } else {
      it = segment_outputs.erase(it);
    }
  }
  return idle;
}

uint64_t Context::row_count_address(uint32_t rows) {
  // Every value has a 256-byte-aligned address, so warm launches bind an
  // immutable count without copying metadata or synchronizing a stream.
  std::call_once(row_count_once, [&] {
    auto h2d = device.h2d_stream();
    const auto address = allocator.allocate(kRowCountTableBytes, h2d);
    try {
      const auto chunk_bytes = std::min(kRowCountTableBytes, pinned.largest_free_block()) & ~uint64_t{255};
      TORCH_CHECK(chunk_bytes >= 256, "GaudiRT: no pinned memory for row counts");
      void* staging = pinned.allocate(chunk_bytes);
      try {
        for (uint64_t offset = 0; offset < kRowCountTableBytes; offset += chunk_bytes) {
          const auto bytes = std::min(chunk_bytes, kRowCountTableBytes - offset);
          std::memset(staging, 0, bytes);
          auto* values = static_cast<int32_t*>(staging);
          for (uint64_t item = 0; item < bytes; item += 256)
            values[item / sizeof(int32_t)] = static_cast<int32_t>((offset + item) / 256 + 1);
          pinned.copy_h2d(address + offset, staging, bytes, h2d, false);
        }
      } catch (...) {
        pinned.free(staging);
        throw;
      }
      pinned.free(staging);
    } catch (...) {
      allocator.free(address);
      throw;
    }
    row_count_table = address;
  });
  return row_count_table + static_cast<uint64_t>(rows - 1) * 256;
}

std::filesystem::path Context::cache_path() {
  const char* path = std::getenv("GAUDIRT_RECIPE_CACHE");
  return path ? std::filesystem::path(path) : std::filesystem::path{};
}

runtime::DeviceRuntime::Config Context::config(uint32_t module) {
  runtime::DeviceRuntime::Config cfg{};
  cfg.module_id = module;
  cfg.uclamp_min = 0;
  return cfg;
}

}  // namespace gaudirt::generic
