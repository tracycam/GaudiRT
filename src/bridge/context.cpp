// SPDX-License-Identifier: Apache-2.0
#include "context.hpp"

#include <c10/util/Exception.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>

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
      allocator(device.device_id(), hbm_bytes), pinned(device.device_id(), pinned_bytes), cache(cache_path()) {}

Context::~Context() {
  try { streams.synchronize(); } catch (...) {}
  synStreamSynchronize(device.h2d_stream());
  synStreamSynchronize(device.d2h_stream());
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
