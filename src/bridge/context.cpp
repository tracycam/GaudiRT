// SPDX-License-Identifier: Apache-2.0
#include "context.hpp"

#include <c10/util/Exception.h>

#include <atomic>
#include <cstdlib>

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
