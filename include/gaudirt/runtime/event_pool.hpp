// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include <stdexcept>
#include <unordered_map>

namespace gaudirt::runtime {

// The owning allocator serializes records and waits and outlives its streams.
// A stream's latest record covers all earlier work on that stream. Synapse
// snapshots the event state when a wait is submitted, so later records do not
// change already-submitted waits. Sharing one event per producer stream may
// wait for extra work, but does not permit a block to be reused too early.
class EventPool {
 public:
  class Event {
   public:
    Event() = default;
    synEventHandle get() const { return handle_; }
    bool operator==(const Event& other) const { return handle_ == other.handle_; }

   private:
    friend class EventPool;
    explicit Event(synEventHandle handle) : handle_(handle) {}
    synEventHandle handle_ = nullptr;
  };

  explicit EventPool(synDeviceId device) : device_(device) {}
  ~EventPool() {
    for (const auto& [stream, event] : events_) {
      (void)stream;
      synEventDestroy(event);
    }
  }
  EventPool(const EventPool&) = delete;
  EventPool& operator=(const EventPool&) = delete;

  Event acquire(synStreamHandle stream) {
    const auto existing = events_.find(stream);
    if (existing != events_.end()) return Event(existing->second);
    synEventHandle event = nullptr;
    if (synEventCreate(&event, device_, 0) != synSuccess)
      throw std::runtime_error("EventPool: event creation failed");
    try {
      events_.emplace(stream, event);
    } catch (...) {
      synEventDestroy(event);
      throw;
    }
    return Event(event);
  }

 private:
  synDeviceId device_;
  std::unordered_map<synStreamHandle, synEventHandle> events_;
};

}  // namespace gaudirt::runtime
