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
  struct State {
    synEventHandle handle = nullptr;
    bool needs_wait = false;
  };

 public:
  class Event {
   public:
    Event() = default;
    synEventHandle get() const { return state_ ? state_->handle : nullptr; }
    bool operator==(const Event& other) const { return state_ == other.state_; }

   private:
    friend class EventPool;
    explicit Event(State* state) : state_(state) {}
    State* state_ = nullptr;
  };

  explicit EventPool(synDeviceId device) : device_(device) {}
  ~EventPool() {
    for (const auto& [stream, state] : events_) {
      (void)stream;
      synEventDestroy(state.handle);
    }
  }
  EventPool(const EventPool&) = delete;
  EventPool& operator=(const EventPool&) = delete;

  Event record(synStreamHandle stream) {
    auto existing = events_.find(stream);
    if (existing == events_.end()) {
      synEventHandle event = nullptr;
      if (synEventCreate(&event, device_, 0) != synSuccess)
        throw std::runtime_error("EventPool: event creation failed");
      try {
        existing = events_.emplace(stream, State{event, false}).first;
      } catch (...) {
        synEventDestroy(event);
        throw;
      }
    }
    auto& state = existing->second;
    // Synapse requires every recorded event to be waited upon before it is
    // re-recorded. A same-stream wait consumes an otherwise unobserved
    // record without blocking the host or changing stream ordering.
    if (state.needs_wait && synStreamWaitEvent(stream, state.handle, 0) != synSuccess)
      throw std::runtime_error("EventPool: event reuse wait failed");
    state.needs_wait = false;
    if (synEventRecord(state.handle, stream) != synSuccess)
      throw std::runtime_error("EventPool: event record failed");
    state.needs_wait = true;
    return Event(&state);
  }

  synStatus wait(Event event, synStreamHandle stream) {
    const auto status = synStreamWaitEvent(stream, event.get(), 0);
    if (status == synSuccess) event.state_->needs_wait = false;
    return status;
  }

  synStatus synchronize(Event event) {
    const auto status = synEventSynchronize(event.get());
    if (status == synSuccess) event.state_->needs_wait = false;
    return status;
  }

 private:
  synDeviceId device_;
  std::unordered_map<synStreamHandle, State> events_;
};

}  // namespace gaudirt::runtime
