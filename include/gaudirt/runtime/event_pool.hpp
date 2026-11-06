// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

namespace gaudirt::runtime {

// The owning allocator serializes access and outlives all borrowed events.
// Events are retained until allocator teardown, avoiding driver and control
// block allocation once the maximum number of concurrent fences is reached.
class EventPool {
  struct Record {
    synEventHandle handle = nullptr;
    std::size_t references = 0;
    EventPool* owner = nullptr;
  };

 public:
  class Event {
   public:
    Event() = default;
    Event(const Event& other) : record_(other.record_) { retain(); }
    Event(Event&& other) noexcept : record_(other.record_) { other.record_ = nullptr; }
    Event& operator=(const Event& other) {
      if (this != &other) {
        release();
        record_ = other.record_;
        retain();
      }
      return *this;
    }
    Event& operator=(Event&& other) noexcept {
      if (this != &other) {
        release();
        record_ = other.record_;
        other.record_ = nullptr;
      }
      return *this;
    }
    ~Event() { release(); }
    synEventHandle get() const { return record_->handle; }
    bool operator==(const Event& other) const { return record_ == other.record_; }

   private:
    friend class EventPool;
    explicit Event(Record* record) : record_(record) { retain(); }
    void retain() { if (record_) ++record_->references; }
    void release() {
      if (record_ && --record_->references == 0)
        record_->owner->available_.push_back(record_);
      record_ = nullptr;
    }
    Record* record_ = nullptr;
  };

  explicit EventPool(synDeviceId device) : device_(device) {}
  ~EventPool() {
    for (const auto& record : records_) synEventDestroy(record->handle);
  }
  EventPool(const EventPool&) = delete;
  EventPool& operator=(const EventPool&) = delete;

  Event acquire() {
    if (!available_.empty()) {
      auto* record = available_.back();
      available_.pop_back();
      return Event(record);
    }
    // release() cannot allocate or throw, including during stack unwinding.
    available_.reserve(records_.size() + 1);
    records_.reserve(records_.size() + 1);
    auto record = std::make_unique<Record>();
    record->owner = this;
    if (synEventCreate(&record->handle, device_, 0) != synSuccess)
      throw std::runtime_error("EventPool: event creation failed");
    auto* raw = record.get();
    records_.push_back(std::move(record));
    return Event(raw);
  }

 private:
  synDeviceId device_;
  std::vector<std::unique_ptr<Record>> records_;
  std::vector<Record*> available_;
};

}  // namespace gaudirt::runtime
