// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include "gaudirt/runtime/event_pool.hpp"

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

namespace gaudirt::runtime {

class StreamEvent {
 public:
  explicit StreamEvent(synDeviceId device);
  ~StreamEvent();
  StreamEvent(const StreamEvent&) = delete;
  StreamEvent& operator=(const StreamEvent&) = delete;
  void record_and_wait(synStreamHandle producer, synStreamHandle consumer);
  void record_and_synchronize(synStreamHandle stream);
  void synchronize() const;
 private:
  synEventHandle event_ = nullptr;
  mutable std::mutex mutex_;
};

// Buffers come from one synHostMalloc allocation and are already DMA mapped.
// A nonblocking transfer keeps its block alive until the transfer completes,
// even if free() is called before then. The caller must not access an output
// buffer until its D2H operation completes.
class PinnedHostPool {
 public:
  PinnedHostPool(synDeviceId device, uint64_t bytes);
  ~PinnedHostPool();
  PinnedHostPool(const PinnedHostPool&) = delete;
  PinnedHostPool& operator=(const PinnedHostPool&) = delete;

  void* allocate(uint64_t bytes);
  void free(void* address);
  void record_stream(const void* address, synStreamHandle stream);
  bool contains_live_range(const void* address, uint64_t bytes) const;
  uint64_t memory_allocated() const;
  uint64_t largest_free_block() const;
  void copy_h2d(uint64_t dst, const void* src, uint64_t bytes,
                synStreamHandle stream, bool non_blocking);
  void copy_d2h(void* dst, uint64_t src, uint64_t bytes,
                synStreamHandle stream, bool non_blocking);
  static void copy_d2d(uint64_t dst, uint64_t src, uint64_t bytes,
                       synStreamHandle stream, bool non_blocking);

 private:
  using Event = EventPool::Event;
  struct Fence {
    synStreamHandle stream;
    uint64_t sequence;
    Event event;
  };
  struct Block {
    uint64_t bytes = 0;
    bool allocated = false;
    std::vector<synStreamHandle> streams;
    std::vector<Fence> fences;
  };
  void copy(void* host, uint64_t device, uint64_t bytes, synStreamHandle stream,
            bool non_blocking, synDmaDir direction);
  synDeviceId device_;
  uint64_t capacity_;
  void* base_ = nullptr;
  uint64_t allocated_ = 0;
  uint64_t next_fence_sequence_ = 0;
  EventPool events_;
  std::map<uint64_t, Block> blocks_;
  mutable std::mutex mutex_;
};

}  // namespace gaudirt::runtime
