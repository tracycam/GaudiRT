// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

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
  void record(synStreamHandle stream);
  void wait(synStreamHandle stream) const;
  void synchronize() const;
 private:
  synEventHandle event_ = nullptr;
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
  void copy_h2d(uint64_t dst, const void* src, uint64_t bytes,
                synStreamHandle stream, bool non_blocking);
  void copy_d2h(void* dst, uint64_t src, uint64_t bytes,
                synStreamHandle stream, bool non_blocking);
  static void copy_d2d(uint64_t dst, uint64_t src, uint64_t bytes,
                       synStreamHandle stream, bool non_blocking);

 private:
  struct EventDeleter { void operator()(void* event) const noexcept; };
  using Event = std::shared_ptr<void>;
  struct Block {
    uint64_t bytes = 0;
    bool allocated = false;
    std::vector<synStreamHandle> streams;
    std::vector<Event> fences;
  };
  void copy(void* host, uint64_t device, uint64_t bytes, synStreamHandle stream,
            bool non_blocking, synDmaDir direction);
  synDeviceId device_;
  uint64_t capacity_;
  void* base_ = nullptr;
  std::map<uint64_t, Block> blocks_;
  std::mutex mutex_;
};

}  // namespace gaudirt::runtime
