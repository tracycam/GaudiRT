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

// A single device allocation subdivided into 256-byte blocks. Every stream
// which uses an allocation must be recorded before that allocation is freed.
// The DeviceRuntime and its streams must outlive this allocator.
class BlockCachingAllocator {
 public:
  static constexpr uint64_t kAlignment = 256;

  BlockCachingAllocator(synDeviceId device, uint64_t budget_bytes);
  ~BlockCachingAllocator();
  BlockCachingAllocator(const BlockCachingAllocator&) = delete;
  BlockCachingAllocator& operator=(const BlockCachingAllocator&) = delete;

  uint64_t allocate(uint64_t bytes, synStreamHandle stream);
  void prepare_stream(synStreamHandle stream);
  void record_stream(uint64_t address, synStreamHandle stream);
  void free(uint64_t address);

  // Wait for free blocks' streams. A completely unused slab is returned to
  // Synapse; the next allocation reacquires the configured budget.
  void empty_cache();
  uint64_t memory_allocated() const;
  uint64_t max_memory_allocated() const;
  uint64_t largest_free_block() const;
  uint64_t capacity_bytes() const noexcept { return capacity_; }
  bool contains_live_range(uint64_t address, uint64_t bytes) const;

 private:
  friend class GraphPool;
  struct BorrowedSlab {
    EventPool* shared_events = nullptr;
  };
  BlockCachingAllocator(synDeviceId device, uint64_t address, uint64_t bytes,
                        BorrowedSlab borrowed);
  bool release_idle_borrowed_slab(std::vector<synStreamHandle>& pending_out);
  struct Block {
    uint64_t bytes = 0;
    bool allocated = false;
    std::vector<synStreamHandle> streams;
    std::vector<synStreamHandle> pending_streams;
  };
  using Blocks = std::map<uint64_t, Block>;
  void reserve_slab();
  Blocks::iterator merge_free(Blocks::iterator block);

  synDeviceId device_;
  uint64_t capacity_;
  uint64_t base_ = 0;
  uint64_t allocated_ = 0;
  uint64_t peak_ = 0;
  bool owns_slab_ = true;
  std::unique_ptr<EventPool> owned_events_;
  EventPool& events_;
  Blocks blocks_;
  mutable std::mutex mutex_;
};

// Reserves a region of the parent slab exclusively for graph temporaries.
// The graph pool must be destroyed before its parent and its device streams.
class GraphPool {
 public:
  GraphPool(BlockCachingAllocator& parent, uint64_t bytes, synStreamHandle stream);
  ~GraphPool();
  GraphPool(const GraphPool&) = delete;
  GraphPool& operator=(const GraphPool&) = delete;
  BlockCachingAllocator& allocator() noexcept { return *pool_; }
  uint64_t allocate(uint64_t bytes, synStreamHandle stream);
  void prepare_stream(synStreamHandle stream);
  void free(uint64_t address);
  bool contains_live_range(uint64_t address, uint64_t bytes) const;
  void record_stream(uint64_t address, synStreamHandle stream);
  uint64_t capacity_bytes() const;
  uint64_t memory_allocated() const;
 private:
  struct Region {
    uint64_t reservation;
    std::unique_ptr<BlockCachingAllocator> pool;
  };
  BlockCachingAllocator& parent_;
  EventPool extra_events_;
  uint64_t reservation_ = 0;
  std::unique_ptr<BlockCachingAllocator> pool_;
  std::vector<Region> extra_regions_;
  std::vector<synStreamHandle> streams_;
  mutable std::mutex mutex_;
  void prepare_stream_locked(synStreamHandle stream);
};

}  // namespace gaudirt::runtime
