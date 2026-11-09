// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/block_caching_allocator.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace gaudirt::runtime {
namespace {
uint64_t aligned(uint64_t n) {
  if (!n || n > std::numeric_limits<uint64_t>::max() - 255)
    throw std::invalid_argument("BlockCachingAllocator: invalid size");
  return (n + 255) & ~uint64_t{255};
}
void check(synStatus status, const char* action) {
  if (status != synSuccess)
    throw std::runtime_error(std::string("BlockCachingAllocator: ") + action +
                             ", status " + std::to_string(static_cast<int>(status)));
}
}  // namespace

BlockCachingAllocator::BlockCachingAllocator(synDeviceId device, uint64_t budget_bytes)
    : device_(device), capacity_(budget_bytes & ~uint64_t{255}), events_(device) {
  if (!capacity_) throw std::invalid_argument("BlockCachingAllocator: budget below 256 bytes");
  reserve_slab();
}

BlockCachingAllocator::BlockCachingAllocator(synDeviceId device, uint64_t address,
                                             uint64_t bytes, BorrowedSlab)
    : device_(device), capacity_(bytes), base_(address), owns_slab_(false), events_(device) {
  if (!address || address % kAlignment || !bytes || bytes % kAlignment)
    throw std::invalid_argument("BlockCachingAllocator: invalid graph pool region");
  blocks_.emplace(base_, Block{capacity_, false, {}, {}});
}

void BlockCachingAllocator::reserve_slab() {
  uint64_t free_bytes = 0, total_bytes = 0;
  check(synDeviceGetMemoryInfo(device_, &free_bytes, &total_bytes), "memory query failed");
  if (capacity_ > free_bytes)
    throw std::runtime_error("BlockCachingAllocator: budget exceeds available HBM");
  check(synDeviceMalloc(device_, capacity_, 0, 0, &base_), "slab allocation failed");
  if (!base_ || base_ % kAlignment) {
    if (base_) synDeviceFree(device_, base_, 0);
    base_ = 0;
    throw std::runtime_error("BlockCachingAllocator: slab is not 256-byte aligned");
  }
  try {
    blocks_.emplace(base_, Block{capacity_, false, {}, {}});
  } catch (...) {
    synDeviceFree(device_, base_, 0);
    base_ = 0;
    throw;
  }
}

BlockCachingAllocator::~BlockCachingAllocator() {
  // Synchronize only streams/events that can still refer to this slab.
  for (auto& [address, block] : blocks_) {
    (void)address;
    for (auto stream : block.streams) synStreamSynchronize(stream);
    for (auto stream : block.pending_streams) synStreamSynchronize(stream);
  }
  blocks_.clear();
  if (base_ && owns_slab_) synDeviceFree(device_, base_, 0);
}

uint64_t BlockCachingAllocator::allocate(uint64_t bytes, synStreamHandle stream) {
  const uint64_t size = aligned(bytes);
  if (!stream) throw std::invalid_argument("BlockCachingAllocator: null stream");
  if (size > capacity_) throw std::runtime_error("BlockCachingAllocator: request exceeds budget");
  std::lock_guard lock(mutex_);
  if (!base_) reserve_slab();
  auto best = blocks_.end();
  for (auto it = blocks_.begin(); it != blocks_.end(); ++it)
    if (!it->second.allocated && it->second.bytes >= size &&
        (best == blocks_.end() || it->second.bytes < best->second.bytes)) best = it;
  if (best == blocks_.end())
    throw std::runtime_error("BlockCachingAllocator: out of HBM: requested=" +
                             std::to_string(size) + ", allocated=" +
                             std::to_string(allocated_) + ", capacity=" +
                             std::to_string(capacity_));
  // All uses were submitted before free. FIFO orders same-stream reuse;
  // for another stream, record after those uses and wait before recycling.
  for (auto producer : best->second.pending_streams) {
    if (producer == stream) continue;
    auto event = events_.record(producer);
    check(events_.wait(event, stream), "stream wait failed");
  }
  const uint64_t address = best->first;
  if (best->second.bytes > size) {
    Block remainder{best->second.bytes - size, false, {}, {}};
    remainder.pending_streams = best->second.pending_streams;
    blocks_.emplace(address + size, std::move(remainder));
  }
  auto& block = best->second;
  block.bytes = size;
  block.allocated = true;
  block.pending_streams.clear();
  block.streams = {stream};
  allocated_ += size;
  peak_ = std::max(peak_, allocated_);
  return address;
}

void BlockCachingAllocator::record_stream(uint64_t address, synStreamHandle stream) {
  if (!stream) throw std::invalid_argument("BlockCachingAllocator: null stream");
  std::lock_guard lock(mutex_);
  auto it = blocks_.upper_bound(address);
  if (it == blocks_.begin())
    throw std::invalid_argument("BlockCachingAllocator: unknown live allocation");
  --it;
  if (!it->second.allocated || address - it->first >= it->second.bytes)
    throw std::invalid_argument("BlockCachingAllocator: unknown live allocation");
  auto& streams = it->second.streams;
  if (std::find(streams.begin(), streams.end(), stream) == streams.end())
    streams.push_back(stream);
}

BlockCachingAllocator::Blocks::iterator BlockCachingAllocator::merge_free(Blocks::iterator it) {
  const auto retain_streams = [](auto& to, const auto& from) {
    for (auto stream : from) {
      if (std::find(to.begin(), to.end(), stream) == to.end()) to.push_back(stream);
    }
  };
  auto next = std::next(it);
  if (next != blocks_.end() && !next->second.allocated) {
    it->second.bytes += next->second.bytes;
    retain_streams(it->second.pending_streams, next->second.pending_streams);
    blocks_.erase(next);
  }
  if (it != blocks_.begin()) {
    auto prev = std::prev(it);
    if (!prev->second.allocated) {
      prev->second.bytes += it->second.bytes;
      retain_streams(prev->second.pending_streams, it->second.pending_streams);
      blocks_.erase(it);
      it = prev;
    }
  }
  return it;
}

void BlockCachingAllocator::free(uint64_t address) {
  std::lock_guard lock(mutex_);
  auto it = blocks_.find(address);
  if (it == blocks_.end() || !it->second.allocated)
    throw std::invalid_argument("BlockCachingAllocator: unknown or freed allocation");
  for (auto stream : it->second.streams) events_.prepare(stream);
  allocated_ -= it->second.bytes;
  it->second.allocated = false;
  it->second.pending_streams.swap(it->second.streams);
  merge_free(it);
}

void BlockCachingAllocator::empty_cache() {
  std::lock_guard lock(mutex_);
  for (auto& [address, block] : blocks_) {
    (void)address;
    for (auto stream : block.pending_streams)
      check(synStreamSynchronize(stream), "stream wait failed");
    block.pending_streams.clear();
  }
  if (base_ && owns_slab_ && allocated_ == 0) {
    check(synDeviceFree(device_, base_, 0), "slab release failed");
    blocks_.clear();
    base_ = 0;
  }
}

uint64_t BlockCachingAllocator::memory_allocated() const {
  std::lock_guard lock(mutex_);
  return allocated_;
}
uint64_t BlockCachingAllocator::max_memory_allocated() const {
  std::lock_guard lock(mutex_);
  return peak_;
}
uint64_t BlockCachingAllocator::largest_free_block() const {
  std::lock_guard lock(mutex_);
  if (!base_) return capacity_;
  uint64_t largest = 0;
  for (const auto& [address, block] : blocks_) {
    (void)address;
    if (!block.allocated) largest = std::max(largest, block.bytes);
  }
  return largest;
}
bool BlockCachingAllocator::contains_live_range(uint64_t address, uint64_t bytes) const {
  std::lock_guard lock(mutex_);
  auto it = blocks_.upper_bound(address);
  if (it == blocks_.begin()) return false;
  --it;
  return it->second.allocated && address >= it->first &&
         address - it->first <= it->second.bytes &&
         bytes <= it->second.bytes - (address - it->first);
}

GraphPool::GraphPool(BlockCachingAllocator& parent, uint64_t bytes, synStreamHandle stream)
    : parent_(parent) {
  const uint64_t size = aligned(bytes);
  reservation_ = parent_.allocate(size, stream);
  try {
    pool_.reset(new BlockCachingAllocator(parent_.device_, reservation_, size,
                                           BlockCachingAllocator::BorrowedSlab{}));
  } catch (...) {
    parent_.free(reservation_);
    reservation_ = 0;
    throw;
  }
}
GraphPool::~GraphPool() {
  for (auto& region : extra_regions_) {
    region.pool.reset();
    parent_.free(region.reservation);
  }
  extra_regions_.clear();
  pool_.reset();
  if (reservation_) parent_.free(reservation_);
}

uint64_t GraphPool::allocate(uint64_t bytes, synStreamHandle stream) {
  const uint64_t size = aligned(bytes);
  std::lock_guard lock(mutex_);
  BlockCachingAllocator* best = nullptr;
  uint64_t best_free = std::numeric_limits<uint64_t>::max();
  const auto consider = [&](BlockCachingAllocator& pool) {
    const auto available = pool.largest_free_block();
    if (available >= size && available < best_free) {
      best = &pool;
      best_free = available;
    }
  };
  consider(*pool_);
  for (auto& region : extra_regions_) consider(*region.pool);
  if (!best) {
    const auto available = parent_.largest_free_block();
    if (available < size) throw std::runtime_error("GraphPool: out of HBM for graph temporaries");
    const auto region_bytes = std::min(available, std::max(size, pool_->capacity_bytes()));
    const auto reservation = parent_.allocate(region_bytes, stream);
    try {
      auto pool = std::unique_ptr<BlockCachingAllocator>(new BlockCachingAllocator(
          parent_.device_, reservation, region_bytes, BlockCachingAllocator::BorrowedSlab{}));
      extra_regions_.push_back(Region{reservation, std::move(pool)});
      best = extra_regions_.back().pool.get();
    } catch (...) {
      parent_.free(reservation);
      throw;
    }
  }
  return best->allocate(size, stream);
}

void GraphPool::free(uint64_t address) {
  std::lock_guard lock(mutex_);
  if (pool_->contains_live_range(address, 1)) return pool_->free(address);
  for (auto& region : extra_regions_)
    if (region.pool->contains_live_range(address, 1)) return region.pool->free(address);
  throw std::invalid_argument("GraphPool: unknown or freed allocation");
}

bool GraphPool::contains_live_range(uint64_t address, uint64_t bytes) const {
  std::lock_guard lock(mutex_);
  if (pool_->contains_live_range(address, bytes)) return true;
  for (const auto& region : extra_regions_)
    if (region.pool->contains_live_range(address, bytes)) return true;
  return false;
}

void GraphPool::record_stream(uint64_t address, synStreamHandle stream) {
  std::lock_guard lock(mutex_);
  if (pool_->contains_live_range(address, 1)) return pool_->record_stream(address, stream);
  for (const auto& region : extra_regions_)
    if (region.pool->contains_live_range(address, 1)) return region.pool->record_stream(address, stream);
  throw std::invalid_argument("GraphPool: unknown or freed allocation");
}

uint64_t GraphPool::capacity_bytes() const {
  std::lock_guard lock(mutex_);
  uint64_t total = pool_->capacity_bytes();
  for (const auto& region : extra_regions_) total += region.pool->capacity_bytes();
  return total;
}

uint64_t GraphPool::memory_allocated() const {
  std::lock_guard lock(mutex_);
  uint64_t total = pool_->memory_allocated();
  for (const auto& region : extra_regions_) total += region.pool->memory_allocated();
  return total;
}

}  // namespace gaudirt::runtime
