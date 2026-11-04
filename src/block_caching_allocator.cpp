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

void BlockCachingAllocator::EventDeleter::operator()(void* event) const noexcept {
  if (event) synEventDestroy(static_cast<synEventHandle>(event));
}

BlockCachingAllocator::BlockCachingAllocator(synDeviceId device, uint64_t budget_bytes)
    : device_(device), capacity_(budget_bytes & ~uint64_t{255}) {
  if (!capacity_) throw std::invalid_argument("BlockCachingAllocator: budget below 256 bytes");
  reserve_slab();
}

BlockCachingAllocator::BlockCachingAllocator(synDeviceId device, uint64_t address,
                                             uint64_t bytes, BorrowedSlab)
    : device_(device), capacity_(bytes), base_(address), owns_slab_(false) {
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
    for (const auto& event : block.fences)
      synEventSynchronize(static_cast<synEventHandle>(event.get()));
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
  for (const auto& event : best->second.fences)
    check(synStreamWaitEvent(stream, static_cast<synEventHandle>(event.get()), 0),
          "stream wait failed");
  const uint64_t address = best->first;
  if (best->second.bytes > size) {
    Block remainder{best->second.bytes - size, false, {}, {}};
    remainder.fences = best->second.fences;
    blocks_.emplace(address + size, std::move(remainder));
  }
  auto& block = best->second;
  block.bytes = size;
  block.allocated = true;
  block.fences.clear();
  block.streams = {stream};
  allocated_ += size;
  peak_ = std::max(peak_, allocated_);
  return address;
}

void BlockCachingAllocator::record_stream(uint64_t address, synStreamHandle stream) {
  if (!stream) throw std::invalid_argument("BlockCachingAllocator: null stream");
  std::lock_guard lock(mutex_);
  auto it = blocks_.find(address);
  if (it == blocks_.end() || !it->second.allocated)
    throw std::invalid_argument("BlockCachingAllocator: unknown live allocation");
  auto& streams = it->second.streams;
  if (std::find(streams.begin(), streams.end(), stream) == streams.end())
    streams.push_back(stream);
}

BlockCachingAllocator::Blocks::iterator BlockCachingAllocator::merge_free(Blocks::iterator it) {
  const auto append_unique = [](auto& to, const auto& from) {
    for (const auto& event : from)
      if (std::find(to.begin(), to.end(), event) == to.end()) to.push_back(event);
  };
  auto next = std::next(it);
  if (next != blocks_.end() && !next->second.allocated) {
    it->second.bytes += next->second.bytes;
    append_unique(it->second.fences, next->second.fences);
    blocks_.erase(next);
  }
  if (it != blocks_.begin()) {
    auto prev = std::prev(it);
    if (!prev->second.allocated) {
      prev->second.bytes += it->second.bytes;
      append_unique(prev->second.fences, it->second.fences);
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
  std::vector<Event> events;
  events.reserve(it->second.streams.size());
  for (auto stream : it->second.streams) {
    synEventHandle raw = nullptr;
    check(synEventCreate(&raw, device_, 0), "event creation failed");
    Event event(raw, EventDeleter{});
    check(synEventRecord(raw, stream), "event record failed");
    events.push_back(std::move(event));
  }
  allocated_ -= it->second.bytes;
  it->second.allocated = false;
  it->second.streams.clear();
  it->second.fences = std::move(events);
  merge_free(it);
}

void BlockCachingAllocator::empty_cache() {
  std::lock_guard lock(mutex_);
  for (auto& [address, block] : blocks_) {
    (void)address;
    for (const auto& event : block.fences)
      check(synEventSynchronize(static_cast<synEventHandle>(event.get())), "event wait failed");
    block.fences.clear();
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
  pool_.reset();
  if (reservation_) parent_.free(reservation_);
}

}  // namespace gaudirt::runtime
