// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/pinned_host_pool.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>

namespace gaudirt::runtime {
namespace {
void check(synStatus status, const char* action) {
  if (status != synSuccess)
    throw std::runtime_error(std::string("PinnedHostPool: ") + action + ", status " +
                             std::to_string(static_cast<int>(status)));
}
uint64_t aligned(uint64_t n) {
  if (!n || n > std::numeric_limits<uint64_t>::max() - 255)
    throw std::invalid_argument("PinnedHostPool: invalid size");
  return (n + 255) & ~uint64_t{255};
}
void check_transfer(uint64_t dst, uint64_t src, uint64_t bytes, synStreamHandle stream) {
  if (!dst || !src || !bytes || !stream)
    throw std::invalid_argument("PinnedHostPool: invalid transfer");
}
}  // namespace

StreamEvent::StreamEvent(synDeviceId device) {
  check(synEventCreate(&event_, device, 0), "event creation failed");
}
StreamEvent::~StreamEvent() { if (event_) synEventDestroy(event_); }
void StreamEvent::record(synStreamHandle stream) {
  if (!stream) throw std::invalid_argument("StreamEvent: null stream");
  check(synEventRecord(event_, stream), "event record failed");
}
void StreamEvent::wait(synStreamHandle stream) const {
  if (!stream) throw std::invalid_argument("StreamEvent: null stream");
  check(synStreamWaitEvent(stream, event_, 0), "stream wait failed");
}
void StreamEvent::synchronize() const { check(synEventSynchronize(event_), "event wait failed"); }

void PinnedHostPool::EventDeleter::operator()(void* event) const noexcept {
  if (event) synEventDestroy(static_cast<synEventHandle>(event));
}
PinnedHostPool::PinnedHostPool(synDeviceId device, uint64_t bytes)
    : device_(device), capacity_(bytes & ~uint64_t{255}) {
  if (!capacity_) throw std::invalid_argument("PinnedHostPool: budget below 256 bytes");
  check(synHostMalloc(device_, capacity_, 0, &base_), "host allocation failed");
  try {
    blocks_.emplace(reinterpret_cast<uint64_t>(base_), Block{capacity_, false, {}, {}});
  } catch (...) {
    synHostFree(device_, base_, 0);
    base_ = nullptr;
    throw;
  }
}
PinnedHostPool::~PinnedHostPool() {
  for (auto& [address, block] : blocks_) {
    (void)address;
    for (auto stream : block.streams) synStreamSynchronize(stream);
    for (const auto& event : block.fences)
      synEventSynchronize(static_cast<synEventHandle>(event.get()));
  }
  blocks_.clear();
  if (base_) synHostFree(device_, base_, 0);
}
void* PinnedHostPool::allocate(uint64_t bytes) {
  uint64_t size = aligned(bytes);
  std::lock_guard lock(mutex_);
  auto best = blocks_.end();
  for (auto it = blocks_.begin(); it != blocks_.end(); ++it)
    if (!it->second.allocated && it->second.bytes >= size &&
        (best == blocks_.end() || it->second.bytes < best->second.bytes)) best = it;
  if (best == blocks_.end()) throw std::runtime_error("PinnedHostPool: out of pinned memory");
  // A host pointer can be dereferenced immediately, so a stream wait is not enough.
  for (const auto& event : best->second.fences)
    check(synEventSynchronize(static_cast<synEventHandle>(event.get())), "reuse wait failed");
  const uint64_t address = best->first;
  if (best->second.bytes > size)
    blocks_.emplace(address + size, Block{best->second.bytes - size, false, {}, {}});
  best->second.bytes = size;
  best->second.allocated = true;
  best->second.fences.clear();
  allocated_ += size;
  return reinterpret_cast<void*>(address);
}
void PinnedHostPool::free(void* address) {
  std::lock_guard lock(mutex_);
  auto it = blocks_.find(reinterpret_cast<uint64_t>(address));
  if (it == blocks_.end() || !it->second.allocated)
    throw std::invalid_argument("PinnedHostPool: unknown or freed allocation");
  std::vector<Event> events;
  for (auto stream : it->second.streams) {
    synEventHandle raw = nullptr;
    check(synEventCreate(&raw, device_, 0), "event creation failed");
    Event event(raw, EventDeleter{});
    check(synEventRecord(raw, stream), "event record failed");
    events.push_back(std::move(event));
  }
  it->second.allocated = false;
  allocated_ -= it->second.bytes;
  it->second.streams.clear();
  it->second.fences = std::move(events);
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
    }
  }
}
void PinnedHostPool::record_stream(const void* address, synStreamHandle stream) {
  if (!stream) throw std::invalid_argument("PinnedHostPool: null stream");
  std::lock_guard lock(mutex_);
  auto it = blocks_.upper_bound(reinterpret_cast<uint64_t>(address));
  if (it == blocks_.begin()) throw std::invalid_argument("PinnedHostPool: invalid pointer");
  --it;
  const auto offset = reinterpret_cast<uint64_t>(address) - it->first;
  if (!it->second.allocated || offset >= it->second.bytes)
    throw std::invalid_argument("PinnedHostPool: invalid pointer");
  auto& streams = it->second.streams;
  if (std::find(streams.begin(), streams.end(), stream) == streams.end()) streams.push_back(stream);
}
bool PinnedHostPool::contains_live_range(const void* address, uint64_t bytes) const {
  std::lock_guard lock(mutex_);
  const auto value = reinterpret_cast<uint64_t>(address);
  auto it = blocks_.upper_bound(value);
  if (it == blocks_.begin()) return false;
  --it;
  const auto offset = value - it->first;
  return it->second.allocated && offset <= it->second.bytes && bytes <= it->second.bytes - offset;
}
uint64_t PinnedHostPool::memory_allocated() const {
  std::lock_guard lock(mutex_);
  return allocated_;
}
uint64_t PinnedHostPool::largest_free_block() const {
  std::lock_guard lock(mutex_);
  uint64_t largest = 0;
  for (const auto& [address, block] : blocks_) {
    (void)address;
    if (!block.allocated) largest = std::max(largest, block.bytes);
  }
  return largest;
}
void PinnedHostPool::copy(void* host, uint64_t device, uint64_t bytes,
                          synStreamHandle stream, bool non_blocking, synDmaDir direction) {
  uint64_t address = reinterpret_cast<uint64_t>(host);
  check_transfer(address, device, bytes, stream);
  std::lock_guard lock(mutex_);
  auto it = blocks_.upper_bound(address);
  if (it == blocks_.begin()) throw std::invalid_argument("PinnedHostPool: invalid host range");
  --it;
  if (!it->second.allocated || address - it->first > it->second.bytes ||
      bytes > it->second.bytes - (address - it->first))
    throw std::invalid_argument("PinnedHostPool: invalid host range");
  auto& streams = it->second.streams;
  if (std::find(streams.begin(), streams.end(), stream) == streams.end()) streams.push_back(stream);
  const uint64_t src = direction == HOST_TO_DRAM ? address : device;
  const uint64_t dst = direction == HOST_TO_DRAM ? device : address;
  check(synMemCopyAsync(stream, src, bytes, dst, direction), "DMA failed");
  if (!non_blocking) check(synStreamSynchronize(stream), "DMA sync failed");
}
void PinnedHostPool::copy_h2d(uint64_t dst, const void* src, uint64_t bytes,
                              synStreamHandle stream, bool non_blocking) {
  copy(const_cast<void*>(src), dst, bytes, stream, non_blocking, HOST_TO_DRAM);
}
void PinnedHostPool::copy_d2h(void* dst, uint64_t src, uint64_t bytes,
                              synStreamHandle stream, bool non_blocking) {
  copy(dst, src, bytes, stream, non_blocking, DRAM_TO_HOST);
}
void PinnedHostPool::copy_d2d(uint64_t dst, uint64_t src, uint64_t bytes,
                              synStreamHandle stream, bool non_blocking) {
  check_transfer(dst, src, bytes, stream);
  check(synMemCopyAsync(stream, src, bytes, dst, DRAM_TO_DRAM), "D2D failed");
  if (!non_blocking) check(synStreamSynchronize(stream), "D2D sync failed");
}
}  // namespace gaudirt::runtime
