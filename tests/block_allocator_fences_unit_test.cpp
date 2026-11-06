// Exercise allocator fence ordering without requiring a device.
#include "gaudirt/runtime/block_caching_allocator.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

namespace {
struct RecordedEvent { synStreamHandle stream = nullptr; uint64_t ordinal = 0; };
std::map<synStreamHandle, uint64_t> recorded;
std::vector<RecordedEvent> waited;
uint64_t live_events = 0;
uint64_t created_events = 0;
auto stream(unsigned id) { return reinterpret_cast<synStreamHandle>(uintptr_t{id}); }
void reset() { assert(live_events == 0); created_events = 0; recorded.clear(); waited.clear(); }
}

extern "C" {
synStatus synDeviceGetMemoryInfo(synDeviceId, uint64_t* free, uint64_t* total) {
  *free = *total = 4096;
  return synSuccess;
}
synStatus synDeviceMalloc(synDeviceId, uint64_t, uint64_t, uint32_t, uint64_t* p) {
  *p = 4096;
  return synSuccess;
}
synStatus synDeviceFree(synDeviceId, uint64_t, uint32_t) { return synSuccess; }
synStatus synEventCreate(synEventHandle* p, synDeviceId, uint32_t) {
  *p = new RecordedEvent;
  ++live_events;
  ++created_events;
  return synSuccess;
}
synStatus synEventDestroy(synEventHandle p) {
  delete static_cast<RecordedEvent*>(p);
  --live_events;
  return synSuccess;
}
synStatus synEventRecord(synEventHandle p, synStreamHandle s) {
  *static_cast<RecordedEvent*>(p) = {s, ++recorded[s]};
  return synSuccess;
}
synStatus synEventSynchronize(synEventHandle) { return synSuccess; }
synStatus synStreamWaitEvent(synStreamHandle, synEventHandle p, uint32_t) {
  waited.push_back(*static_cast<RecordedEvent*>(p));
  return synSuccess;
}
synStatus synStreamSynchronize(synStreamHandle) { return synSuccess; }
}

int main() {
  using gaudirt::runtime::BlockCachingAllocator;
  {
    BlockCachingAllocator pool(0, 4096);
    for (uint64_t i = 0; i < 1000; ++i) {
      const auto p = pool.allocate(256, stream(1));
      pool.free(p);
    }
    // A producer stream uses one persistent event, including when a split
    // remainder and the freed block both retain its most recent record.
    assert(waited.size() == 999 && live_events == 1 && created_events == 1);
    for (uint64_t i = 0; i < waited.size(); ++i)
      assert(waited[i].stream == stream(1) && waited[i].ordinal == i + 1);
  }
  reset();
  for (bool reverse : {false, true}) {
    reset();
    BlockCachingAllocator pool(0, 4096);
    const auto left = pool.allocate(256, stream(1));
    const auto right = pool.allocate(256, stream(1));
    pool.free(reverse ? right : left);
    pool.free(reverse ? left : right);
    waited.clear();
    const auto merged = pool.allocate(512, stream(2));
    assert(waited.size() == 1 && waited[0].stream == stream(1) && waited[0].ordinal == 2);
    pool.free(merged);
  }
  reset();
  {
    BlockCachingAllocator pool(0, 4096);
    const auto a = pool.allocate(256, stream(1));
    const auto gap = pool.allocate(256, stream(1));
    const auto b = pool.allocate(256, stream(1));
    const auto guard = pool.allocate(256, stream(1));
    pool.free(a);
    pool.free(b);
    assert(created_events == 1);
    waited.clear();
    const auto reused = pool.allocate(256, stream(2));
    assert(reused == a && waited.size() == 1 && waited[0].stream == stream(1));
    assert(waited[0].ordinal == 2);
    pool.free(gap);  // Record the same producer event again.
    assert(waited[0].ordinal == 2);  // A submitted wait captured its own state.
    pool.free(reused);
    pool.free(guard);
  }
  reset();
  {
    BlockCachingAllocator pool(0, 4096);
    auto p = pool.allocate(256, stream(1));
    pool.record_stream(p, stream(2));
    pool.free(p);
    p = pool.allocate(256, stream(1));
    pool.record_stream(p, stream(2));
    pool.free(p);
    waited.clear();
    p = pool.allocate(256, stream(3));
    assert(waited.size() == 2);
    std::map<synStreamHandle, uint64_t> latest;
    for (const auto& event : waited) latest[event.stream] = event.ordinal;
    assert(latest[stream(1)] == 2 && latest[stream(2)] == 2);
    pool.free(p);
  }
  assert(live_events == 0);
  std::cout << "Allocator retains the latest fence per stream across split and merge\n";
}
