// Exercise allocator fence ordering without requiring a device.
#include "gaudirt/runtime/block_caching_allocator.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <map>
#include <vector>

namespace {
struct RecordedEvent { synStreamHandle stream = nullptr; uint64_t ordinal = 0; bool needs_wait = false; };
std::map<synStreamHandle, uint64_t> recorded;
std::vector<RecordedEvent> waited;
std::vector<synStreamHandle> synchronized;
uint64_t live_events = 0;
uint64_t created_events = 0;
auto stream(unsigned id) { return reinterpret_cast<synStreamHandle>(uintptr_t{id}); }
void reset() { assert(live_events == 0); created_events = 0; recorded.clear(); waited.clear(); synchronized.clear(); }
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
  auto& event = *static_cast<RecordedEvent*>(p);
  assert(!event.needs_wait && "recorded event was reused without a wait");
  event = {s, ++recorded[s], true};
  return synSuccess;
}
synStatus synEventSynchronize(synEventHandle p) {
  static_cast<RecordedEvent*>(p)->needs_wait = false;
  return synSuccess;
}
synStatus synStreamWaitEvent(synStreamHandle, synEventHandle p, uint32_t) {
  auto& event = *static_cast<RecordedEvent*>(p);
  waited.push_back(event);
  event.needs_wait = false;
  return synSuccess;
}
synStatus synStreamSynchronize(synStreamHandle s) { synchronized.push_back(s); return synSuccess; }
}

int main() {
  using gaudirt::runtime::BlockCachingAllocator;
  {
    BlockCachingAllocator pool(0, 4096);
    for (unsigned i = 0; i < 1000; ++i) {
      const auto p = pool.allocate(256, stream(1));
      pool.free(p);
    }
    assert(created_events == 1 && recorded.empty() && waited.empty());
  }
  reset();
  for (bool reverse : {false, true}) {
    {
      BlockCachingAllocator pool(0, 4096);
      const auto left = pool.allocate(256, stream(1));
      const auto right = pool.allocate(256, stream(1));
      pool.free(reverse ? right : left);
      pool.free(reverse ? left : right);
      const auto merged = pool.allocate(512, stream(2));
      assert(merged == left && waited.size() == 1);
      assert(waited[0].stream == stream(1) && waited[0].ordinal == 1);
      pool.free(merged);
    }
    reset();
  }
  {
    BlockCachingAllocator pool(0, 4096);
    const auto a = pool.allocate(256, stream(1));
    const auto gap = pool.allocate(256, stream(1));
    const auto b = pool.allocate(256, stream(1));
    const auto guard = pool.allocate(256, stream(1));
    pool.free(a);
    pool.free(b);
    assert(created_events == 1);
    const auto first = pool.allocate(256, stream(2));
    assert(first == a && waited.size() == 1 && waited[0].ordinal == 1);
    const auto second = pool.allocate(256, stream(3));
    assert(second == b && waited.size() == 2 && waited[1].ordinal == 2);
    assert(waited[0].stream == stream(1) && waited[1].stream == stream(1));
    pool.free(gap);
    pool.free(first);
    pool.free(second);
    pool.free(guard);
  }
  reset();
  {
    BlockCachingAllocator pool(0, 4096);
    auto p = pool.allocate(256, stream(1));
    pool.record_stream(p, stream(2));
    pool.free(p);
    p = pool.allocate(256, stream(1));
    assert(waited.size() == 1 && waited[0].stream == stream(2));
    pool.record_stream(p, stream(2));
    pool.free(p);
    waited.clear();
    p = pool.allocate(256, stream(3));
    assert(waited.size() == 2);
    assert(waited[0].stream == stream(1) && waited[1].stream == stream(2));
    assert(waited[1].ordinal == 2);
    pool.free(p);
  }
  reset();
  {
    BlockCachingAllocator pool(0, 4096);
    auto p = pool.allocate(256, stream(1));
    pool.free(p);
    pool.empty_cache();
    assert(synchronized.size() == 1 && synchronized[0] == stream(1));
    assert(created_events == 1 && recorded.empty());
    p = pool.allocate(256, stream(1));
    pool.free(p);
  }
  assert(live_events == 0);
  std::cout << "Same-stream reuse has no event records; cross-stream dependencies survive split and merge\n";
}
