// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/block_caching_allocator.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/pinned_host_pool.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using namespace gaudirt::runtime;

static void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

int main() {
  try {
    DeviceRuntime::Config cfg{};
    cfg.uclamp_min = 0;
    DeviceRuntime rt(cfg);
    constexpr uint64_t slab_bytes = 4 * 1024 * 1024;
    BlockCachingAllocator pool(rt.device_id(), slab_bytes);
    auto s = rt.compute_stream();
    auto a = pool.allocate(1, s);
    auto b = pool.allocate(257, s);
    auto c = pool.allocate(9, s);
    require(a % 256 == 0 && b % 256 == 0 && c % 256 == 0, "unaligned allocation");
    require(pool.memory_allocated() == 1024, "allocation accounting");
    require(pool.contains_live_range(b + 1, 511), "live range rejected");
    pool.record_stream(b + 1, s);
    require(!pool.contains_live_range(b + 1, 512), "out-of-range allocation accepted");
    pool.free(b);
    pool.free(a);
    auto merged = pool.allocate(768, s);
    require(merged == a, "adjacent free blocks did not merge");
    pool.free(merged);
    pool.free(c);
    auto entire = pool.allocate(slab_bytes, s);
    require(entire == a, "full slab did not coalesce");
    pool.free(entire);

    std::mt19937 rng(12345);
    std::vector<uint64_t> live;
    for (int i = 0; i < 2000; ++i) {
      if (!live.empty() && rng() % 2) {
        auto index = rng() % live.size();
        pool.free(live[index]);
        live.erase(live.begin() + index);
      } else {
        live.push_back(pool.allocate(1 + rng() % 2048, s));
      }
    }
    for (auto ptr : live) pool.free(ptr);
    entire = pool.allocate(slab_bytes, s);
    pool.free(entire);

    {
      GraphPool graph(pool, 1024 * 1024, s);
      auto ga = graph.allocator().allocate(513, s);
      auto gb = graph.allocator().allocate(256, s);
      require(pool.memory_allocated() == 1024 * 1024, "graph reservation accounting");
      graph.allocator().free(ga);
      graph.allocator().free(gb);
      auto whole_graph = graph.allocator().allocate(1024 * 1024, s);
      graph.allocator().free(whole_graph);
      graph.allocator().empty_cache();
      auto first = graph.allocate(800 * 1024, s);
      auto second = graph.allocate(700 * 1024, s);
      require(first % 256 == 0 && second % 256 == 0, "graph growth alignment");
      require(graph.capacity_bytes() == 2 * 1024 * 1024, "graph pool did not grow");
      require(graph.memory_allocated() == 1500 * 1024, "graph growth accounting");
      graph.free(first);
      graph.free(second);
      auto reused_graph = graph.allocate(1024 * 1024, s);
      require(graph.capacity_bytes() == 2 * 1024 * 1024, "graph pool failed to reuse a region");
      graph.free(reused_graph);
      require(graph.memory_allocated() == 0, "graph allocation leak");
    }
    entire = pool.allocate(slab_bytes, s);
    pool.free(entire);

    constexpr size_t bytes = 64 * 1024;
    PinnedHostPool host(rt.device_id(), 2 * bytes);
    void* input = host.allocate(bytes);
    void* output = host.allocate(bytes);
    auto* in = static_cast<uint8_t*>(input);
    auto* out = static_cast<uint8_t*>(output);
    for (size_t i = 0; i < bytes; ++i) in[i] = static_cast<uint8_t>(i * 13 + 7);
    std::fill(out, out + bytes, 0);
    auto src = pool.allocate(bytes, rt.h2d_stream());
    auto dst = pool.allocate(bytes, s);
    StreamEvent h2d_done(rt.device_id()), d2d_done(rt.device_id());
    host.copy_h2d(src, in, bytes, rt.h2d_stream(), true);
    h2d_done.record(rt.h2d_stream());
    h2d_done.wait(s);
    host.copy_d2d(dst, src, bytes, s, true);
    pool.record_stream(src, s);
    d2d_done.record(s);
    d2d_done.wait(rt.d2h_stream());
    host.copy_d2h(out, dst, bytes, rt.d2h_stream(), true);
    pool.record_stream(dst, rt.d2h_stream());
    pool.free(src);
    pool.free(dst);
    auto reused = pool.allocate(2 * bytes, s);
    require(reused == src, "cross-stream blocks not reused");
    // The allocator's wait on s must protect both source reads and D2H.
    host.copy_h2d(reused, in, bytes, s, true);
    host.free(output);
    output = host.allocate(bytes);
    require(output == out, "pinned block not reused");
    require(synStreamSynchronize(s) == synSuccess, "reuse sync");
    require(std::equal(in, in + bytes, out), "cross-stream copy mismatch");
    pool.free(reused);
    auto sync_device = pool.allocate(bytes, s);
    host.copy_h2d(sync_device, in, bytes, s, false);
    std::fill(out, out + bytes, 0);
    host.copy_d2h(out, sync_device, bytes, s, false);
    require(std::equal(in, in + bytes, out), "synchronous copy mismatch");
    pool.free(sync_device);
    host.free(input);
    host.free(output);
    require(pool.memory_allocated() == 0, "allocation leak");
    require(pool.max_memory_allocated() == slab_bytes, "peak accounting");
    pool.empty_cache();
    auto reacquired = pool.allocate(256, s);
    require(reacquired % 256 == 0, "slab reacquisition");
    pool.free(reacquired);
    pool.empty_cache();
    std::cout << "block allocator and cross-stream DMA passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
