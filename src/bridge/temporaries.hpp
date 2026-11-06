// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "context.hpp"

namespace gaudirt::generic {
using runtime::GraphPool;

struct TemporaryDevice {
  TemporaryDevice(Context& ctx, uint64_t bytes)
      : ctx(ctx), address(ctx.allocator.allocate(bytes, ctx.stream())) {}
  ~TemporaryDevice() { ctx.allocator.free(address); }
  Context& ctx;
  uint64_t address;
};
struct TemporaryHost {
  TemporaryHost(Context& ctx, uint64_t bytes) : ctx(ctx), address(ctx.pinned.allocate(bytes)) {}
  ~TemporaryHost() { ctx.pinned.free(address); }
  Context& ctx;
  void* address;
};

struct GraphTemporary {
  GraphTemporary(GraphPool& pool, synStreamHandle stream, uint64_t bytes)
      : pool(pool), address(pool.allocate(bytes, stream)) {}
  ~GraphTemporary() { pool.free(address); }
  GraphPool& pool;
  uint64_t address;
};

}  // namespace gaudirt::generic
