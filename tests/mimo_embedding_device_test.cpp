// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "model_embedding.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main() {
  using gaudirt::runtime::DeviceRuntime;
  using gaudirt::runtime::HbmArena;
  DeviceRuntime::Config config{};
  config.module_id = 0;
  config.rank = 0;
  config.world_size = 1;
  DeviceRuntime runtime(config);
  HbmArena arena(runtime.device_id(), 2ULL * 1024 * 1024 * 1024);
  constexpr uint64_t kHidden = 6144;
  constexpr uint64_t kVocab = 152576;
  constexpr uint64_t kRowBytes = kHidden * sizeof(uint16_t);
  const uint64_t table = arena.allocate(kVocab * kRowBytes);
  const uint64_t token = arena.allocate(sizeof(int32_t));
  const uint64_t embedding = arena.allocate(kRowBytes);
  const uint64_t status = arena.allocate(sizeof(int32_t));
  gaudirt::mimo::ModelEmbedding graph(arena, table);
  constexpr std::array<int32_t, 3> kRows{0, 76311, 152575};
  std::vector<uint16_t> expected(kHidden), actual(kHidden);
  for (const int32_t row : kRows) {
    for (uint64_t index = 0; index < kHidden; ++index) {
      expected[index] = static_cast<uint16_t>(0x3f00U + ((index + row) % 256));
    }
    arena.copy_h2d_sync(runtime.h2d_stream(), expected.data(), table + row * kRowBytes, kRowBytes);
    arena.copy_h2d_sync(runtime.h2d_stream(), &row, token, sizeof(row));
    graph.execute(token, embedding, status, runtime.compute_stream());
    runtime.synchronize_compute();
    int32_t result = -1;
    arena.copy_d2h_sync(runtime.d2h_stream(), status, &result, sizeof(result));
    arena.copy_d2h_sync(runtime.d2h_stream(), embedding, actual.data(), kRowBytes);
    if (result != 0 || actual != expected) throw std::runtime_error("valid embedding lookup mismatch");
  }
  for (const int32_t row : std::array<int32_t, 2>{-1, static_cast<int32_t>(kVocab)}) {
    arena.copy_h2d_sync(runtime.h2d_stream(), &row, token, sizeof(row));
    graph.execute(token, embedding, status, runtime.compute_stream());
    runtime.synchronize_compute();
    int32_t result = 0;
    arena.copy_d2h_sync(runtime.d2h_stream(), status, &result, sizeof(result));
    if (result != 1) throw std::runtime_error("invalid token was not reported");
  }
  std::printf("MiMo device embedding and invalid-token status passed.\n");
}
