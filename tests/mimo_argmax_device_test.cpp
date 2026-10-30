// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "model_argmax.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
float to_float(uint16_t bits) {
  const uint32_t raw = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &raw, sizeof(value));
  return value;
}

int32_t reference(const std::vector<uint16_t>& logits, int32_t status) {
  if (status) return -1;
  int32_t index = 0;
  float best = to_float(logits[0]);
  for (size_t i = 0; i < logits.size(); ++i) {
    float value = to_float(logits[i]);
    if (!std::isfinite(value)) return -1;
    if (value > best) {
      best = value;
      index = static_cast<int32_t>(i);
    }
  }
  return index;
}
}  // namespace

int main() {
  using gaudirt::runtime::DeviceRuntime;
  using gaudirt::runtime::HbmArena;
  DeviceRuntime::Config config{};
  config.module_id = 0;
  DeviceRuntime runtime(config);
  HbmArena arena(runtime.device_id(), 64ULL * 1024 * 1024);
  constexpr size_t kVocab = 152576;
  const uint64_t logits_dev = arena.allocate(kVocab * sizeof(uint16_t));
  const uint64_t status_dev = arena.allocate(sizeof(int32_t));
  const uint64_t token_dev = arena.allocate(sizeof(int32_t));
  gaudirt::mimo::ModelArgmax graph(arena);
  std::vector<uint16_t> logits(kVocab, 0xc000);  // -2
  auto check = [&](int32_t status) {
    const int32_t expected = reference(logits, status);
    arena.copy_h2d_sync(runtime.h2d_stream(), logits.data(), logits_dev, logits.size() * 2);
    arena.copy_h2d_sync(runtime.h2d_stream(), &status, status_dev, sizeof(status));
    graph.execute(logits_dev, status_dev, token_dev, runtime.compute_stream());
    runtime.synchronize_compute();
    int32_t actual = -2;
    arena.copy_d2h_sync(runtime.d2h_stream(), token_dev, &actual, sizeof(actual));
    if (actual != expected) {
      std::fprintf(stderr, "argmax expected %d, got %d\n", expected, actual);
      throw std::runtime_error("MiMo device argmax mismatch");
    }
  };
  for (size_t index : std::array<size_t, 9>{0, 1, 63, 64, 127, 128, 6399, 6400, kVocab - 1}) {
    std::fill(logits.begin(), logits.end(), uint16_t{0xc000});
    logits[index] = 0x4000;  // +2
    check(0);
  }
  std::fill(logits.begin(), logits.end(), uint16_t{0xc000});
  logits[63] = logits[64] = logits[6400] = logits[kVocab - 1] = 0x4000;
  check(0);
  std::fill(logits.begin(), logits.end(), uint16_t{0xc000});
  logits[0] = 0x8000;  // -0
  logits[1] = 0x0000;  // +0, same mathematical value
  check(0);
  uint32_t random = 0x91d3c7a5;
  for (auto& value : logits) {
    random ^= random << 13;
    random ^= random >> 17;
    random ^= random << 5;
    value = static_cast<uint16_t>(random);
    if ((value & 0x7f80) == 0x7f80) value &= 0xfeff;
  }
  check(0);
  check(1);
  for (uint16_t nonfinite : std::array<uint16_t, 4>{0x7f80, 0xff80, 0x7fc0, 0xffc0}) {
    logits[kVocab / 2] = nonfinite;
    check(0);
  }
  std::printf("MiMo device argmax, ties, nonfinite values and status passed.\n");
}
