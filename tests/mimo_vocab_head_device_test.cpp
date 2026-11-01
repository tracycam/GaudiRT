// SPDX-License-Identifier: Apache-2.0
// stdin supplies the BF16 final norm followed by the contiguous BF16 lm_head.
#include "gaudirt/common/synapse_raii.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "model_head.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
constexpr uint64_t kHidden = 6144;
constexpr uint64_t kVocab = gaudirt::mimo::ModelHead::kVocabSize;
constexpr uint64_t kShard = gaudirt::mimo::ModelHead::kLocalVocabSize;

void read_exact(void* data, size_t size) {
  auto* next = static_cast<uint8_t*>(data);
  while (size != 0) {
    const size_t bytes = std::fread(next, 1, size, stdin);
    if (bytes == 0) throw std::runtime_error("model head input ended early");
    next += bytes;
    size -= bytes;
  }
}

uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const auto mid = values.size() / 2;
  return (values[mid - 1] + values[mid]) / 2;
}
}  // namespace

int main() {
  using gaudirt::common::TimedEventPair;
  using gaudirt::mimo::ModelHead;
  using gaudirt::runtime::DeviceRuntime;
  using gaudirt::runtime::HbmArena;

  DeviceRuntime::Config config{};
  config.module_id = 0;
  DeviceRuntime runtime(config);
  HbmArena arena(runtime.device_id(), 2ULL * 1024 * 1024 * 1024);
  const uint64_t weight_bytes = kVocab * kHidden * sizeof(uint16_t);
  const uint64_t weight = arena.allocate(weight_bytes);
  const uint64_t gamma = arena.allocate(kHidden * sizeof(uint16_t));
  const uint64_t gathered = arena.allocate(8 * kHidden * sizeof(float));
  const uint64_t residual = arena.allocate(kHidden * sizeof(uint16_t));
  const uint64_t full_normal = arena.allocate(kHidden * sizeof(uint16_t));
  const uint64_t part_normal = arena.allocate(kHidden * sizeof(uint16_t));
  const uint64_t full_logits = arena.allocate(kVocab * sizeof(uint16_t));
  const uint64_t part_logits = arena.allocate(kVocab * sizeof(uint16_t));

  std::vector<uint16_t> host_gamma(kHidden), host_residual(kHidden);
  std::vector<float> host_gathered(8 * kHidden);
  read_exact(host_gamma.data(), host_gamma.size() * sizeof(uint16_t));
  arena.copy_h2d_sync(runtime.h2d_stream(), host_gamma.data(), gamma,
                      host_gamma.size() * sizeof(uint16_t));
  std::vector<uint8_t> block(16 * 1024 * 1024);
  for (uint64_t offset = 0; offset < weight_bytes; offset += block.size()) {
    const size_t bytes = std::min<uint64_t>(block.size(), weight_bytes - offset);
    read_exact(block.data(), bytes);
    arena.copy_h2d_sync(runtime.h2d_stream(), block.data(), weight + offset, bytes);
  }
  for (size_t i = 0; i < kHidden; ++i) {
    host_residual[i] = bf16(static_cast<float>(static_cast<int>(i % 97) - 48) / 71);
    for (size_t rank = 0; rank < 8; ++rank) {
      const int value = static_cast<int>((i * 13 + rank * 19) % 113) - 56;
      host_gathered[rank * kHidden + i] = static_cast<float>(value) / (31 + rank * 7);
    }
  }
  arena.copy_h2d_sync(runtime.h2d_stream(), host_residual.data(), residual,
                      host_residual.size() * sizeof(uint16_t));
  arena.copy_h2d_sync(runtime.h2d_stream(), host_gathered.data(), gathered,
                      host_gathered.size() * sizeof(float));

  const auto stream = runtime.compute_stream();
  TimedEventPair timer(runtime.device_id());
  std::vector<double> full_us, shard_us;
  {
    std::fprintf(stderr, "compiling full vocabulary head\n");
    ModelHead full(arena, gamma, weight, nullptr, true, kVocab);
    auto launch = full.prepare(gathered, residual, full_normal, full_logits);
    for (int i = 0; i < 4; ++i) launch.execute(stream);
    runtime.synchronize_compute();
    for (int i = 0; i < 32; ++i) {
      timer.record_start(stream);
      launch.execute(stream);
      timer.record_end(stream);
      full_us.push_back(timer.elapsed_us());
    }
  }
  {
    std::fprintf(stderr, "compiling sliced vocabulary head\n");
    ModelHead shard(arena, gamma, weight, nullptr, true, kShard);
    auto launch = shard.prepare(gathered, residual, part_normal, part_logits);
    auto* shard_weight = static_cast<synLaunchTensorInfo*>(nullptr);
    auto* shard_output = static_cast<synLaunchTensorInfo*>(nullptr);
    for (auto& tensor : launch.tensors) {
      if (std::strcmp(tensor.tensorName, "model_lm_head_weight") == 0) shard_weight = &tensor;
      if (std::strcmp(tensor.tensorName, "model_logits") == 0) shard_output = &tensor;
    }
    if (!shard_weight || !shard_output) throw std::runtime_error("missing head launch tensor");
    for (uint64_t rank = 0; rank < 8; ++rank) {
      shard_weight->pTensorAddress = weight + rank * kShard * kHidden * sizeof(uint16_t);
      shard_output->pTensorAddress = part_logits + rank * kShard * sizeof(uint16_t);
      launch.execute(stream);
    }
    runtime.synchronize_compute();
    shard_weight->pTensorAddress = weight;
    shard_output->pTensorAddress = part_logits;
    for (int i = 0; i < 4; ++i) launch.execute(stream);
    runtime.synchronize_compute();
    for (int i = 0; i < 32; ++i) {
      timer.record_start(stream);
      launch.execute(stream);
      timer.record_end(stream);
      shard_us.push_back(timer.elapsed_us());
    }
  }
  std::vector<uint16_t> expected(kVocab), actual(kVocab);
  std::vector<uint16_t> expected_norm(kHidden), actual_norm(kHidden);
  arena.copy_d2h_sync(runtime.d2h_stream(), full_logits, expected.data(), kVocab * sizeof(uint16_t));
  arena.copy_d2h_sync(runtime.d2h_stream(), part_logits, actual.data(), kVocab * sizeof(uint16_t));
  arena.copy_d2h_sync(runtime.d2h_stream(), full_normal, expected_norm.data(), kHidden * sizeof(uint16_t));
  arena.copy_d2h_sync(runtime.d2h_stream(), part_normal, actual_norm.data(), kHidden * sizeof(uint16_t));
  uint64_t mismatches = 0, nonfinite = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    mismatches += expected[i] != actual[i];
    nonfinite += (expected[i] & 0x7f80) == 0x7f80 || (actual[i] & 0x7f80) == 0x7f80;
  }
  const bool norm_equal = expected_norm == actual_norm;
  std::printf("{\"same_logits_bitwise\":%s,\"same_normalized_bitwise\":%s,"
              "\"mismatches\":%llu,\"nonfinite\":%llu,\"full_p50_us\":%.3f,"
              "\"shard_p50_us\":%.3f,\"samples\":32}\n",
              mismatches == 0 ? "true" : "false", norm_equal ? "true" : "false",
              static_cast<unsigned long long>(mismatches),
              static_cast<unsigned long long>(nonfinite), median(full_us), median(shard_us));
  return mismatches == 0 && nonfinite == 0 && norm_equal ? 0 : 1;
}
