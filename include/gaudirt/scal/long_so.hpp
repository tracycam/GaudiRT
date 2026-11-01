// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <optional>
#include <vector>

#include "gaudirt/scal/gaudi2_arc_sched_packets.h"

namespace gaudirt::scal {

constexpr uint64_t kLongSoChunkMask = 0x7FFF;
constexpr unsigned kLongSoChunkBits = 15;
constexpr unsigned kLongSoChunks = 4;

inline bool long_so_target_valid(uint64_t target) {
  return target < (1ULL << (kLongSoChunkBits * kLongSoChunks));
}

inline uint16_t long_so_chunk(uint64_t target, unsigned index) {
  return index < kLongSoChunks
             ? static_cast<uint16_t>((target >> (index * kLongSoChunkBits)) &
                                     kLongSoChunkMask)
             : 0;
}

inline uint64_t long_so_target(const std::array<uint16_t, kLongSoChunks>& chunks) {
  uint64_t target = 0;
  for (unsigned i = 0; i < kLongSoChunks; ++i) {
    target |= static_cast<uint64_t>(chunks[i] & kLongSoChunkMask)
              << (i * kLongSoChunkBits);
  }
  return target;
}

inline std::array<uint16_t, kLongSoChunks> long_so_chunks(uint64_t target) {
  std::array<uint16_t, kLongSoChunks> chunks{};
  for (unsigned i = 0; i < kLongSoChunks; ++i) chunks[i] = long_so_chunk(target, i);
  return chunks;
}

inline uint64_t wait_for_ext_signal_target(
    const sched_arc_cmd_wait_for_ext_signal_t& wait) {
  return long_so_target({static_cast<uint16_t>(wait.target_value0),
                         static_cast<uint16_t>(wait.target_value1),
                         static_cast<uint16_t>(wait.target_value2),
                         static_cast<uint16_t>(wait.target_value3)});
}

inline bool set_wait_for_ext_signal_target(
    sched_arc_cmd_wait_for_ext_signal_t& wait, uint64_t target) {
  if (!long_so_target_valid(target)) return false;
  const auto chunks = long_so_chunks(target);
  wait.target_value0 = chunks[0];
  wait.target_value1 = chunks[1];
  wait.target_value2 = chunks[2];
  wait.target_value3 = chunks[3];
  return wait_for_ext_signal_target(wait) == target;
}

inline bool monitor_arm_address(uint32_t addr) {
  for (const uint32_t base : {0xFC100000U, 0xFC300000U, 0xFC500000U,
                              0xFC700000U}) {
    if (addr >= base + 0xE000 && addr < base + 0x10000) return true;
  }
  return false;
}

inline std::optional<unsigned> monitor_arm_chunk(uint32_t addr) {
  if (!monitor_arm_address(addr) || (addr & 3) != 0) return std::nullopt;
  return (addr >> 2) & 3;
}

inline uint32_t monitor_arm_base(uint32_t addr) {
  const auto chunk = monitor_arm_chunk(addr);
  return chunk ? addr - 4 * *chunk : addr;
}

inline uint32_t monitor_arm_data(uint32_t low_arm_data, unsigned chunk,
                                 uint64_t target) {
  const uint32_t preserved = chunk == 0 ? 0x1FFFFU : 0x10000U;
  return (low_arm_data & preserved) |
         (static_cast<uint32_t>(long_so_chunk(target, chunk)) << 17);
}

class LongMonitorState {
 public:
  void observe(uint32_t addr, uint32_t data) {
    const auto chunk = monitor_arm_chunk(addr);
    if (chunk) values_[monitor_arm_base(addr)][*chunk] = data;
  }

  bool rewrite(std::vector<sched_arc_lbw_write_t>& writes, uint64_t target) {
    if (!long_so_target_valid(target)) return false;
    std::vector<sched_arc_lbw_write_t> result;
    result.reserve(writes.size() + 3);
    for (size_t i = 0; i < writes.size();) {
      const auto first_chunk = monitor_arm_chunk(writes[i].addr);
      if (!first_chunk) {
        result.push_back(writes[i++]);
        continue;
      }
      const uint32_t base = monitor_arm_base(writes[i].addr);
      std::array<std::optional<sched_arc_lbw_write_t>, kLongSoChunks> group;
      while (i < writes.size() && monitor_arm_address(writes[i].addr) &&
             monitor_arm_base(writes[i].addr) == base) {
        const auto chunk = monitor_arm_chunk(writes[i].addr);
        if (!chunk || group[*chunk]) return false;
        group[*chunk] = writes[i++];
      }
      if (!group[0]) return false;

      auto state = values_.find(base);
      if (state == values_.end()) state = values_.emplace(base, ArmValues{}).first;
      for (unsigned chunk = kLongSoChunks; chunk-- > 0;) {
        sched_arc_lbw_write_t write = group[chunk].value_or(group[0].value());
        write.addr = base + 4 * chunk;
        write.data = monitor_arm_data(group[0]->data, chunk, target);
        if (chunk == 0 || !state->second[chunk] ||
            *state->second[chunk] != write.data) {
          result.push_back(write);
          const uint32_t data = write.data;
          state->second[chunk] = data;
          if (chunk == 1) ++chunk1_writes_;
          if (chunk != 0) ++high_chunk_writes_;
        }
      }
    }
    writes = std::move(result);
    return true;
  }

  std::optional<uint16_t> target_chunk(uint32_t base, unsigned chunk) const {
    const auto state = values_.find(base);
    if (chunk >= kLongSoChunks || state == values_.end() ||
        !state->second[chunk]) {
      return std::nullopt;
    }
    return (*state->second[chunk] >> 17) & kLongSoChunkMask;
  }

  uint64_t chunk1_writes() const { return chunk1_writes_; }
  uint64_t high_chunk_writes() const { return high_chunk_writes_; }

 private:
  using ArmValues = std::array<std::optional<uint32_t>, kLongSoChunks>;
  std::map<uint32_t, ArmValues> values_;
  uint64_t chunk1_writes_ = 0;
  uint64_t high_chunk_writes_ = 0;
};

}  // namespace gaudirt::scal
