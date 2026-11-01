// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "step_metadata.hpp"

namespace gaudirt::mimo {

// Shared by the 70 same-position graphs. Every Synapse tensor begins on a
// separate aligned boundary; Full Attention gathers up to the final row and
// applies an additional per-row causal mask inside its softmax.
template <uint32_t Rows>
struct alignas(256) PrefillMetadataT {
  static_assert(Rows == 32 || Rows == 512 || Rows == 1024);
  static constexpr uint32_t kRows = Rows;
  alignas(256) std::array<uint16_t, kRows * 64> full_cos{};
  alignas(256) std::array<uint16_t, kRows * 64> full_sin{};
  alignas(256) std::array<uint16_t, kRows * 64> swa_cos{};
  alignas(256) std::array<uint16_t, kRows * 64> swa_sin{};
  alignas(256) std::array<int32_t, kRows> positions{};
  alignas(256) std::array<int32_t, kRows> slots{};
  alignas(256) std::array<int32_t, 64> full_pages{};
  alignas(256) std::array<int32_t, 64> full_usage{};
  alignas(256) std::array<int32_t, 64> full_groups{};
  alignas(256) std::array<int32_t, kRows * 2> window_pages{};
  alignas(256) std::array<int32_t, kRows * 2> window_groups{};
};
using PrefillMetadata = PrefillMetadataT<32>;
using LargePrefillMetadata = PrefillMetadataT<512>;
using XLargePrefillMetadata = PrefillMetadataT<1024>;
static_assert(std::is_trivially_copyable_v<PrefillMetadata>);
static_assert(std::is_trivially_copyable_v<LargePrefillMetadata>);
static_assert(std::is_trivially_copyable_v<XLargePrefillMetadata>);

template <uint32_t Rows>
inline void fill_prefill_metadata(PrefillMetadataT<Rows>& result, uint32_t first_position,
                                  uint32_t rope_rows, const uint16_t* full_rope,
                                  const uint16_t* swa_rope,
                                  const std::array<int32_t, 64>& pages,
                                  uint32_t cache_slots) {
  if (first_position % 32 || first_position > UINT32_MAX - Rows + 1)
    throw std::invalid_argument("MiMo prefill: invalid chunk position");
  const uint32_t last = first_position + Rows - 1;
  validate_step_metadata(last, 0, rope_rows, full_rope, swa_rope, pages, cache_slots);
  StepMetadata step{};
  for (uint32_t row = 0; row < Rows; ++row) {
    fill_step_metadata_prevalidated(step, first_position + row, 0,
                                    full_rope, swa_rope, pages);
    std::copy(step.full_cos.begin(), step.full_cos.end(), result.full_cos.begin() + row * 64);
    std::copy(step.full_sin.begin(), step.full_sin.end(), result.full_sin.begin() + row * 64);
    std::copy(step.swa_cos.begin(), step.swa_cos.end(), result.swa_cos.begin() + row * 64);
    std::copy(step.swa_sin.begin(), step.swa_sin.end(), result.swa_sin.begin() + row * 64);
    result.positions[row] = step.position;
    result.slots[row] = step.slot;
    for (unsigned page = 0; page < 2; ++page) {
      result.window_pages[row * 2 + page] = step.window_pages[page];
      result.window_groups[row * 2 + page] = step.window_groups[page] == 0 ? row : -1;
    }
  }
  result.full_pages = step.full_pages;
  result.full_usage = step.full_usage;
  result.full_groups = step.full_groups;
}

}  // namespace gaudirt::mimo
