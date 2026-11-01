// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace gaudirt::mimo {

// Each launch tensor starts at a 256-byte-aligned offset. The complete object
// is copied to the device once per step and shared by all isomorphic layers.
struct alignas(256) StepMetadata {
  alignas(256) std::array<uint16_t, 64> full_cos{};
  alignas(256) std::array<uint16_t, 64> full_sin{};
  alignas(256) std::array<uint16_t, 64> swa_cos{};
  alignas(256) std::array<uint16_t, 64> swa_sin{};
  alignas(256) int32_t position = 0;
  alignas(256) int32_t slot = 0;
  alignas(256) std::array<int32_t, 64> full_pages{};
  alignas(256) std::array<int32_t, 64> full_usage{};
  alignas(256) std::array<int32_t, 64> full_groups{};
  alignas(256) std::array<int32_t, 2> window_pages{};
  alignas(256) std::array<int32_t, 2> window_groups{};
};

static_assert(std::is_trivially_copyable_v<StepMetadata>);
static_assert(sizeof(StepMetadata) <= 4096);

// The scheduler may add pages, but already-written logical pages cannot move.
inline void validate_page_update(const std::array<int32_t, 64>& previous,
                                 const std::array<int32_t, 64>& next,
                                 uint32_t written_tokens, uint32_t cache_slots) {
  if (written_tokens > 8192 || cache_slots == 0 || cache_slots % 128 != 0)
    throw std::invalid_argument("MiMo pages: invalid geometry");
  const uint32_t written_pages = (written_tokens + 127) / 128;
  for (size_t i = 0; i < next.size(); ++i) {
    if (next[i] < 0 || static_cast<uint32_t>(next[i]) >= cache_slots / 128 ||
        std::find(next.begin(), next.begin() + i, next[i]) != next.begin() + i ||
        (i < written_pages && next[i] != previous[i]))
      throw std::invalid_argument("MiMo pages: invalid, aliased, or moved page");
  }
}

// Validate the complete page range once when the fixed mapping is registered.
inline void validate_step_metadata(uint32_t position, uint32_t first_position,
                                   uint32_t rope_rows, const uint16_t* full_rope,
                                   const uint16_t* swa_rope,
                                   const std::array<int32_t, 64>& pages,
                                   uint32_t cache_slots) {
  if (!full_rope || !swa_rope || rope_rows == 0 || position < first_position ||
      position - first_position >= rope_rows || position >= 64 * 128 ||
      cache_slots == 0 || cache_slots > INT32_MAX || cache_slots % 128 != 0) {
    throw std::invalid_argument("MiMo step: invalid position, RoPE, or cache geometry");
  }
  const uint32_t current_page = position / 128;
  const uint32_t physical_pages = cache_slots / 128;
  for (uint32_t page = 0; page <= current_page; ++page) {
    if (pages[page] < 0 || static_cast<uint32_t>(pages[page]) >= physical_pages ||
        std::find(pages.begin(), pages.begin() + page, pages[page]) != pages.begin() + page) {
      throw std::invalid_argument("MiMo step: invalid or duplicate active physical page");
    }
  }
}

// RoPE rows contain 32 cosine and 32 sine BF16 values. The caller has already
// validated the fixed tables and the page mapping through the last position.
inline void fill_step_metadata_prevalidated(StepMetadata& result, uint32_t position,
                                            uint32_t first_position,
                                            const uint16_t* full_rope,
                                            const uint16_t* swa_rope,
                                            const std::array<int32_t, 64>& pages) {
  const uint32_t current_page = position / 128;
  const size_t row = static_cast<size_t>(position - first_position) * 64;
  for (size_t i = 0; i < 64; ++i) {
    result.full_cos[i] = full_rope[row + i % 32];
    result.full_sin[i] = full_rope[row + 32 + i % 32];
    result.swa_cos[i] = swa_rope[row + i % 32];
    result.swa_sin[i] = swa_rope[row + 32 + i % 32];
  }
  result.position = static_cast<int32_t>(position);
  result.slot = pages[current_page] * 128 + static_cast<int32_t>(position % 128);
  result.full_pages = pages;
  for (uint32_t page = 0; page < 64; ++page) {
    const int64_t count = static_cast<int64_t>(position) + 1 - static_cast<int64_t>(page) * 128;
    result.full_usage[page] = static_cast<int32_t>(std::clamp<int64_t>(count, 1, 128));
    result.full_groups[page] = count > 0 ? 0 : -1;
  }
  const uint32_t window_first = (position >= 127 ? position - 127 : 0) / 128;
  result.window_pages[0] = pages[window_first];
  result.window_groups[0] = 0;
  const bool second = window_first < current_page;
  result.window_pages[1] = second ? pages[window_first + 1] : -1;
  result.window_groups[1] = second ? 0 : -1;
}

inline void fill_step_metadata(StepMetadata& result, uint32_t position,
                               uint32_t first_position, uint32_t rope_rows,
                               const uint16_t* full_rope, const uint16_t* swa_rope,
                               const std::array<int32_t, 64>& pages,
                               uint32_t cache_slots) {
  validate_step_metadata(position, first_position, rope_rows, full_rope, swa_rope,
                         pages, cache_slots);
  fill_step_metadata_prevalidated(result, position, first_position, full_rope,
                                   swa_rope, pages);
}

}  // namespace gaudirt::mimo
