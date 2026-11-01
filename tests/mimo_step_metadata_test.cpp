// SPDX-License-Identifier: Apache-2.0
#include "step_metadata.hpp"
#include "prefill_metadata.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>

namespace {
void require(bool value) {
  if (!value) throw std::runtime_error("MiMo metadata contract failed");
}
}  // namespace

int main() {
  using gaudirt::mimo::StepMetadata;
  using gaudirt::mimo::fill_step_metadata;
  using gaudirt::mimo::fill_step_metadata_prevalidated;
  using gaudirt::mimo::validate_step_metadata;
  constexpr uint32_t kRows = 144;
  std::array<uint16_t, kRows * 64> full{}, swa{};
  for (size_t i = 0; i < full.size(); ++i) {
    full[i] = static_cast<uint16_t>(i);
    swa[i] = static_cast<uint16_t>(i + 1);
  }
  std::array<int32_t, 64> pages{};
  for (size_t i = 0; i < pages.size(); ++i) pages[i] = static_cast<int32_t>(i + 34);
  StepMetadata step{};
  StepMetadata prevalidated{};
  validate_step_metadata(4096 + kRows - 1, 4096, kRows, full.data(), swa.data(), pages, 205184);
  for (uint32_t i = 0; i < kRows; ++i) {
    fill_step_metadata(step, 4096 + i, 4096, kRows, full.data(), swa.data(), pages, 205184);
    fill_step_metadata_prevalidated(prevalidated, 4096 + i, 4096, full.data(), swa.data(), pages);
    require(prevalidated.position == step.position && prevalidated.slot == step.slot &&
            prevalidated.full_cos == step.full_cos && prevalidated.full_sin == step.full_sin &&
            prevalidated.swa_cos == step.swa_cos && prevalidated.swa_sin == step.swa_sin &&
            prevalidated.full_pages == step.full_pages && prevalidated.full_usage == step.full_usage &&
            prevalidated.full_groups == step.full_groups && prevalidated.window_pages == step.window_pages &&
            prevalidated.window_groups == step.window_groups);
    require(step.position == static_cast<int32_t>(4096 + i));
    const bool next_page = i >= 128;
    const int32_t expected_slot = next_page
        ? 67 * 128 + static_cast<int32_t>(i - 128)
        : 66 * 128 + static_cast<int32_t>(i);
    require(step.slot == expected_slot);
    require(step.full_usage[31] == 128 &&
            step.full_usage[32] == static_cast<int32_t>(next_page ? 128 : i + 1));
    require(step.full_usage[33] == static_cast<int32_t>(next_page ? i - 127 : 1) &&
            step.full_groups[32] == 0 && step.full_groups[33] == (next_page ? 0 : -1));
    require(step.full_cos[0] == full[i * 64] && step.full_cos[32] == full[i * 64]);
    require(step.full_sin[63] == full[i * 64 + 63]);
    require(step.swa_cos[31] == swa[i * 64 + 31]);
    require(step.swa_sin[32] == swa[i * 64 + 32]);
    if (i < 127) {
      require(step.window_pages[0] == 65 && step.window_pages[1] == 66);
      require(step.window_groups[0] == 0 && step.window_groups[1] == 0);
    } else if (i == 127) {
      require(step.window_pages[0] == 66 && step.window_groups[0] == 0);
      require(step.window_pages[1] == -1 && step.window_groups[1] == -1);
    } else {
      require(step.window_pages[0] == 66 && step.window_pages[1] == 67);
      require(step.window_groups[0] == 0 && step.window_groups[1] == 0);
    }
  }
  pages[32] = pages[31];
  bool rejected = false;
  try {
    fill_step_metadata(step, 4096, 4096, kRows, full.data(), swa.data(), pages, 205184);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected);
  pages[32] = 66;
  pages[33] = pages[32];
  rejected = false;
  try {
    validate_step_metadata(4096 + kRows - 1, 4096, kRows, full.data(), swa.data(), pages, 205184);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected);
  pages[33] = 67;
  rejected = false;
  try {
    fill_step_metadata(step, 4240, 4096, kRows, full.data(), swa.data(), pages, 205184);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected);

  // Serving registers RoPE from position zero so shorter prompts decode at
  // their real end rather than at the maximum supported prompt length.
  std::array<uint16_t, 4224 * 64> serving_rope{};
  for (size_t i = 0; i < serving_rope.size(); ++i)
    serving_rope[i] = static_cast<uint16_t>(i);
  for (uint32_t prompt : {1536u, 2080u, 4096u}) {
    for (uint32_t offset : {0u, 127u}) {
      const uint32_t position = prompt + offset;
      fill_step_metadata(step, position, 0, 4224, serving_rope.data(),
                         serving_rope.data(), pages, 205184);
      require(step.position == static_cast<int32_t>(position));
      require(step.slot == pages[position / 128] * 128 + static_cast<int32_t>(position % 128));
      require(step.full_cos[0] == serving_rope[position * 64]);
      require(step.full_sin[63] == serving_rope[position * 64 + 63]);
    }
  }

  std::array<int32_t, 64> updated{};
  for (size_t i = 0; i < updated.size(); ++i) updated[i] = static_cast<int32_t>(i + 34);
  auto original = updated;
  std::swap(updated[33], updated[63]);
  gaudirt::mimo::validate_page_update(original, updated, 4096, 205184);
  rejected = false;
  try {
    gaudirt::mimo::validate_page_update(original, updated, 4225, 205184);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected);
  updated = original;
  updated[63] = updated[1];
  rejected = false;
  try {
    gaudirt::mimo::validate_page_update(original, updated, 0, 205184);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected);

  // The M512 SWA kernel consumes exactly two adjacent metadata entries per
  // row. Compare that layout with both the step contract and a full scan.
  std::array<uint16_t, 4096 * 64> prefill_rope{};
  std::array<int32_t, 64> permuted{};
  for (size_t i = 0; i < permuted.size(); ++i) permuted[i] = (i * 17) % 64;
  gaudirt::mimo::LargePrefillMetadata batch{};
  for (uint32_t first = 0; first < 4096; first += 512) {
    gaudirt::mimo::fill_prefill_metadata(batch, first, 4096, prefill_rope.data(),
                                        prefill_rope.data(), permuted, 8192);
    for (uint32_t row = 0; row < 512; ++row) {
      fill_step_metadata(step, first + row, 0, 4096, prefill_rope.data(),
                         prefill_rope.data(), permuted, 8192);
      int32_t scanned[2]{-1, -1};
      unsigned found = 0;
      for (size_t i = 0; i < batch.window_groups.size(); ++i) {
        if (batch.window_groups[i] == static_cast<int32_t>(row)) {
          require(found < 2);
          scanned[found++] = batch.window_pages[i];
        }
      }
      require(batch.window_groups[row * 2] == static_cast<int32_t>(row));
      require(batch.window_pages[row * 2] == step.window_pages[0] &&
              batch.window_pages[row * 2 + 1] == step.window_pages[1]);
      require(scanned[0] == step.window_pages[0] && scanned[1] == step.window_pages[1]);
      require(found == (step.window_pages[1] < 0 ? 1u : 2u));
    }
  }
}
