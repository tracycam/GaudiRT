// SPDX-License-Identifier: Apache-2.0
#include "step_metadata.hpp"

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
  constexpr uint32_t kRows = 144;
  std::array<uint16_t, kRows * 64> full{}, swa{};
  for (size_t i = 0; i < full.size(); ++i) {
    full[i] = static_cast<uint16_t>(i);
    swa[i] = static_cast<uint16_t>(i + 1);
  }
  std::array<int32_t, 64> pages{};
  for (size_t i = 0; i < pages.size(); ++i) pages[i] = static_cast<int32_t>(i + 34);
  StepMetadata step{};
  for (uint32_t i = 0; i < kRows; ++i) {
    fill_step_metadata(step, 4096 + i, 4096, kRows, full.data(), swa.data(), pages, 205184);
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
  rejected = false;
  try {
    fill_step_metadata(step, 4240, 4096, kRows, full.data(), swa.data(), pages, 205184);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected);
}
