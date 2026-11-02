// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gaudirt::mimo {
struct PrefillChunk { uint32_t offset, index, rows, valid_rows; };
inline std::vector<PrefillChunk> prefill_chunks(uint32_t count) {
  if (!count || count > 4096) throw std::invalid_argument("invalid prefill chunk count");
  std::vector<PrefillChunk> result;
  uint32_t small = 0, large = 0, xlarge = 0;
  for (uint32_t offset = 0; offset < count;) {
    const uint32_t rows = count - offset >= 2048 ? 2048 : count - offset >= 512 ? 512 : 32;
    const uint32_t valid_rows = std::min(rows, count - offset);
    result.push_back({offset, rows == 2048 ? xlarge++ : rows == 512 ? large++ : small++, rows, valid_rows});
    offset += valid_rows;
  }
  return result;
}
}
