// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "step_metadata.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gaudirt::mimo {

// Layout of the tensors consumed by B4/B8 decode graphs. Each request owns
// its active logical pages; padding rows write only to the dedicated page.
template <uint32_t Rows>
struct alignas(256) BatchStepMetadata {
  static_assert(Rows == 4 || Rows == 8);
  alignas(256) std::array<uint16_t, Rows * 64> full_cos{};
  alignas(256) std::array<uint16_t, Rows * 64> full_sin{};
  alignas(256) std::array<uint16_t, Rows * 64> swa_cos{};
  alignas(256) std::array<uint16_t, Rows * 64> swa_sin{};
  alignas(256) std::array<int32_t, Rows> positions{};
  alignas(256) std::array<int32_t, Rows> slots{};
  alignas(256) std::array<int32_t, Rows> full_row_groups{};
  alignas(256) std::array<int32_t, Rows * 64> full_pages{};
  alignas(256) std::array<int32_t, Rows * 64> full_usage{};
  alignas(256) std::array<int32_t, Rows * 64> full_groups{};
  alignas(256) std::array<int32_t, Rows * 2> window_pages{};
  alignas(256) std::array<int32_t, Rows * 2> window_groups{};
};

template <uint32_t Rows>
struct alignas(256) BatchDecodeInput {
  BatchStepMetadata<Rows> metadata{};
  alignas(256) std::array<int32_t, Rows> tokens{};
};

struct BatchRequestStep {
  uint32_t position;
  // Only the allocated prefix is used. An allocation includes the slot to
  // which this step will write, even when it is the first slot in a new page.
  std::array<int32_t, 64> pages;
  uint32_t page_count;
};

// Call at the scheduling boundary whenever request/page ownership changes.
template <uint32_t Rows>
inline void validate_batch_step(const BatchRequestStep* requests, uint32_t active,
                                uint32_t rope_rows, const uint16_t* full_rope,
                                const uint16_t* swa_rope, uint32_t physical_pages,
                                uint32_t scratch_first_slot) {
  if (!requests || !active || active > Rows || !full_rope || !swa_rope || !rope_rows ||
      !physical_pages || physical_pages > static_cast<uint32_t>(INT32_MAX / 128) ||
      scratch_first_slot != physical_pages * 128 ||
      scratch_first_slot > static_cast<uint32_t>(INT32_MAX) - Rows)
    throw std::invalid_argument("MiMo batch step: invalid batch or KV geometry");
  std::vector<bool> owned(physical_pages, false);
  for (uint32_t row=0; row<active; ++row) {
    const auto& request=requests[row];
    if (request.position >= 64 * 128 || request.position >= rope_rows || request.page_count > 64 ||
        request.page_count < request.position / 128 + 1)
      throw std::invalid_argument("MiMo batch step: invalid position or page count");
    for (uint32_t p=0; p<request.page_count; ++p) {
      const int32_t physical=request.pages[p];
      if (physical < 0 || static_cast<uint32_t>(physical) >= physical_pages || owned[physical])
        throw std::invalid_argument("MiMo batch step: invalid or shared physical page");
      owned[physical]=true;
    }
  }
}

// The caller has already validated request and page ownership.
template <uint32_t Rows>
inline void fill_batch_step_metadata_prevalidated(BatchStepMetadata<Rows>& result,
                                                  const BatchRequestStep* requests,
                                                  uint32_t active, const uint16_t* full_rope,
                                                  const uint16_t* swa_rope,
                                                  uint32_t scratch_first_slot) {
  result = {};
  result.full_pages.fill(-1);
  result.full_groups.fill(-1);
  result.window_pages.fill(-1);
  result.window_groups.fill(-1);
  result.full_row_groups.fill(-1);
  for (uint32_t row=0; row<Rows; ++row) {
    if (row >= active) {
      result.slots[row]=static_cast<int32_t>(scratch_first_slot + row-active);
      continue;
    }
    const auto& req=requests[row];
    const uint32_t pos=req.position, current=pos/128;
    const size_t rope_offset=static_cast<size_t>(pos)*64;
    for (size_t i=0;i<64;++i) {
      result.full_cos[row*64+i]=full_rope[rope_offset+i%32];
      result.full_sin[row*64+i]=full_rope[rope_offset+32+i%32];
      result.swa_cos[row*64+i]=swa_rope[rope_offset+i%32];
      result.swa_sin[row*64+i]=swa_rope[rope_offset+32+i%32];
    }
    result.positions[row]=static_cast<int32_t>(pos);
    result.slots[row]=req.pages[current]*128+static_cast<int32_t>(pos%128);
    result.full_row_groups[row]=static_cast<int32_t>(row);
    for (uint32_t p=0;p<=current;++p) {
      const size_t index=row*64+p;
      result.full_pages[index]=req.pages[p];
      result.full_usage[index]=static_cast<int32_t>(std::min<uint32_t>(128,pos+1-p*128));
      result.full_groups[index]=static_cast<int32_t>(row);
    }
    const uint32_t first=(pos>=127?pos-127:0)/128;
    result.window_pages[row*2]=req.pages[first];
    result.window_groups[row*2]=static_cast<int32_t>(row);
    if (first<current) {
      result.window_pages[row*2+1]=req.pages[current];
      result.window_groups[row*2+1]=static_cast<int32_t>(row);
    }
  }
}

}  // namespace gaudirt::mimo
