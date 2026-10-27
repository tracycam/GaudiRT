// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/hbm_arena.hpp"

#include <array>
#include <cstdint>

namespace gaudirt::runtime {

class TpReduce {
 public:
  TpReduce(HbmArena& arena, uint32_t elements, synDataType gathered_type);
  ~TpReduce();

  TpReduce(const TpReduce&) = delete;
  TpReduce& operator=(const TpReduce&) = delete;

  void execute(uint64_t gathered, uint64_t output, synStreamHandle stream);
  uint64_t workspace_bytes() const noexcept { return workspace_bytes_; }

 private:
  HbmArena& arena_;
  uint32_t elements_ = 0;
  uint64_t rank_bytes_ = 0;
  synGraphHandle graph_ = nullptr;
  synRecipeHandle recipe_ = nullptr;
  std::array<synSectionHandle, 8> rank_sections_{};
  synSectionHandle output_section_ = nullptr;
  uint64_t workspace_bytes_ = 0;
};

}  // namespace gaudirt::runtime
