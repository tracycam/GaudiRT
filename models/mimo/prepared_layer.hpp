// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/tp_collective.hpp"

#include <hccl.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace gaudirt::mimo {

struct PreparedLayer {
  std::vector<runtime::PreparedLaunch> attention;
  runtime::PreparedLaunch attention_reduce;
  std::vector<runtime::PreparedLaunch> feed_forward;
  runtime::PreparedLaunch output_reduce;
  uint64_t attention_local = 0;
  uint64_t attention_gathered = 0;
  uint64_t feed_forward_local = 0;
  uint64_t feed_forward_gathered = 0;
  hcclDataType_t attention_type = hcclFloat32;
  hcclComm_t comm = nullptr;
  uint32_t rows = 1;

  void validate() const {
    if ((rows != 1 && rows != 4 && rows != 8 && rows != 32 && rows != 512 && rows != 1024 && rows != 2048) || !comm || attention.empty() || feed_forward.empty() ||
        !attention_local || !attention_gathered ||
        !feed_forward_local || !feed_forward_gathered ||
        (rows < 512 && (attention_local == attention_gathered ||
                        feed_forward_local == feed_forward_gathered)) ||
        (rows >= 512 && (attention_local != attention_gathered ||
                         feed_forward_local != feed_forward_gathered)))
      throw std::invalid_argument("PreparedLayer: invalid collective bindings");
  }

  template <bool fused>
  void execute_graphs(synStreamHandle stream) const {
    if constexpr (!fused) {
      if (rows >= 512) throw std::invalid_argument("PreparedLayer: large prefill requires fused execution");
    }
    for (const auto& launch : attention) launch.execute(stream);
    runtime::enqueue_tp8_collective(attention_local, attention_gathered, rows, 6144,
                                    attention_type, comm, stream);
    if constexpr (!fused) attention_reduce.execute(stream);
    for (const auto& launch : feed_forward) launch.execute(stream);
    runtime::enqueue_tp8_collective(feed_forward_local, feed_forward_gathered, rows, 6144,
                                    hcclFloat32, comm, stream);
    if constexpr (!fused) output_reduce.execute(stream);
  }

  void execute(synStreamHandle stream) const { execute_graphs<false>(stream); }
  void execute_fused(synStreamHandle stream) const { execute_graphs<true>(stream); }
};

}  // namespace gaudirt::mimo
