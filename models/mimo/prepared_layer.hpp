// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/prepared_launch.hpp"

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

  void validate() const {
    if (!comm || attention.empty() || feed_forward.empty() ||
        !attention_local || !attention_gathered ||
        !feed_forward_local || !feed_forward_gathered ||
        attention_local == attention_gathered ||
        feed_forward_local == feed_forward_gathered)
      throw std::invalid_argument("PreparedLayer: invalid collective bindings");
  }

  void execute(synStreamHandle stream) const {
    for (const auto& launch : attention) launch.execute(stream);
    if (hcclAllGather(reinterpret_cast<const void*>(attention_local),
                      reinterpret_cast<void*>(attention_gathered), 6144,
                      attention_type, comm, stream) != hcclSuccess)
      throw std::runtime_error("PreparedLayer: attention AllGather failed");
    attention_reduce.execute(stream);
    for (const auto& launch : feed_forward) launch.execute(stream);
    if (hcclAllGather(reinterpret_cast<const void*>(feed_forward_local),
                      reinterpret_cast<void*>(feed_forward_gathered), 6144,
                      hcclFloat32, comm, stream) != hcclSuccess)
      throw std::runtime_error("PreparedLayer: feed-forward AllGather failed");
    output_reduce.execute(stream);
  }
};

}  // namespace gaudirt::mimo
