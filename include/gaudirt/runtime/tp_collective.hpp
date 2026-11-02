// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <hccl.h>
#include <synapse_api.h>

#include <cstdint>
#include <stdexcept>

namespace gaudirt::runtime {

// The large-row producer supplies FP32, including a widened BF16 boundary
// where its unaggregated output is BF16. The result remains in that buffer.
inline void enqueue_tp8_collective(uint64_t local, uint64_t gathered,
                                   uint32_t rows, uint32_t elements_per_row, hcclDataType_t type,
                                   hcclComm_t comm, synStreamHandle stream) {
  const auto status = rows >= 512
      ? hcclAllReduce(reinterpret_cast<void*>(local), reinterpret_cast<void*>(local),
                      static_cast<uint64_t>(elements_per_row) * rows, hcclFloat32, hcclSum, comm, stream)
      : hcclAllGather(reinterpret_cast<const void*>(local), reinterpret_cast<void*>(gathered),
                      static_cast<uint64_t>(elements_per_row) * rows, type, comm, stream);
  if (status != hcclSuccess) throw std::runtime_error("TP8 collective failed");
}

}  // namespace gaudirt::runtime
