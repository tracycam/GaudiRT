// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>
#include <synapse_api.h>

#include <cstdint>

namespace gaudirt::generic {

struct Layout {
  explicit Layout(const at::Tensor& tensor);
  uint64_t offset(uint64_t index) const;
  const at::Tensor& tensor;
  uint64_t run;
};

void host_copy(uint8_t* packed, const at::Tensor& tensor, uint64_t start, uint64_t elements, bool pack);
void dma_copy(uint64_t dst, const Layout* dst_layout, uint64_t src, const Layout* src_layout,
              uint64_t start, uint64_t elements, uint64_t item, synStreamHandle stream, synDmaDir direction);

}  // namespace gaudirt::generic
