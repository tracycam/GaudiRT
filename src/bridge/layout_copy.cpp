// SPDX-License-Identifier: Apache-2.0
#include "layout_copy.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>

namespace gaudirt::generic {

// Addresses come from host-visible tensor metadata; D2D data stays in HBM.
Layout::Layout(const at::Tensor& value) : tensor(value), run(1) {
  for (int64_t i = tensor.dim() - 1; i >= 0; --i) {
    const auto size = tensor.size(i);
    if (size != 1 && tensor.stride(i) != static_cast<int64_t>(run)) break;
    run *= static_cast<uint64_t>(size);
  }
}

uint64_t Layout::offset(uint64_t index) const {
  uint64_t result = 0;
  for (int64_t i = tensor.dim() - 1; i >= 0; --i) {
    const auto size = static_cast<uint64_t>(tensor.size(i));
    result += (index % size) * static_cast<uint64_t>(tensor.stride(i));
    index /= size;
  }
  return result;
}

void host_copy(uint8_t* packed, const at::Tensor& tensor, uint64_t start,
               uint64_t elements, bool pack) {
  const Layout layout(tensor);
  const uint64_t item = tensor.element_size();
  auto* base = static_cast<uint8_t*>(tensor.data_ptr());
  for (uint64_t done = 0; done < elements;) {
    const uint64_t index = start + done;
    const uint64_t count = std::min(elements - done, layout.run - index % layout.run);
    auto* slot = base + layout.offset(index) * item;
    if (pack) std::memcpy(packed + done * item, slot, count * item);
    else std::memcpy(slot, packed + done * item, count * item);
    done += count;
  }
}

void dma_copy(uint64_t dst, const Layout* dst_layout, uint64_t src,
              const Layout* src_layout, uint64_t start, uint64_t elements,
              uint64_t item, synStreamHandle stream, synDmaDir direction) {
  std::array<uint64_t, 128> sources{}, sizes{}, destinations{};
  size_t used = 0;
  auto flush = [&] {
    if (used && synMemCopyAsyncMultiple(stream, sources.data(), sizes.data(), destinations.data(),
                                        direction, used) != synSuccess)
      throw std::runtime_error("GaudiRT: strided DMA failed");
    used = 0;
  };
  for (uint64_t done = 0; done < elements;) {
    const uint64_t index = start + done;
    uint64_t count = elements - done;
    if (src_layout) count = std::min(count, src_layout->run - index % src_layout->run);
    if (dst_layout) count = std::min(count, dst_layout->run - index % dst_layout->run);
    sources[used] = src + (src_layout ? src_layout->offset(index) : done) * item;
    destinations[used] = dst + (dst_layout ? dst_layout->offset(index) : done) * item;
    sizes[used] = count * item;
    ++used;
    if (used == sources.size()) flush();
    done += count;
  }
  flush();
}

}  // namespace gaudirt::generic
