// SPDX-License-Identifier: Apache-2.0
#include "prefill_bindings.hpp"

#include <stdexcept>
#include <vector>

int main() {
  using gaudirt::mimo::PreparedLayer;
  std::vector<PreparedLayer> layers(2);
  layers[0].attention.resize(1);
  layers[0].attention[0].tensors.resize(3);
  layers[0].attention[0].tensors[0].pTensorAddress = 0x1000;
  layers[0].attention[0].tensors[1].pTensorAddress = 0x1020;
  layers[0].attention[0].tensors[2].pTensorAddress = 0x8000;
  layers[1].feed_forward.resize(1);
  layers[1].feed_forward[0].tensors.resize(2);
  layers[1].feed_forward[0].tensors[0].pTensorAddress = 0x1040;
  layers[1].feed_forward[0].tensors[1].pTensorAddress = 0x1100;
  const auto slots = gaudirt::mimo::prefill_metadata_bindings(layers, 0x1000, 0x100);
  if (slots.size() != 3) throw std::runtime_error("wrong metadata slot count");
  for (auto [address, offset] : slots) *address = 0x2000 + offset;
  if (layers[0].attention[0].tensors[0].pTensorAddress != 0x2000 ||
      layers[0].attention[0].tensors[1].pTensorAddress != 0x2020 ||
      layers[1].feed_forward[0].tensors[0].pTensorAddress != 0x2040 ||
      layers[0].attention[0].tensors[2].pTensorAddress != 0x8000 ||
      layers[1].feed_forward[0].tensors[1].pTensorAddress != 0x1100)
    throw std::runtime_error("metadata update changed a static binding");
  for (auto [address, offset] : slots) *address = 0x3000 + offset;
  if (layers[1].feed_forward[0].tensors[0].pTensorAddress != 0x3040)
    throw std::runtime_error("metadata offset not preserved across chunks");
}
