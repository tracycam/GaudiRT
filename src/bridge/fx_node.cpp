// SPDX-License-Identifier: Apache-2.0
#include "fx_node.hpp"

#include <bit>
#include <type_traits>

namespace gaudirt::generic {

void FxNode::append_normalized(std::ostream& output) const {
  output << "n" << static_cast<int>(op);
  for (auto input : inputs) output << ":i" << input;
  std::visit([&](const auto& value) {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, FxLinear>) output << ":t" << value.transpose_weight;
    else if constexpr (std::is_same_v<T, FxSlice>)
      output << ":s" << value.start << ":" << value.width << ":" << value.total;
    else if constexpr (std::is_same_v<T, FxEpsilon>) output << ":e" << std::bit_cast<uint32_t>(value.value);
    else if constexpr (std::is_same_v<T, FxRotary>)
      output << ":r" << value.context << ":" << value.head_dim << ":" << value.rotary_dim << ":" << value.neox;
    else if constexpr (std::is_same_v<T, FxCast>) output << ":c" << value.to_float;
    else if constexpr (std::is_same_v<T, FxShape>) {
      output << ":v";
      for (auto dimension : value.dimensions) output << ":" << dimension;
    } else if constexpr (std::is_same_v<T, FxTranspose>) output << ":p" << value.dim0 << ":" << value.dim1;
    else if constexpr (std::is_same_v<T, FxScalar>)
      output << ":s" << std::bit_cast<uint32_t>(value.value) << ":" << value.reversed;
  }, parameters);
  output << ";";
}

}  // namespace gaudirt::generic
