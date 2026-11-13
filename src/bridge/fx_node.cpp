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
    else if constexpr (std::is_same_v<T, FxRmsNorm>)
      output << ":e" << std::bit_cast<uint32_t>(value.epsilon)
             << ":" << std::bit_cast<uint32_t>(value.weight_offset);
    else if constexpr (std::is_same_v<T, FxRotary>)
      output << ":r" << value.head_dim << ":" << value.rotary_dim << ":" << value.neox;
    else if constexpr (std::is_same_v<T, FxCast>) output << ":c" << value.to_float;
    else if constexpr (std::is_same_v<T, FxCastDtype>) output << ":cd" << value.scalar_type;
    else if constexpr (std::is_same_v<T, FxShape>) {
      output << ":v";
      for (auto dimension : value.dimensions) output << ":" << dimension;
    } else if constexpr (std::is_same_v<T, FxTranspose>) output << ":p" << value.dim0 << ":" << value.dim1;
    else if constexpr (std::is_same_v<T, FxScalar>)
      output << ":s" << std::bit_cast<uint32_t>(value.value) << ":" << value.reversed;
    else if constexpr (std::is_same_v<T, FxShardEmbedding>)
      output << ":se" << value.org_start << ":" << value.org_end << ":"
             << value.added_start << ":" << value.added_end << ":" << value.added_offset;
    else if constexpr (std::is_same_v<T, FxUnary>)
      output << ":u" << value.op_code << ":" << std::bit_cast<uint32_t>(value.alpha) << ":"
             << std::bit_cast<uint32_t>(value.beta);
    else if constexpr (std::is_same_v<T, FxGeluMul>) output << ":gm" << value.approximate;
    else if constexpr (std::is_same_v<T, FxAxis>) output << ":ax" << value.dim;
    else if constexpr (std::is_same_v<T, FxReduce>)
      output << ":rd" << value.dim << ":" << value.keepdim;
    else if constexpr (std::is_same_v<T, FxLayerNorm>)
      output << ":ln" << value.has_weight << ":" << value.has_bias << ":"
             << std::bit_cast<uint32_t>(value.epsilon);
    else if constexpr (std::is_same_v<T, FxSliceDim>)
      output << ":sd" << value.dim << ":" << value.start << ":" << value.length;
    else if constexpr (std::is_same_v<T, FxConstant>)
      output << ":k" << std::bit_cast<uint32_t>(value.float_value) << ":" << value.int_value << ":"
             << value.scalar_type << ":" << value.is_integral;
  }, parameters);
  output << ";";
}

}  // namespace gaudirt::generic
