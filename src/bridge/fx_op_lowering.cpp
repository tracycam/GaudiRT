// SPDX-License-Identifier: Apache-2.0
#include "fx_op_lowering.hpp"

#include "bridge_internal.hpp"
#include "bucket_program.hpp"

#include <c10/util/Exception.h>
#include <perf_lib_layer_params.h>
#include <synapse_api.h>
#include <synapse_common_types.hpp>

#include <algorithm>
#include <bit>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace gaudirt::generic {
namespace {

static_assert(sizeof(EwiseUnaryParameters) == 12, "ABI mismatch for EwiseUnaryParameters");
static_assert(sizeof(GeluAndMulParameters) == 4, "ABI mismatch for GeluAndMulParameters");
static_assert(sizeof(RmsNormParameters) == 8, "ABI mismatch for RmsNormParameters");

synTensor make_tensor_2d(SynapseGraphBuilder& builder, synDataType type, uint64_t fcd,
                         uint64_t rows, int slot = -1) {
  return builder.tensor(type, {fcd, rows}, slot);
}

void add_single_out_node(SynapseGraphBuilder& builder, std::vector<synTensor> inputs,
                         synTensor output, NodeParameters params, std::string guid) {
  builder.node(NodeSpec{std::move(guid), std::move(inputs), {output}, std::move(params)});
}

synDataType syn_dtype_for(at::ScalarType dtype) {
  if (dtype == at::kFloat8_e4m3fn) return syn_type_fp8_143;
  return cast_type(dtype).syn;
}

uint64_t syn_axis_from_pt(size_t rank, int64_t pt_dim) {
  const int64_t resolved = pt_dim < 0 ? pt_dim + static_cast<int64_t>(rank) : pt_dim;
  TORCH_CHECK(resolved >= 0 && resolved < static_cast<int64_t>(rank), "GaudiRT: axis out of bounds");
  return static_cast<uint64_t>(rank - 1 - static_cast<size_t>(resolved));
}

}  // namespace

std::vector<uint64_t> flatten_2d_syn_dims(const std::vector<uint64_t>& syn_dims) {
  TORCH_CHECK(!syn_dims.empty(), "GaudiRT: empty tensor dimensions");
  if (syn_dims.size() == 1) return {syn_dims[0], 1};
  if (syn_dims.size() == 2) return syn_dims;
  uint64_t rows = 1;
  for (size_t i = 1; i < syn_dims.size(); ++i) rows *= syn_dims[i];
  return {syn_dims[0], rows};
}

synTensor emit_constant(SynapseGraphBuilder& builder, at::ScalarType dtype,
                        const std::vector<uint64_t>& syn_dims,
                        float float_value, int64_t int_value, bool is_integral,
                        synTensor target_out) {
  auto out = target_out ? target_out : builder.tensor(syn_dtype_for(dtype), syn_dims, -1);
  if (dtype == at::kFloat) {
    ns_ConstantKernel::ParamsV3 p{};
    p.constant.f = is_integral ? static_cast<float>(int_value) : float_value;
    add_single_out_node(builder, {}, out, p, "constant_f32");
    return out;
  }
  if (dtype == at::kBFloat16) {
    auto f32 = builder.tensor(syn_type_float, syn_dims, -1);
    ns_ConstantKernel::ParamsV3 p{};
    p.constant.f = is_integral ? static_cast<float>(int_value) : float_value;
    add_single_out_node(builder, {}, f32, p, "constant_f32");
    return emit_cast(builder, f32, syn_dims, at::kFloat, at::kBFloat16, out);
  }
  if (dtype == at::kInt) {
    ns_ConstantKernel::ParamsV3 p{};
    p.constant.i = is_integral ? static_cast<int32_t>(int_value) : static_cast<int32_t>(float_value);
    add_single_out_node(builder, {}, out, p, "constant_i32");
    return out;
  }
  if (dtype == at::kLong) {
    auto i32 = builder.tensor(syn_type_int32, syn_dims, -1);
    ns_ConstantKernel::ParamsV3 p{};
    p.constant.i = is_integral ? static_cast<int32_t>(int_value) : static_cast<int32_t>(float_value);
    add_single_out_node(builder, {}, i32, p, "constant_i32");
    ns_CastKernel::ParamsV3 cast{};
    cast.round_mode = CAST_ROUND_HALF_NE;
    add_single_out_node(builder, {i32}, out, cast, "cast_i32_to_i64");
    return out;
  }
  TORCH_CHECK(dtype == at::kBool, "GaudiRT: unsupported constant dtype");
  const bool flag = is_integral ? (int_value != 0) : (float_value != 0.f);
  auto lhs = builder.tensor(syn_type_int32, syn_dims, -1);
  auto rhs = builder.tensor(syn_type_int32, syn_dims, -1);
  ns_ConstantKernel::ParamsV3 pl{}, pr{};
  pl.constant.i = flag ? 1 : 0;
  pr.constant.i = 1;
  add_single_out_node(builder, {}, lhs, pl, "constant_i32");
  add_single_out_node(builder, {}, rhs, pr, "constant_i32");
  add_single_out_node(builder, {lhs, rhs}, out, std::monostate{}, "equal_fwd_i32");
  return out;
}

synTensor emit_cast(SynapseGraphBuilder& builder, synTensor input,
                    const std::vector<uint64_t>& syn_dims,
                    at::ScalarType from, at::ScalarType to,
                    synTensor target_out) {
  if (from == to) {
    if (!target_out) return input;
    auto zero = emit_constant(builder, from, syn_dims, 0.f, 0, from == at::kInt || from == at::kLong, nullptr);
    if (from == at::kBool) {
      add_single_out_node(builder, {input, zero}, target_out, std::monostate{}, "or_fwd_u8");
    } else {
      add_single_out_node(builder, {input, zero}, target_out, std::monostate{},
                          "add_fwd_" + std::string(cast_type(from).name));
    }
    return target_out;
  }
  auto out = target_out ? target_out : builder.tensor(syn_dtype_for(to), syn_dims, -1);
  ns_CastKernel::ParamsV3 cast{};
  cast.round_mode = CAST_ROUND_HALF_NE;

  if (to == at::kBool) {
    auto zero = emit_constant(builder, from, syn_dims, 0.f, 0, from == at::kInt || from == at::kLong, nullptr);
    auto eq_u8 = builder.tensor(syn_type_uint8, syn_dims, -1);
    add_single_out_node(builder, {input, zero}, eq_u8, std::monostate{},
                        "equal_fwd_" + std::string(cast_type(from).name));
    add_single_out_node(builder, {eq_u8}, out, std::monostate{}, "not_fwd_u8");
    return out;
  }
  if (from == at::kBool) {
    if (to == at::kFloat) {
      add_single_out_node(builder, {input}, out, cast, "cast_u8_to_f32");
      return out;
    }
    if (to == at::kInt) {
      auto one = emit_constant(builder, at::kInt, syn_dims, 0.f, 1, true);
      auto zero = emit_constant(builder, at::kInt, syn_dims, 0.f, 0, true);
      add_single_out_node(builder, {input, one, zero}, out, std::monostate{}, "where_fwd_i32");
      return out;
    }
    auto f32 = builder.tensor(syn_type_float, syn_dims, -1);
    add_single_out_node(builder, {input}, f32, cast, "cast_u8_to_f32");
    return emit_cast(builder, f32, syn_dims, at::kFloat, to, out);
  }
  if (from == at::kBFloat16 && to == at::kFloat) {
    add_single_out_node(builder, {input}, out, cast, "cast_bf16_to_f32");
    return out;
  }
  if (from == at::kFloat && to == at::kBFloat16) {
    if (syn_dims.size() == 2 && syn_dims[1] <= kMaxBucketRows) {
      add_single_out_node(builder, {input}, out, std::monostate{}, "gk_cast_f32_to_bf16_v1");
    } else {
      add_single_out_node(builder, {input}, out, cast, "cast_f32_to_bf16");
    }
    return out;
  }
  if ((from == at::kInt && to == at::kLong) || (from == at::kLong && to == at::kInt) ||
      (from == at::kInt && to == at::kFloat) || (from == at::kFloat && to == at::kInt)) {
    add_single_out_node(builder, {input}, out, cast,
                        "cast_" + std::string(cast_type(from).name) + "_to_" + cast_type(to).name);
    return out;
  }
  if (from == at::kBFloat16) {
    auto f32 = emit_cast(builder, input, syn_dims, at::kBFloat16, at::kFloat);
    return emit_cast(builder, f32, syn_dims, at::kFloat, to, out);
  }
  if (to == at::kBFloat16) {
    auto f32 = emit_cast(builder, input, syn_dims, from, at::kFloat);
    return emit_cast(builder, f32, syn_dims, at::kFloat, at::kBFloat16, out);
  }
  // Int64 <-> Float32 via Int32
  auto i32 = emit_cast(builder, input, syn_dims, from, at::kInt);
  return emit_cast(builder, i32, syn_dims, at::kInt, to, out);
}

synTensor emit_broadcast_to(SynapseGraphBuilder& builder, synTensor input, synDataType dtype,
                            const std::vector<uint64_t>& src_syn_dims,
                            const std::vector<uint64_t>& dst_syn_dims,
                            synTensor target_out) {
  if (src_syn_dims == dst_syn_dims) {
    if (!target_out) return input;
    add_single_out_node(builder, {input}, target_out, std::monostate{}, "reshape");
    return target_out;
  }
  auto current = input;
  auto cur_dims = src_syn_dims;
  if (cur_dims.size() < dst_syn_dims.size()) {
    while (cur_dims.size() < dst_syn_dims.size()) cur_dims.push_back(1);
    const bool done = (cur_dims == dst_syn_dims);
    auto reshaped = (done && target_out) ? target_out : builder.tensor(dtype, cur_dims, -1);
    add_single_out_node(builder, {current}, reshaped, std::monostate{}, "reshape");
    if (done) return reshaped;
    current = reshaped;
  }
  TORCH_CHECK(cur_dims.size() == dst_syn_dims.size(), "GaudiRT: invalid broadcast rank");
  if (dtype != syn_type_int64) {
    auto out = target_out ? target_out : builder.tensor(dtype, dst_syn_dims, -1);
    add_single_out_node(builder, {current}, out, std::monostate{}, "broadcast");
    return out;
  }
  // P-RT-92: Synapse "broadcast" on syn_type_int64 silently truncates the upper 32 bits;
  // lower all int64 broadcasts via local "reinterpret_cast" to syn_type_int32 "broadcast".
  if (cur_dims.size() <= 4) {
    std::vector<uint64_t> in64_dims{1};
    std::vector<uint64_t> in32_dims{2};
    std::vector<uint64_t> out32_dims{2};
    std::vector<uint64_t> out64_dims{1};
    in64_dims.insert(in64_dims.end(), cur_dims.begin(), cur_dims.end());
    in32_dims.insert(in32_dims.end(), cur_dims.begin(), cur_dims.end());
    out32_dims.insert(out32_dims.end(), dst_syn_dims.begin(), dst_syn_dims.end());
    out64_dims.insert(out64_dims.end(), dst_syn_dims.begin(), dst_syn_dims.end());
    auto in64 = builder.tensor(syn_type_int64, in64_dims, -1);
    auto in32 = builder.tensor(syn_type_int32, in32_dims, -1);
    auto out32 = builder.tensor(syn_type_int32, out32_dims, -1);
    auto out64 = builder.tensor(syn_type_int64, out64_dims, -1);
    auto out = target_out ? target_out : builder.tensor(syn_type_int64, dst_syn_dims, -1);
    add_single_out_node(builder, {current}, in64, std::monostate{}, "reshape");
    add_single_out_node(builder, {in64}, in32, std::monostate{}, "reinterpret_cast");
    add_single_out_node(builder, {in32}, out32, std::monostate{}, "broadcast");
    add_single_out_node(builder, {out32}, out64, std::monostate{}, "reinterpret_cast");
    add_single_out_node(builder, {out64}, out, std::monostate{}, "reshape");
    return out;
  }
  TORCH_CHECK(cur_dims.size() == 5, "GaudiRT: int64 broadcast rank exceeds Synapse 5D limit");
  if (cur_dims[0] != dst_syn_dims[0]) {
    TORCH_CHECK(cur_dims[0] == 1, "GaudiRT: invalid int64 FCD broadcast");
    const uint64_t rest = cur_dims[1] * cur_dims[2] * cur_dims[3] * cur_dims[4];
    auto in64 = builder.tensor(syn_type_int64, {1, 1, rest}, -1);
    auto in32 = builder.tensor(syn_type_int32, {2, 1, rest}, -1);
    auto out32 = builder.tensor(syn_type_int32, {2, dst_syn_dims[0], rest}, -1);
    auto out64 = builder.tensor(syn_type_int64, {1, dst_syn_dims[0], rest}, -1);
    add_single_out_node(builder, {current}, in64, std::monostate{}, "reshape");
    add_single_out_node(builder, {in64}, in32, std::monostate{}, "reinterpret_cast");
    add_single_out_node(builder, {in32}, out32, std::monostate{}, "broadcast");
    add_single_out_node(builder, {out32}, out64, std::monostate{}, "reinterpret_cast");
    cur_dims[0] = dst_syn_dims[0];
    const bool done = (cur_dims == dst_syn_dims);
    auto reshaped = (done && target_out) ? target_out : builder.tensor(syn_type_int64, cur_dims, -1);
    add_single_out_node(builder, {out64}, reshaped, std::monostate{}, "reshape");
    if (done) return reshaped;
    current = reshaped;
  }
  std::vector<uint64_t> in32_dims = cur_dims;
  std::vector<uint64_t> out32_dims = dst_syn_dims;
  in32_dims[0] *= 2;
  out32_dims[0] *= 2;
  auto cur_local = builder.tensor(syn_type_int64, cur_dims, -1);
  auto in32 = builder.tensor(syn_type_int32, in32_dims, -1);
  auto out32 = builder.tensor(syn_type_int32, out32_dims, -1);
  auto out64 = builder.tensor(syn_type_int64, dst_syn_dims, -1);
  add_single_out_node(builder, {current}, cur_local, std::monostate{}, "reshape");
  add_single_out_node(builder, {cur_local}, in32, std::monostate{}, "reinterpret_cast");
  add_single_out_node(builder, {in32}, out32, std::monostate{}, "broadcast");
  add_single_out_node(builder, {out32}, out64, std::monostate{}, "reinterpret_cast");
  if (!target_out) return out64;
  add_single_out_node(builder, {out64}, target_out, std::monostate{}, "reshape");
  return target_out;
}

void emit_ewise_unary(SynapseGraphBuilder& builder, synTensor input, synTensor output,
                      at::ScalarType dtype, const std::vector<uint64_t>& syn_dims,
                      int32_t op_code, float alpha, float beta) {
  if (dtype == at::kInt) {
    if (op_code == static_cast<int32_t>(FxUnaryOp::neg)) {
      auto minus_one = emit_constant(builder, at::kInt, syn_dims, 0.f, -1, true);
      add_single_out_node(builder, {input, minus_one}, output, std::monostate{}, "mult_fwd_i32");
      return;
    }
    TORCH_CHECK(op_code == static_cast<int32_t>(FxUnaryOp::abs),
                "GaudiRT: int32 ewise_unary supports only neg and abs");
    auto zero = emit_constant(builder, at::kInt, syn_dims, 0.f, 0, true);
    auto minus_one = emit_constant(builder, at::kInt, syn_dims, 0.f, -1, true);
    auto is_neg = builder.tensor(syn_type_uint8, syn_dims, -1);
    auto negated = builder.tensor(syn_type_int32, syn_dims, -1);
    add_single_out_node(builder, {input, zero}, is_neg, std::monostate{}, "less_fwd_i32");
    add_single_out_node(builder, {input, minus_one}, negated, std::monostate{}, "mult_fwd_i32");
    add_single_out_node(builder, {is_neg, negated, input}, output, std::monostate{}, "where_fwd_i32");
    return;
  }
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kFloat,
              "GaudiRT: ewise_unary requires BF16 or FP32 tensor");
  const auto syn = syn_dtype_for(dtype);
  const auto guid = dtype == at::kBFloat16 ? "gk_ewise_unary_bf16_v1" : "gk_ewise_unary_f32_v1";
  const auto packed = pack_node_params(EwiseUnaryParameters{op_code, alpha, beta});
  if (syn_dims.size() == 2) {
    add_single_out_node(builder, {input}, output, packed, guid);
    return;
  }
  const auto flat_dims = flatten_2d_syn_dims(syn_dims);
  auto flat_in = builder.tensor(syn, flat_dims, -1);
  auto flat_out = builder.tensor(syn, flat_dims, -1);
  add_single_out_node(builder, {input}, flat_in, std::monostate{}, "reshape");
  add_single_out_node(builder, {flat_in}, flat_out, packed, guid);
  add_single_out_node(builder, {flat_out}, output, std::monostate{}, "reshape");
}

void emit_isnan(SynapseGraphBuilder& builder, synTensor input, synTensor output,
                at::ScalarType dtype) {
  TORCH_CHECK(dtype == at::kBFloat16 || dtype == at::kFloat, "GaudiRT: isnan requires BF16 or FP32");
  add_single_out_node(builder, {input}, output, std::monostate{},
                      "isnan_fwd_" + std::string(cast_type(dtype).name));
}

void emit_logical_not(SynapseGraphBuilder& builder, synTensor input, synTensor output) {
  add_single_out_node(builder, {input}, output, std::monostate{}, "not_fwd_u8");
}

void emit_binary_pointwise(SynapseGraphBuilder& builder, FxOp op, synTensor left, synTensor right,
                           synTensor output, at::ScalarType in_dtype,
                           const std::vector<uint64_t>& left_dims,
                           const std::vector<uint64_t>& right_dims,
                           const std::vector<uint64_t>& out_dims) {
  const auto type = cast_type(in_dtype);
  auto lhs = emit_broadcast_to(builder, left, type.syn, left_dims, out_dims);
  auto rhs = emit_broadcast_to(builder, right, type.syn, right_dims, out_dims);

  if (op == FxOp::add) {
    const auto guid = (in_dtype == at::kBFloat16 && out_dims.size() == 2 && out_dims[1] <= kMaxBucketRows)
                          ? std::string("gk_add_bf16_v1")
                          : "add_fwd_" + std::string(type.name);
    add_single_out_node(builder, {lhs, rhs}, output, std::monostate{}, guid);
    return;
  }
  if (op == FxOp::sub) {
    if (in_dtype == at::kLong) {
      auto minus_one = emit_constant(builder, at::kLong, out_dims, 0.f, -1, true);
      auto neg_rhs = builder.tensor(syn_type_int64, out_dims, -1);
      add_single_out_node(builder, {rhs, minus_one}, neg_rhs, std::monostate{}, "mult_fwd_i64");
      add_single_out_node(builder, {lhs, neg_rhs}, output, std::monostate{}, "add_fwd_i64");
      return;
    }
    add_single_out_node(builder, {lhs, rhs}, output, std::monostate{}, "sub_fwd_" + std::string(type.name));
    return;
  }
  if (op == FxOp::mul) {
    add_single_out_node(builder, {lhs, rhs}, output, std::monostate{}, "mult_fwd_" + std::string(type.name));
    return;
  }
  if (op == FxOp::div) {
    if (in_dtype == at::kBFloat16) {
      auto nl = emit_cast(builder, lhs, out_dims, at::kBFloat16, at::kFloat);
      auto nr = emit_cast(builder, rhs, out_dims, at::kBFloat16, at::kFloat);
      auto q = builder.tensor(syn_type_float, out_dims, -1);
      add_single_out_node(builder, {nl, nr}, q, std::monostate{}, "div_fwd_f32");
      emit_cast(builder, q, out_dims, at::kFloat, at::kBFloat16, output);
      return;
    }
    add_single_out_node(builder, {lhs, rhs}, output, std::monostate{}, "div_fwd_f32");
    return;
  }
  if (op == FxOp::minimum || op == FxOp::maximum) {
    const std::string base = op == FxOp::minimum ? "min_fwd_" : "max_fwd_";
    add_single_out_node(builder, {lhs, rhs}, output, std::monostate{}, base + type.name);
    return;
  }
  if (op == FxOp::logical_and || (op == FxOp::bitwise_and && in_dtype == at::kBool)) {
    add_single_out_node(builder, {lhs, rhs}, output, std::monostate{}, "and_fwd_u8");
    return;
  }
  if (op == FxOp::logical_or || (op == FxOp::bitwise_or && in_dtype == at::kBool)) {
    add_single_out_node(builder, {lhs, rhs}, output, std::monostate{}, "or_fwd_u8");
    return;
  }
  if (op == FxOp::bitwise_and) {
    add_single_out_node(builder, {lhs, rhs}, output, std::monostate{}, "bitwise_and_fwd_i32");
    return;
  }
  if (op == FxOp::bitwise_or) {
    add_single_out_node(builder, {lhs, rhs}, output, std::monostate{}, "bitwise_or_fwd_i32");
    return;
  }
  // Comparisons: eq, ne, lt, le, gt, ge
  if (op == FxOp::ne) {
    auto eq_u8 = builder.tensor(syn_type_uint8, out_dims, -1);
    add_single_out_node(builder, {lhs, rhs}, eq_u8, std::monostate{}, "equal_fwd_" + std::string(type.name));
    add_single_out_node(builder, {eq_u8}, output, std::monostate{}, "not_fwd_u8");
    return;
  }
  if (in_dtype == at::kLong) {
    if (op == FxOp::gt) {
      add_single_out_node(builder, {rhs, lhs}, output, std::monostate{}, "less_fwd_i64");
      return;
    }
    if (op == FxOp::le) {
      add_single_out_node(builder, {rhs, lhs}, output, std::monostate{}, "greater_equal_fwd_i64");
      return;
    }
  }
  const char* prefix = op == FxOp::eq   ? "equal_fwd_"
                       : op == FxOp::lt ? "less_fwd_"
                       : op == FxOp::le ? "less_equal_fwd_"
                       : op == FxOp::gt ? "greater_fwd_"
                                        : "greater_equal_fwd_";
  add_single_out_node(builder, {lhs, rhs}, output, std::monostate{}, std::string(prefix) + type.name);
}

void emit_where(SynapseGraphBuilder& builder, synTensor cond, synTensor x, synTensor y,
                synTensor output, at::ScalarType dtype,
                const std::vector<uint64_t>& cond_dims,
                const std::vector<uint64_t>& x_dims,
                const std::vector<uint64_t>& y_dims,
                const std::vector<uint64_t>& out_dims) {
  auto c_b = emit_broadcast_to(builder, cond, syn_type_uint8, cond_dims, out_dims);
  if (dtype == at::kBool) {
    auto x_b = emit_broadcast_to(builder, x, syn_type_uint8, x_dims, out_dims);
    auto y_b = emit_broadcast_to(builder, y, syn_type_uint8, y_dims, out_dims);
    auto not_c = builder.tensor(syn_type_uint8, out_dims, -1);
    auto lhs = builder.tensor(syn_type_uint8, out_dims, -1);
    auto rhs = builder.tensor(syn_type_uint8, out_dims, -1);
    add_single_out_node(builder, {c_b}, not_c, std::monostate{}, "not_fwd_u8");
    add_single_out_node(builder, {c_b, x_b}, lhs, std::monostate{}, "and_fwd_u8");
    add_single_out_node(builder, {not_c, y_b}, rhs, std::monostate{}, "and_fwd_u8");
    add_single_out_node(builder, {lhs, rhs}, output, std::monostate{}, "or_fwd_u8");
    return;
  }
  const auto syn = syn_dtype_for(dtype);
  auto x_b = emit_broadcast_to(builder, x, syn, x_dims, out_dims);
  auto y_b = emit_broadcast_to(builder, y, syn, y_dims, out_dims);
  if (dtype == at::kLong) {
    uint64_t flat_elems = 1;
    for (const auto d : out_dims) flat_elems *= d;
    const std::vector<uint64_t> flat_i64_dims{1, flat_elems};
    const std::vector<uint64_t> lanes_dims{2, flat_elems};
    const std::vector<uint64_t> c_lanes_dims{1, flat_elems};
    auto xf = builder.tensor(syn_type_int64, flat_i64_dims, -1);
    auto yf = builder.tensor(syn_type_int64, flat_i64_dims, -1);
    auto xl = builder.tensor(syn_type_int32, lanes_dims, -1);
    auto yl = builder.tensor(syn_type_int32, lanes_dims, -1);
    auto c1 = builder.tensor(syn_type_uint8, c_lanes_dims, -1);
    auto cl = builder.tensor(syn_type_uint8, lanes_dims, -1);
    auto ol = builder.tensor(syn_type_int32, lanes_dims, -1);
    auto of = builder.tensor(syn_type_int64, flat_i64_dims, -1);
    add_single_out_node(builder, {x_b}, xf, std::monostate{}, "reshape");
    add_single_out_node(builder, {y_b}, yf, std::monostate{}, "reshape");
    add_single_out_node(builder, {xf}, xl, std::monostate{}, "reinterpret_cast");
    add_single_out_node(builder, {yf}, yl, std::monostate{}, "reinterpret_cast");
    add_single_out_node(builder, {c_b}, c1, std::monostate{}, "reshape");
    add_single_out_node(builder, {c1}, cl, std::monostate{}, "broadcast");
    add_single_out_node(builder, {cl, xl, yl}, ol, std::monostate{}, "where_fwd_i32");
    add_single_out_node(builder, {ol}, of, std::monostate{}, "reinterpret_cast");
    add_single_out_node(builder, {of}, output, std::monostate{}, "reshape");
    return;
  }
  add_single_out_node(builder, {c_b, x_b, y_b}, output, std::monostate{},
                      "where_fwd_" + std::string(cast_type(dtype).name));
}

void emit_rms_norm(SynapseGraphBuilder& builder, synTensor input, synTensor residual,
                   synTensor weight, synTensor out_norm, synTensor out_residual,
                   const std::vector<uint64_t>& syn_dims, float epsilon, float weight_offset) {
  const auto params = pack_node_params(RmsNormParameters{epsilon, weight_offset});
  const bool fused = (residual != nullptr);
  const std::string guid = fused ? "gk_fused_add_rms_norm_bf16_v1" : "gk_rms_norm_bf16_v1";
  if (syn_dims.size() == 2) {
    if (fused) {
      builder.node(NodeSpec{guid, {input, residual, weight}, {out_residual, out_norm}, params});
    } else {
      add_single_out_node(builder, {input, weight}, out_norm, params, guid);
    }
    return;
  }
  const auto flat_dims = flatten_2d_syn_dims(syn_dims);
  auto flat_in = builder.tensor(syn_type_bf16, flat_dims, -1);
  add_single_out_node(builder, {input}, flat_in, std::monostate{}, "reshape");
  auto flat_out = builder.tensor(syn_type_bf16, flat_dims, -1);
  if (fused) {
    auto flat_res = builder.tensor(syn_type_bf16, flat_dims, -1);
    auto flat_out_res = builder.tensor(syn_type_bf16, flat_dims, -1);
    add_single_out_node(builder, {residual}, flat_res, std::monostate{}, "reshape");
    builder.node(NodeSpec{guid, {flat_in, flat_res, weight}, {flat_out_res, flat_out}, params});
    add_single_out_node(builder, {flat_out_res}, out_residual, std::monostate{}, "reshape");
  } else {
    add_single_out_node(builder, {flat_in, weight}, flat_out, params, guid);
  }
  add_single_out_node(builder, {flat_out}, out_norm, std::monostate{}, "reshape");
}

void emit_gated_mul(SynapseGraphBuilder& builder, bool is_gelu, int32_t approximate,
                    synTensor input, synTensor output,
                    const std::vector<uint64_t>& in_syn_dims,
                    const std::vector<uint64_t>& out_syn_dims) {
  const std::string guid = is_gelu ? "gk_gelu_and_mul_bf16_v1" : "gk_silu_and_mul_bf16_v1";
  NodeParameters params = is_gelu ? NodeParameters{pack_node_params(GeluAndMulParameters{approximate})}
                                  : NodeParameters{std::monostate{}};
  if (in_syn_dims.size() == 2) {
    add_single_out_node(builder, {input}, output, std::move(params), guid);
    return;
  }
  const auto flat_in_dims = flatten_2d_syn_dims(in_syn_dims);
  const auto flat_out_dims = flatten_2d_syn_dims(out_syn_dims);
  auto flat_in = builder.tensor(syn_type_bf16, flat_in_dims, -1);
  auto flat_out = builder.tensor(syn_type_bf16, flat_out_dims, -1);
  add_single_out_node(builder, {input}, flat_in, std::monostate{}, "reshape");
  add_single_out_node(builder, {flat_in}, flat_out, std::move(params), guid);
  add_single_out_node(builder, {flat_out}, output, std::monostate{}, "reshape");
}

void emit_rotary_pos(SynapseGraphBuilder& builder, synTensor q, synTensor k, synTensor cache,
                     synTensor positions, synTensor q_out, synTensor k_out,
                     const std::vector<uint64_t>& q_syn_dims,
                     const std::vector<uint64_t>& k_syn_dims,
                     at::ScalarType pos_dtype, uint64_t rows,
                     int32_t head_dim, int32_t rotary_dim, bool is_neox) {
  auto pos_i32 = positions;
  if (pos_dtype == at::kLong) {
    pos_i32 = emit_cast(builder, positions, {rows}, at::kLong, at::kInt);
  }
  auto q_2d = q;
  auto k_2d = k;
  auto q_out_2d = q_out;
  auto k_out_2d = k_out;
  const bool need_q_reshape = (q_syn_dims.size() != 2);
  const bool need_k_reshape = (k_syn_dims.size() != 2);
  if (need_q_reshape) {
    TORCH_CHECK(q_syn_dims.size() == 3, "GaudiRT: rotary_pos expects rank 2 or 3 query");
    const std::vector<uint64_t> q_flat{q_syn_dims[0] * q_syn_dims[1], q_syn_dims[2]};
    q_2d = builder.tensor(syn_type_bf16, q_flat, -1);
    q_out_2d = builder.tensor(syn_type_bf16, q_flat, -1);
    add_single_out_node(builder, {q}, q_2d, std::monostate{}, "reshape");
  }
  if (need_k_reshape) {
    TORCH_CHECK(k_syn_dims.size() == 3, "GaudiRT: rotary_pos expects rank 2 or 3 key");
    const std::vector<uint64_t> k_flat{k_syn_dims[0] * k_syn_dims[1], k_syn_dims[2]};
    k_2d = builder.tensor(syn_type_bf16, k_flat, -1);
    k_out_2d = builder.tensor(syn_type_bf16, k_flat, -1);
    add_single_out_node(builder, {k}, k_2d, std::monostate{}, "reshape");
  }
  const RotaryParameters params{head_dim, rotary_dim, is_neox ? 1 : 0};
  builder.node(NodeSpec{"gk_rotary_pos_bf16_v1", {q_2d, k_2d, pos_i32, cache},
                        {q_out_2d, k_out_2d}, params});
  if (need_q_reshape) add_single_out_node(builder, {q_out_2d}, q_out, std::monostate{}, "reshape");
  if (need_k_reshape) add_single_out_node(builder, {k_out_2d}, k_out, std::monostate{}, "reshape");
}

void emit_reduction(SynapseGraphBuilder& builder, FxOp op, synTensor input, synTensor output,
                    at::ScalarType dtype, const std::vector<uint64_t>& in_syn_dims,
                    const std::vector<uint64_t>& out_syn_dims, int64_t pt_dim, bool keepdim) {
  const uint64_t syn_axis = syn_axis_from_pt(in_syn_dims.size(), pt_dim);
  std::vector<uint64_t> reduced_dims = in_syn_dims;
  reduced_dims[syn_axis] = 1;
  auto src_f32 = dtype == at::kFloat ? input : emit_cast(builder, input, in_syn_dims, dtype, at::kFloat);
  const bool direct_out = (dtype == at::kFloat && keepdim);
  auto red_f32 = direct_out ? output : builder.tensor(syn_type_float, reduced_dims, -1);
  const char* guid = op == FxOp::sum_dim  ? "reduce_sum_f32"
                     : op == FxOp::mean_dim ? "reduce_mean_f32"
                                            : "reduce_max_f32";
  add_single_out_node(builder, {src_f32}, red_f32,
                      ns_Reduction::Params{static_cast<unsigned int>(syn_axis)}, guid);
  if (direct_out) return;
  auto shaped = red_f32;
  if (!keepdim) {
    if (dtype == at::kFloat) {
      add_single_out_node(builder, {red_f32}, output, std::monostate{}, "reshape");
      return;
    }
    shaped = builder.tensor(syn_type_float, out_syn_dims, -1);
    add_single_out_node(builder, {red_f32}, shaped, std::monostate{}, "reshape");
  }
  emit_cast(builder, shaped, out_syn_dims, at::kFloat, dtype, output);
}

void emit_softmax(SynapseGraphBuilder& builder, bool is_log, synTensor input, synTensor output,
                  at::ScalarType dtype, const std::vector<uint64_t>& syn_dims, int64_t pt_dim) {
  const uint64_t syn_axis = syn_axis_from_pt(syn_dims.size(), pt_dim);
  std::vector<uint64_t> red_dims = syn_dims;
  red_dims[syn_axis] = 1;
  const ns_Reduction::Params red_params{static_cast<unsigned int>(syn_axis)};

  auto x_f32 = dtype == at::kFloat ? input : emit_cast(builder, input, syn_dims, dtype, at::kFloat);
  auto max_val = builder.tensor(syn_type_float, red_dims, -1);
  add_single_out_node(builder, {x_f32}, max_val, red_params, "reduce_max_f32");
  auto max_b = emit_broadcast_to(builder, max_val, syn_type_float, red_dims, syn_dims);
  auto shifted = builder.tensor(syn_type_float, syn_dims, -1);
  add_single_out_node(builder, {x_f32, max_b}, shifted, std::monostate{}, "sub_fwd_f32");

  // P-TPC-01: all exp nodes MUST route through gk_ewise_unary_*_v1 (op_code=0).
  auto exps = builder.tensor(syn_type_float, syn_dims, -1);
  emit_ewise_unary(builder, shifted, exps, at::kFloat, syn_dims,
                   static_cast<int32_t>(FxUnaryOp::exp));
  auto denom = builder.tensor(syn_type_float, red_dims, -1);
  add_single_out_node(builder, {exps}, denom, red_params, "reduce_sum_f32");

  auto result_f32 = dtype == at::kFloat ? output : builder.tensor(syn_type_float, syn_dims, -1);
  if (is_log) {
    auto log_denom = builder.tensor(syn_type_float, red_dims, -1);
    add_single_out_node(builder, {denom}, log_denom, std::monostate{}, "log_fwd_f32");
    auto log_denom_b = emit_broadcast_to(builder, log_denom, syn_type_float, red_dims, syn_dims);
    add_single_out_node(builder, {shifted, log_denom_b}, result_f32, std::monostate{}, "sub_fwd_f32");
  } else {
    auto denom_b = emit_broadcast_to(builder, denom, syn_type_float, red_dims, syn_dims);
    add_single_out_node(builder, {exps, denom_b}, result_f32, std::monostate{}, "div_fwd_f32");
  }
  if (dtype != at::kFloat) {
    emit_cast(builder, result_f32, syn_dims, at::kFloat, dtype, output);
  }
}

void emit_layer_norm(SynapseGraphBuilder& builder, synTensor input, synTensor weight,
                     synTensor bias, synTensor output, synTensor mean_out, synTensor rstd_out,
                     at::ScalarType dtype, const std::vector<uint64_t>& syn_dims, float eps) {
  std::vector<uint64_t> red_dims = syn_dims;
  red_dims[0] = 1;
  const ns_Reduction::Params last_axis{0};
  auto x_f32 = dtype == at::kFloat ? input : emit_cast(builder, input, syn_dims, dtype, at::kFloat);

  auto mean_f32 = (mean_out && dtype == at::kFloat) ? mean_out
                                                   : builder.tensor(syn_type_float, red_dims, -1);
  add_single_out_node(builder, {x_f32}, mean_f32, last_axis, "reduce_mean_f32");
  if (mean_out && dtype != at::kFloat) {
    emit_cast(builder, mean_f32, red_dims, at::kFloat, dtype, mean_out);
  }
  auto mean_b = emit_broadcast_to(builder, mean_f32, syn_type_float, red_dims, syn_dims);
  auto diff = builder.tensor(syn_type_float, syn_dims, -1);
  add_single_out_node(builder, {x_f32, mean_b}, diff, std::monostate{}, "sub_fwd_f32");

  auto sq = builder.tensor(syn_type_float, syn_dims, -1);
  add_single_out_node(builder, {diff, diff}, sq, std::monostate{}, "mult_fwd_f32");
  auto var = builder.tensor(syn_type_float, red_dims, -1);
  add_single_out_node(builder, {sq}, var, last_axis, "reduce_mean_f32");
  auto eps_c = emit_constant(builder, at::kFloat, red_dims, eps, 0, false);
  auto var_eps = builder.tensor(syn_type_float, red_dims, -1);
  add_single_out_node(builder, {var, eps_c}, var_eps, std::monostate{}, "add_fwd_f32");

  auto rstd_f32 = (rstd_out && dtype == at::kFloat) ? rstd_out
                                                   : builder.tensor(syn_type_float, red_dims, -1);
  emit_ewise_unary(builder, var_eps, rstd_f32, at::kFloat, red_dims,
                   static_cast<int32_t>(FxUnaryOp::rsqrt));
  if (rstd_out && dtype != at::kFloat) {
    emit_cast(builder, rstd_f32, red_dims, at::kFloat, dtype, rstd_out);
  }
  auto rstd_b = emit_broadcast_to(builder, rstd_f32, syn_type_float, red_dims, syn_dims);
  auto normed = builder.tensor(syn_type_float, syn_dims, -1);
  add_single_out_node(builder, {diff, rstd_b}, normed, std::monostate{}, "mult_fwd_f32");

  const std::vector<uint64_t> param_dims{syn_dims[0]};
  auto cur = normed;
  if (weight) {
    auto w_f32 = dtype == at::kFloat ? weight : emit_cast(builder, weight, param_dims, dtype, at::kFloat);
    auto w_b = emit_broadcast_to(builder, w_f32, syn_type_float, param_dims, syn_dims);
    auto scaled = (!bias && dtype == at::kFloat) ? output : builder.tensor(syn_type_float, syn_dims, -1);
    add_single_out_node(builder, {cur, w_b}, scaled, std::monostate{}, "mult_fwd_f32");
    cur = scaled;
  }
  if (bias) {
    auto b_f32 = dtype == at::kFloat ? bias : emit_cast(builder, bias, param_dims, dtype, at::kFloat);
    auto b_b = emit_broadcast_to(builder, b_f32, syn_type_float, param_dims, syn_dims);
    auto shifted = dtype == at::kFloat ? output : builder.tensor(syn_type_float, syn_dims, -1);
    add_single_out_node(builder, {cur, b_b}, shifted, std::monostate{}, "add_fwd_f32");
    cur = shifted;
  }
  if (cur != output) {
    emit_cast(builder, cur, syn_dims, at::kFloat, dtype, output);
  }
}

void lower_shard_embedding(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& spec,
                           uint32_t bucket, synTensor row_count,
                           const std::vector<synTensor>& values, synTensor output) {
  const auto type = cast_type(spec.dtype);
  const uint64_t width = spec.sizes.back();
  const auto& se = std::get<FxShardEmbedding>(op.parameters);
  const int32_t org_start = static_cast<int32_t>(se.org_start);
  const int32_t org_end = static_cast<int32_t>(se.org_end);
  const int32_t org_len = org_end - org_start;
  const int32_t added_start = static_cast<int32_t>(se.added_start);
  const int32_t added_end = static_cast<int32_t>(se.added_end);
  const int32_t added_len = added_end - added_start;

  auto tok_in = values[op.inputs[0]];
  auto input = builder.tensor(syn_type_int32, {1, bucket}, -1);
  auto masked = builder.tensor(syn_type_int32, {bucket}, -1);
  auto status = builder.tensor(syn_type_int32, {bucket}, -1);
  add_single_out_node(builder, {tok_in}, input, std::monostate{}, "reshape");
  builder.node(NodeSpec{"gk_scheduler_tokens_i32_v1", {input, row_count},
                        {masked, status}, int32_t{1}});

  auto const_i32 = [&](int32_t value) {
    auto out = builder.tensor(syn_type_int32, {bucket}, -1);
    ns_ConstantKernel::ParamsV3 p{};
    p.constant.i = value;
    add_single_out_node(builder, {}, out, p, "constant_i32");
    return out;
  };
  const auto zero_i32 = const_i32(0);

  auto range_mask = [&](int32_t lo, int32_t hi) {
    auto ge_u8 = builder.tensor(syn_type_uint8, {bucket}, -1);
    auto lt_u8 = builder.tensor(syn_type_uint8, {bucket}, -1);
    auto mask_u8 = builder.tensor(syn_type_uint8, {bucket}, -1);
    add_single_out_node(builder, {masked, const_i32(lo)}, ge_u8, std::monostate{},
                        "greater_equal_fwd_i32");
    add_single_out_node(builder, {masked, const_i32(hi)}, lt_u8, std::monostate{}, "less_fwd_i32");
    add_single_out_node(builder, {ge_u8, lt_u8}, mask_u8, std::monostate{}, "and_fwd_u8");
    return mask_u8;
  };

  synTensor valid_u8 = nullptr;
  synTensor safe_idx = zero_i32;
  if (org_len > 0) {
    auto org_u8 = range_mask(org_start, org_end);
    auto org_idx = org_start == 0 ? masked : builder.tensor(syn_type_int32, {bucket}, -1);
    if (org_start != 0) {
      add_single_out_node(builder, {masked, const_i32(org_start)}, org_idx, std::monostate{},
                          "sub_fwd_i32");
    }
    auto sel_idx = builder.tensor(syn_type_int32, {bucket}, -1);
    add_single_out_node(builder, {org_u8, org_idx, zero_i32}, sel_idx, std::monostate{},
                        "where_fwd_i32");
    safe_idx = sel_idx;
    valid_u8 = org_u8;
  }
  if (added_len > 0) {
    auto add_u8 = range_mask(added_start, added_end);
    const int32_t add_shift = static_cast<int32_t>(se.added_offset);
    auto add_idx = add_shift == 0 ? masked : builder.tensor(syn_type_int32, {bucket}, -1);
    if (add_shift != 0) {
      add_single_out_node(builder, {masked, const_i32(add_shift)}, add_idx, std::monostate{},
                          "sub_fwd_i32");
    }
    auto merged_idx = builder.tensor(syn_type_int32, {bucket}, -1);
    add_single_out_node(builder, {add_u8, add_idx, safe_idx}, merged_idx, std::monostate{},
                        "where_fwd_i32");
    safe_idx = merged_idx;
    if (!valid_u8) {
      valid_u8 = add_u8;
    } else {
      const auto one_i32 = const_i32(1);
      auto org_flag = builder.tensor(syn_type_int32, {bucket}, -1);
      auto any_flag = builder.tensor(syn_type_int32, {bucket}, -1);
      auto any_u8 = builder.tensor(syn_type_uint8, {bucket}, -1);
      add_single_out_node(builder, {valid_u8, one_i32, zero_i32}, org_flag, std::monostate{},
                          "where_fwd_i32");
      add_single_out_node(builder, {add_u8, one_i32, org_flag}, any_flag, std::monostate{},
                          "where_fwd_i32");
      add_single_out_node(builder, {any_flag, one_i32}, any_u8, std::monostate{}, "equal_fwd_i32");
      valid_u8 = any_u8;
    }
  }

  if (!valid_u8) {
    emit_constant(builder, spec.dtype, {width, bucket}, 0.f, 0, false, output);
    return;
  }

  auto gathered = make_tensor_2d(builder, type.syn, width, bucket);
  ns_GatherKernel::ParamsV2 gparams{};
  gparams.axis = 1;
  gparams.batchDims = 0;
  add_single_out_node(builder, {values[op.inputs[1]], safe_idx}, gathered, gparams,
                      "gather_fwd_" + std::string(type.name));

  auto mask_1d_2d = builder.tensor(syn_type_uint8, {1, bucket}, -1);
  auto mask_2d = builder.tensor(syn_type_uint8, {width, bucket}, -1);
  add_single_out_node(builder, {valid_u8}, mask_1d_2d, std::monostate{}, "reshape");
  add_single_out_node(builder, {mask_1d_2d}, mask_2d, std::monostate{}, "broadcast");

  auto zero_embed = emit_constant(builder, spec.dtype, {width, bucket}, 0.f, 0, false);
  add_single_out_node(builder, {mask_2d, gathered, zero_embed}, output, std::monostate{},
                      "where_fwd_" + std::string(type.name));
}

void lower_feature_slice(SynapseGraphBuilder& builder, size_t op_index,
                         const std::vector<int64_t>& roles,
                         const std::vector<FxNode>& operations,
                         const std::vector<int64_t>& outputs,
                         const std::vector<TensorSpec>& specs, uint32_t bucket,
                         std::vector<synTensor>& values) {
  const auto& op = operations[op_index];
  const auto id = roles.size() + op_index;
  const auto type = cast_type(specs[id].dtype);
  const auto src_syn_dims = specs[op.inputs[0]].synapse_sizes(bucket);
  const int64_t total = specs[op.inputs[0]].sizes.back();
  const auto& feature = std::get<FxSlice>(op.parameters);
  struct FeaturePart {
    int64_t start;
    int64_t width;
    size_t id;
  };
  std::vector<FeaturePart> selected{{feature.start, feature.width, id}};
  for (size_t j = op_index + 1; j < operations.size(); ++j) {
    const auto& candidate = operations[j];
    if (candidate.op != FxOp::feature_slice || candidate.inputs[0] != op.inputs[0] ||
        values[roles.size() + j]) {
      continue;
    }
    const auto& other = std::get<FxSlice>(candidate.parameters);
    const auto overlaps = std::any_of(selected.begin(), selected.end(), [&](const FeaturePart& part) {
      return other.start < part.start + part.width && part.start < other.start + other.width;
    });
    if (!overlaps) selected.push_back({other.start, other.width, roles.size() + j});
  }
  std::sort(selected.begin(), selected.end(), [](const FeaturePart& a, const FeaturePart& b) {
    return a.start < b.start;
  });
  auto slice_tensor = [&](int64_t width, int slot) {
    auto dims = src_syn_dims;
    dims[0] = static_cast<uint64_t>(width);
    return builder.tensor(type.syn, dims, slot);
  };
  std::vector<synTensor> parts;
  int64_t offset = 0;
  for (const auto& part : selected) {
    if (part.start > offset) {
      parts.push_back(slice_tensor(part.start - offset, -1));
    }
    if (!values[part.id]) {
      const auto out_it = std::find(outputs.begin(), outputs.end(), static_cast<int64_t>(part.id));
      const int slot = out_it == outputs.end()
                           ? -1
                           : static_cast<int>(roles.size() + std::distance(outputs.begin(), out_it));
      values[part.id] = slice_tensor(part.width, slot);
    }
    parts.push_back(values[part.id]);
    offset = part.start + part.width;
  }
  if (offset < total) {
    parts.push_back(slice_tensor(total - offset, -1));
  }
  synSplitParams params{0};
  builder.node(NodeSpec{"split", {values[op.inputs[0]]}, std::move(parts), params});
}

void lower_slice_dim(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& in_spec,
                     const TensorSpec& /*out_spec*/, uint32_t bucket,
                     synTensor input, synTensor output) {
  const auto& p = std::get<FxSliceDim>(op.parameters);
  const auto syn = syn_dtype_for(in_spec.dtype);
  const auto in_dims = in_spec.synapse_sizes(bucket);
  const int64_t pt_dim = p.dim < 0 ? p.dim + static_cast<int64_t>(in_spec.sizes.size()) : p.dim;
  const unsigned int syn_axis = static_cast<unsigned int>(syn_axis_from_pt(in_dims.size(), pt_dim));
  const int64_t total = in_spec.sizes[pt_dim];
  if (p.start == 0 && p.length == total) {
    add_single_out_node(builder, {input}, output, std::monostate{}, "reshape");
    return;
  }
  std::vector<synTensor> parts;
  auto make_part = [&](uint64_t dim_len) {
    auto d = in_dims;
    d[syn_axis] = dim_len;
    return builder.tensor(syn, d, -1);
  };
  if (p.start > 0) parts.push_back(make_part(static_cast<uint64_t>(p.start)));
  parts.push_back(output);
  const int64_t tail = total - (p.start + p.length);
  if (tail > 0) parts.push_back(make_part(static_cast<uint64_t>(tail)));
  synSplitParams params{syn_axis};
  builder.node(NodeSpec{"split", {input}, std::move(parts), params});
}

void lower_scalar_arith(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& spec,
                        uint32_t bucket, const std::vector<synTensor>& values, synTensor output) {
  const auto syn_dims = spec.synapse_sizes(bucket);
  const auto& scalar_params = std::get<FxScalar>(op.parameters);
  const bool bf16 = spec.dtype == at::kBFloat16;
  const bool wide = bf16 && (op.op == FxOp::scalar_mul || op.op == FxOp::scalar_div);
  const auto arith_dtype = wide ? at::kFloat : spec.dtype;
  auto input = values[op.inputs[0]];
  if (wide) input = emit_cast(builder, input, syn_dims, at::kBFloat16, at::kFloat);
  const bool is_int = (arith_dtype == at::kInt || arith_dtype == at::kLong);
  auto scalar = emit_constant(builder, arith_dtype, syn_dims, scalar_params.value,
                              static_cast<int64_t>(scalar_params.value), is_int);
  const bool reversed = scalar_params.reversed;
  const auto left = reversed ? scalar : input;
  const auto right = reversed ? input : scalar;
  auto result = wide ? builder.tensor(syn_type_float, syn_dims, -1) : output;
  const FxOp binary_op = op.op == FxOp::scalar_add   ? FxOp::add
                         : op.op == FxOp::scalar_sub ? FxOp::sub
                         : op.op == FxOp::scalar_mul ? FxOp::mul
                                                     : FxOp::div;
  emit_binary_pointwise(builder, binary_op, left, right, result, arith_dtype, syn_dims, syn_dims, syn_dims);
  if (wide) emit_cast(builder, result, syn_dims, at::kFloat, at::kBFloat16, output);
}

void lower_scalar_cmp(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& in_spec,
                      const TensorSpec& out_spec, uint32_t bucket,
                      const std::vector<synTensor>& values, synTensor output) {
  const auto syn_dims = in_spec.synapse_sizes(bucket);
  const auto& scalar_params = std::get<FxScalar>(op.parameters);
  const bool is_int = (in_spec.dtype == at::kInt || in_spec.dtype == at::kLong || in_spec.dtype == at::kBool);
  auto scalar = emit_constant(builder, in_spec.dtype, syn_dims, scalar_params.value,
                              static_cast<int64_t>(scalar_params.value), is_int);
  const auto left = scalar_params.reversed ? scalar : values[op.inputs[0]];
  const auto right = scalar_params.reversed ? values[op.inputs[0]] : scalar;
  const FxOp cmp_op = op.op == FxOp::scalar_eq   ? FxOp::eq
                      : op.op == FxOp::scalar_ne ? FxOp::ne
                      : op.op == FxOp::scalar_lt ? FxOp::lt
                      : op.op == FxOp::scalar_le ? FxOp::le
                      : op.op == FxOp::scalar_gt ? FxOp::gt
                                                 : FxOp::ge;
  emit_binary_pointwise(builder, cmp_op, left, right, output, in_spec.dtype,
                        syn_dims, syn_dims, out_spec.synapse_sizes(bucket));
}

void lower_binary_arith(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& spec,
                        const std::vector<TensorSpec>& specs, uint32_t bucket,
                        const std::vector<synTensor>& values, synTensor output) {
  const auto& left_spec = specs[op.inputs[0]];
  const auto& right_spec = specs[op.inputs[1]];
  emit_binary_pointwise(builder, op.op, values[op.inputs[0]], values[op.inputs[1]], output,
                        left_spec.dtype, left_spec.synapse_sizes(bucket),
                        right_spec.synapse_sizes(bucket), spec.synapse_sizes(bucket));
}

void lower_block_fp8_linear(SynapseGraphBuilder& builder, synDeviceId device_id, const FxNode& op,
                            const std::vector<TensorSpec>& specs, uint32_t bucket,
                            const std::vector<synTensor>& values, synTensor output) {
  const auto& input_spec = specs[op.inputs[0]];
  const auto& weight_spec = specs[op.inputs[1]];
  const uint64_t groups = weight_spec.sizes[0];
  const uint64_t columns = weight_spec.sizes[1];
  const uint64_t input_width = groups * 128;
  const uint64_t token_rows = bucket * (input_spec.sizes.size() == 3 ? input_spec.sizes[1] : 1);
  TORCH_CHECK(token_rows > 0 && token_rows <= kMaxBucketRows,
              "GaudiRT FP8: flattened token rows outside 1..8192");
  auto local = [&](synDataType dtype, std::vector<uint64_t> dimensions, bool rmw = false) {
    return builder.tensor(dtype, dimensions, -1, -1, 0, false, {}, rmw);
  };
  if (token_rows >= 128) {
    // Hardware-budgeted single-pass 1K FP16 SRAM-bundled prefill lowering:
    // - Keep the single FP32 accumulator (c0) <= 256 MiB per (row_tile, col_tile)
    //   so wide projections (e.g. N=44032 at M=4096) split into 2048-row tiles
    //   without K-chunk accumulator spills;
    // - Chunk K-groups so each GEMM's FP16 activation operand stays <= 29 MiB,
    //   allowing MantaRayBundlizer to bundle both MME operands directly in on-chip
    //   SRAM with zero HBM prequant intermediates (P-RT-81, P-NUM-42).
    constexpr uint64_t kMaxTileColumns = 65536;
    constexpr uint64_t kMaxAccumElements = 4096ULL * 16384ULL;
    constexpr uint64_t kMaxAct1kBytes = 29ULL << 20;

    const uint64_t raw_col_tiles = (columns + kMaxTileColumns - 1) / kMaxTileColumns;
    const uint64_t tile_columns = std::min<uint64_t>(
        columns, (((columns + raw_col_tiles - 1) / raw_col_tiles + 127) / 128) * 128);
    const uint64_t column_tiles = (columns + tile_columns - 1) / tile_columns;

    uint64_t tile_rows = std::min<uint64_t>(token_rows, 4096ULL);
    if (tile_rows > 2048 && tile_rows * tile_columns > kMaxAccumElements) {
      tile_rows = 2048;
    }
    const uint64_t row_tiles = (token_rows + tile_rows - 1) / tile_rows;

    const uint64_t gc_max = std::max<uint64_t>(1, kMaxAct1kBytes / (256ULL * tile_rows));
    const uint64_t k_chunks = (groups + gc_max - 1) / gc_max;
    const uint64_t gc = (groups + k_chunks - 1) / k_chunks;
    const uint64_t g_cnt = (groups + gc - 1) / gc;

    auto input = values[op.inputs[0]];
    if (input_spec.sizes.size() == 3) {
      auto flat = local(syn_type_bf16, {input_width, token_rows});
      add_single_out_node(builder, {input}, flat, std::monostate{}, "reshape");
      input = flat;
    }
    synTensor bias = nullptr;
    if (op.inputs.size() == 4) {
      bias = values[op.inputs[3]];
      if (specs[op.inputs[3]].dtype == at::kBFloat16) {
        auto wide_bias = local(syn_type_float, {columns});
        ns_CastKernel::ParamsV3 cast{};
        cast.round_mode = CAST_ROUND_HALF_NE;
        add_single_out_node(builder, {bias}, wide_bias, cast, "cast_bf16_to_f32");
        bias = wide_bias;
      }
    }
    const synAxisParams column_axis{0}, feature_axis{1}, group_axis{2};
    int32_t tag_seq = 0;

    std::vector<synTensor> inputs, row_scales;
    inputs.reserve(row_tiles);
    row_scales.reserve(row_tiles);
    for (uint64_t r = 0; r < row_tiles; ++r) {
      const auto rows = std::min(tile_rows, token_rows - r * tile_rows);
      inputs.push_back(row_tiles == 1 ? input : local(syn_type_bf16, {input_width, rows}));
    }
    if (row_tiles > 1) builder.node(NodeSpec{"split", {input}, inputs, feature_axis});
    for (uint64_t r = 0; r < row_tiles; ++r) {
      const auto rows = std::min(tile_rows, token_rows - r * tile_rows);
      auto rs = local(syn_type_float, {1, rows});
      add_single_out_node(builder, {inputs[r]}, rs, tag_seq++, "gk_fp8_act_row_scale_f32_v1");
      row_scales.push_back(rs);
    }

    std::vector<synTensor> weights, weight_scales, blk_scales, biases;
    weights.reserve(column_tiles);
    weight_scales.reserve(column_tiles);
    blk_scales.reserve(column_tiles);
    if (bias) biases.reserve(column_tiles);
    for (uint64_t c = 0; c < column_tiles; ++c) {
      const auto width = std::min(tile_columns, columns - c * tile_columns);
      weights.push_back(column_tiles == 1 ? values[op.inputs[1]]
                                          : local(syn_type_fp8_143, {128, width, groups}));
      weight_scales.push_back(column_tiles == 1 ? values[op.inputs[2]]
                                                : local(syn_type_float, {groups, width / 128}));
      if (bias) biases.push_back(column_tiles == 1 ? bias : local(syn_type_float, {width}));
    }
    if (column_tiles > 1) {
      builder.node(NodeSpec{"split", {values[op.inputs[1]]}, weights, feature_axis});
      builder.node(NodeSpec{"split", {values[op.inputs[2]]}, weight_scales, feature_axis});
      if (bias) builder.node(NodeSpec{"split", {bias}, biases, column_axis});
    }
    for (uint64_t c = 0; c < column_tiles; ++c) {
      const auto width = std::min(tile_columns, columns - c * tile_columns);
      auto bs = local(syn_type_float, {1, width / 128});
      add_single_out_node(builder, {weight_scales[c]}, bs, tag_seq++,
                          "gk_fp8_weight_blk_scale_f32_v1");
      blk_scales.push_back(bs);
    }

    std::vector<std::vector<synTensor>> w_g(column_tiles), ws_g(column_tiles);
    for (uint64_t c = 0; c < column_tiles; ++c) {
      const auto width = std::min(tile_columns, columns - c * tile_columns);
      for (uint64_t g = 0; g < g_cnt; ++g) {
        const auto gl = std::min(gc, groups - g * gc);
        w_g[c].push_back(g_cnt == 1 ? weights[c] : local(syn_type_fp8_143, {128, width, gl}));
        ws_g[c].push_back(g_cnt == 1 ? weight_scales[c]
                                     : local(syn_type_float, {gl, width / 128}));
      }
      if (g_cnt > 1) {
        builder.node(NodeSpec{"split", {weights[c]}, w_g[c], group_axis});
        builder.node(NodeSpec{"split", {weight_scales[c]}, ws_g[c], column_axis});
      }
    }

    const auto flat_output =
        input_spec.sizes.size() == 3 ? local(syn_type_bf16, {columns, token_rows}) : output;
    const synGEMMParams gemm{false, true};
    std::vector<synTensor> row_products;
    row_products.reserve(row_tiles);
    for (uint64_t r = 0; r < row_tiles; ++r) {
      const auto rows = std::min(tile_rows, token_rows - r * tile_rows);
      std::vector<synTensor> col_products;
      col_products.reserve(column_tiles);
      for (uint64_t c = 0; c < column_tiles; ++c) {
        const auto width = std::min(tile_columns, columns - c * tile_columns);
        // Distinct non-negative scalar tag per GEMM prevents Synapse CSE from merging
        // dedicated 1-output TPC producers across tiles or K-chunks (P-RT-81).
        synTensor c0 = nullptr;
        for (uint64_t g = 0; g < g_cnt; ++g) {
          const auto gl = std::min(gc, groups - g * gc);
          const int32_t tile_tag = tag_seq++;
          const Fp8QuantActParameters qa_params{static_cast<int32_t>(g * gc * 128), tile_tag};
          auto act_1k = local(syn_type_fp16, {gl * 128, rows});
          auto w_1k = local(syn_type_fp16, {gl * 128, width});
          auto next_c0 = local(syn_type_float, {width, rows});
          add_single_out_node(builder, {inputs[r], row_scales[r]}, act_1k, qa_params,
                              "gk_fp8_quant_act_fp16_1k_v1");
          add_single_out_node(builder, {w_g[c][g], ws_g[c][g], blk_scales[c]}, w_1k, tile_tag,
                              "gk_fp8_dequant_weight_fp16_1k_v1");
          if (g == 0) {
            add_single_out_node(builder, {act_1k, w_1k}, next_c0, gemm, "gemm");
          } else {
            add_single_out_node(builder, {act_1k, w_1k, c0}, next_c0, gemm, "gemm");
          }
          c0 = next_c0;
        }

        auto tile_out = (row_tiles == 1 && column_tiles == 1)
                            ? flat_output
                            : local(syn_type_bf16, {width, rows});
        if (bias) {
          add_single_out_node(builder, {c0, row_scales[r], blk_scales[c], biases[c]}, tile_out,
                              std::monostate{}, "gk_fp8_epilogue_bias_bf16_v1");
        } else {
          add_single_out_node(builder, {c0, row_scales[r], blk_scales[c]}, tile_out,
                              std::monostate{}, "gk_fp8_epilogue_bf16_v1");
        }
        col_products.push_back(tile_out);
      }
      if (column_tiles == 1) {
        row_products.push_back(col_products[0]);
      } else {
        auto row_out = row_tiles == 1 ? flat_output : local(syn_type_bf16, {columns, rows});
        builder.node(NodeSpec{"concat", col_products, {row_out}, column_axis});
        row_products.push_back(row_out);
      }
    }
    if (row_tiles > 1) {
      builder.node(NodeSpec{"concat", row_products, {flat_output}, feature_axis});
    }
    if (input_spec.sizes.size() == 3) {
      add_single_out_node(builder, {flat_output}, output, std::monostate{}, "reshape");
    }
    return;
  }

  uint64_t rmw_limit = 0;
  synDeviceAttribute attribute = DEVICE_ATTRIBUTE_MAX_RMW_SIZE;
  TORCH_CHECK(synDeviceGetAttribute(&rmw_limit, &attribute, 1, device_id) == synSuccess,
              "GaudiRT FP8: cannot query RMW capacity");
  const uint64_t rows_fit = rmw_limit / (128 * groups * sizeof(float));
  TORCH_CHECK(rows_fit > 0, "GaudiRT FP8: a minimum tile exceeds device RMW capacity");
  const uint64_t tile_rows = std::min<uint64_t>(token_rows, std::bit_floor(rows_fit));
  const uint64_t tile_columns =
      std::min<uint64_t>(columns, (rmw_limit / (tile_rows * groups * sizeof(float)) / 128) * 128);
  TORCH_CHECK(tile_columns >= 128, "GaudiRT FP8: invalid device tile capacity");
  const uint64_t row_tiles = (token_rows + tile_rows - 1) / tile_rows;
  const uint64_t column_tiles = (columns + tile_columns - 1) / tile_columns;
  auto input = values[op.inputs[0]];
  if (input_spec.sizes.size() == 3) {
    auto flat = local(syn_type_bf16, {input_width, token_rows});
    add_single_out_node(builder, {input}, flat, std::monostate{}, "reshape");
    input = flat;
  }
  auto bias = op.inputs.size() == 4 ? values[op.inputs[3]] : local(syn_type_float, {columns});
  if (op.inputs.size() == 3) {
    ns_ConstantKernel::ParamsV3 zero{};
    zero.constant.f = 0.f;
    add_single_out_node(builder, {}, bias, zero, "constant_f32");
  } else if (specs[op.inputs[3]].dtype == at::kBFloat16) {
    auto wide_bias = local(syn_type_float, {columns});
    ns_CastKernel::ParamsV3 cast{};
    cast.round_mode = CAST_ROUND_HALF_NE;
    add_single_out_node(builder, {bias}, wide_bias, cast, "cast_bf16_to_f32");
    bias = wide_bias;
  }
  auto quantized = local(syn_type_fp8_143, {128, token_rows, groups});
  auto activation_scales = local(syn_type_float, {1, token_rows, groups});
  builder.node(NodeSpec{"gk_fp8_block_quant_bf16_v1", {input}, {quantized, activation_scales}, {}});
  std::vector<synTensor> weights, weight_scales, biases, activations, scales, result_rows;
  weights.reserve(column_tiles);
  weight_scales.reserve(column_tiles);
  biases.reserve(column_tiles);
  for (uint64_t i = 0; i < column_tiles; ++i) {
    const auto width = std::min(tile_columns, columns - i * tile_columns);
    weights.push_back(column_tiles == 1 ? values[op.inputs[1]]
                                        : local(syn_type_fp8_143, {128, width, groups}));
    weight_scales.push_back(column_tiles == 1 ? values[op.inputs[2]]
                                              : local(syn_type_float, {groups, width / 128}));
    biases.push_back(column_tiles == 1 ? bias : local(syn_type_float, {width}));
  }
  const synAxisParams feature_axis{1}, column_axis{0};
  if (column_tiles > 1) {
    builder.node(NodeSpec{"split", {values[op.inputs[1]]}, weights, feature_axis});
    builder.node(NodeSpec{"split", {values[op.inputs[2]]}, weight_scales, feature_axis});
    builder.node(NodeSpec{"split", {bias}, biases, column_axis});
  }
  const auto flat_output =
      input_spec.sizes.size() == 3 ? local(syn_type_bf16, {columns, token_rows}) : output;
  for (uint64_t r = 0; r < row_tiles; ++r) {
    const auto height = std::min(tile_rows, token_rows - r * tile_rows);
    activations.push_back(row_tiles == 1 ? quantized : local(syn_type_fp8_143, {128, height, groups}));
    scales.push_back(row_tiles == 1 ? activation_scales : local(syn_type_float, {1, height, groups}));
    result_rows.push_back(row_tiles == 1 ? flat_output : local(syn_type_bf16, {columns, height}));
  }
  if (row_tiles > 1) {
    builder.node(NodeSpec{"split", {quantized}, activations, feature_axis});
    builder.node(NodeSpec{"split", {activation_scales}, scales, feature_axis});
  }
  std::vector<std::vector<synTensor>> products(row_tiles);
  const synGEMMParams gemm{false, true};
  for (uint64_t i = 0; i < column_tiles; ++i) {
    const auto width = std::min(tile_columns, columns - i * tile_columns);
    for (uint64_t r = 0; r < row_tiles; ++r) {
      const auto height = std::min(tile_rows, token_rows - r * tile_rows);
      auto result = column_tiles == 1 ? result_rows[r] : local(syn_type_bf16, {width, height});
      products[r].push_back(result);
      auto partial = local(syn_type_float, {width, height, groups}, true);
      add_single_out_node(builder, {activations[r], weights[i]}, partial, gemm, "batch_gemm");
      add_single_out_node(builder, {partial, scales[r], weight_scales[i], biases[i]}, result,
                          std::monostate{}, "gk_fp8_block_reduce_bf16_v1");
    }
  }
  if (column_tiles > 1) {
    for (uint64_t r = 0; r < row_tiles; ++r) {
      builder.node(NodeSpec{"concat", products[r], {result_rows[r]}, column_axis});
    }
  }
  if (row_tiles > 1) builder.node(NodeSpec{"concat", result_rows, {flat_output}, feature_axis});
  if (input_spec.sizes.size() == 3) {
    add_single_out_node(builder, {flat_output}, output, std::monostate{}, "reshape");
  }
}

void lower_fp8_linear(SynapseGraphBuilder& builder, synDeviceId device_id, const FxNode& op,
                      const std::vector<TensorSpec>& specs, uint32_t bucket,
                      const std::vector<synTensor>& values, synTensor output) {
  const auto& input_spec = specs[op.inputs[0]];
  const auto& weight_spec = specs[op.inputs[1]];
  const uint64_t columns = weight_spec.sizes[0];
  const uint64_t input_width = weight_spec.sizes[1];
  const uint64_t token_rows = bucket * (input_spec.sizes.size() == 3 ? input_spec.sizes[1] : 1);
  TORCH_CHECK(token_rows > 0 && token_rows <= kMaxBucketRows,
              "GaudiRT FP8: flattened token rows outside 1..8192");
  uint64_t rmw_limit = 0;
  synDeviceAttribute attribute = DEVICE_ATTRIBUTE_MAX_RMW_SIZE;
  TORCH_CHECK(synDeviceGetAttribute(&rmw_limit, &attribute, 1, device_id) == synSuccess,
              "GaudiRT FP8: cannot query RMW capacity");
  const uint64_t rows_fit = rmw_limit / (128 * sizeof(float));
  TORCH_CHECK(rows_fit > 0, "GaudiRT FP8: a minimum tile exceeds device RMW capacity");
  const uint64_t tile_rows = std::min<uint64_t>(token_rows, std::bit_floor(rows_fit));
  const uint64_t tile_columns =
      std::min<uint64_t>(columns, (rmw_limit / (tile_rows * sizeof(float)) / 128) * 128);
  TORCH_CHECK(tile_columns >= 128, "GaudiRT FP8: invalid device tile capacity");
  const uint64_t row_tiles = (token_rows + tile_rows - 1) / tile_rows;
  const uint64_t column_tiles = (columns + tile_columns - 1) / tile_columns;
  auto local = [&](synDataType dtype, std::vector<uint64_t> dimensions, bool rmw = false) {
    return builder.tensor(dtype, dimensions, -1, -1, 0, false, {}, rmw);
  };
  auto input = values[op.inputs[0]];
  if (input_spec.sizes.size() == 3) {
    auto flat = local(syn_type_bf16, {input_width, token_rows});
    add_single_out_node(builder, {input}, flat, std::monostate{}, "reshape");
    input = flat;
  }
  auto bias = op.inputs.size() == 4 ? values[op.inputs[3]] : local(syn_type_float, {columns});
  if (op.inputs.size() == 3) {
    ns_ConstantKernel::ParamsV3 zero{};
    zero.constant.f = 0.f;
    add_single_out_node(builder, {}, bias, zero, "constant_f32");
  } else if (specs[op.inputs[3]].dtype == at::kBFloat16) {
    auto wide_bias = local(syn_type_float, {columns});
    ns_CastKernel::ParamsV3 cast{};
    cast.round_mode = CAST_ROUND_HALF_NE;
    add_single_out_node(builder, {bias}, wide_bias, cast, "cast_bf16_to_f32");
    bias = wide_bias;
  }
  auto quantized = local(syn_type_fp8_143, {input_width, token_rows});
  auto activation_scales = local(syn_type_float, {1, token_rows});
  builder.node(NodeSpec{"gk_fp8_token_quant_bf16_v1", {input}, {quantized, activation_scales}, {}});
  std::vector<synTensor> weights, weight_scales, biases, activations, scales, result_rows;
  for (uint64_t i = 0; i < column_tiles; ++i) {
    const auto width = std::min(tile_columns, columns - i * tile_columns);
    weights.push_back(column_tiles == 1 ? values[op.inputs[1]]
                                        : local(syn_type_fp8_143, {input_width, width}));
    weight_scales.push_back(column_tiles == 1 ? values[op.inputs[2]]
                                              : local(syn_type_float, {width}));
    biases.push_back(column_tiles == 1 ? bias : local(syn_type_float, {width}));
  }
  const synAxisParams row_axis{1}, column_axis{0};
  if (column_tiles > 1) {
    builder.node(NodeSpec{"split", {values[op.inputs[1]]}, weights, row_axis});
    builder.node(NodeSpec{"split", {values[op.inputs[2]]}, weight_scales, column_axis});
    builder.node(NodeSpec{"split", {bias}, biases, column_axis});
  }
  const auto flat_output =
      input_spec.sizes.size() == 3 ? local(syn_type_bf16, {columns, token_rows}) : output;
  for (uint64_t r = 0; r < row_tiles; ++r) {
    const auto height = std::min(tile_rows, token_rows - r * tile_rows);
    activations.push_back(row_tiles == 1 ? quantized
                                         : local(syn_type_fp8_143, {input_width, height}));
    scales.push_back(row_tiles == 1 ? activation_scales : local(syn_type_float, {1, height}));
    result_rows.push_back(row_tiles == 1 ? flat_output : local(syn_type_bf16, {columns, height}));
  }
  if (row_tiles > 1) {
    builder.node(NodeSpec{"split", {quantized}, activations, row_axis});
    builder.node(NodeSpec{"split", {activation_scales}, scales, row_axis});
  }
  std::vector<std::vector<synTensor>> products(row_tiles);
  const synGEMMParams gemm{false, true};
  for (uint64_t i = 0; i < column_tiles; ++i) {
    const auto width = std::min(tile_columns, columns - i * tile_columns);
    for (uint64_t r = 0; r < row_tiles; ++r) {
      const auto height = std::min(tile_rows, token_rows - r * tile_rows);
      auto result = column_tiles == 1 ? result_rows[r] : local(syn_type_bf16, {width, height});
      products[r].push_back(result);
      auto partial = local(syn_type_float, {width, height}, true);
      add_single_out_node(builder, {activations[r], weights[i]}, partial, gemm, "gemm");
      add_single_out_node(builder, {partial, scales[r], weight_scales[i], biases[i]}, result,
                          std::monostate{}, "gk_fp8_channel_finish_bf16_v1");
    }
  }
  if (column_tiles > 1) {
    for (uint64_t r = 0; r < row_tiles; ++r) {
      builder.node(NodeSpec{"concat", products[r], {result_rows[r]}, column_axis});
    }
  }
  if (row_tiles > 1) builder.node(NodeSpec{"concat", result_rows, {flat_output}, row_axis});
  if (input_spec.sizes.size() == 3) {
    add_single_out_node(builder, {flat_output}, output, std::monostate{}, "reshape");
  }
}

}  // namespace gaudirt::generic
