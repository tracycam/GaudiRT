// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <optional>
#include <ostream>
#include <utility>
#include <variant>
#include <vector>

namespace gaudirt::generic {

enum class FxOp {
  linear, add, rms_norm, silu_and_mul, feature_slice, fused_add_rms_norm,
  fused_residual, rotary, rotary_key, embedding, gather_rows, cast, sub, mul,
  div, concat, bmm, reshape, transpose, copy, scalar_add, scalar_sub,
  scalar_mul, scalar_div, block_fp8_linear, fp8_linear, shard_embedding,
  ewise_unary, gelu_and_mul, gemma_rms_norm, fused_add_gemma_rms_norm,
  rotary_pos, minimum, maximum, where, eq, ne, lt, le, gt, ge,
  scalar_eq, scalar_ne, scalar_lt, scalar_le, scalar_gt, scalar_ge,
  isnan, logical_and, logical_or, logical_not, bitwise_and, bitwise_or,
  expand, slice_dim, sum_dim, mean_dim, amax_dim, softmax, log_softmax,
  layer_norm, full_like,
};

enum class FxUnaryOp : int32_t {
  exp = 0,
  tanh = 1,
  sigmoid = 2,
  silu = 3,
  gelu_exact = 4,
  gelu_tanh = 5,
  softcap = 6,
  clamp = 7,
  rsqrt = 8,
  sqrt = 9,
  neg = 10,
  abs = 11,
};

struct FxLinear { bool transpose_weight; };
struct FxSlice { int64_t start, width, total; };
struct FxEpsilon { float value; };
struct FxRotary { int64_t context, head_dim, rotary_dim; bool neox; };
struct FxCast { bool to_float; };
struct FxCastDtype { int32_t scalar_type; };
struct FxShape { std::vector<int64_t> dimensions; };
struct FxTranspose { int64_t dim0, dim1; };
struct FxScalar { float value; bool reversed; };
struct FxShardEmbedding { int64_t org_start, org_end, added_start, added_end, added_offset; };
struct FxUnary { int32_t op_code; float alpha; float beta; };
struct FxGeluMul { int32_t approximate; };
struct FxAxis { int64_t dim; };
struct FxReduce { int64_t dim; bool keepdim; };
struct FxLayerNorm { bool has_weight; bool has_bias; float epsilon; };
struct FxSliceDim { int64_t dim, start, length; };
struct FxConstant { float float_value; int64_t int_value; int32_t scalar_type; bool is_integral; };
using FxParameters = std::variant<std::monostate, FxLinear, FxSlice, FxEpsilon,
                                  FxRotary, FxCast, FxCastDtype, FxShape, FxTranspose, FxScalar,
                                  FxShardEmbedding, FxUnary, FxGeluMul, FxAxis, FxReduce,
                                  FxLayerNorm, FxSliceDim, FxConstant>;

struct FxNode {
  FxOp op;
  std::vector<int64_t> inputs;
  FxParameters parameters;

  static FxNode unary(FxOp op, int64_t input) { return {op, {input}, {}}; }
  static FxNode binary(FxOp op, int64_t left, int64_t right) { return {op, {left, right}, {}}; }
  static FxNode ternary(FxOp op, int64_t a, int64_t b, int64_t c) { return {op, {a, b, c}, {}}; }
  static FxNode linear(int64_t input, int64_t weight, std::optional<int64_t> bias, bool transpose) {
    std::vector<int64_t> inputs{input, weight};
    if (bias) inputs.push_back(*bias);
    return {FxOp::linear, std::move(inputs), FxLinear{transpose}};
  }
  static FxNode block_fp8(int64_t input, int64_t weight, int64_t scales, std::optional<int64_t> bias) {
    std::vector<int64_t> inputs{input, weight, scales};
    if (bias) inputs.push_back(*bias);
    return {FxOp::block_fp8_linear, std::move(inputs), {}};
  }
  static FxNode fp8(int64_t input, int64_t weight, int64_t scales, std::optional<int64_t> bias) {
    std::vector<int64_t> inputs{input, weight, scales};
    if (bias) inputs.push_back(*bias);
    return {FxOp::fp8_linear, std::move(inputs), {}};
  }
  static FxNode slice(int64_t input, int64_t start, int64_t width, int64_t total) {
    return {FxOp::feature_slice, {input}, FxSlice{start, width, total}};
  }
  static FxNode slice_dim(int64_t input, int64_t dim, int64_t start, int64_t length) {
    return {FxOp::slice_dim, {input}, FxSliceDim{dim, start, length}};
  }
  static FxNode norm(int64_t input, int64_t weight, float epsilon) {
    return {FxOp::rms_norm, {input, weight}, FxEpsilon{epsilon}};
  }
  static FxNode gemma_norm(int64_t input, int64_t weight, float epsilon) {
    return {FxOp::gemma_rms_norm, {input, weight}, FxEpsilon{epsilon}};
  }
  static FxNode fused_norm(int64_t input, int64_t residual, int64_t weight, float epsilon) {
    return {FxOp::fused_add_rms_norm, {input, residual, weight}, FxEpsilon{epsilon}};
  }
  static FxNode fused_gemma_norm(int64_t input, int64_t residual, int64_t weight, float epsilon) {
    return {FxOp::fused_add_gemma_rms_norm, {input, residual, weight}, FxEpsilon{epsilon}};
  }
  static FxNode rotary(int64_t q, int64_t k, int64_t cache, int64_t positions, int64_t header,
                       int64_t context, int64_t head_dim, int64_t rotary_dim, bool neox) {
    return {FxOp::rotary, {q, k, cache, positions, header}, FxRotary{context, head_dim, rotary_dim, neox}};
  }
  static FxNode rotary_pos(int64_t q, int64_t k, int64_t cache, int64_t positions,
                           int64_t head_dim, int64_t rotary_dim, bool neox) {
    return {FxOp::rotary_pos, {q, k, cache, positions}, FxRotary{0, head_dim, rotary_dim, neox}};
  }
  static FxNode cast(int64_t input, bool to_float) { return {FxOp::cast, {input}, FxCast{to_float}}; }
  static FxNode cast_dtype(int64_t input, int32_t scalar_type) {
    return {FxOp::cast, {input}, FxCastDtype{scalar_type}};
  }
  static FxNode concat(std::vector<int64_t> inputs) { return {FxOp::concat, std::move(inputs), {}}; }
  static FxNode concat_dim(std::vector<int64_t> inputs, int64_t dim) {
    return {FxOp::concat, std::move(inputs), FxAxis{dim}};
  }
  static FxNode reshape(int64_t input, std::vector<int64_t> dimensions) {
    return {FxOp::reshape, {input}, FxShape{std::move(dimensions)}};
  }
  static FxNode expand(int64_t input, std::vector<int64_t> dimensions) {
    return {FxOp::expand, {input}, FxShape{std::move(dimensions)}};
  }
  static FxNode transpose(int64_t input, int64_t dim0, int64_t dim1) {
    return {FxOp::transpose, {input}, FxTranspose{dim0, dim1}};
  }
  static FxNode scalar(FxOp op, int64_t input, float value, bool reversed) {
    return {op, {input}, FxScalar{value, reversed}};
  }
  static FxNode ewise_unary(int64_t input, int32_t op_code, float alpha = 0.f, float beta = 0.f) {
    return {FxOp::ewise_unary, {input}, FxUnary{op_code, alpha, beta}};
  }
  static FxNode gelu_and_mul(int64_t input, int32_t approximate) {
    return {FxOp::gelu_and_mul, {input}, FxGeluMul{approximate}};
  }
  static FxNode reduce(FxOp op, int64_t input, int64_t dim, bool keepdim) {
    return {op, {input}, FxReduce{dim, keepdim}};
  }
  static FxNode softmax(FxOp op, int64_t input, int64_t dim) {
    return {op, {input}, FxAxis{dim}};
  }
  static FxNode layer_norm(int64_t input, std::optional<int64_t> weight,
                           std::optional<int64_t> bias, float epsilon) {
    std::vector<int64_t> inputs{input};
    if (weight) inputs.push_back(*weight);
    if (bias) inputs.push_back(*bias);
    return {FxOp::layer_norm, std::move(inputs),
            FxLayerNorm{weight.has_value(), bias.has_value(), epsilon}};
  }
  static FxNode full_like(int64_t ref, float float_value, int64_t int_value,
                          int32_t scalar_type, bool is_integral) {
    return {FxOp::full_like, {ref}, FxConstant{float_value, int_value, scalar_type, is_integral}};
  }
  static FxNode shard_embedding(int64_t input, int64_t weight,
                                int64_t org_start, int64_t org_end,
                                int64_t added_start, int64_t added_end,
                                int64_t added_offset) {
    return {FxOp::shard_embedding, {input, weight},
            FxShardEmbedding{org_start, org_end, added_start, added_end, added_offset}};
  }
  void append_normalized(std::ostream& output) const;
};

}  // namespace gaudirt::generic
