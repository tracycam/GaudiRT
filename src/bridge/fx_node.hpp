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
  scalar_mul, scalar_div, block_fp8_linear, fp8_linear,
};

struct FxLinear { bool transpose_weight; };
struct FxSlice { int64_t start, width, total; };
struct FxEpsilon { float value; };
struct FxRotary { int64_t context, head_dim, rotary_dim; bool neox; };
struct FxCast { bool to_float; };
struct FxShape { std::vector<int64_t> dimensions; };
struct FxTranspose { int64_t dim0, dim1; };
struct FxScalar { float value; bool reversed; };
using FxParameters = std::variant<std::monostate, FxLinear, FxSlice, FxEpsilon,
                                  FxRotary, FxCast, FxShape, FxTranspose, FxScalar>;

struct FxNode {
  FxOp op;
  std::vector<int64_t> inputs;
  FxParameters parameters;

  static FxNode unary(FxOp op, int64_t input) { return {op, {input}, {}}; }
  static FxNode binary(FxOp op, int64_t left, int64_t right) { return {op, {left, right}, {}}; }
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
  static FxNode norm(int64_t input, int64_t weight, float epsilon) {
    return {FxOp::rms_norm, {input, weight}, FxEpsilon{epsilon}};
  }
  static FxNode fused_norm(int64_t input, int64_t residual, int64_t weight, float epsilon) {
    return {FxOp::fused_add_rms_norm, {input, residual, weight}, FxEpsilon{epsilon}};
  }
  static FxNode rotary(int64_t q, int64_t k, int64_t cache, int64_t positions, int64_t header,
                       int64_t context, int64_t head_dim, int64_t rotary_dim, bool neox) {
    return {FxOp::rotary, {q, k, cache, positions, header}, FxRotary{context, head_dim, rotary_dim, neox}};
  }
  static FxNode cast(int64_t input, bool to_float) { return {FxOp::cast, {input}, FxCast{to_float}}; }
  static FxNode concat(std::vector<int64_t> inputs) { return {FxOp::concat, std::move(inputs), {}}; }
  static FxNode reshape(int64_t input, std::vector<int64_t> dimensions) {
    return {FxOp::reshape, {input}, FxShape{std::move(dimensions)}};
  }
  static FxNode transpose(int64_t input, int64_t dim0, int64_t dim1) {
    return {FxOp::transpose, {input}, FxTranspose{dim0, dim1}};
  }
  static FxNode scalar(FxOp op, int64_t input, float value, bool reversed) {
    return {op, {input}, FxScalar{value, reversed}};
  }
  void append_normalized(std::ostream& output) const;
};

}  // namespace gaudirt::generic
