// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>
#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <cstdint>
#include <initializer_list>
#include <string>

namespace gaudirt::mimo {

// Three compensated products preserve the high/low BF16 split without a
// persistent decoded weight. Callers supply the appropriate activation split.
template <class Intermediate, class Node>
void append_block_fp8_mme(Intermediate tensor, Node node, synTensor x_hi,
                          synTensor x_lo, synTensor w_hi, synTensor w_lo,
                          synTensor bias, synTensor output, uint64_t width,
                          uint64_t rows, bool round_bf16, const char* prefix) {
  const auto name = [&](const char* suffix) { return std::string(prefix) + suffix; };
  auto make = [&](const char* suffix) {
    return tensor(name(suffix).c_str(), syn_type_float, {width, rows});
  };
  synTensor main = make("_main");
  synTensor extra = make("_extra");
  synTensor weight_extra = make("_weight_extra");
  synTensor combined = make("_combined");
  synTensor combined_all = make("_combined_all");
  const synGEMMParams params{false, true};
  node({x_hi, w_hi}, {main}, &params, sizeof(params), "gemm", name("_main").c_str());
  node({x_lo, w_hi}, {extra}, &params, sizeof(params), "gemm", name("_extra").c_str());
  node({x_hi, w_lo}, {weight_extra}, &params, sizeof(params), "gemm", name("_weight_extra").c_str());
  node({main, extra}, {combined}, nullptr, 0, "add_fwd_f32", name("_combine").c_str());
  node({combined, weight_extra}, {combined_all}, nullptr, 0, "add_fwd_f32", name("_combine_weight").c_str());
  synTensor biased = round_bf16 ? make("_biased") : output;
  node({combined_all, bias}, {biased}, nullptr, 0, "add_fwd_f32", name("_bias").c_str());
  if (round_bf16) {
    ns_CastKernel::ParamsV3 cast{};
    cast.round_mode = CAST_ROUND_HALF_NE;
    node({biased}, {output}, &cast, sizeof(cast), "cast_f32_to_bf16", name("_round").c_str());
  }
}

}  // namespace gaudirt::mimo
