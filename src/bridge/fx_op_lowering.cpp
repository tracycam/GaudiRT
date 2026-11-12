// SPDX-License-Identifier: Apache-2.0
#include "fx_op_lowering.hpp"

#include "bridge_internal.hpp"

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

synTensor make_tensor_2d(SynapseGraphBuilder& builder, synDataType type, uint64_t fcd,
                         uint64_t rows, int slot = -1) {
  return builder.tensor(type, {fcd, rows}, slot);
}

void add_single_out_node(SynapseGraphBuilder& builder, std::vector<synTensor> inputs,
                         synTensor output, NodeParameters params, std::string guid) {
  builder.node(NodeSpec{std::move(guid), std::move(inputs), {output}, std::move(params)});
}

}  // namespace

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

  auto input = builder.tensor(syn_type_int32, {1, bucket}, -1);
  auto masked = builder.tensor(syn_type_int32, {bucket}, -1);
  auto status = builder.tensor(syn_type_int32, {bucket}, -1);
  add_single_out_node(builder, {values[op.inputs[0]]}, input, std::monostate{}, "reshape");
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

  ns_ConstantKernel::ParamsV3 zparams{};
  zparams.constant.f = 0.f;
  const auto const_guid = "constant_" + std::string(type.name);
  if (!valid_u8) {
    add_single_out_node(builder, {}, output, zparams, const_guid);
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

  auto zero_embed = make_tensor_2d(builder, type.syn, width, bucket);
  add_single_out_node(builder, {}, zero_embed, zparams, const_guid);
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
  const auto total = specs[op.inputs[0]].sizes[1];
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
  std::vector<synTensor> parts;
  int64_t offset = 0;
  for (const auto& part : selected) {
    if (part.start > offset) {
      parts.push_back(make_tensor_2d(builder, type.syn, part.start - offset, bucket));
    }
    if (!values[part.id]) {
      const auto out_it = std::find(outputs.begin(), outputs.end(), static_cast<int64_t>(part.id));
      const int slot = out_it == outputs.end()
                           ? -1
                           : static_cast<int>(roles.size() + std::distance(outputs.begin(), out_it));
      values[part.id] = make_tensor_2d(builder, type.syn, part.width, bucket, slot);
    }
    parts.push_back(values[part.id]);
    offset = part.start + part.width;
  }
  if (offset < total) {
    parts.push_back(make_tensor_2d(builder, type.syn, total - offset, bucket));
  }
  synSplitParams params{0};
  builder.node(NodeSpec{"split", {values[op.inputs[0]]}, std::move(parts), params});
}

void lower_scalar_arith(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& spec,
                        uint32_t bucket, const std::vector<synTensor>& values, synTensor output) {
  const auto type = cast_type(spec.dtype);
  const uint64_t width = spec.sizes.back();
  const uint64_t rows =
      spec.sizes.size() == 3 ? spec.sizes[1] : (spec.dynamic() ? bucket : spec.sizes[0]);
  const auto& scalar_params = std::get<FxScalar>(op.parameters);
  const bool bf16 = spec.dtype == at::kBFloat16;
  const bool wide = bf16 && (op.op == FxOp::scalar_mul || op.op == FxOp::scalar_div);
  const auto arithmetic_type = wide ? syn_type_float : type.syn;
  const auto arithmetic_name = wide ? std::string("f32") : std::string(type.name);
  auto input = values[op.inputs[0]];
  ns_CastKernel::ParamsV3 cast{};
  cast.round_mode = CAST_ROUND_HALF_NE;
  if (wide) {
    input = make_tensor_2d(builder, syn_type_float, width, rows);
    add_single_out_node(builder, {values[op.inputs[0]]}, input, cast, "cast_bf16_to_f32");
  }
  auto scalar = make_tensor_2d(builder, arithmetic_type, width, rows);
  ns_ConstantKernel::ParamsV3 cparams{};
  cparams.constant.f = scalar_params.value;
  add_single_out_node(builder, {}, scalar, cparams, "constant_" + arithmetic_name);
  const bool reversed = scalar_params.reversed;
  const auto left = reversed ? scalar : input;
  const auto right = reversed ? input : scalar;
  auto result = wide ? make_tensor_2d(builder, syn_type_float, width, rows) : output;
  const auto guid = op.op == FxOp::scalar_add && bf16
                        ? std::string("gk_add_bf16_v1")
                        : std::string(op.op == FxOp::scalar_add   ? "add_fwd_"
                                      : op.op == FxOp::scalar_sub ? "sub_fwd_"
                                      : op.op == FxOp::scalar_mul ? "mult_fwd_"
                                                                  : "div_fwd_") +
                              arithmetic_name;
  add_single_out_node(builder, {left, right}, result, std::monostate{}, guid);
  if (wide) {
    add_single_out_node(builder, {result}, output, std::monostate{}, "gk_cast_f32_to_bf16_v1");
  }
}

void lower_binary_arith(SynapseGraphBuilder& builder, const FxNode& op, const TensorSpec& spec,
                        const std::vector<TensorSpec>& specs, uint32_t bucket,
                        const std::vector<synTensor>& values, synTensor output) {
  const auto type = cast_type(spec.dtype);
  const uint64_t width = spec.sizes.back();
  const uint64_t rows =
      spec.sizes.size() == 3 ? spec.sizes[1] : (spec.dynamic() ? bucket : spec.sizes[0]);
  auto expand = [&](int64_t input_id) {
    const auto& input = specs[input_id];
    if (input.dynamic()) return values[input_id];
    auto source = values[input_id];
    if (input.sizes.size() == 1) {
      auto reshaped = make_tensor_2d(builder, type.syn, width, 1);
      add_single_out_node(builder, {source}, reshaped, std::monostate{}, "reshape");
      source = reshaped;
    }
    auto expanded = make_tensor_2d(builder, type.syn, width, bucket);
    ns_BroadcastNonFcd::Params params{1};
    add_single_out_node(builder, {source}, expanded, params,
                        "broadcast_non_fcd_" + std::string(type.name));
    return expanded;
  };
  auto left = expand(op.inputs[0]);
  auto right = expand(op.inputs[1]);
  if (op.op == FxOp::add) {
    const auto guid = spec.dtype == at::kBFloat16 ? std::string("gk_add_bf16_v1")
                                                  : "add_fwd_" + std::string(type.name);
    add_single_out_node(builder, {left, right}, output, std::monostate{}, guid);
    return;
  }
  if (op.op == FxOp::div && spec.dtype == at::kBFloat16) {
    auto numerator = make_tensor_2d(builder, syn_type_float, width, rows);
    auto denominator = make_tensor_2d(builder, syn_type_float, width, rows);
    auto quotient = make_tensor_2d(builder, syn_type_float, width, rows);
    ns_CastKernel::ParamsV3 params{};
    params.round_mode = CAST_ROUND_HALF_NE;
    add_single_out_node(builder, {left}, numerator, params, "cast_bf16_to_f32");
    add_single_out_node(builder, {right}, denominator, params, "cast_bf16_to_f32");
    add_single_out_node(builder, {numerator, denominator}, quotient, std::monostate{}, "div_fwd_f32");
    add_single_out_node(builder, {quotient}, output, std::monostate{}, "gk_cast_f32_to_bf16_v1");
    return;
  }
  const auto operation =
      op.op == FxOp::sub ? "sub_fwd_" : (op.op == FxOp::mul ? "mult_fwd_" : "div_fwd_");
  add_single_out_node(builder, {left, right}, output, std::monostate{},
                      std::string(operation) + type.name);
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
  TORCH_CHECK(token_rows > 0 && token_rows <= 4096,
              "GaudiRT FP8: flattened token rows outside 1..4096");
  auto local = [&](synDataType dtype, std::vector<uint64_t> dimensions, bool rmw = false) {
    return builder.tensor(dtype, dimensions, -1, -1, 0, false, {}, rmw);
  };
  if (token_rows >= 512) {
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
  TORCH_CHECK(token_rows > 0 && token_rows <= 4096,
              "GaudiRT FP8: flattened token rows outside 1..4096");
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
