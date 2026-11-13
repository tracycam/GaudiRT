// SPDX-License-Identifier: Apache-2.0
#include "fx_geometry.hpp"

#include "bucket_program.hpp"
#include "gaudirt/common/sha256.hpp"

#include <c10/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace gaudirt::generic {
namespace {

int64_t resolve_axis(int64_t dim, size_t rank, const char* context) {
  const auto r = static_cast<int64_t>(rank);
  TORCH_CHECK(dim >= -r && dim < r, context);
  return dim < 0 ? dim + r : dim;
}

bool is_supported_value_dtype(at::ScalarType dtype) {
  return dtype == at::kBFloat16 || dtype == at::kFloat || dtype == at::kInt ||
         dtype == at::kLong || dtype == at::kBool;
}

TensorSpec broadcast_specs(const TensorSpec& a, const TensorSpec& b, at::ScalarType out_dtype) {
  const size_t rank = std::max(a.sizes.size(), b.sizes.size());
  TORCH_CHECK(rank >= 1 && rank <= 5, "GaudiRT FX: broadcast rank must be 1..5");
  std::vector<int64_t> out(rank);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t da = (i + a.sizes.size() >= rank) ? a.sizes[i + a.sizes.size() - rank] : 1;
    const int64_t db = (i + b.sizes.size() >= rank) ? b.sizes[i + b.sizes.size() - rank] : 1;
    if (da == -1 || db == -1) {
      TORCH_CHECK(i == 0 && (da == db || da == 1 || db == 1),
                  "GaudiRT FX: unsupported dynamic broadcast geometry");
      out[i] = -1;
    } else {
      TORCH_CHECK(da == db || da == 1 || db == 1,
                  "GaudiRT FX: incompatible elementwise broadcast dimensions");
      out[i] = std::max(da, db);
    }
  }
  const bool dynamic = !out.empty() && out[0] == -1;
  return {out_dtype, std::move(out), dynamic};
}

}  // namespace

std::vector<bool> rotary_cache_inputs(const std::vector<int64_t>& roles,
                                      const std::vector<FxNode>& operations) {
  std::vector<bool> caches(roles.size(), false), other_uses(roles.size(), false);
  for (const auto& op : operations) {
    for (size_t index = 0; index < op.inputs.size(); ++index) {
      const auto id = op.inputs[index];
      if (id < 0 || id >= static_cast<int64_t>(roles.size())) continue;
      if (op.op == FxOp::rotary && index == 2 &&
          std::get<FxRotary>(op.parameters).context <= static_cast<int64_t>(kRotaryStaticRows))
        caches[id] = true;
      else
        other_uses[id] = true;
    }
  }
  for (size_t i = 0; i < caches.size(); ++i) caches[i] = caches[i] && !roles[i] && !other_uses[i];
  return caches;
}

FxGeometry infer_fx_geometry(const std::vector<at::Tensor>& inputs, const std::vector<int64_t>& roles,
                             const std::vector<FxNode>& operations, const std::vector<int64_t>& outputs) {
  TORCH_CHECK(inputs.size() == roles.size(), "GaudiRT FX: wrong input count");
  std::vector<TensorSpec> specs;
  specs.reserve(inputs.size() + operations.size());
  std::ostringstream normalized;
  const auto caches = rotary_cache_inputs(roles, operations);
  normalized << (std::any_of(caches.begin(), caches.end(), [](bool value) { return value; })
                     ? "fx_program_rotary_static_v3;" : "fx_program_v17;");
  if (std::any_of(operations.begin(), operations.end(),
                  [](const FxNode& op) { return op.op == FxOp::block_fp8_linear; })) {
    normalized << "block_fp8_fp16_1k_v1;";
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& tensor = inputs[i];
    const bool valid_dtype = is_supported_value_dtype(tensor.scalar_type());
    const bool prepared_fp8 = tensor.scalar_type() == at::kFloat8_e4m3fn;
    TORCH_CHECK(tensor.device().type() == c10::kPrivateUse1 && tensor.device().index() >= 0 &&
                tensor.device().index() == inputs[0].device().index() &&
                (valid_dtype || (prepared_fp8 && !roles[i])),
                "GaudiRT FX: unsupported input device or dtype");
    TORCH_CHECK(tensor.dim() >= 1 && tensor.dim() <= 5, "GaudiRT FX: unsupported input rank");
    std::vector<int64_t> sizes(tensor.sizes().begin(), tensor.sizes().end());
    for (auto size : sizes) TORCH_CHECK(size > 0, "GaudiRT FX: empty input dimension");
    if (roles[i]) sizes[0] = -1;
    normalized << "i" << roles[i] << ":" << static_cast<int>(tensor.scalar_type());
    for (size_t dim = 0; dim < sizes.size(); ++dim) {
      if (caches[i] && dim == 0) normalized << ":rotary_capacity";
      else normalized << ":" << sizes[dim];
    }
    normalized << ";";
    specs.push_back({tensor.scalar_type(), std::move(sizes), roles[i] == 1});
  }
  for (const auto& op : operations) {
    op.append_normalized(normalized);
    const auto& a = specs[op.inputs[0]];
    if (op.op == FxOp::scalar_add || op.op == FxOp::scalar_sub ||
        op.op == FxOp::scalar_mul || op.op == FxOp::scalar_div) {
      const float val = std::get<FxScalar>(op.parameters).value;
      const bool ok_dtype = a.dtype == at::kBFloat16 || a.dtype == at::kFloat ||
                            ((op.op != FxOp::scalar_div) && (a.dtype == at::kInt || a.dtype == at::kLong));
      TORCH_CHECK(a.sizes.size() >= 1 && a.sizes.size() <= 5 && ok_dtype && std::isfinite(val),
                  "GaudiRT FX: scalar arithmetic requires a finite scalar and rank 1..5 tensor");
      specs.push_back(a);
      continue;
    }
    if (op.op == FxOp::scalar_eq || op.op == FxOp::scalar_ne || op.op == FxOp::scalar_lt ||
        op.op == FxOp::scalar_le || op.op == FxOp::scalar_gt || op.op == FxOp::scalar_ge) {
      const float val = std::get<FxScalar>(op.parameters).value;
      TORCH_CHECK(a.sizes.size() >= 1 && a.sizes.size() <= 5 && is_supported_value_dtype(a.dtype) &&
                  !std::isnan(val),
                  "GaudiRT FX: scalar comparison requires non-NaN scalar and rank 1..5 tensor");
      specs.push_back({at::kBool, a.sizes, a.dynamic()});
      continue;
    }
    if (op.op == FxOp::ewise_unary) {
      const auto& p = std::get<FxUnary>(op.parameters);
      const bool is_int_neg_abs = (a.dtype == at::kInt) && (p.op_code == static_cast<int32_t>(FxUnaryOp::neg) ||
                                                             p.op_code == static_cast<int32_t>(FxUnaryOp::abs));
      TORCH_CHECK(a.sizes.size() >= 1 && a.sizes.size() <= 5 &&
                  (a.dtype == at::kBFloat16 || a.dtype == at::kFloat || is_int_neg_abs),
                  "GaudiRT FX: ewise_unary requires BF16/FP32 (or I32 neg/abs) rank 1..5 tensor");
      TORCH_CHECK(p.op_code >= 0 && p.op_code <= 11, "GaudiRT FX: unsupported ewise_unary opcode");
      if (p.op_code == static_cast<int32_t>(FxUnaryOp::softcap)) {
        TORCH_CHECK(std::isfinite(p.alpha) && p.alpha > 0.f, "GaudiRT FX: softcap requires finite positive cap");
      } else if (p.op_code == static_cast<int32_t>(FxUnaryOp::clamp)) {
        TORCH_CHECK(!std::isnan(p.alpha) && !std::isnan(p.beta) && p.alpha <= p.beta,
                    "GaudiRT FX: clamp requires ordered non-NaN bounds");
      }
      specs.push_back(a);
      continue;
    }
    if (op.op == FxOp::isnan) {
      TORCH_CHECK(a.sizes.size() >= 1 && a.sizes.size() <= 5 &&
                  (a.dtype == at::kBFloat16 || a.dtype == at::kFloat),
                  "GaudiRT FX: isnan requires BF16 or FP32 tensor");
      specs.push_back({at::kBool, a.sizes, a.dynamic()});
      continue;
    }
    if (op.op == FxOp::logical_not) {
      TORCH_CHECK(a.sizes.size() >= 1 && a.sizes.size() <= 5 && a.dtype == at::kBool,
                  "GaudiRT FX: logical_not requires boolean tensor");
      specs.push_back(a);
      continue;
    }
    if (op.op == FxOp::copy) {
      TORCH_CHECK(a.sizes.size() >= 1 && a.sizes.size() <= 5 && is_supported_value_dtype(a.dtype),
                  "GaudiRT FX: copy requires rank 1..5 supported tensor");
      specs.push_back(a);
      continue;
    }
    if (op.op == FxOp::full_like) {
      const auto& c = std::get<FxConstant>(op.parameters);
      const auto dtype = static_cast<at::ScalarType>(c.scalar_type);
      TORCH_CHECK(is_supported_value_dtype(dtype) && a.sizes.size() >= 1 && a.sizes.size() <= 5,
                  "GaudiRT FX: invalid full_like constant target");
      specs.push_back({dtype, a.sizes, a.dynamic()});
      continue;
    }
    if (op.op == FxOp::transpose) {
      const auto& params = std::get<FxTranspose>(op.parameters);
      TORCH_CHECK(a.sizes.size() >= 2 && a.sizes.size() <= 5 &&
                  params.dim0 >= 0 && params.dim0 < static_cast<int64_t>(a.sizes.size()) &&
                  params.dim1 >= 0 && params.dim1 < static_cast<int64_t>(a.sizes.size()) &&
                  (!a.dynamic() || (params.dim0 != 0 && params.dim1 != 0)),
                  "GaudiRT FX: invalid feature transpose geometry");
      auto sizes = a.sizes;
      std::swap(sizes[params.dim0], sizes[params.dim1]);
      specs.push_back({a.dtype, std::move(sizes), a.dynamic()});
      continue;
    }
    if (op.op == FxOp::reshape) {
      TORCH_CHECK(!a.sizes.empty() && a.sizes.size() <= 5, "GaudiRT FX: invalid reshape input");
      const auto& dims = std::get<FxShape>(op.parameters).dimensions;
      if (a.dynamic()) {
        TORCH_CHECK(dims.size() <= 4, "GaudiRT FX: invalid reshape rank");
        int64_t source_width = 1, target_width = 1;
        for (size_t i = 1; i < a.sizes.size(); ++i) {
          TORCH_CHECK(source_width <= INT64_MAX / a.sizes[i], "GaudiRT FX: reshape input is too large");
          source_width *= a.sizes[i];
        }
        std::vector<int64_t> sizes{-1};
        for (auto size : dims) {
          TORCH_CHECK(size > 0 && target_width <= INT64_MAX / size, "GaudiRT FX: reshape output is too large");
          target_width *= size;
          sizes.push_back(size);
        }
        TORCH_CHECK(sizes.size() <= 5 && source_width == target_width,
                    "GaudiRT FX: reshape changes the token width");
        specs.push_back({a.dtype, std::move(sizes), true});
      } else {
        TORCH_CHECK(!dims.empty() && dims.size() <= 5, "GaudiRT FX: invalid reshape rank");
        int64_t source_numel = 1, target_numel = 1;
        for (auto s : a.sizes) {
          TORCH_CHECK(source_numel <= INT64_MAX / s, "GaudiRT FX: reshape overflow");
          source_numel *= s;
        }
        for (auto s : dims) {
          TORCH_CHECK(s > 0 && target_numel <= INT64_MAX / s, "GaudiRT FX: invalid static reshape dim");
          target_numel *= s;
        }
        TORCH_CHECK(source_numel == target_numel, "GaudiRT FX: static reshape element count mismatch");
        specs.push_back({a.dtype, dims, false});
      }
      continue;
    }
    if (op.op == FxOp::expand) {
      const auto& dims = std::get<FxShape>(op.parameters).dimensions;
      TORCH_CHECK(dims.size() >= a.sizes.size() && dims.size() <= 5,
                  "GaudiRT FX: expand rank must be >= source rank and <= 5");
      const size_t offset = dims.size() - a.sizes.size();
      for (size_t i = 0; i < dims.size(); ++i) {
        const int64_t src = i >= offset ? a.sizes[i - offset] : 1;
        const int64_t dst = dims[i];
        if (i == 0 && dst == -1) {
          TORCH_CHECK(src == -1 || src == 1, "GaudiRT FX: expand dynamic dim requires source dim 1 or -1");
        } else {
          TORCH_CHECK(dst > 0 && (src == dst || src == 1),
                      "GaudiRT FX: incompatible expand dimension");
        }
      }
      specs.push_back({a.dtype, dims, dims[0] == -1});
      continue;
    }
    if (op.op == FxOp::concat) {
      TORCH_CHECK(op.inputs.size() >= 2 && a.sizes.size() >= 1 && a.sizes.size() <= 5,
                  "GaudiRT FX: invalid cat input");
      const int64_t raw_dim = std::holds_alternative<FxAxis>(op.parameters)
                                  ? std::get<FxAxis>(op.parameters).dim
                                  : static_cast<int64_t>(a.sizes.size()) - 1;
      const int64_t dim = resolve_axis(raw_dim, a.sizes.size(), "GaudiRT FX: invalid cat axis");
      TORCH_CHECK(!a.dynamic() || dim != 0, "GaudiRT FX: cannot concat along dynamic token axis");
      int64_t cat_size = 0;
      for (auto input_id : op.inputs) {
        const auto& input = specs[input_id];
        TORCH_CHECK(input.dtype == a.dtype && input.sizes.size() == a.sizes.size() &&
                    input.dynamic() == a.dynamic() &&
                    input.sizes[dim] <= INT64_MAX - cat_size,
                    "GaudiRT FX: invalid cat input geometry");
        for (size_t d = 0; d < a.sizes.size(); ++d) {
          if (static_cast<int64_t>(d) != dim) {
            TORCH_CHECK(input.sizes[d] == a.sizes[d], "GaudiRT FX: mismatched cat dimension");
          }
        }
        cat_size += input.sizes[dim];
      }
      auto out_sizes = a.sizes;
      out_sizes[dim] = cat_size;
      specs.push_back({a.dtype, std::move(out_sizes), a.dynamic()});
      continue;
    }
    if (op.op == FxOp::gather_rows) {
      const auto& index = specs[op.inputs[1]];
      TORCH_CHECK(a.dynamic() && a.sizes.size() >= 2 && a.sizes.size() <= 5 &&
                  !index.dynamic() && (index.dtype == at::kInt || index.dtype == at::kLong) &&
                  index.sizes.size() == 1 && index.sizes[0] > 0 && index.sizes[0] <= kMaxBucketRows,
                  "GaudiRT FX: invalid validated row gather geometry");
      auto sizes = a.sizes;
      sizes[0] = index.sizes[0];
      specs.push_back({a.dtype, std::move(sizes), false});
      continue;
    }
    if (op.op == FxOp::cast) {
      at::ScalarType dtype;
      if (std::holds_alternative<FxCastDtype>(op.parameters)) {
        dtype = static_cast<at::ScalarType>(std::get<FxCastDtype>(op.parameters).scalar_type);
        TORCH_CHECK(is_supported_value_dtype(dtype) && is_supported_value_dtype(a.dtype),
                    "GaudiRT FX: unsupported cast dtype");
      } else {
        dtype = std::get<FxCast>(op.parameters).to_float ? at::kFloat : at::kBFloat16;
        TORCH_CHECK((a.dtype == at::kBFloat16 || a.dtype == at::kFloat) && a.dtype != dtype,
                    "GaudiRT FX: invalid BF16/FP32 cast");
      }
      TORCH_CHECK(a.sizes.size() >= 1 && a.sizes.size() <= 5, "GaudiRT FX: invalid cast rank");
      specs.push_back({dtype, a.sizes, a.dynamic()});
      continue;
    }
    if (op.op == FxOp::embedding) {
      const auto& weight = specs[op.inputs[1]];
      TORCH_CHECK((a.dtype == at::kInt || a.dtype == at::kLong) && a.dynamic() && a.sizes.size() == 1 &&
                  (weight.dtype == at::kBFloat16 || weight.dtype == at::kFloat) &&
                  !weight.dynamic() && weight.sizes.size() == 2,
                  "GaudiRT FX: invalid validated embedding geometry");
      specs.push_back({weight.dtype, {-1, weight.sizes[1]}, true});
      continue;
    }
    if (op.op == FxOp::shard_embedding) {
      const auto& weight = specs[op.inputs[1]];
      const auto& p = std::get<FxShardEmbedding>(op.parameters);
      const int64_t min_rows =
          (p.added_end > p.added_start) ? (p.added_end - p.added_offset) : (p.org_end - p.org_start);
      TORCH_CHECK((a.dtype == at::kInt || a.dtype == at::kLong) && a.dynamic() && a.sizes.size() == 1 &&
                  (weight.dtype == at::kBFloat16 || weight.dtype == at::kFloat) &&
                  !weight.dynamic() && weight.sizes.size() == 2 &&
                  weight.sizes[0] > 0 && weight.sizes[0] >= min_rows,
                  "GaudiRT FX: invalid shard_embedding geometry");
      specs.push_back({weight.dtype, {-1, weight.sizes[1]}, true});
      continue;
    }
    if (op.op == FxOp::rotary_key) {
      specs.push_back(specs[operations[op.inputs[0] - roles.size()].inputs[1]]);
      continue;
    }
    if (op.op == FxOp::rotary) {
      const auto& params = std::get<FxRotary>(op.parameters);
      const auto& key = specs[op.inputs[1]];
      const auto& cache = specs[op.inputs[2]];
      const auto& positions = specs[op.inputs[3]];
      const auto& header = specs[op.inputs[4]];
      TORCH_CHECK(a.dtype == at::kBFloat16 && a.dynamic() && a.sizes.size() == 2 &&
                  key.dtype == a.dtype && key.dynamic() && key.sizes.size() == 2 &&
                  !cache.dynamic() && cache.dtype == a.dtype && cache.sizes.size() == 2 &&
                  cache.sizes[0] >= params.context && cache.sizes[1] == params.rotary_dim &&
                  positions.dtype == at::kInt && positions.dynamic() && positions.sizes.size() == 1 &&
                  header.dtype == at::kInt && !header.dynamic() && header.sizes == std::vector<int64_t>{1} &&
                  params.head_dim <= INT32_MAX && params.rotary_dim <= params.head_dim &&
                  params.rotary_dim % 2 == 0 && a.sizes[1] % params.head_dim == 0 &&
                  key.sizes[1] % params.head_dim == 0,
                  "GaudiRT FX: invalid rotary geometry or validated metadata");
      specs.push_back(a);
      continue;
    }
    if (op.op == FxOp::rotary_pos) {
      const auto& params = std::get<FxRotary>(op.parameters);
      const auto& key = specs[op.inputs[1]];
      const auto& cache = specs[op.inputs[2]];
      const auto& positions = specs[op.inputs[3]];
      TORCH_CHECK(a.dtype == at::kBFloat16 && a.dynamic() && (a.sizes.size() == 2 || a.sizes.size() == 3) &&
                  key.dtype == a.dtype && key.dynamic() && key.sizes.size() == a.sizes.size() &&
                  !cache.dynamic() && cache.dtype == a.dtype && cache.sizes.size() == 2 &&
                  cache.sizes[1] == params.rotary_dim &&
                  (positions.dtype == at::kInt || positions.dtype == at::kLong) &&
                  positions.dynamic() && positions.sizes.size() == 1 &&
                  params.head_dim > 0 && params.head_dim <= INT32_MAX &&
                  params.rotary_dim > 0 && params.rotary_dim <= params.head_dim &&
                  params.rotary_dim % 2 == 0,
                  "GaudiRT FX: invalid standalone rotary_pos geometry");
      if (a.sizes.size() == 3) {
        TORCH_CHECK(a.sizes[2] == params.head_dim && key.sizes[2] == params.head_dim,
                    "GaudiRT FX: 3D rotary_pos trailing dimension must equal head_dim");
      } else {
        TORCH_CHECK(a.sizes[1] % params.head_dim == 0 && key.sizes[1] % params.head_dim == 0,
                    "GaudiRT FX: 2D rotary_pos width must be divisible by head_dim");
      }
      specs.push_back(a);
      continue;
    }
    if (op.op == FxOp::fused_residual) {
      specs.push_back(a);
      continue;
    }
    if (op.op == FxOp::feature_slice) {
      const auto& p = std::get<FxSlice>(op.parameters);
      const int64_t last = a.sizes.back();
      TORCH_CHECK(a.sizes.size() >= 2 && a.sizes.size() <= 5 && p.total == last &&
                  p.start >= 0 && p.start < last && p.width > 0 && p.width <= last - p.start &&
                  (p.start > 0 || p.width < last),
                  "GaudiRT FX: invalid feature split geometry");
      auto sizes = a.sizes;
      sizes.back() = p.width;
      specs.push_back({a.dtype, std::move(sizes), a.dynamic()});
      continue;
    }
    if (op.op == FxOp::slice_dim) {
      const auto& p = std::get<FxSliceDim>(op.parameters);
      const int64_t dim = resolve_axis(p.dim, a.sizes.size(), "GaudiRT FX: invalid slice_dim axis");
      TORCH_CHECK(!a.dynamic() || dim != 0, "GaudiRT FX: cannot slice dynamic token axis");
      TORCH_CHECK(p.start >= 0 && p.length > 0 && p.start <= a.sizes[dim] &&
                  p.length <= a.sizes[dim] - p.start,
                  "GaudiRT FX: slice_dim out of bounds");
      auto sizes = a.sizes;
      sizes[dim] = p.length;
      specs.push_back({a.dtype, std::move(sizes), a.dynamic()});
      continue;
    }
    if (op.op == FxOp::silu_and_mul || op.op == FxOp::gelu_and_mul) {
      if (op.op == FxOp::gelu_and_mul) {
        const auto& gm = std::get<FxGeluMul>(op.parameters);
        TORCH_CHECK(gm.approximate == 0 || gm.approximate == 1,
                    "GaudiRT FX: gelu_and_mul approximate must be 0 or 1");
      }
      TORCH_CHECK(a.dtype == at::kBFloat16 && a.dynamic() &&
                  a.sizes.size() >= 2 && a.sizes.size() <= 5 && a.sizes.back() % 2 == 0,
                  "GaudiRT FX: invalid BF16 gated activation geometry");
      auto sizes = a.sizes;
      sizes.back() /= 2;
      specs.push_back({a.dtype, std::move(sizes), true});
      continue;
    }
    if (op.op == FxOp::sum_dim || op.op == FxOp::mean_dim || op.op == FxOp::amax_dim) {
      const auto& r = std::get<FxReduce>(op.parameters);
      const int64_t dim = resolve_axis(r.dim, a.sizes.size(), "GaudiRT FX: invalid reduction axis");
      TORCH_CHECK(!a.dynamic() || dim != 0, "GaudiRT FX: cannot reduce along dynamic token axis");
      TORCH_CHECK(a.dtype == at::kBFloat16 || a.dtype == at::kFloat ||
                  (op.op == FxOp::sum_dim && a.dtype == at::kInt),
                  "GaudiRT FX: unsupported reduction dtype");
      auto sizes = a.sizes;
      if (r.keepdim) {
        sizes[dim] = 1;
      } else {
        TORCH_CHECK(sizes.size() > 1, "GaudiRT FX: rank-1 non-keepdim reduction requires keepdim=True");
        sizes.erase(sizes.begin() + dim);
      }
      specs.push_back({a.dtype, std::move(sizes), a.dynamic()});
      continue;
    }
    if (op.op == FxOp::softmax || op.op == FxOp::log_softmax) {
      const int64_t dim = resolve_axis(std::get<FxAxis>(op.parameters).dim, a.sizes.size(),
                                       "GaudiRT FX: invalid softmax axis");
      TORCH_CHECK(!a.dynamic() || dim != 0, "GaudiRT FX: cannot softmax along dynamic token axis");
      TORCH_CHECK(a.dtype == at::kBFloat16 || a.dtype == at::kFloat,
                  "GaudiRT FX: softmax requires BF16 or FP32 tensor");
      specs.push_back(a);
      continue;
    }
    if (op.op == FxOp::layer_norm) {
      const auto& ln = std::get<FxLayerNorm>(op.parameters);
      TORCH_CHECK((a.dtype == at::kBFloat16 || a.dtype == at::kFloat) &&
                  a.sizes.size() >= (a.dynamic() ? 2u : 1u) && a.sizes.size() <= 5 &&
                  std::isfinite(ln.epsilon) && ln.epsilon > 0.f,
                  "GaudiRT FX: invalid layer_norm geometry or epsilon");
      const size_t expected_inputs = 1 + static_cast<size_t>(ln.has_weight) + static_cast<size_t>(ln.has_bias);
      TORCH_CHECK(op.inputs.size() == expected_inputs, "GaudiRT FX: layer_norm input count mismatch");
      for (size_t idx = 1; idx < op.inputs.size(); ++idx) {
        const auto& param = specs[op.inputs[idx]];
        TORCH_CHECK(!param.dynamic() && param.dtype == a.dtype && param.sizes.size() == 1 &&
                    param.sizes[0] == a.sizes.back(),
                    "GaudiRT FX: invalid layer_norm affine parameter");
      }
      specs.push_back(a);
      continue;
    }
    if (op.op == FxOp::block_fp8_linear) {
      const auto& weight = specs[op.inputs[1]];
      const auto& scales = specs[op.inputs[2]];
      TORCH_CHECK(a.dtype == at::kBFloat16 && a.dynamic() && (a.sizes.size() == 2 || a.sizes.size() == 3) &&
                  !weight.dynamic() && weight.dtype == at::kFloat8_e4m3fn && weight.sizes.size() == 3 &&
                  weight.sizes[0] > 0 && weight.sizes[1] > 0 && weight.sizes[1] % 128 == 0 &&
                  weight.sizes[2] == 128 && a.sizes.back() == weight.sizes[0] * 128 &&
                  !scales.dynamic() && scales.dtype == at::kFloat && scales.sizes.size() == 2 &&
                  scales.sizes[0] == weight.sizes[1] / 128 && scales.sizes[1] == weight.sizes[0],
                  "GaudiRT FX: invalid prepared block FP8 geometry");
      if (op.inputs.size() == 4) {
        const auto& bias = specs[op.inputs[3]];
        TORCH_CHECK(!bias.dynamic() && (bias.dtype == at::kBFloat16 || bias.dtype == at::kFloat) &&
                    bias.sizes.size() == 1 && bias.sizes[0] == weight.sizes[1],
                    "GaudiRT FX: invalid block FP8 bias");
      }
      auto sizes = a.sizes;
      sizes.back() = weight.sizes[1];
      specs.push_back({at::kBFloat16, std::move(sizes), true});
      continue;
    }
    if (op.op == FxOp::fp8_linear) {
      const auto& weight = specs[op.inputs[1]];
      const auto& scales = specs[op.inputs[2]];
      TORCH_CHECK(a.dtype == at::kBFloat16 && a.dynamic() && (a.sizes.size() == 2 || a.sizes.size() == 3) &&
                  !weight.dynamic() && weight.dtype == at::kFloat8_e4m3fn && weight.sizes.size() == 2 &&
                  weight.sizes[0] > 0 && weight.sizes[0] % 128 == 0 && weight.sizes[1] > 0 &&
                  weight.sizes[1] % 128 == 0 && a.sizes.back() == weight.sizes[1] &&
                  !scales.dynamic() && scales.dtype == at::kFloat && scales.sizes ==
                      std::vector<int64_t>{weight.sizes[0]}, "GaudiRT FX: invalid prepared FP8 channel geometry");
      if (op.inputs.size() == 4) {
        const auto& bias = specs[op.inputs[3]];
        TORCH_CHECK(!bias.dynamic() && (bias.dtype == at::kBFloat16 || bias.dtype == at::kFloat) &&
                    bias.sizes.size() == 1 && bias.sizes[0] == weight.sizes[0],
                    "GaudiRT FX: invalid FP8 linear bias");
      }
      auto sizes = a.sizes;
      sizes.back() = weight.sizes[0];
      specs.push_back({at::kBFloat16, std::move(sizes), true});
      continue;
    }
    if (op.op == FxOp::where) {
      const auto& x = specs[op.inputs[1]];
      const auto& y = specs[op.inputs[2]];
      TORCH_CHECK(a.dtype == at::kBool && x.dtype == y.dtype && is_supported_value_dtype(x.dtype),
                  "GaudiRT FX: where requires boolean condition and matching branch dtypes");
      const auto first = broadcast_specs(a, x, x.dtype);
      specs.push_back(broadcast_specs(first, y, x.dtype));
      continue;
    }
    const auto& b = specs[op.inputs[1]];
    TORCH_CHECK(a.dtype == b.dtype, "GaudiRT FX: input dtype mismatch");
    if (op.op == FxOp::fused_add_rms_norm) {
      const auto& weight = specs[op.inputs[2]];
      const auto& rms = std::get<FxRmsNorm>(op.parameters);
      TORCH_CHECK(a.dtype == at::kBFloat16 && a.dynamic() && b.dynamic() &&
                  a.sizes.size() >= 2 && a.sizes.size() <= 5 &&
                  a.sizes == b.sizes && a.sizes.back() <= 8192 && !weight.dynamic() &&
                  weight.dtype == a.dtype && weight.sizes.size() == 1 &&
                  weight.sizes[0] == a.sizes.back() &&
                  std::isfinite(rms.epsilon) && rms.epsilon > 0 &&
                  std::isfinite(rms.weight_offset),
                  "GaudiRT FX: invalid fused RMSNorm geometry or parameters");
      specs.push_back(a);
    } else if (op.op == FxOp::rms_norm) {
      const auto& rms = std::get<FxRmsNorm>(op.parameters);
      TORCH_CHECK(a.dtype == at::kBFloat16 && a.dynamic() &&
                  a.sizes.size() >= 2 && a.sizes.size() <= 5 &&
                  a.sizes.back() <= 8192 && !b.dynamic() && b.sizes.size() == 1 &&
                  b.sizes[0] == a.sizes.back() && std::isfinite(rms.epsilon) &&
                  rms.epsilon > 0 && std::isfinite(rms.weight_offset),
                  "GaudiRT FX: invalid BF16 RMSNorm geometry or parameters");
      specs.push_back(a);
    } else if (op.op == FxOp::linear) {
      const bool transposed = std::get<FxLinear>(op.parameters).transpose_weight;
      TORCH_CHECK(a.sizes.size() >= 2 && a.sizes.size() <= 4 && !b.dynamic() && b.sizes.size() == 2 &&
                  a.sizes.back() == b.sizes[transposed ? 1 : 0], "GaudiRT FX: invalid linear geometry");
      const auto output_width = b.sizes[transposed ? 0 : 1];
      if (op.inputs.size() == 3) {
        const auto& bias = specs[op.inputs[2]];
        TORCH_CHECK(!bias.dynamic() && bias.sizes.size() == 1 && bias.sizes[0] == output_width &&
                    bias.dtype == a.dtype, "GaudiRT FX: invalid linear bias");
      }
      auto sizes = a.sizes;
      sizes.back() = output_width;
      specs.push_back({a.dtype, std::move(sizes), a.dynamic()});
    } else if (op.op == FxOp::bmm) {
      TORCH_CHECK(a.dynamic() && b.dynamic() && a.sizes.size() >= 3 && a.sizes.size() <= 5 &&
                  b.sizes.size() == a.sizes.size() &&
                  std::equal(a.sizes.begin(), a.sizes.end() - 2, b.sizes.begin()) &&
                  a.sizes.back() == b.sizes[b.sizes.size() - 2], "GaudiRT FX: invalid batch GEMM geometry");
      auto sizes = a.sizes;
      sizes.back() = b.sizes.back();
      specs.push_back({a.dtype, std::move(sizes), true});
    } else if (op.op == FxOp::add || op.op == FxOp::sub || op.op == FxOp::mul ||
               op.op == FxOp::div || op.op == FxOp::minimum || op.op == FxOp::maximum) {
      if (op.op == FxOp::div) {
        TORCH_CHECK(a.dtype == at::kBFloat16 || a.dtype == at::kFloat,
                    "GaudiRT FX: div requires BF16 or FP32 tensor");
      } else if (op.op == FxOp::minimum || op.op == FxOp::maximum) {
        TORCH_CHECK(a.dtype == at::kBFloat16 || a.dtype == at::kFloat || a.dtype == at::kInt,
                    "GaudiRT FX: minimum/maximum requires BF16, FP32, or I32");
      } else {
        TORCH_CHECK(a.dtype == at::kBFloat16 || a.dtype == at::kFloat ||
                    a.dtype == at::kInt || a.dtype == at::kLong,
                    "GaudiRT FX: binary arithmetic requires BF16, FP32, I32, or I64");
      }
      specs.push_back(broadcast_specs(a, b, a.dtype));
    } else if (op.op == FxOp::eq || op.op == FxOp::ne || op.op == FxOp::lt ||
               op.op == FxOp::le || op.op == FxOp::gt || op.op == FxOp::ge) {
      TORCH_CHECK(is_supported_value_dtype(a.dtype), "GaudiRT FX: unsupported comparison dtype");
      specs.push_back(broadcast_specs(a, b, at::kBool));
    } else if (op.op == FxOp::logical_and || op.op == FxOp::logical_or) {
      TORCH_CHECK(a.dtype == at::kBool, "GaudiRT FX: logical op requires boolean inputs");
      specs.push_back(broadcast_specs(a, b, at::kBool));
    } else if (op.op == FxOp::bitwise_and || op.op == FxOp::bitwise_or) {
      TORCH_CHECK(a.dtype == at::kInt || a.dtype == at::kBool,
                  "GaudiRT FX: bitwise op requires I32 or boolean inputs");
      specs.push_back(broadcast_specs(a, b, a.dtype));
    } else {
      TORCH_CHECK(false, "GaudiRT FX: unsupported operation");
    }
  }
  for (auto id : outputs) normalized << "o" << id << ";";
  const auto text = normalized.str();
  return {std::move(specs), gaudirt::common::sha256_bytes(
                                reinterpret_cast<const uint8_t*>(text.data()), text.size())};
}

}  // namespace gaudirt::generic
