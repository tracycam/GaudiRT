// SPDX-License-Identifier: Apache-2.0
#include "fx_geometry.hpp"

#include "gaudirt/common/sha256.hpp"

#include <c10/util/Exception.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace gaudirt::generic {

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
                     ? "fx_program_rotary_static_v2;" : "fx_program_v16;");
  if (std::any_of(operations.begin(), operations.end(),
                  [](const FxNode& op) { return op.op == FxOp::block_fp8_linear; })) {
    normalized << "block_fp8_fp16_1k_v1;";
  }
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& tensor = inputs[i];
    const bool floating = tensor.scalar_type() == at::kBFloat16 || tensor.scalar_type() == at::kFloat;
    const bool prepared_fp8 = tensor.scalar_type() == at::kFloat8_e4m3fn;
    const bool integer_vector = tensor.scalar_type() == at::kInt && tensor.dim() == 1;
    TORCH_CHECK(tensor.device().type() == c10::kPrivateUse1 && tensor.device().index() >= 0 &&
                tensor.device().index() == inputs[0].device().index() &&
                (floating || integer_vector || prepared_fp8), "GaudiRT FX: unsupported input device or dtype");
    TORCH_CHECK(((floating || (prepared_fp8 && !roles[i])) && tensor.dim() >= 1 && tensor.dim() <= 5) || integer_vector,
                "GaudiRT FX: unsupported input rank");
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
      TORCH_CHECK(a.dynamic() && a.sizes.size() == 2 &&
                  (a.dtype == at::kBFloat16 || a.dtype == at::kFloat),
                  "GaudiRT FX: scalar arithmetic requires a finite FP32 scalar and rank-two input");
      specs.push_back(a);
      continue;
    }
    if (op.op == FxOp::copy) {
      TORCH_CHECK(a.dynamic() && a.sizes.size() == 2 &&
                  (a.dtype == at::kBFloat16 || a.dtype == at::kFloat),
                  "GaudiRT FX: copy requires a dynamic rank-two BF16 or FP32 input");
      specs.push_back(a);
      continue;
    }
    if (op.op == FxOp::transpose) {
      const auto& params = std::get<FxTranspose>(op.parameters);
      TORCH_CHECK(a.dynamic() && a.sizes.size() >= 3 && a.sizes.size() <= 5 &&
                  params.dim0 < static_cast<int64_t>(a.sizes.size()) &&
                  params.dim1 < static_cast<int64_t>(a.sizes.size()),
                  "GaudiRT FX: feature transpose requires a dynamic rank-three through rank-five input");
      auto sizes = a.sizes;
      std::swap(sizes[params.dim0], sizes[params.dim1]);
      specs.push_back({a.dtype, std::move(sizes), true});
      continue;
    }
    if (op.op == FxOp::reshape) {
      TORCH_CHECK(a.dynamic() && !a.sizes.empty() && a.sizes.size() <= 5,
                  "GaudiRT FX: invalid reshape input");
      int64_t source_width = 1, target_width = 1;
      for (size_t i = 1; i < a.sizes.size(); ++i) {
        TORCH_CHECK(source_width <= INT64_MAX / a.sizes[i], "GaudiRT FX: reshape input is too large");
        source_width *= a.sizes[i];
      }
      std::vector<int64_t> sizes{-1};
      for (auto size : std::get<FxShape>(op.parameters).dimensions) {
        TORCH_CHECK(target_width <= INT64_MAX / size, "GaudiRT FX: reshape output is too large");
        target_width *= size;
        sizes.push_back(size);
      }
      TORCH_CHECK(source_width == target_width, "GaudiRT FX: reshape changes the token width");
      specs.push_back({a.dtype, std::move(sizes), true});
      continue;
    }
    if (op.op == FxOp::concat) {
      TORCH_CHECK(a.dynamic() && a.sizes.size() == 2, "GaudiRT FX: invalid cat input");
      int64_t width = 0;
      for (auto input_id : op.inputs) {
        const auto& input = specs[input_id];
        TORCH_CHECK(input.dynamic() && input.dtype == a.dtype && input.sizes.size() == 2 &&
                    input.sizes[1] <= INT64_MAX - width, "GaudiRT FX: invalid cat input geometry");
        width += input.sizes[1];
      }
      specs.push_back({a.dtype, {-1, width}, true});
      continue;
    }
    if (op.op == FxOp::gather_rows) {
      const auto& index = specs[op.inputs[1]];
      TORCH_CHECK(a.dynamic() && a.sizes.size() == 2 && !index.dynamic() && index.dtype == at::kInt &&
                  index.sizes.size() == 1 && index.sizes[0] > 0 && index.sizes[0] <= 4096,
                  "GaudiRT FX: invalid validated row gather geometry");
      specs.push_back({a.dtype, {index.sizes[0], a.sizes[1]}, false});
      continue;
    }
    if (op.op == FxOp::cast) {
      const auto dtype = std::get<FxCast>(op.parameters).to_float ? at::kFloat : at::kBFloat16;
      TORCH_CHECK(a.sizes.size() == 2 && (a.dtype == at::kBFloat16 || a.dtype == at::kFloat) &&
                  a.dtype != dtype, "GaudiRT FX: invalid BF16/FP32 cast");
      specs.push_back({dtype, a.sizes, a.dynamic()});
      continue;
    }
    if (op.op == FxOp::embedding) {
      const auto& weight = specs[op.inputs[1]];
      TORCH_CHECK(a.dtype == at::kInt && a.dynamic() && a.sizes.size() == 1 &&
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
      TORCH_CHECK(a.dtype == at::kInt && a.dynamic() && a.sizes.size() == 1 &&
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
    if (op.op == FxOp::fused_residual) {
      specs.push_back(a);
      continue;
    }
    if (op.op == FxOp::feature_slice) {
      const auto& p = std::get<FxSlice>(op.parameters);
      TORCH_CHECK(a.dynamic() && a.sizes.size() == 2 && p.total == a.sizes[1] && p.start < a.sizes[1] &&
                  p.width <= a.sizes[1] - p.start && (p.start > 0 || p.width < a.sizes[1]),
                  "GaudiRT FX: invalid feature split geometry");
      specs.push_back({a.dtype, {-1, p.width}, true});
      continue;
    }
    if (op.op == FxOp::silu_and_mul) {
      TORCH_CHECK(a.dtype == at::kBFloat16 && a.dynamic() && a.sizes.size() == 2 &&
                  a.sizes[1] % 2 == 0, "GaudiRT FX: invalid BF16 SiLU/multiply geometry");
      specs.push_back({a.dtype, {-1, a.sizes[1] / 2}, true});
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
    const auto& b = specs[op.inputs[1]];
    TORCH_CHECK(a.dtype == b.dtype, "GaudiRT FX: input dtype mismatch");
    if (op.op == FxOp::fused_add_rms_norm) {
      const auto& weight = specs[op.inputs[2]];
      const float epsilon = std::get<FxEpsilon>(op.parameters).value;
      TORCH_CHECK(a.dtype == at::kBFloat16 && a.dynamic() && b.dynamic() && a.sizes.size() == 2 &&
                  a.sizes == b.sizes && a.sizes[1] <= 8192 && !weight.dynamic() &&
                  weight.dtype == a.dtype && weight.sizes.size() == 1 && weight.sizes[0] == a.sizes[1] &&
                  std::isfinite(epsilon) && epsilon > 0, "GaudiRT FX: invalid fused RMSNorm geometry or epsilon");
      specs.push_back(a);
    } else if (op.op == FxOp::linear) {
      const bool transposed = std::get<FxLinear>(op.parameters).transpose_weight;
      TORCH_CHECK(a.sizes.size() == 2 && !b.dynamic() && b.sizes.size() == 2 &&
                  a.sizes[1] == b.sizes[transposed ? 1 : 0], "GaudiRT FX: invalid linear geometry");
      const auto output_width = b.sizes[transposed ? 0 : 1];
      if (op.inputs.size() == 3) {
        const auto& bias = specs[op.inputs[2]];
        TORCH_CHECK(!bias.dynamic() && bias.sizes.size() == 1 && bias.sizes[0] == output_width &&
                    bias.dtype == a.dtype, "GaudiRT FX: invalid linear bias");
      }
      specs.push_back({a.dtype, {a.sizes[0], output_width}, a.dynamic()});
    } else if (op.op == FxOp::bmm) {
      TORCH_CHECK(a.dynamic() && b.dynamic() && a.sizes.size() >= 3 && a.sizes.size() <= 5 &&
                  b.sizes.size() == a.sizes.size() &&
                  std::equal(a.sizes.begin(), a.sizes.end() - 2, b.sizes.begin()) &&
                  a.sizes.back() == b.sizes[b.sizes.size() - 2], "GaudiRT FX: invalid batch GEMM geometry");
      auto sizes = a.sizes;
      sizes.back() = b.sizes.back();
      specs.push_back({a.dtype, std::move(sizes), true});
    } else if (op.op == FxOp::add || op.op == FxOp::sub || op.op == FxOp::mul || op.op == FxOp::div) {
      const auto compatible = [](const TensorSpec& dynamic, const TensorSpec& other) {
        return dynamic.dynamic() && dynamic.sizes.size() == 2 &&
               ((other.dynamic() && other.sizes == dynamic.sizes) ||
                (!other.dynamic() && ((other.sizes.size() == 1 && other.sizes[0] == dynamic.sizes[1]) ||
                                    (other.sizes.size() == 2 && other.sizes[0] == 1 &&
                                     other.sizes[1] == dynamic.sizes[1]))));
      };
      TORCH_CHECK(compatible(a, b) || compatible(b, a), "GaudiRT FX: unsupported elementwise broadcast geometry");
      specs.push_back(a.dynamic() ? a : b);
    } else {
      TORCH_CHECK(op.op == FxOp::rms_norm, "GaudiRT FX: unsupported operation");
      const float epsilon = std::get<FxEpsilon>(op.parameters).value;
      TORCH_CHECK(a.dtype == at::kBFloat16 && a.dynamic() && a.sizes.size() == 2 &&
                  a.sizes[1] <= 8192 && !b.dynamic() && b.sizes.size() == 1 &&
                  b.sizes[0] == a.sizes[1] && std::isfinite(epsilon) && epsilon > 0,
                  "GaudiRT FX: invalid BF16 RMSNorm geometry or epsilon");
      specs.push_back(a);
    }
  }
  for (auto id : outputs) normalized << "o" << id << ";";
  const auto text = normalized.str();
  return {std::move(specs), gaudirt::common::sha256_bytes(
                                reinterpret_cast<const uint8_t*>(text.data()), text.size())};
}

}  // namespace gaudirt::generic
