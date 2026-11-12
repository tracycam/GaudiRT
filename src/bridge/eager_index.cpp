// SPDX-License-Identifier: Apache-2.0
#include "bridge_internal.hpp"
#include "eager_copy.hpp"
#include "eager_ops.hpp"
#include "eager_index.hpp"
#include "argmax_vocab.hpp"
#include "graph_builder.hpp"
#include "layout_copy.hpp"
#include "temporaries.hpp"
#include "gaudirt/common/sha256.hpp"

#include <ATen/EmptyTensor.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <perf_lib_layer_params.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gaudirt::generic {
using namespace runtime;

ColdRecipe& invalid_reduce_recipe(Context& ctx, uint32_t count) {
  const std::string key = "cold_index_check_" + std::to_string(count);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>();
  {
    SynapseGraphBuilder builder(key, *recipe);
    auto in = builder.tensor(syn_type_float, {count}, 0);
    auto out = builder.tensor(syn_type_float, {1}, 1);
    builder.node(NodeSpec{"reduce_sum_f32", {in}, {out}, ns_Reduction::Params{0}});
    builder.compile(ctx.cache, key);
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate index check recipe");
  return *it->second;
}

void validate_indices(const at::Tensor& index, int64_t upper) {
  if (!index.numel()) return;
  TORCH_CHECK(index.numel() <= UINT32_MAX && upper >= 0 && upper <= INT32_MAX,
              "GaudiRT: index dimensions exceed the supported range");
  auto low = compare_scalar(index, at::Scalar(0), "less");
  auto high = compare_scalar(index, at::Scalar(upper), "greater_equal");
  const std::vector<int64_t> shape(index.sizes().begin(), index.sizes().end());
  auto invalid = pointwise({low, high}, shape, at::kBool, "or_fwd_u8",
                           {syn_type_uint8, syn_type_uint8, syn_type_uint8});
  auto values = invalid.to(at::kFloat);
  auto ctx = current();
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = invalid_reduce_recipe(*ctx, index.numel());
  auto stream = ctx->stream();
  const auto input = address_of(values, *ctx);
  ctx->allocator.record_stream(input, stream);
  TemporaryDevice result(*ctx, sizeof(float));
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  auto bindings = recipe.tensors;
  bindings[0].pTensorAddress = input;
  bindings[1].pTensorAddress = result.address;
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: index check launch failed");
  auto d2h = ctx->device.d2h_stream();
  auto& ready = ctx->streams.state(active_stream_id).d2h_reusable;
  ready.record_and_wait(stream, d2h);
  ctx->allocator.record_stream(result.address, d2h);
  TemporaryHost host(*ctx, sizeof(float));
  ctx->pinned.copy_d2h(host.address, result.address, sizeof(float), d2h, false);
  float failures;
  std::memcpy(&failures, host.address, sizeof(failures));
  TORCH_CHECK(std::isfinite(failures) && failures == 0.0f, "GaudiRT: index out of bounds");
}

ColdRecipe& gather_recipe(Context& ctx, const at::Tensor& input, const at::Tensor& output,
                          uint32_t indices, int64_t axis, bool wide_indices) {
  std::string signature = std::to_string(static_cast<int>(input.scalar_type())) + ":" +
                          std::to_string(axis) + ":" + std::to_string(indices) + ":" +
                          std::to_string(wide_indices);
  for (auto size : input.sizes()) signature += ":" + std::to_string(size);
  const auto hash = common::sha256_bytes(reinterpret_cast<const uint8_t*>(signature.data()),
                                          signature.size());
  const std::string key = "cold_gather_" + hash;
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(3);
  {
    auto synapse_dims = [](const at::Tensor& shaped) {
      std::vector<uint64_t> dims(shaped.dim());
      for (int64_t dim = 0; dim < shaped.dim(); ++dim) {
        const auto size = static_cast<uint64_t>(shaped.size(shaped.dim() - dim - 1));
        TORCH_CHECK(size <= UINT32_MAX, "GaudiRT: gather dimension too large");
        dims[dim] = size;
      }
      return dims;
    };
    const auto in_dims = synapse_dims(input);
    const auto out_dims = synapse_dims(output);
    const auto type = cast_type(input.scalar_type());
    SynapseGraphBuilder builder(key, *recipe);
    auto in = builder.tensor(type.syn, in_dims, 0);
    auto idx = builder.tensor(wide_indices ? syn_type_int64 : syn_type_int32, {indices}, 1);
    auto out = builder.tensor(type.syn, out_dims, 2);
    if (wide_indices) {
      auto idx_i32 = builder.tensor(syn_type_int32, {indices}, -1);
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_HALF_NE;
      builder.node(NodeSpec{"cast_i64_to_i32", {idx}, {idx_i32}, cast});
      idx = idx_i32;
    }
    ns_GatherKernel::ParamsV2 params{};
    params.axis = input.dim() - axis - 1;
    params.batchDims = 0;
    builder.node(NodeSpec{"gather_fwd_" + std::string(type.name), {in, idx}, {out}, params});
    builder.compile(ctx.cache, key);
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate gather recipe");
  return *it->second;
}

at::Tensor index_select_impl(const at::Tensor& input, int64_t axis, const at::Tensor& index, bool validated) {
  TORCH_CHECK(input.device() == index.device() && input.device().type() == c10::kPrivateUse1 &&
              (index.scalar_type() == at::kInt || index.scalar_type() == at::kLong),
              "GaudiRT: index_select requires device int32 or int64 indices");
  TORCH_CHECK(input.dim() > 0 && input.dim() <= 5 && index.dim() <= 1,
              "GaudiRT: index_select rank is not supported");
  cast_type(input.scalar_type());
  axis = at::maybe_wrap_dim(axis, input.dim());
  const auto count = index.numel();
  if (!validated) validate_indices(index, input.size(axis));
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  shape[axis] = count;
  auto result = at::empty(shape, input.options());
  if (!result.numel()) return result;
  auto ctx = current();
  auto source = input;
  auto source_address = address_of(source, *ctx);
  if (!source.is_contiguous() || source_address % BlockCachingAllocator::kAlignment) {
    source = at::empty(input.sizes(), input.options());
    copy(source, input, true);
  }
  auto ids = index;
  source_address = address_of(source, *ctx);
  const auto ids_address = address_of(ids, *ctx);
  const bool overlaps = ids_address < source_address + source.nbytes() &&
                        source_address < ids_address + ids.nbytes();
  if (!ids.is_contiguous() || ids_address % BlockCachingAllocator::kAlignment || overlaps) {
    auto staged = at::empty(ids.sizes(), ids.options());
    copy(staged, ids, true);
    ids = std::move(staged);
  }
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = gather_recipe(*ctx, source, result, count, axis, ids.scalar_type() == at::kLong);
  auto stream = ctx->stream();
  auto bindings = recipe.tensors;
  bindings[0].pTensorAddress = address_of(source, *ctx);
  bindings[1].pTensorAddress = address_of(ids, *ctx);
  bindings[2].pTensorAddress = address_of(result, *ctx);
  for (const auto& binding : bindings) ctx->allocator.record_stream(binding.pTensorAddress, stream);
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: gather launch failed");
  return result;
}

at::Tensor index_select(const at::Tensor& input, int64_t axis, const at::Tensor& index) {
  return index_select_impl(input, axis, index, false);
}

at::Tensor index_select_validated(const at::Tensor& input, int64_t axis, const at::Tensor& index) {
  return index_select_impl(input, axis, index, true);
}

ColdRecipe& index_put_recipe(Context& ctx, uint32_t count, uint32_t slots, bool wide_indices) {
  const auto key = "cold_index_put_i64_" + std::to_string(count) + "_" + std::to_string(slots) +
                   (wide_indices ? "_idx64" : "_idx32");
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto compiled = std::make_unique<ColdRecipe>(3);
  {
    SynapseGraphBuilder builder(key, *compiled);
    auto idx = builder.tensor(wide_indices ? syn_type_int64 : syn_type_int32, {count}, 0);
    auto vals = builder.tensor(syn_type_int32, {2, count}, 1);
    auto out = builder.tensor(syn_type_int32, {2, slots}, 2);
    if (wide_indices) {
      auto idx_i32 = builder.tensor(syn_type_int32, {count}, -1);
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_HALF_NE;
      builder.node(NodeSpec{"cast_i64_to_i32", {idx}, {idx_i32}, cast});
      idx = idx_i32;
    }
    builder.node(NodeSpec{"gk_index_put_i64_v1", {idx, vals}, {out}, {}});
    builder.compile(ctx.cache, key);
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(compiled));
  TORCH_CHECK(inserted, "GaudiRT index put: duplicate recipe");
  return *it->second;
}

at::Tensor index_tensor(const at::Tensor& input, const c10::List<std::optional<at::Tensor>>& indices) {
  TORCH_CHECK(indices.size() == 1 && indices.get(0).has_value() && indices.get(0)->dim() == 1,
              "GaudiRT: index supports one one-dimensional index tensor");
  return gaudirt::generic::index_select(input, 0, *indices.get(0));
}

at::Tensor& index_put_impl(at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices,
                           const at::Tensor& values, bool accumulate, bool unsafe, bool validated) {
  TORCH_CHECK(!accumulate && !unsafe && indices.size() == 1 && indices.get(0).has_value(),
              "GaudiRT: index put supports one safe non-accumulating index");
  auto index = *indices.get(0);
  auto incoming = values;
  if (self.device().type() == c10::kPrivateUse1 && values.device().is_cpu() && values.dim() == 0) {
    incoming = at::empty({}, self.options());
    fill(incoming, values.item());
  }
  TORCH_CHECK(self.device().type() == c10::kPrivateUse1 && self.device() == incoming.device() &&
              self.device() == index.device() && self.scalar_type() == at::kLong &&
              incoming.scalar_type() == at::kLong && self.dim() == 1 && incoming.dim() <= 1 &&
              index.dim() == 1 && (index.scalar_type() == at::kInt || index.scalar_type() == at::kLong) &&
              (incoming.numel() == index.numel() || incoming.numel() == 1) && index.numel() <= 4096,
              "GaudiRT: index put supports INT64 vectors with matching device indices and values; ",
              "self=", self.scalar_type(), "/", self.dim(), " values=", incoming.scalar_type(), "/", incoming.dim(),
              "/", incoming.numel(), " indices=", index.scalar_type(), "/", index.dim(), "/", index.numel());
  if (!validated) validate_indices(index, self.numel());
  if (!index.numel()) return self;
  auto ctx = current();
  auto output = self;
  auto output_address = address_of(output, *ctx);
  if (!output.is_contiguous() || output_address % BlockCachingAllocator::kAlignment) {
    output = at::empty(self.sizes(), self.options());
    copy(output, self, true);
    output_address = address_of(output, *ctx);
  }
  auto ids = index;
  auto id_address = address_of(ids, *ctx);
  auto overlaps = [](uint64_t a, uint64_t a_bytes, uint64_t b, uint64_t b_bytes) {
    return a < b + b_bytes && b < a + a_bytes;
  };
  if (!ids.is_contiguous() || id_address % BlockCachingAllocator::kAlignment ||
      overlaps(id_address, ids.nbytes(), output_address, output.nbytes())) {
    auto staged = at::empty(ids.sizes(), ids.options());
    copy(staged, ids, true);
    ids = std::move(staged);
    id_address = address_of(ids, *ctx);
  }
  auto source = incoming.expand({index.numel()});
  auto source_address = address_of(source, *ctx);
  if (!source.is_contiguous() || source_address % BlockCachingAllocator::kAlignment ||
      overlaps(source_address, source.nbytes(), output_address, output.nbytes()) ||
      overlaps(source_address, source.nbytes(), id_address, ids.nbytes())) {
    auto staged = at::empty(source.sizes(), source.options());
    copy(staged, source, true);
    source = std::move(staged);
    source_address = address_of(source, *ctx);
  }
  {
    std::lock_guard lock(ctx->cold_mutex);
    auto& recipe = index_put_recipe(*ctx, index.numel(), self.numel(), ids.scalar_type() == at::kLong);
    auto stream = ctx->stream();
    auto bindings = recipe.tensors;
    bindings[0].pTensorAddress = id_address;
    bindings[1].pTensorAddress = source_address;
    bindings[2].pTensorAddress = output_address;
    for (const auto& binding : bindings) ctx->allocator.record_stream(binding.pTensorAddress, stream);
    std::unique_ptr<TemporaryDevice> workspace;
    if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
    TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                          workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
                "GaudiRT: index put launch failed");
  }
  if (output.unsafeGetTensorImpl() != self.unsafeGetTensorImpl()) copy(self, output, true);
  return self;
}

at::Tensor& index_put(at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices,
                      const at::Tensor& values, bool accumulate, bool unsafe) {
  return index_put_impl(self, indices, values, accumulate, unsafe, false);
}

at::Tensor& index_put_validated(at::Tensor& self, const at::Tensor& index, const at::Tensor& values) {
  c10::List<std::optional<at::Tensor>> indices;
  indices.push_back(index);
  return index_put_impl(self, indices, values, false, false, true);
}

at::Tensor embedding(const at::Tensor& weight, const at::Tensor& index, int64_t padding_idx,
                     bool scale_grad_by_freq, bool sparse) {
  (void)padding_idx;
  TORCH_CHECK(weight.dim() == 2 && !scale_grad_by_freq && !sparse,
              "GaudiRT: unsupported embedding configuration");
  auto flat = index.reshape({-1});
  auto selected = gaudirt::generic::index_select(weight, 0, flat);
  std::vector<int64_t> shape(index.sizes().begin(), index.sizes().end());
  shape.push_back(weight.size(1));
  return selected.view(shape);
}

ColdRecipe& argmax_recipe(Context& ctx, const at::Tensor& input, const at::Tensor& reduced,
                          int64_t axis, bool values_only = false) {
  std::string signature = std::to_string(static_cast<int>(input.scalar_type())) + ":" +
                          std::to_string(axis);
  for (auto size : input.sizes()) signature += ":" + std::to_string(size);
  const auto hash = common::sha256_bytes(reinterpret_cast<const uint8_t*>(signature.data()),
                                          signature.size());
  const std::string key = (values_only ? "cold_max_i32_" : "cold_argmax_i64_") + hash;
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>();
  {
    auto synapse_dims = [](const at::Tensor& shaped) {
      std::vector<uint64_t> dims(shaped.dim());
      for (int64_t dim = 0; dim < shaped.dim(); ++dim) {
        const auto size = static_cast<uint64_t>(shaped.size(shaped.dim() - dim - 1));
        TORCH_CHECK(size <= UINT32_MAX, "GaudiRT: argmax dimension too large");
        dims[dim] = size;
      }
      return dims;
    };
    const auto in_dims = synapse_dims(input);
    const auto red_dims = synapse_dims(reduced);
    const auto in_type = cast_type(input.scalar_type());
    SynapseGraphBuilder builder(key, *recipe);
    auto in = builder.tensor(in_type.syn, in_dims, 0);
    auto out = builder.tensor(values_only ? syn_type_int32 : syn_type_int64, red_dims, 1);

    ns_Reduction::Params params{static_cast<unsigned int>(input.dim() - axis - 1)};
    const bool floating = !values_only &&
                          (input.scalar_type() == at::kFloat || input.scalar_type() == at::kBFloat16);
    auto idx_i32 = values_only ? nullptr : builder.tensor(syn_type_int32, red_dims, -1);
    auto ord_idx = floating ? builder.tensor(syn_type_int32, red_dims, -1)
                            : (values_only ? out : idx_i32);
    const auto guid = values_only ? std::string("reduce_max_i32")
                                  : "argmax_fwd_" + std::string(in_type.name);
    builder.node(NodeSpec{guid, {in}, {ord_idx}, params});
    if (floating) {
      auto nan_u8 = builder.tensor(syn_type_uint8, in_dims, -1);
      auto nan_f32 = builder.tensor(syn_type_float, in_dims, -1);
      auto nan_max = builder.tensor(syn_type_float, red_dims, -1);
      auto zero_f32 = builder.tensor(syn_type_float, red_dims, -1);
      auto has_nan = builder.tensor(syn_type_uint8, red_dims, -1);
      auto nan_idx = builder.tensor(syn_type_int32, red_dims, -1);
      auto neg_inf_val = builder.tensor(in_type.syn, in_dims, -1);
      auto is_inf_u8 = builder.tensor(syn_type_uint8, in_dims, -1);
      auto is_inf_f32 = builder.tensor(syn_type_float, in_dims, -1);
      auto inf_min = builder.tensor(syn_type_float, red_dims, -1);
      auto all_inf = builder.tensor(syn_type_uint8, red_dims, -1);
      auto zero_idx = builder.tensor(syn_type_int32, red_dims, -1);
      auto safe_idx = builder.tensor(syn_type_int32, red_dims, -1);

      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_HALF_NE;
      builder.node(NodeSpec{"isnan_fwd_" + std::string(in_type.name), {in}, {nan_u8}, {}});
      builder.node(NodeSpec{"cast_u8_to_f32", {nan_u8}, {nan_f32}, cast});
      builder.node(NodeSpec{"argmax_fwd_f32", {nan_f32}, {nan_idx}, params});
      builder.node(NodeSpec{"reduce_max_f32", {nan_f32}, {nan_max}, params});
      ns_ConstantKernel::ParamsV3 zero{};
      builder.node(NodeSpec{"constant_f32", {}, {zero_f32}, zero});
      builder.node(NodeSpec{"greater_fwd_f32", {nan_max, zero_f32}, {has_nan}, {}});
      ns_ConstantKernel::ParamsV3 negative_inf{};
      negative_inf.constant.f = -std::numeric_limits<float>::infinity();
      builder.node(NodeSpec{"constant_" + std::string(in_type.name), {}, {neg_inf_val}, negative_inf});
      builder.node(NodeSpec{"equal_fwd_" + std::string(in_type.name), {in, neg_inf_val}, {is_inf_u8}, {}});
      builder.node(NodeSpec{"cast_u8_to_f32", {is_inf_u8}, {is_inf_f32}, cast});
      builder.node(NodeSpec{"reduce_min_f32", {is_inf_f32}, {inf_min}, params});
      builder.node(NodeSpec{"greater_fwd_f32", {inf_min, zero_f32}, {all_inf}, {}});
      builder.node(NodeSpec{"cast_f32_to_i32", {zero_f32}, {zero_idx}, cast});
      builder.node(NodeSpec{"where_fwd_i32", {all_inf, zero_idx, ord_idx}, {safe_idx}, {}});
      builder.node(NodeSpec{"where_fwd_i32", {has_nan, nan_idx, safe_idx}, {idx_i32}, {}});
    }
    if (!values_only) {
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_HALF_NE;
      builder.node(NodeSpec{"cast_i32_to_i64", {idx_i32}, {out}, cast});
    }
    builder.compile(ctx.cache, key);
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate argmax recipe");
  return *it->second;
}

at::Tensor reduce_max_i32(const at::Tensor& input, int64_t axis) {
  auto ctx = current();
  auto source = input;
  auto address = address_of(source, *ctx);
  if (!source.is_contiguous() || address % BlockCachingAllocator::kAlignment) {
    source = at::empty(input.sizes(), input.options());
    copy(source, input, true);
  }
  std::vector<int64_t> shape(source.sizes().begin(), source.sizes().end());
  shape[axis] = 1;
  auto result = at::empty(shape, source.options());
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = argmax_recipe(*ctx, source, result, axis, true);
  auto stream = ctx->stream();
  auto bindings = recipe.tensors;
  bindings[0].pTensorAddress = address_of(source, *ctx);
  bindings[1].pTensorAddress = address_of(result, *ctx);
  for (const auto& binding : bindings) ctx->allocator.record_stream(binding.pTensorAddress, stream);
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: integer maximum launch failed");
  return result;
}

at::Tensor argmax(const at::Tensor& input, std::optional<int64_t> dimension, bool keepdim) {
  TORCH_CHECK(input.device().type() == c10::kPrivateUse1 && input.numel(),
              "GaudiRT: argmax requires a nonempty device tensor");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kFloat ||
              input.scalar_type() == at::kInt || input.scalar_type() == at::kLong,
              "GaudiRT: argmax dtype is not implemented");
  TORCH_CHECK(input.dim() <= 5, "GaudiRT: argmax rank is not supported");
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  auto source = input;
  if (!input.is_contiguous()) {
    source = at::empty(input.sizes(), input.options());
    copy(source, input, true);
  }
  if (source.dim() == 0) source = source.view({1});
  int64_t axis = 0;
  if (!dimension) {
    source = source.view({-1});
    if (keepdim) std::fill(shape.begin(), shape.end(), 1);
    else shape.clear();
  } else {
    axis = at::maybe_wrap_dim(*dimension, source.dim());
    if (input.dim() > 0) {
      if (keepdim) shape[axis] = 1;
      else shape.erase(shape.begin() + axis);
    }
  }
  TORCH_CHECK(source.size(axis) <= INT32_MAX, "GaudiRT: argmax dimension too large");
  if (source.scalar_type() == at::kLong) {
    static_assert(std::endian::native == std::endian::little);
    std::vector<int64_t> source_shape(source.sizes().begin(), source.sizes().end());
    auto lane_shape = source_shape;
    lane_shape.push_back(2);
    auto words = source.view({-1}).view(at::kInt).view(lane_shape);
    auto low = words.select(-1, 0);
    auto high = words.select(-1, 1);
    auto high_max = reduce_max_i32(high, axis);
    auto high_matches = compare(high, high_max, "equal");
    auto sign_bit = at::empty({}, source.options().dtype(at::kInt));
    fill(sign_bit, at::Scalar(INT32_MIN));
    auto ordered_low = pointwise({low, sign_bit}, source_shape, at::kInt, "bitwise_xor_fwd_i32",
                                 {syn_type_int32, syn_type_int32, syn_type_int32});
    auto eligible = gaudirt::generic::where(high_matches, ordered_low, sign_bit);
    auto low_max = reduce_max_i32(eligible, axis);
    auto low_matches = compare(ordered_low, low_max, "equal");
    auto matches = pointwise({high_matches, low_matches}, source_shape, at::kBool, "and_fwd_u8",
                             {syn_type_uint8, syn_type_uint8, syn_type_uint8});
    auto scores = matches.to(at::kFloat);
    return gaudirt::generic::argmax(scores, axis, true).view(shape);
  }
  std::vector<int64_t> reduced_shape(source.sizes().begin(), source.sizes().end());
  reduced_shape[axis] = 1;
  auto reduced = at::empty(reduced_shape, source.options().dtype(at::kLong));
  auto ctx = current();
  auto source_address = address_of(source, *ctx);
  if (source_address % BlockCachingAllocator::kAlignment) {
    auto staged = at::empty(source.sizes(), source.options());
    copy(staged, source, true);
    source = std::move(staged);
  }
  {
    std::lock_guard lock(ctx->cold_mutex);
    const bool vocabulary = source.dim() == 2 && axis == 1 && source.size(1) <= INT_MAX - 127 &&
                            (source.scalar_type() == at::kFloat || source.scalar_type() == at::kBFloat16);
    auto& recipe = vocabulary ? vocabulary_argmax_recipe(*ctx, source)
                              : argmax_recipe(*ctx, source, reduced, axis);
    auto stream = ctx->stream();
    auto bindings = recipe.tensors;
    bindings[0].pTensorAddress = address_of(source, *ctx);
    bindings[1].pTensorAddress = address_of(reduced, *ctx);
    for (const auto& binding : bindings) ctx->allocator.record_stream(binding.pTensorAddress, stream);
    std::unique_ptr<TemporaryDevice> workspace;
    if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
    TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                          workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
                "GaudiRT: argmax launch failed");
  }
  return reduced.view(shape);
}

}  // namespace gaudirt::generic
