// SPDX-License-Identifier: Apache-2.0
#include "bridge_internal.hpp"
#include "eager_copy.hpp"
#include "eager_index.hpp"
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

ColdRecipe& cat_recipe(Context& ctx, const std::vector<at::Tensor>& inputs,
                       const at::Tensor& output, int64_t axis) {
  std::string signature = std::to_string(static_cast<int>(output.scalar_type())) + ":" +
                          std::to_string(axis);
  for (const auto& tensor : inputs) {
    signature += ";";
    for (auto size : tensor.sizes()) signature += std::to_string(size) + ",";
  }
  const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(signature.data()),
                                            signature.size());
  const std::string key = "cold_cat_" + digest;
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(inputs.size() + 1);
  {
    SynapseGraphBuilder builder(key, *recipe);
    std::vector<synTensor> cat_inputs;
    cat_inputs.reserve(inputs.size());
    synTensor out_tensor = nullptr;
    for (size_t i = 0; i <= inputs.size(); ++i) {
      const auto& tensor = i < inputs.size() ? inputs[i] : output;
      std::vector<uint64_t> dims(tensor.dim());
      for (int64_t dim = 0; dim < tensor.dim(); ++dim)
        dims[dim] = static_cast<uint64_t>(tensor.size(tensor.dim() - dim - 1));
      auto handle = builder.tensor(cast_type(tensor.scalar_type()).syn, dims, static_cast<int>(i));
      if (i < inputs.size()) cat_inputs.push_back(handle);
      else out_tensor = handle;
    }
    synConcatenateParams params{static_cast<unsigned>(output.dim() - axis - 1)};
    builder.node(NodeSpec{"concat", std::move(cat_inputs), {out_tensor}, params});
    builder.compile(ctx.cache, key);
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate cold cat recipe");
  return *it->second;
}

at::Tensor cat(const at::ITensorListRef& tensors, int64_t axis) {
  TORCH_CHECK(!tensors.empty(), "GaudiRT: cat requires at least one tensor");
  std::vector<at::Tensor> all;
  for (const auto& tensor : tensors) all.push_back(tensor);
  const at::Tensor* reference = &all.front();
  for (const auto& tensor : all)
    if (!(tensor.dim() == 1 && tensor.size(0) == 0)) { reference = &tensor; break; }
  TORCH_CHECK(reference->dim() > 0 && reference->dim() <= 5,
              "GaudiRT: cat requires tensors of rank 1 to 5");
  axis = at::maybe_wrap_dim(axis, reference->dim());
  auto dtype = reference->scalar_type();
  for (const auto& tensor : all) dtype = c10::promoteTypes(dtype, tensor.scalar_type());
  cast_type(dtype);
  std::vector<int64_t> sizes(reference->sizes().begin(), reference->sizes().end());
  sizes[axis] = 0;
  std::vector<at::Tensor> active;
  for (const auto& tensor : all) {
    TORCH_CHECK(tensor.device() == reference->device() &&
                tensor.device().type() == c10::kPrivateUse1,
                "GaudiRT: cat requires same-device tensors");
    if (tensor.dim() == 1 && tensor.size(0) == 0 && reference->dim() != 1) continue;
    TORCH_CHECK(tensor.dim() == reference->dim(), "GaudiRT: cat rank mismatch");
    for (int64_t dim = 0; dim < tensor.dim(); ++dim)
      TORCH_CHECK(dim == axis || tensor.size(dim) == sizes[dim], "GaudiRT: cat size mismatch");
    TORCH_CHECK(tensor.size(axis) <= std::numeric_limits<int64_t>::max() - sizes[axis],
                "GaudiRT: cat size overflow");
    sizes[axis] += tensor.size(axis);
    if (tensor.numel()) active.push_back(tensor.scalar_type() == dtype ? tensor : tensor.to(dtype));
  }
  auto result = at::empty(sizes, reference->options().dtype(dtype));
  if (!result.numel()) return result;
  TORCH_CHECK(!active.empty(), "GaudiRT: cat has no nonempty inputs");
  if (active.size() == 1) {
    copy(result, active.front(), true);
    return result;
  }
  // One-dimensional concatenation is a sequence of copies into disjoint
  // output ranges.  Keeping it on the compute stream avoids a shape-specific
  // graph for changing metadata lengths, and also handles strided inputs.
  if (reference->dim() == 1) {
    int64_t offset = 0;
    for (const auto& tensor : active) {
      auto slice = result.narrow(0, offset, tensor.numel());
      copy(slice, tensor, true);
      offset += tensor.numel();
    }
    return result;
  }
  auto ctx = current();
  std::vector<std::pair<uint64_t, uint64_t>> regions;
  for (auto& tensor : active) {
    auto address = address_of(tensor, *ctx);
    const bool overlaps = std::any_of(regions.begin(), regions.end(), [&](const auto& region) {
      return address < region.first + region.second && region.first < address + tensor.nbytes();
    });
    if (!tensor.is_contiguous() || address % BlockCachingAllocator::kAlignment || overlaps) {
      auto staged = at::empty(tensor.sizes(), tensor.options());
      copy(staged, tensor, true);
      tensor = std::move(staged);
      address = address_of(tensor, *ctx);
    }
    regions.emplace_back(address, tensor.nbytes());
  }
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = cat_recipe(*ctx, active, result, axis);
  auto stream = ctx->stream();
  auto bindings = recipe.tensors;
  for (size_t i = 0; i < active.size(); ++i) {
    bindings[i].pTensorAddress = address_of(active[i], *ctx);
    ctx->allocator.record_stream(bindings[i].pTensorAddress, stream);
  }
  bindings.back().pTensorAddress = address_of(result, *ctx);
  ctx->allocator.record_stream(bindings.back().pTensorAddress, stream);
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: cat launch failed");
  return result;
}

at::Tensor& cat_out(const at::ITensorListRef& tensors, int64_t axis, at::Tensor& out) {
  auto value = gaudirt::generic::cat(tensors, axis);
  TORCH_CHECK(out.sizes() == value.sizes() && out.scalar_type() == value.scalar_type(),
              "GaudiRT: cat out must already have the expected shape and dtype");
  return copy(out, value, true);
}

ColdRecipe& range_recipe(Context& ctx, uint32_t count, at::ScalarType dtype, bool wide) {
  const auto type = cast_type(dtype);
  const std::string key = "cold_range_v3_" + std::string(type.name) + "_" + std::to_string(count) +
                          (wide ? "_wide" : "_narrow");
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(3);
  {
    SynapseGraphBuilder builder(key, *recipe);
    auto start = builder.tensor(syn_type_int32, {1}, 0);
    auto step = builder.tensor(syn_type_int32, {1}, 1);
    auto out = builder.tensor(type.syn, {count}, 2);
    auto idx_i32 = builder.tensor(syn_type_int32, {count}, -1);
    ns_RangeKernel::Params params{};
    params.start.i = 0;
    params.limit.i = count;
    params.delta.i = 1;
    builder.node(NodeSpec{"range_i32", {}, {idx_i32}, params});
    ns_CastKernel::ParamsV3 cast{};
    cast.round_mode = CAST_ROUND_ZERO;
    if (wide) {
      auto idx_i64 = builder.tensor(syn_type_int64, {count}, -1);
      auto start_i64 = builder.tensor(syn_type_int64, {1}, -1);
      auto step_i64 = builder.tensor(syn_type_int64, {1}, -1);
      auto prod_i64 = builder.tensor(syn_type_int64, {count}, -1);
      auto sum_i64 = dtype == at::kInt ? builder.tensor(syn_type_int64, {count}, -1) : out;
      builder.node(NodeSpec{"cast_i32_to_i64", {idx_i32}, {idx_i64}, cast});
      builder.node(NodeSpec{"cast_i32_to_i64", {start}, {start_i64}, cast});
      builder.node(NodeSpec{"cast_i32_to_i64", {step}, {step_i64}, cast});
      builder.node(NodeSpec{"mult_fwd_i64", {idx_i64, step_i64}, {prod_i64}, {}});
      builder.node(NodeSpec{"add_fwd_i64", {prod_i64, start_i64}, {sum_i64}, {}});
      if (dtype == at::kInt) builder.node(NodeSpec{"cast_i64_to_i32", {sum_i64}, {out}, cast});
    } else {
      auto prod_i32 = builder.tensor(syn_type_int32, {count}, -1);
      auto sum_i32 = dtype == at::kLong ? builder.tensor(syn_type_int32, {count}, -1) : out;
      builder.node(NodeSpec{"mult_fwd_i32", {idx_i32, step}, {prod_i32}, {}});
      builder.node(NodeSpec{"add_fwd_i32", {prod_i32, start}, {sum_i32}, {}});
      if (dtype == at::kLong) builder.node(NodeSpec{"cast_i32_to_i64", {sum_i32}, {out}, cast});
    }
    builder.compile(ctx.cache, key);
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate cold range recipe");
  return *it->second;
}

ColdRecipe& range64_recipe(Context& ctx, uint32_t count, at::ScalarType dtype) {
  const auto type = cast_type(dtype);
  const std::string key = "cold_range64_v1_" + std::string(type.name) + "_" + std::to_string(count);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(3);
  {
    SynapseGraphBuilder builder(key, *recipe);
    auto start = builder.tensor(syn_type_int64, {1}, 0);
    auto step = builder.tensor(syn_type_int64, {1}, 1);
    auto out = builder.tensor(type.syn, {count}, 2);
    auto idx_i32 = builder.tensor(syn_type_int32, {count}, -1);
    auto idx_i64 = builder.tensor(syn_type_int64, {count}, -1);
    auto prod_i64 = builder.tensor(syn_type_int64, {count}, -1);
    auto sum_i64 = dtype == at::kInt ? builder.tensor(syn_type_int64, {count}, -1) : out;
    ns_RangeKernel::Params params{};
    params.start.i = 0;
    params.limit.i = count;
    params.delta.i = 1;
    builder.node(NodeSpec{"range_i32", {}, {idx_i32}, params});
    ns_CastKernel::ParamsV3 cast{};
    cast.round_mode = CAST_ROUND_ZERO;
    builder.node(NodeSpec{"cast_i32_to_i64", {idx_i32}, {idx_i64}, cast});
    builder.node(NodeSpec{"mult_fwd_i64", {idx_i64, step}, {prod_i64}, {}});
    builder.node(NodeSpec{"add_fwd_i64", {prod_i64, start}, {sum_i64}, {}});
    if (dtype == at::kInt) builder.node(NodeSpec{"cast_i64_to_i32", {sum_i64}, {out}, cast});
    builder.compile(ctx.cache, key);
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate wide-parameter range recipe");
  return *it->second;
}

at::Tensor pointwise(std::vector<at::Tensor> inputs, const std::vector<int64_t>& shape,
                     at::ScalarType output_type, const std::string& op,
                     const std::vector<synDataType>& types);
at::Tensor& arange_out(const at::Scalar& start_value, const at::Scalar& end_value,
                       const at::Scalar& step_value, at::Tensor& out);

at::Tensor& arange_dyadic_out(const at::Scalar& start_value, const at::Scalar& end_value,
                              const at::Scalar& step_value, at::Tensor& out) {
  TORCH_CHECK(out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16,
              "GaudiRT: noninteger arange parameters require a floating output");
  const double start = start_value.toDouble(), end = end_value.toDouble(), step = step_value.toDouble();
  TORCH_CHECK(std::isfinite(start) && std::isfinite(end) && std::isfinite(step) && step != 0 &&
              !(start == 0.0 && std::signbit(start)), "GaudiRT: unsupported floating arange bounds");
  TORCH_CHECK((step > 0 && start <= end) || (step < 0 && start >= end),
              "GaudiRT: arange bounds and step are inconsistent");
  const double count_double = std::ceil((end - start) / step);
  TORCH_CHECK(std::isfinite(count_double) && count_double >= 0 && count_double <= (1 << 24),
              "GaudiRT: arange is too large");
  const int64_t count = static_cast<int64_t>(count_double);
  const std::array<int64_t, 1> shape{count};
  const int64_t exact_limit = out.scalar_type() == at::kBFloat16 ? 1 << 8 : 1 << 24;
  int scale_shift = -1;
  int64_t scaled_start = 0, scaled_step = 0;
  for (int shift = 0; shift <= 16; ++shift) {
    const double a = std::ldexp(start, shift), s = std::ldexp(step, shift);
    if (!std::isfinite(a) || !std::isfinite(s) || std::trunc(a) != a || std::trunc(s) != s ||
        std::abs(a) > exact_limit || std::abs(s) > exact_limit) continue;
    const auto ai = static_cast<int64_t>(a), si = static_cast<int64_t>(s);
    const __int128 offset = static_cast<__int128>(count - 1) * si;
    const __int128 last = static_cast<__int128>(ai) + offset;
    if (offset < -exact_limit || offset > exact_limit || last < -exact_limit || last > exact_limit) continue;
    scale_shift = shift;
    scaled_start = ai;
    scaled_step = si;
    break;
  }
  TORCH_CHECK(!count || scale_shift >= 0, "GaudiRT: floating arange exceeds the verified dyadic range");
  if (out.sizes() != at::IntArrayRef(shape)) {
    TORCH_CHECK(out.numel() == 0, "GaudiRT: nonempty arange out has the wrong shape");
    auto replacement = at::empty(shape, out.options());
    set_storage(out, replacement.storage(), 0, shape, {});
  }
  if (!count) return out;
  auto integral = at::empty(shape, out.options().dtype(at::kInt));
  const int64_t scaled_end = scaled_start + count * scaled_step;
  arange_out(at::Scalar(scaled_start), at::Scalar(scaled_end), at::Scalar(scaled_step), integral);
  auto values = integral.to(at::kFloat);
  auto scale = at::empty({}, out.options().dtype(at::kFloat));
  fill(scale, at::Scalar(std::ldexp(1.0, -scale_shift)));
  const std::vector<int64_t> result_shape{count};
  auto result = pointwise({values, scale}, result_shape, at::kFloat, "mult_fwd_f32",
                          {syn_type_float, syn_type_float, syn_type_float});
  return copy(out, result, true);
}

at::Tensor& arange_out(const at::Scalar& start_value, const at::Scalar& end_value,
                       const at::Scalar& step_value, at::Tensor& out) {
  if (!start_value.isIntegral(false) || !end_value.isIntegral(false) || !step_value.isIntegral(false))
    return arange_dyadic_out(start_value, end_value, step_value, out);
  TORCH_CHECK(out.scalar_type() == at::kInt || out.scalar_type() == at::kLong ||
              out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16,
              "GaudiRT: arange dtype is not implemented");
  const int64_t start = start_value.toLong(), end = end_value.toLong(), step = step_value.toLong();
  TORCH_CHECK(step != 0, "GaudiRT: arange step must be nonzero");
  TORCH_CHECK((step > 0 && start <= end) || (step < 0 && start >= end),
              "GaudiRT: arange bounds and step are inconsistent");
  const __int128 distance = step > 0 ? static_cast<__int128>(end) - start :
                                       static_cast<__int128>(start) - end;
  const __int128 magnitude = step > 0 ? static_cast<__int128>(step) : -static_cast<__int128>(step);
  __int128 count128 = distance / magnitude + (distance % magnitude != 0);
  if (out.scalar_type() == at::kLong) {
    const __int128 difference = static_cast<__int128>(end) - start;
    const __int128 sum = difference + step;
    const __int128 adjusted = sum - (step > 0 ? 1 : -1);
    TORCH_CHECK(difference >= INT64_MIN && difference <= INT64_MAX &&
                sum >= INT64_MIN && sum <= INT64_MAX &&
                adjusted >= INT64_MIN && adjusted <= INT64_MAX,
                "GaudiRT: arange int64 size calculation exceeds the supported range");
  }
  if (out.scalar_type() == at::kInt) {
    // PyTorch computes the INT32 output size from FP64 scalar bounds.
    const double count_double = std::ceil((static_cast<double>(end) - static_cast<double>(start)) /
                                          static_cast<double>(step));
    TORCH_CHECK(count_double >= 0.0 && count_double <= (1 << 24), "GaudiRT: arange is too large");
    count128 = static_cast<int64_t>(count_double);
  }
  TORCH_CHECK(count128 <= (1 << 24), "GaudiRT: arange is too large");
  const int64_t count = static_cast<int64_t>(count128);
  const __int128 last_offset = count ? static_cast<__int128>(count - 1) * step : 0;
  const __int128 last_value = static_cast<__int128>(start) + last_offset;
  const bool wide = last_offset < INT32_MIN || last_offset > INT32_MAX ||
                    last_value < INT32_MIN || last_value > INT32_MAX;
  const bool wide_parameters = start < INT32_MIN || start > INT32_MAX ||
                               step < INT32_MIN || step > INT32_MAX;
  const bool floating = out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16;
  if (floating && count) {
    const int64_t exact_limit = out.scalar_type() == at::kBFloat16 ? 1 << 8 : 1 << 24;
    const auto in_exact_range = [exact_limit](__int128 value) {
      return value >= -exact_limit && value <= exact_limit;
    };
    TORCH_CHECK(in_exact_range(start) && in_exact_range(step) &&
                in_exact_range(last_offset) && in_exact_range(last_value),
                "GaudiRT: floating arange exceeds the verified exact integer range");
  }
  const std::array<int64_t, 1> shape{count};
  if (out.sizes() != at::IntArrayRef(shape)) {
    TORCH_CHECK(out.numel() == 0, "GaudiRT: nonempty arange out has the wrong shape");
    auto replacement = at::empty(shape, out.options());
    set_storage(out, replacement.storage(), 0, shape, {});
  }
  if (!count) return out;
  if (floating) {
    auto integral = at::empty(shape, out.options().dtype(at::kInt));
    arange_out(start_value, end_value, step_value, integral);
    return copy(out, integral, true);
  }
  at::assert_no_internal_overlap(out);
  auto ctx = current();
  const auto target = address_of(out, *ctx);
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = wide_parameters ? range64_recipe(*ctx, count, out.scalar_type()) :
                                   range_recipe(*ctx, count, out.scalar_type(), wide);
  auto stream = ctx->stream();
  ctx->allocator.record_stream(target, stream);
  TemporaryDevice scalars(*ctx, 512);
  TemporaryHost host(*ctx, 512);
  if (wide_parameters) {
    std::memcpy(host.address, &start, sizeof(start));
    std::memcpy(static_cast<uint8_t*>(host.address) + 256, &step, sizeof(step));
  } else {
    const int32_t start32 = start, step32 = step;
    std::memcpy(host.address, &start32, sizeof(start32));
    std::memcpy(static_cast<uint8_t*>(host.address) + 256, &step32, sizeof(step32));
  }
  auto h2d = ctx->device.h2d_stream();
  ctx->allocator.record_stream(scalars.address, h2d);
  auto& reusable = ctx->streams.state(active_stream_id).h2d_reusable;
  reusable.record_and_wait(stream, h2d);
  ctx->pinned.copy_h2d(scalars.address, host.address, 512, h2d, true);
  auto& ready = ctx->streams.state(active_stream_id).h2d_ready;
  ready.record_and_wait(h2d, stream);
  std::unique_ptr<TemporaryDevice> output, workspace;
  if (!out.is_contiguous() || target % BlockCachingAllocator::kAlignment)
    output = std::make_unique<TemporaryDevice>(*ctx, out.nbytes());
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  auto bindings = recipe.tensors;
  bindings[0].pTensorAddress = scalars.address;
  bindings[1].pTensorAddress = scalars.address + 256;
  bindings[2].pTensorAddress = output ? output->address : target;
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: arange launch failed");
  if (output) {
    const Layout layout(out);
    dma_copy(target, &layout, output->address, nullptr, 0, count, out.element_size(),
             stream, DRAM_TO_DRAM);
  }
  return out;
}

at::Tensor& arange_end_out(const at::Scalar& end, at::Tensor& out) {
  return arange_out(at::Scalar(0), end, at::Scalar(1), out);
}

ColdRecipe& pointwise_recipe(Context& ctx, const std::string& op,
                             const std::vector<synDataType>& types, uint32_t count) {
  std::string key = "cold_pointwise_" + op + "_" + std::to_string(count);
  for (auto type : types) key += "_" + std::to_string(static_cast<int>(type));
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(types.size());
  {
    SynapseGraphBuilder builder(key, *recipe);
    std::vector<synTensor> inputs;
    inputs.reserve(types.size() - 1);
    synTensor out = nullptr;
    for (size_t i = 0; i < types.size(); ++i) {
      auto handle = builder.tensor(types[i], {count}, static_cast<int>(i));
      if (i + 1 < types.size()) inputs.push_back(handle);
      else out = handle;
    }
    if (op.rfind("not_equal_fwd_", 0) == 0) {
      auto eq_u8 = builder.tensor(syn_type_uint8, {count}, -1);
      builder.node(NodeSpec{"equal_fwd_" + op.substr(std::strlen("not_equal_fwd_")),
                            std::move(inputs), {eq_u8}, {}});
      builder.node(NodeSpec{"not_fwd_u8", {eq_u8}, {out}, {}});
    } else {
      builder.node(NodeSpec{op, std::move(inputs), {out}, {}});
    }
    builder.compile(ctx.cache, key);
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate cold pointwise recipe");
  return *it->second;
}

at::Tensor pointwise(std::vector<at::Tensor> inputs, const std::vector<int64_t>& shape,
                     at::ScalarType output_type, const std::string& op,
                     const std::vector<synDataType>& types) {
  auto result = at::empty(shape, inputs[0].options().dtype(output_type));
  if (!result.numel()) return result;
  TORCH_CHECK(static_cast<uint64_t>(result.numel()) <= UINT32_MAX,
              "GaudiRT: pointwise tensor is too large");
  auto ctx = current();
  std::vector<std::pair<uint64_t, uint64_t>> regions;
  for (auto& input : inputs) {
    auto expanded = input.expand(shape);
    auto address = address_of(expanded, *ctx);
    const bool overlaps = std::any_of(regions.begin(), regions.end(), [&](const auto& region) {
      return address < region.first + region.second && region.first < address + expanded.nbytes();
    });
    if (!expanded.is_contiguous() || address % BlockCachingAllocator::kAlignment || overlaps) {
      auto staged = at::empty(shape, input.options());
      copy(staged, expanded, true);
      input = std::move(staged);
    } else {
      input = std::move(expanded);
    }
    regions.emplace_back(address_of(input, *ctx), input.nbytes());
  }
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = pointwise_recipe(*ctx, op, types, result.numel());
  auto stream = ctx->stream();
  auto bindings = recipe.tensors;
  for (size_t i = 0; i < inputs.size(); ++i) {
    bindings[i].pTensorAddress = address_of(inputs[i], *ctx);
    ctx->allocator.record_stream(bindings[i].pTensorAddress, stream);
  }
  bindings.back().pTensorAddress = address_of(result, *ctx);
  ctx->allocator.record_stream(bindings.back().pTensorAddress, stream);
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: pointwise launch failed");
  return result;
}

at::Tensor compare(const at::Tensor& a, const at::Tensor& b, const char* op) {
  TORCH_CHECK(a.device() == b.device() && a.device().type() == c10::kPrivateUse1,
              "GaudiRT: comparison requires same-device tensors");
  const auto dtype = at::result_type(a, b);
  auto left = a.scalar_type() == dtype ? a : a.to(dtype);
  auto right = b.scalar_type() == dtype ? b : b.to(dtype);
  if (dtype == at::kLong && std::strcmp(op, "greater") == 0)
    return compare(right, left, "less");
  if (dtype == at::kLong && std::strcmp(op, "less_equal") == 0)
    return compare(right, left, "greater_equal");
  const auto type = cast_type(dtype);
  const auto inferred = at::infer_size(left.sizes(), right.sizes());
  const std::vector<int64_t> shape(inferred.begin(), inferred.end());
  return pointwise({left, right}, shape, at::kBool, std::string(op) + "_fwd_" + type.name,
                   {type.syn, type.syn, syn_type_uint8});
}

at::Tensor compare_scalar(const at::Tensor& a, const at::Scalar& scalar, const char* op) {
  const auto dtype = at::result_type(a, scalar);
  auto input = a.scalar_type() == dtype ? a : a.to(dtype);
  auto rhs = at::empty({}, input.options());
  if (dtype == at::kInt && scalar.isIntegral(false)) {
    const auto bits = static_cast<uint32_t>(scalar.toLong());
    fill(rhs, at::Scalar(std::bit_cast<int32_t>(bits)));
  } else {
    fill(rhs, scalar);
  }
  return compare(input, rhs, op);
}

#define GAUDIRT_COMPARE(name, guid) \
  at::Tensor name##_tensor(const at::Tensor& a, const at::Tensor& b) { return compare(a, b, guid); } \
  at::Tensor name##_scalar(const at::Tensor& a, const at::Scalar& b) { return compare_scalar(a, b, guid); }
GAUDIRT_COMPARE(eq, "equal")
GAUDIRT_COMPARE(ne, "not_equal")
GAUDIRT_COMPARE(lt, "less")
GAUDIRT_COMPARE(le, "less_equal")
GAUDIRT_COMPARE(gt, "greater")
GAUDIRT_COMPARE(ge, "greater_equal")
#undef GAUDIRT_COMPARE

at::Tensor subtract_scalar(const at::Tensor& a, const at::Scalar& b, const at::Scalar& alpha);

at::Tensor subtract(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  if (a.device().type() == c10::kPrivateUse1 && b.device().is_cpu() && b.dim() == 0)
    return subtract_scalar(a, b.item(), alpha);
  if (b.device().type() == c10::kPrivateUse1 && a.device().is_cpu() && a.dim() == 0) {
    auto scalar = at::empty({}, b.options().dtype(at::result_type(a, b)));
    fill(scalar, a.item());
    return gaudirt::generic::subtract(scalar, b, alpha);
  }
  TORCH_CHECK(a.device() == b.device() && a.device().type() == c10::kPrivateUse1,
              "GaudiRT: subtraction requires same-device tensors");
  TORCH_CHECK(alpha.toDouble() == 1.0, "GaudiRT: subtraction supports alpha=1 only");
  const auto dtype = at::result_type(a, b);
  TORCH_CHECK(dtype == at::kInt || dtype == at::kLong || dtype == at::kFloat || dtype == at::kBFloat16,
              "GaudiRT: subtraction dtype is not supported");
  auto left = a.scalar_type() == dtype ? a : a.to(dtype);
  auto right = b.scalar_type() == dtype ? b : b.to(dtype);
  const auto inferred = at::infer_size(left.sizes(), right.sizes());
  const std::vector<int64_t> shape(inferred.begin(), inferred.end());
  const auto type = cast_type(dtype);
  if (dtype == at::kLong) {
    auto minus_one = at::empty({}, left.options());
    fill(minus_one, at::Scalar(int64_t{-1}));
    auto negated = pointwise({right, minus_one}, shape, dtype, "mult_fwd_i64",
                             {type.syn, type.syn, type.syn});
    return pointwise({left, negated}, shape, dtype, "add_fwd_i64", {type.syn, type.syn, type.syn});
  }
  return pointwise({left, right}, shape, dtype, std::string("sub_fwd_") + type.name,
                   {type.syn, type.syn, type.syn});
}

at::Tensor subtract_scalar(const at::Tensor& a, const at::Scalar& b, const at::Scalar& alpha) {
  const auto dtype = at::result_type(a, b);
  auto left = a.scalar_type() == dtype ? a : a.to(dtype);
  auto right = at::empty({}, left.options());
  fill(right, b);
  return gaudirt::generic::subtract(left, right, alpha);
}

at::Tensor& subtract_out(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha, at::Tensor& out) {
  auto value = gaudirt::generic::subtract(a, b, alpha);
  TORCH_CHECK(out.device() == value.device() && out.scalar_type() == value.scalar_type() &&
              out.sizes() == value.sizes(), "GaudiRT: subtraction output geometry or dtype mismatch");
  return copy(out, value, true);
}

at::Tensor& subtract_scalar_out(const at::Tensor& a, const at::Scalar& b, const at::Scalar& alpha,
                                at::Tensor& out) {
  auto value = subtract_scalar(a, b, alpha);
  TORCH_CHECK(out.device() == value.device() && out.scalar_type() == value.scalar_type() &&
              out.sizes() == value.sizes(), "GaudiRT: subtraction output geometry or dtype mismatch");
  return copy(out, value, true);
}

at::Tensor add_scalar(const at::Tensor& a, const at::Scalar& b, const at::Scalar& alpha);

at::Tensor add(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  if (a.device().type() == c10::kPrivateUse1 && b.device().is_cpu() && b.dim() == 0)
    return add_scalar(a, b.item(), alpha);
  if (b.device().type() == c10::kPrivateUse1 && a.device().is_cpu() && a.dim() == 0)
    return add_scalar(b, a.item(), alpha);
  TORCH_CHECK(a.device() == b.device() && a.device().type() == c10::kPrivateUse1,
              "GaudiRT: addition requires same-device tensors");
  TORCH_CHECK(alpha.toDouble() == 1.0, "GaudiRT: addition supports alpha=1 only");
  const auto dtype = at::result_type(a, b);
  TORCH_CHECK(dtype == at::kInt || dtype == at::kLong || dtype == at::kFloat || dtype == at::kBFloat16,
              "GaudiRT: addition dtype is not supported");
  auto left = a.scalar_type() == dtype ? a : a.to(dtype);
  auto right = b.scalar_type() == dtype ? b : b.to(dtype);
  const auto inferred = at::infer_size(left.sizes(), right.sizes());
  const std::vector<int64_t> shape(inferred.begin(), inferred.end());
  const auto type = cast_type(dtype);
  return pointwise({left, right}, shape, dtype, std::string("add_fwd_") + type.name,
                   {type.syn, type.syn, type.syn});
}

at::Tensor add_scalar(const at::Tensor& a, const at::Scalar& b, const at::Scalar& alpha) {
  const auto dtype = at::result_type(a, b);
  auto left = a.scalar_type() == dtype ? a : a.to(dtype);
  auto right = at::empty({}, left.options());
  fill(right, b);
  return gaudirt::generic::add(left, right, alpha);
}

at::Tensor& add_out(const at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha, at::Tensor& out) {
  auto value = gaudirt::generic::add(a, b, alpha);
  TORCH_CHECK(out.device() == value.device() && out.scalar_type() == value.scalar_type() &&
              out.sizes() == value.sizes(), "GaudiRT: addition output geometry or dtype mismatch");
  return copy(out, value, true);
}

at::Tensor& add_scalar_out(const at::Tensor& a, const at::Scalar& b, const at::Scalar& alpha,
                           at::Tensor& out) {
  auto value = add_scalar(a, b, alpha);
  TORCH_CHECK(out.device() == value.device() && out.scalar_type() == value.scalar_type() &&
              out.sizes() == value.sizes(), "GaudiRT: addition output geometry or dtype mismatch");
  return copy(out, value, true);
}

at::Tensor where(const at::Tensor& condition, const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(condition.scalar_type() == at::kBool && condition.device() == a.device() &&
              a.device() == b.device() && a.device().type() == c10::kPrivateUse1,
              "GaudiRT: where requires a boolean condition and same-device values");
  const auto dtype = at::result_type(a, b);
  if (a.scalar_type() != dtype || b.scalar_type() != dtype) {
    auto left = a.scalar_type() == dtype ? a : a.to(dtype);
    auto right = b.scalar_type() == dtype ? b : b.to(dtype);
    return gaudirt::generic::where(condition, left, right);
  }
  cast_type(a.scalar_type());
  const auto first = at::infer_size(a.sizes(), b.sizes());
  const auto inferred = at::infer_size(first, condition.sizes());
  const std::vector<int64_t> shape(inferred.begin(), inferred.end());
  if (a.scalar_type() == at::kBool)
    return gaudirt::generic::where(condition, a.to(at::kInt), b.to(at::kInt)).to(at::kBool);
  if (a.scalar_type() == at::kLong) {
    auto left = at::empty(shape, a.options());
    auto right = at::empty(shape, b.options());
    if (!left.numel()) return left;
    copy(left, a.expand(shape), true);
    copy(right, b.expand(shape), true);
    std::vector<int64_t> lanes = shape;
    lanes.push_back(2);
    auto left_lanes = left.view({-1}).view(at::kInt).view(lanes);
    auto right_lanes = right.view({-1}).view(at::kInt).view(lanes);
    auto mask_lanes = condition.expand(shape).unsqueeze(-1).expand(lanes);
    auto selected = pointwise({mask_lanes, left_lanes, right_lanes}, lanes, at::kInt,
                              "where_fwd_i32", {syn_type_uint8, syn_type_int32,
                                                syn_type_int32, syn_type_int32});
    return selected.view({-1}).view(at::kLong).view(shape);
  }
  const bool bf16 = a.scalar_type() == at::kBFloat16;
  const auto type = bf16 ? syn_type_bf16 : syn_type_int32;
  return pointwise({condition, a, b}, shape, a.scalar_type(), bf16 ? "where_fwd_bf16" : "where_fwd_i32",
                   {syn_type_uint8, type, type, type});
}

}  // namespace gaudirt::generic

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("record_stream", TORCH_FN(gaudirt::generic::record_stream));
  m.impl("copy_", TORCH_FN(gaudirt::generic::copy));
  m.impl("empty.memory_format", TORCH_FN(gaudirt::generic::empty));
  m.impl("empty_strided", TORCH_FN(gaudirt::generic::empty_strided));
  m.impl("as_strided", TORCH_FN(at::native::as_strided_tensorimpl));
  m.impl("view", TORCH_FN(at::native::view));
  m.impl("_reshape_alias", TORCH_FN(at::native::_reshape_alias));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(gaudirt::generic::set_storage));
  m.impl("fill_.Scalar", TORCH_FN(gaudirt::generic::fill));
  m.impl("zero_", TORCH_FN(gaudirt::generic::zero));
  m.impl("cat", TORCH_FN(gaudirt::generic::cat));
  m.impl("cat.out", TORCH_FN(gaudirt::generic::cat_out));
  m.impl("arange.start_out", TORCH_FN(gaudirt::generic::arange_out));
  m.impl("arange.out", TORCH_FN(gaudirt::generic::arange_end_out));
  m.impl("eq.Tensor", TORCH_FN(gaudirt::generic::eq_tensor));
  m.impl("eq.Scalar", TORCH_FN(gaudirt::generic::eq_scalar));
  m.impl("ne.Tensor", TORCH_FN(gaudirt::generic::ne_tensor));
  m.impl("ne.Scalar", TORCH_FN(gaudirt::generic::ne_scalar));
  m.impl("lt.Tensor", TORCH_FN(gaudirt::generic::lt_tensor));
  m.impl("lt.Scalar", TORCH_FN(gaudirt::generic::lt_scalar));
  m.impl("le.Tensor", TORCH_FN(gaudirt::generic::le_tensor));
  m.impl("le.Scalar", TORCH_FN(gaudirt::generic::le_scalar));
  m.impl("gt.Tensor", TORCH_FN(gaudirt::generic::gt_tensor));
  m.impl("gt.Scalar", TORCH_FN(gaudirt::generic::gt_scalar));
  m.impl("ge.Tensor", TORCH_FN(gaudirt::generic::ge_tensor));
  m.impl("ge.Scalar", TORCH_FN(gaudirt::generic::ge_scalar));
  m.impl("sub.Tensor", TORCH_FN(gaudirt::generic::subtract));
  m.impl("sub.Scalar", TORCH_FN(gaudirt::generic::subtract_scalar));
  m.impl("sub.out", TORCH_FN(gaudirt::generic::subtract_out));
  m.impl("sub.Scalar_out", TORCH_FN(gaudirt::generic::subtract_scalar_out));
  m.impl("add.Tensor", TORCH_FN(gaudirt::generic::add));
  m.impl("add.Scalar", TORCH_FN(gaudirt::generic::add_scalar));
  m.impl("add.out", TORCH_FN(gaudirt::generic::add_out));
  m.impl("add.Scalar_out", TORCH_FN(gaudirt::generic::add_scalar_out));
  m.impl("where.self", TORCH_FN(gaudirt::generic::where));
  m.impl("index_select", TORCH_FN(gaudirt::generic::index_select));
  m.impl("index.Tensor", TORCH_FN(gaudirt::generic::index_tensor));
  m.impl("_index_put_impl_", TORCH_FN(gaudirt::generic::index_put));
  m.impl("embedding", TORCH_FN(gaudirt::generic::embedding));
  m.impl("argmax", TORCH_FN(gaudirt::generic::argmax));
}
