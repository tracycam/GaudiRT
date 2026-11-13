// SPDX-License-Identifier: Apache-2.0
#include "eager_ops.hpp"

#include "bridge_internal.hpp"
#include "bucket_program.hpp"
#include "eager_copy.hpp"
#include "eager_index.hpp"
#include "fx_node.hpp"
#include "fx_op_lowering.hpp"
#include "graph_builder.hpp"
#include "layout_copy.hpp"
#include "temporaries.hpp"
#include "gaudirt/common/sha256.hpp"

#include <ATen/EmptyTensor.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <c10/core/SymIntArrayRef.h>
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
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
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
  const auto type = cast_type(a.scalar_type());
  return pointwise({condition, a, b}, shape, a.scalar_type(),
                   std::string("where_fwd_") + type.name,
                   {syn_type_uint8, type.syn, type.syn, type.syn});
}

namespace {

std::vector<uint64_t> to_syn_dims(const at::Tensor& tensor) {
  if (tensor.dim() == 0) return {1};
  std::vector<uint64_t> dims(tensor.dim());
  for (int64_t d = 0; d < tensor.dim(); ++d) {
    dims[d] = static_cast<uint64_t>(tensor.size(tensor.dim() - 1 - d));
  }
  return dims;
}

std::string format_dims_key(const std::vector<uint64_t>& dims) {
  std::string out;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i) out += "x";
    out += std::to_string(dims[i]);
  }
  return out;
}

template <typename BuildFn>
void launch_cold_recipe(const std::string& prefix, const std::string& raw_signature,
                        std::vector<at::Tensor> inputs,
                        const std::vector<at::Tensor>& outputs, BuildFn&& build_fn) {
  auto ctx = current();
  auto stream = ctx->stream();
  std::vector<std::pair<uint64_t, uint64_t>> regions;
  regions.reserve(inputs.size() + outputs.size());
  for (const auto& out : outputs) {
    TORCH_CHECK(out.device().type() == c10::kPrivateUse1 &&
                out.device().index() == ctx->device_index && out.is_contiguous(),
                "GaudiRT: cold recipe outputs must be contiguous on the active device");
    const auto addr = address_of(out, *ctx);
    TORCH_CHECK(addr % BlockCachingAllocator::kAlignment == 0,
                "GaudiRT: cold recipe outputs must be aligned");
    regions.emplace_back(addr, out.nbytes());
  }
  for (auto& input : inputs) {
    TORCH_CHECK(input.device().type() == c10::kPrivateUse1 &&
                input.device().index() == ctx->device_index,
                "GaudiRT: cold recipe inputs must reside on the active device");
    auto addr = address_of(input, *ctx);
    const bool overlaps = std::any_of(regions.begin(), regions.end(), [&](const auto& region) {
      return addr < region.first + region.second && region.first < addr + input.nbytes();
    });
    if (!input.is_contiguous() || addr % BlockCachingAllocator::kAlignment != 0 || overlaps) {
      auto staged = at::empty(input.sizes(), input.options());
      copy(staged, input, true);
      input = std::move(staged);
      addr = address_of(input, *ctx);
    }
    regions.emplace_back(addr, input.nbytes());
  }

  const auto digest = common::sha256_bytes(
      reinterpret_cast<const uint8_t*>(raw_signature.data()), raw_signature.size());
  const std::string key = prefix + "_" + digest;

  std::lock_guard lock(ctx->cold_mutex);
  ColdRecipe* recipe_ptr = nullptr;
  auto found = ctx->cold_recipes.find(key);
  if (found != ctx->cold_recipes.end()) {
    recipe_ptr = found->second.get();
  } else {
    auto recipe = std::make_unique<ColdRecipe>(inputs.size() + outputs.size());
    {
      SynapseGraphBuilder builder("c" + digest.substr(0, 12), *recipe);
      std::vector<synTensor> syn_inputs;
      std::vector<synTensor> syn_outputs;
      syn_inputs.reserve(inputs.size());
      syn_outputs.reserve(outputs.size());
      int ext_idx = 0;
      for (const auto& t : inputs) {
        syn_inputs.push_back(
            builder.tensor(cast_type(t.scalar_type()).syn, to_syn_dims(t), ext_idx++));
      }
      for (const auto& t : outputs) {
        syn_outputs.push_back(
            builder.tensor(cast_type(t.scalar_type()).syn, to_syn_dims(t), ext_idx++));
      }
      build_fn(builder, syn_inputs, syn_outputs);
      builder.compile(ctx->cache, key);
    }
    auto [it, inserted] = ctx->cold_recipes.emplace(key, std::move(recipe));
    TORCH_CHECK(inserted, "GaudiRT: duplicate cold recipe");
    recipe_ptr = it->second.get();
  }

  auto bindings = recipe_ptr->tensors;
  size_t idx = 0;
  for (const auto& t : inputs) {
    bindings[idx].pTensorAddress = address_of(t, *ctx);
    ctx->allocator.record_stream(bindings[idx].pTensorAddress, stream);
    ++idx;
  }
  for (const auto& t : outputs) {
    bindings[idx].pTensorAddress = address_of(t, *ctx);
    ctx->allocator.record_stream(bindings[idx].pTensorAddress, stream);
    ++idx;
  }
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe_ptr->workspace) {
    workspace = std::make_unique<TemporaryDevice>(*ctx, recipe_ptr->workspace);
  }
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe_ptr->recipe, 0) == synSuccess,
              "GaudiRT: cold recipe launch failed");
}

at::Tensor& copy_inplace(at::Tensor& self, const at::Tensor& value) {
  TORCH_CHECK(self.device() == value.device() && self.sizes() == value.sizes(),
              "GaudiRT: in-place or out tensor geometry mismatch");
  if (value.scalar_type() == self.scalar_type()) return copy(self, value, true);
  return copy(self, value.to(self.scalar_type()), true);
}

at::Tensor align_device_operand(const at::Tensor& tensor, const at::Tensor& peer,
                                at::ScalarType dtype) {
  if (tensor.device().is_cpu() && tensor.dim() == 0) {
    auto scalar = at::empty({}, peer.options().dtype(dtype));
    fill(scalar, tensor.item());
    return scalar;
  }
  TORCH_CHECK(tensor.device() == peer.device() && tensor.device().type() == c10::kPrivateUse1,
              "GaudiRT: binary operation requires same-device PrivateUse1 tensors");
  return tensor.scalar_type() == dtype ? tensor : tensor.to(dtype);
}

at::Tensor scalar_to_device(const at::Tensor& peer, const at::Scalar& scalar,
                            at::ScalarType dtype) {
  auto out = at::empty({}, peer.options().dtype(dtype));
  if (dtype == at::kInt && scalar.isIntegral(false)) {
    const auto bits = static_cast<uint32_t>(scalar.toLong());
    fill(out, at::Scalar(std::bit_cast<int32_t>(bits)));
  } else {
    fill(out, scalar);
  }
  return out;
}

at::Tensor unary_ewise_op(const at::Tensor& self, FxUnaryOp op_kind,
                          float alpha = 0.f, float beta = 0.f) {
  TORCH_CHECK(self.device().type() == c10::kPrivateUse1 && self.dim() <= 5,
              "GaudiRT: unary op requires PrivateUse1 tensor of rank <= 5");
  const bool fp_only = op_kind != FxUnaryOp::neg && op_kind != FxUnaryOp::abs &&
                       op_kind != FxUnaryOp::clamp;
  at::Tensor work = self;
  if (fp_only && self.scalar_type() != at::kBFloat16 && self.scalar_type() != at::kFloat) {
    work = self.to(at::kFloat);
  }
  if (!work.numel()) return at::empty(work.sizes(), work.options());

  if (work.scalar_type() == at::kInt || work.scalar_type() == at::kLong) {
    if (op_kind == FxUnaryOp::neg) {
      auto zero = at::empty({}, work.options());
      zero_(zero);
      return gaudirt::generic::subtract(zero, work, 1);
    }
    if (op_kind == FxUnaryOp::abs) {
      auto zero = at::empty({}, work.options());
      zero_(zero);
      auto negated = gaudirt::generic::subtract(zero, work, 1);
      return gaudirt::generic::where(compare(work, zero, "less"), negated, work);
    }
  }

  TORCH_CHECK(work.scalar_type() == at::kBFloat16 || work.scalar_type() == at::kFloat,
              "GaudiRT: unary op supports BF16/FP32 tensors");
  auto out = at::empty(work.sizes(), work.options());
  const auto syn_dims = to_syn_dims(work);
  const auto op_code = static_cast<int32_t>(op_kind);
  std::ostringstream sig;
  sig << "unary_v1:" << op_code << ":" << std::bit_cast<uint32_t>(alpha) << ":"
      << std::bit_cast<uint32_t>(beta) << ":" << cast_type(work.scalar_type()).name << ":"
      << format_dims_key(syn_dims);
  launch_cold_recipe(
      "cold_unary", sig.str(), {work}, {out},
      [&](SynapseGraphBuilder& builder, const std::vector<synTensor>& in,
          const std::vector<synTensor>& out_t) {
        emit_ewise_unary(builder, in[0], out_t[0], work.scalar_type(), syn_dims, op_code, alpha, beta);
      });
  return out;
}

at::Tensor binary_ewise_op(const at::Tensor& a, const at::Tensor& b, FxOp op,
                           bool true_div = false) {
  const auto& dev_peer = a.device().type() == c10::kPrivateUse1 ? a : b;
  TORCH_CHECK(dev_peer.device().type() == c10::kPrivateUse1,
              "GaudiRT: binary operation requires at least one PrivateUse1 tensor");
  at::ScalarType dtype = at::result_type(a, b);
  if (true_div && c10::isIntegralType(dtype, true)) dtype = at::kFloat;
  if (op == FxOp::logical_and || op == FxOp::logical_or) dtype = at::kBool;
  if (dtype == at::kBool && (op == FxOp::minimum || op == FxOp::maximum)) {
    op = (op == FxOp::minimum) ? FxOp::logical_and : FxOp::logical_or;
  }

  auto left = align_device_operand(a, dev_peer, dtype);
  auto right = align_device_operand(b, dev_peer, dtype);
  TORCH_CHECK(left.dim() <= 5 && right.dim() <= 5,
              "GaudiRT: binary operation requires tensor rank <= 5");
  const auto inferred = at::infer_size(left.sizes(), right.sizes());
  const std::vector<int64_t> shape(inferred.begin(), inferred.end());
  auto out = at::empty(shape, dev_peer.options().dtype(dtype));
  if (!out.numel()) return out;

  if (dtype == at::kLong && op == FxOp::mul) {
    return pointwise({left, right}, shape, dtype, "mult_fwd_i64",
                     {syn_type_int64, syn_type_int64, syn_type_int64});
  }
  if (dtype == at::kLong && (op == FxOp::minimum || op == FxOp::maximum)) {
    return gaudirt::generic::where(
        compare(left, right, op == FxOp::minimum ? "less" : "greater"), left, right);
  }
  if (op == FxOp::bitwise_and || op == FxOp::bitwise_or) {
    TORCH_CHECK(dtype == at::kInt || dtype == at::kBool,
                "GaudiRT: bitwise op supports Int32 or Bool tensors");
    if (dtype == at::kBool) {
      op = (op == FxOp::bitwise_and) ? FxOp::logical_and : FxOp::logical_or;
    }
  }

  const auto l_syn = to_syn_dims(left);
  const auto r_syn = to_syn_dims(right);
  const auto o_syn = to_syn_dims(out);
  std::ostringstream sig;
  sig << "bin_v1:" << static_cast<int>(op) << ":" << cast_type(dtype).name << ":"
      << format_dims_key(l_syn) << ":" << format_dims_key(r_syn) << ":"
      << format_dims_key(o_syn);
  launch_cold_recipe(
      "cold_bin", sig.str(), {left, right}, {out},
      [&](SynapseGraphBuilder& builder, const std::vector<synTensor>& in,
          const std::vector<synTensor>& out_t) {
        emit_binary_pointwise(builder, op, in[0], in[1], out_t[0], dtype, l_syn, r_syn, o_syn);
      });
  return out;
}

at::Tensor softmax_impl(const at::Tensor& self, int64_t dim, bool is_log,
                        std::optional<at::ScalarType> dtype) {
  TORCH_CHECK(self.device().type() == c10::kPrivateUse1 && self.dim() <= 5,
              "GaudiRT: softmax requires PrivateUse1 tensor of rank <= 5");
  auto work = dtype ? self.to(*dtype) : self;
  TORCH_CHECK(work.scalar_type() == at::kBFloat16 || work.scalar_type() == at::kFloat,
              "GaudiRT: softmax requires BF16 or FP32 tensor");
  if (work.dim() == 0) {
    auto out = at::empty({}, work.options());
    fill(out, at::Scalar(is_log ? 0.0 : 1.0));
    return out;
  }
  dim = at::maybe_wrap_dim(dim, work.dim());
  auto out = at::empty(work.sizes(), work.options());
  if (!out.numel()) return out;
  const auto syn_dims = to_syn_dims(work);
  std::ostringstream sig;
  sig << "softmax_v1:" << (is_log ? 1 : 0) << ":" << cast_type(work.scalar_type()).name
      << ":" << dim << ":" << format_dims_key(syn_dims);
  launch_cold_recipe(
      "cold_smax", sig.str(), {work}, {out},
      [&](SynapseGraphBuilder& builder, const std::vector<synTensor>& in,
          const std::vector<synTensor>& out_t) {
        emit_softmax(builder, is_log, in[0], out_t[0], work.scalar_type(), syn_dims, dim);
      });
  return out;
}

at::Tensor reduce_dims_impl(const at::Tensor& self, FxOp op,
                            at::OptionalIntArrayRef dims_opt, bool keepdim,
                            std::optional<at::ScalarType> out_dtype) {
  TORCH_CHECK(self.device().type() == c10::kPrivateUse1 && self.dim() <= 5,
              "GaudiRT: reduction requires PrivateUse1 tensor of rank <= 5");
  at::ScalarType compute_dtype = out_dtype.value_or(self.scalar_type());
  if (op == FxOp::sum_dim && compute_dtype == at::kBool) compute_dtype = at::kInt;
  if (op == FxOp::sum_dim && !out_dtype && self.scalar_type() == at::kBool) compute_dtype = at::kInt;
  if (op == FxOp::amax_dim && compute_dtype == at::kInt) compute_dtype = at::kFloat;
  if (op == FxOp::mean_dim) {
    TORCH_CHECK(c10::isFloatingType(compute_dtype), "GaudiRT: mean requires floating dtype");
  }
  at::Tensor cur = self.scalar_type() == compute_dtype ? self : self.to(compute_dtype);
  if (cur.dim() == 0) {
    if (out_dtype && *out_dtype != cur.scalar_type()) return cur.to(*out_dtype);
    return cur.clone();
  }

  std::vector<int64_t> dims;
  if (!dims_opt.has_value() || dims_opt->empty()) {
    dims.resize(cur.dim());
    std::iota(dims.begin(), dims.end(), 0);
  } else {
    for (int64_t d : *dims_opt) {
      int64_t w = at::maybe_wrap_dim(d, cur.dim());
      TORCH_CHECK(std::find(dims.begin(), dims.end(), w) == dims.end(),
                  "GaudiRT: duplicate dim in reduction");
      dims.push_back(w);
    }
  }
  std::sort(dims.rbegin(), dims.rend());

  for (int64_t d : dims) {
    std::vector<int64_t> out_sizes(cur.sizes().begin(), cur.sizes().end());
    if (keepdim) {
      out_sizes[d] = 1;
    } else {
      out_sizes.erase(out_sizes.begin() + d);
    }
    const bool scalar_out = out_sizes.empty();
    std::vector<int64_t> alloc_sizes = scalar_out ? std::vector<int64_t>{1} : out_sizes;
    auto out = at::empty(alloc_sizes, cur.options());
    if (!cur.numel()) {
      TORCH_CHECK(op == FxOp::sum_dim && cur.size(d) == 0,
                  "GaudiRT: mean/amax cannot reduce zero-length dimension");
      zero_(out);
      cur = scalar_out ? out.view({}) : out;
      continue;
    }
    const auto in_syn = to_syn_dims(cur);
    const auto out_syn = to_syn_dims(out);
    std::ostringstream sig;
    sig << "red_v1:" << static_cast<int>(op) << ":" << cast_type(cur.scalar_type()).name
        << ":" << d << ":" << (keepdim ? 1 : 0) << ":" << format_dims_key(in_syn)
        << ":" << format_dims_key(out_syn);
    launch_cold_recipe(
        "cold_red", sig.str(), {cur}, {out},
        [&](SynapseGraphBuilder& builder, const std::vector<synTensor>& in,
            const std::vector<synTensor>& out_t) {
          emit_reduction(builder, op, in[0], out_t[0], cur.scalar_type(),
                         in_syn, out_syn, d, keepdim);
        });
    cur = scalar_out ? out.view({}) : out;
  }
  const at::ScalarType final_dtype = out_dtype.value_or(
      (op == FxOp::sum_dim && self.scalar_type() == at::kBool) ? at::kLong : self.scalar_type());
  if (cur.scalar_type() != final_dtype) cur = cur.to(final_dtype);
  return cur;
}

at::Tensor rms_norm_common(const at::Tensor& input, const at::Tensor& weight,
                           double epsilon, bool gemma) {
  const float eps = static_cast<float>(epsilon);
  TORCH_CHECK(std::isfinite(epsilon) && std::isfinite(eps) && eps > 0,
              "GaudiRT: RMSNorm epsilon must be a positive finite FP32 value");
  TORCH_CHECK(input.device().type() == c10::kPrivateUse1 && weight.device() == input.device() &&
              input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
              "GaudiRT: RMSNorm requires same-device PrivateUse1 BF16 tensors");
  TORCH_CHECK(input.dim() >= 2 && input.dim() <= 5 && weight.dim() == 1 &&
              weight.size(0) == input.size(-1) && input.size(-1) <= 8192,
              "GaudiRT: invalid RMSNorm tensor geometry");
  if (!input.numel()) return at::empty(input.sizes(), input.options());
  if (input.size(0) <= kMaxBucketRows) {
    const auto node = gemma ? FxNode::gemma_norm(0, 1, eps) : FxNode::norm(0, 1, eps);
    BucketProgram program({1, 0}, {node});
    return program.run({input, weight})[0];
  }
  auto out = at::empty(input.sizes(), input.options());
  const auto syn_dims = to_syn_dims(input);
  const float offset = gemma ? 1.0f : 0.0f;
  std::ostringstream sig;
  sig << "rms_v1:" << (gemma ? 1 : 0) << ":" << std::bit_cast<uint32_t>(eps) << ":"
      << format_dims_key(syn_dims);
  launch_cold_recipe(
      "cold_rms", sig.str(), {input, weight}, {out},
      [&](SynapseGraphBuilder& builder, const std::vector<synTensor>& in,
          const std::vector<synTensor>& out_t) {
        emit_rms_norm(builder, in[0], nullptr, in[1], out_t[0], nullptr, syn_dims, eps, offset);
      });
  return out;
}

std::tuple<at::Tensor, at::Tensor> fused_add_rms_norm_common(
    const at::Tensor& input, const at::Tensor& residual, const at::Tensor& weight,
    double epsilon, bool gemma) {
  const float eps = static_cast<float>(epsilon);
  TORCH_CHECK(std::isfinite(epsilon) && std::isfinite(eps) && eps > 0,
              "GaudiRT: fused RMSNorm epsilon must be a positive finite FP32 value");
  TORCH_CHECK(input.device().type() == c10::kPrivateUse1 && residual.device() == input.device() &&
              weight.device() == input.device() && input.scalar_type() == at::kBFloat16 &&
              residual.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16,
              "GaudiRT: fused RMSNorm requires same-device PrivateUse1 BF16 tensors");
  TORCH_CHECK(input.dim() >= 2 && input.dim() <= 5 && residual.sizes() == input.sizes() &&
              weight.dim() == 1 && weight.size(0) == input.size(-1) && input.size(-1) <= 8192,
              "GaudiRT: invalid fused RMSNorm tensor geometry");
  if (!input.numel()) {
    return {at::empty(input.sizes(), input.options()),
            at::empty(residual.sizes(), residual.options())};
  }
  if (input.size(0) <= kMaxBucketRows) {
    const auto node = gemma ? FxNode::fused_gemma_norm(0, 1, 2, eps)
                            : FxNode::fused_norm(0, 1, 2, eps);
    BucketProgram program({1, 1, 0}, {node, FxNode::unary(FxOp::fused_residual, 3)}, {3, 4});
    auto result = program.run({input, residual, weight});
    return {std::move(result[0]), std::move(result[1])};
  }
  auto out_norm = at::empty(input.sizes(), input.options());
  auto out_res = at::empty(residual.sizes(), residual.options());
  const auto syn_dims = to_syn_dims(input);
  const float offset = gemma ? 1.0f : 0.0f;
  std::ostringstream sig;
  sig << "frms_v1:" << (gemma ? 1 : 0) << ":" << std::bit_cast<uint32_t>(eps) << ":"
      << format_dims_key(syn_dims);
  launch_cold_recipe(
      "cold_frms", sig.str(), {input, residual, weight}, {out_norm, out_res},
      [&](SynapseGraphBuilder& builder, const std::vector<synTensor>& in,
          const std::vector<synTensor>& out_t) {
        emit_rms_norm(builder, in[0], in[1], in[2], out_t[0], out_t[1], syn_dims, eps, offset);
      });
  return {std::move(out_norm), std::move(out_res)};
}

}  // namespace

at::Tensor rms_norm(const at::Tensor& input, const at::Tensor& weight, double epsilon) {
  return rms_norm_common(input, weight, epsilon, false);
}

std::tuple<at::Tensor, at::Tensor> fused_add_rms_norm(
    const at::Tensor& input, const at::Tensor& residual, const at::Tensor& weight, double epsilon) {
  return fused_add_rms_norm_common(input, residual, weight, epsilon, false);
}

at::Tensor gemma_rms_norm(const at::Tensor& input, const at::Tensor& weight, double epsilon) {
  return rms_norm_common(input, weight, epsilon, true);
}

std::tuple<at::Tensor, at::Tensor> fused_add_gemma_rms_norm(
    const at::Tensor& input, const at::Tensor& residual, const at::Tensor& weight, double epsilon) {
  return fused_add_rms_norm_common(input, residual, weight, epsilon, true);
}

at::Tensor gelu_and_mul(const at::Tensor& input, std::string_view approximate) {
  TORCH_CHECK(approximate == "none" || approximate == "tanh",
              "GaudiRT: gelu_and_mul approximate must be 'none' or 'tanh'");
  const int32_t approx = approximate == "tanh" ? 1 : 0;
  TORCH_CHECK(input.device().type() == c10::kPrivateUse1 &&
              input.scalar_type() == at::kBFloat16 &&
              input.dim() >= 2 && input.dim() <= 5 && input.size(-1) % 2 == 0,
              "GaudiRT: invalid BF16 gelu_and_mul input geometry");
  std::vector<int64_t> out_sizes(input.sizes().begin(), input.sizes().end());
  out_sizes.back() /= 2;
  if (!input.numel()) return at::empty(out_sizes, input.options());
  if (input.size(0) <= kMaxBucketRows) {
    BucketProgram program({1}, {FxNode::gelu_and_mul(0, approx)});
    return program.run({input})[0];
  }
  auto out = at::empty(out_sizes, input.options());
  const auto in_syn = to_syn_dims(input);
  const auto out_syn = to_syn_dims(out);
  std::ostringstream sig;
  sig << "gelu_mul_v1:" << approx << ":" << format_dims_key(in_syn);
  launch_cold_recipe(
      "cold_gelumul", sig.str(), {input}, {out},
      [&](SynapseGraphBuilder& builder, const std::vector<synTensor>& in,
          const std::vector<synTensor>& out_t) {
        emit_gated_mul(builder, true, approx, in[0], out_t[0], in_syn, out_syn);
      });
  return out;
}

std::tuple<at::Tensor, at::Tensor> rotary_embedding_positions(
    const at::Tensor& query, const at::Tensor& key, const at::Tensor& cos_sin_cache,
    const at::Tensor& positions, int64_t head_size, int64_t rotary_dim, bool is_neox_style) {
  TORCH_CHECK(head_size > 0 && head_size <= INT32_MAX && rotary_dim > 0 &&
              rotary_dim <= head_size && rotary_dim % 2 == 0,
              "GaudiRT: rotary_embedding_positions invalid head_size or rotary_dim");
  TORCH_CHECK(query.device().type() == c10::kPrivateUse1 && key.device() == query.device() &&
              cos_sin_cache.device() == query.device() && positions.device() == query.device(),
              "GaudiRT: rotary_embedding_positions requires same-device PrivateUse1 tensors");
  TORCH_CHECK(query.scalar_type() == at::kBFloat16 && key.scalar_type() == at::kBFloat16 &&
              cos_sin_cache.scalar_type() == at::kBFloat16 &&
              (positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong),
              "GaudiRT: rotary_embedding_positions expects BF16 query/key/cache and Int32/Int64 positions");
  TORCH_CHECK((query.dim() == 2 || query.dim() == 3) && key.dim() == query.dim() &&
              key.size(0) == query.size(0) && cos_sin_cache.dim() == 2 &&
              cos_sin_cache.size(1) == rotary_dim &&
              positions.dim() == 1 && positions.size(0) == query.size(0),
              "GaudiRT: rotary_embedding_positions shape mismatch");
  if (!query.numel()) {
    return {at::empty(query.sizes(), query.options()), at::empty(key.sizes(), key.options())};
  }
  if (query.size(0) <= kMaxBucketRows) {
    BucketProgram program(
        {1, 1, 0, 1},
        {FxNode::rotary_pos(0, 1, 2, 3, static_cast<int32_t>(head_size),
                            static_cast<int32_t>(rotary_dim), is_neox_style),
         FxNode::unary(FxOp::rotary_key, 4)},
        {4, 5});
    auto result = program.run({query, key, cos_sin_cache, positions});
    return {std::move(result[0]), std::move(result[1])};
  }
  auto q_out = at::empty(query.sizes(), query.options());
  auto k_out = at::empty(key.sizes(), key.options());
  const auto q_syn = to_syn_dims(query);
  const auto k_syn = to_syn_dims(key);
  std::ostringstream sig;
  sig << "rotpos_v1:" << head_size << ":" << rotary_dim << ":" << (is_neox_style ? 1 : 0)
      << ":" << cast_type(positions.scalar_type()).name << ":" << format_dims_key(q_syn)
      << ":" << format_dims_key(k_syn) << ":" << cos_sin_cache.size(0);
  launch_cold_recipe(
      "cold_rotpos", sig.str(), {query, key, cos_sin_cache, positions}, {q_out, k_out},
      [&](SynapseGraphBuilder& builder, const std::vector<synTensor>& in,
          const std::vector<synTensor>& out_t) {
        emit_rotary_pos(builder, in[0], in[1], in[2], in[3], out_t[0], out_t[1], q_syn, k_syn,
                        positions.scalar_type(), static_cast<uint64_t>(query.size(0)),
                        static_cast<int32_t>(head_size), static_cast<int32_t>(rotary_dim),
                        is_neox_style);
      });
  return {std::move(q_out), std::move(k_out)};
}

at::Tensor& softcap_(at::Tensor& logits, double cap) {
  TORCH_CHECK(logits.device().type() == c10::kPrivateUse1 &&
              std::isfinite(cap) && cap > 0.0,
              "GaudiRT: softcap_ requires PrivateUse1 tensor and finite positive cap");
  if (!logits.numel()) return logits;
  return copy_inplace(logits, unary_ewise_op(logits, FxUnaryOp::softcap, static_cast<float>(cap), 0.f));
}

namespace {

at::Tensor tanh_op(const at::Tensor& self) { return unary_ewise_op(self, FxUnaryOp::tanh); }
at::Tensor& tanh_inplace(at::Tensor& self) { return copy_inplace(self, tanh_op(self)); }
at::Tensor sigmoid_op(const at::Tensor& self) { return unary_ewise_op(self, FxUnaryOp::sigmoid); }
at::Tensor& sigmoid_inplace(at::Tensor& self) { return copy_inplace(self, sigmoid_op(self)); }
at::Tensor exp_op(const at::Tensor& self) { return unary_ewise_op(self, FxUnaryOp::exp); }
at::Tensor& exp_inplace(at::Tensor& self) { return copy_inplace(self, exp_op(self)); }
at::Tensor silu_op(const at::Tensor& self) { return unary_ewise_op(self, FxUnaryOp::silu); }
at::Tensor& silu_inplace(at::Tensor& self) { return copy_inplace(self, silu_op(self)); }
at::Tensor rsqrt_op(const at::Tensor& self) { return unary_ewise_op(self, FxUnaryOp::rsqrt); }
at::Tensor& rsqrt_inplace(at::Tensor& self) { return copy_inplace(self, rsqrt_op(self)); }
at::Tensor sqrt_op(const at::Tensor& self) { return unary_ewise_op(self, FxUnaryOp::sqrt); }
at::Tensor& sqrt_inplace(at::Tensor& self) { return copy_inplace(self, sqrt_op(self)); }
at::Tensor neg_op(const at::Tensor& self) { return unary_ewise_op(self, FxUnaryOp::neg); }
at::Tensor& neg_inplace(at::Tensor& self) { return copy_inplace(self, neg_op(self)); }
at::Tensor abs_op(const at::Tensor& self) { return unary_ewise_op(self, FxUnaryOp::abs); }
at::Tensor& abs_inplace(at::Tensor& self) { return copy_inplace(self, abs_op(self)); }

at::Tensor gelu_op(const at::Tensor& self, c10::string_view approximate) {
  TORCH_CHECK(approximate == "none" || approximate == "tanh",
              "GaudiRT: gelu approximate must be 'none' or 'tanh'");
  return unary_ewise_op(self, approximate == "tanh" ? FxUnaryOp::gelu_tanh : FxUnaryOp::gelu_exact);
}

at::Tensor clamp_op(const at::Tensor& self, const std::optional<at::Scalar>& min,
                    const std::optional<at::Scalar>& max) {
  if (self.scalar_type() == at::kInt || self.scalar_type() == at::kLong) {
    at::Tensor cur = self;
    if (min) cur = binary_ewise_op(cur, scalar_to_device(cur, *min, cur.scalar_type()), FxOp::maximum);
    if (max) cur = binary_ewise_op(cur, scalar_to_device(cur, *max, cur.scalar_type()), FxOp::minimum);
    return cur;
  }
  float lo = min ? static_cast<float>(min->toDouble()) : -std::numeric_limits<float>::infinity();
  float hi = max ? static_cast<float>(max->toDouble()) : std::numeric_limits<float>::infinity();
  if (lo > hi) lo = hi;
  return unary_ewise_op(self, FxUnaryOp::clamp, lo, hi);
}

at::Tensor& clamp_inplace(at::Tensor& self, const std::optional<at::Scalar>& min,
                          const std::optional<at::Scalar>& max) {
  return copy_inplace(self, clamp_op(self, min, max));
}

at::Tensor clamp_min_op(const at::Tensor& self, const at::Scalar& min) {
  return clamp_op(self, min, std::nullopt);
}

at::Tensor clamp_max_op(const at::Tensor& self, const at::Scalar& max) {
  return clamp_op(self, std::nullopt, max);
}

at::Tensor isnan_op(const at::Tensor& self) {
  TORCH_CHECK(self.device().type() == c10::kPrivateUse1 && self.dim() <= 5,
              "GaudiRT: isnan requires PrivateUse1 tensor of rank <= 5");
  auto out = at::empty(self.sizes(), self.options().dtype(at::kBool));
  if (!out.numel()) return out;
  if (self.scalar_type() != at::kBFloat16 && self.scalar_type() != at::kFloat) {
    return zero_(out);
  }
  const auto syn_dims = to_syn_dims(self);
  std::ostringstream sig;
  sig << "isnan_v1:" << cast_type(self.scalar_type()).name << ":" << format_dims_key(syn_dims);
  launch_cold_recipe(
      "cold_isnan", sig.str(), {self}, {out},
      [&](SynapseGraphBuilder& builder, const std::vector<synTensor>& in,
          const std::vector<synTensor>& out_t) {
        emit_isnan(builder, in[0], out_t[0], self.scalar_type());
      });
  return out;
}

at::Tensor logical_not_op(const at::Tensor& self) {
  TORCH_CHECK(self.device().type() == c10::kPrivateUse1 && self.dim() <= 5,
              "GaudiRT: logical_not requires PrivateUse1 tensor of rank <= 5");
  if (self.scalar_type() != at::kBool) return compare_scalar(self, 0, "equal");
  auto out = at::empty(self.sizes(), self.options().dtype(at::kBool));
  if (!out.numel()) return out;
  const auto syn_dims = to_syn_dims(self);
  std::ostringstream sig;
  sig << "lnot_v1:" << format_dims_key(syn_dims);
  launch_cold_recipe(
      "cold_lnot", sig.str(), {self}, {out},
      [&](SynapseGraphBuilder& builder, const std::vector<synTensor>& in,
          const std::vector<synTensor>& out_t) { emit_logical_not(builder, in[0], out_t[0]); });
  return out;
}

at::Tensor mul_tensor(const at::Tensor& a, const at::Tensor& b) {
  return binary_ewise_op(a, b, FxOp::mul);
}
at::Tensor mul_scalar(const at::Tensor& a, const at::Scalar& b) {
  const auto dtype = at::result_type(a, b);
  return binary_ewise_op(a.scalar_type() == dtype ? a : a.to(dtype),
                         scalar_to_device(a, b, dtype), FxOp::mul);
}
at::Tensor& mul_inplace(at::Tensor& a, const at::Tensor& b) {
  return copy_inplace(a, mul_tensor(a, b));
}
at::Tensor& mul_scalar_inplace(at::Tensor& a, const at::Scalar& b) {
  return copy_inplace(a, mul_scalar(a, b));
}
at::Tensor& mul_out(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  return copy_inplace(out, mul_tensor(a, b));
}

at::Tensor div_tensor(const at::Tensor& a, const at::Tensor& b) {
  return binary_ewise_op(a, b, FxOp::div, true);
}
at::Tensor div_scalar(const at::Tensor& a, const at::Scalar& b) {
  auto dtype = at::result_type(a, b);
  if (c10::isIntegralType(dtype, true)) dtype = at::kFloat;
  return binary_ewise_op(a.scalar_type() == dtype ? a : a.to(dtype),
                         scalar_to_device(a, b, dtype), FxOp::div, true);
}
at::Tensor& div_inplace(at::Tensor& a, const at::Tensor& b) {
  return copy_inplace(a, div_tensor(a, b));
}
at::Tensor& div_scalar_inplace(at::Tensor& a, const at::Scalar& b) {
  return copy_inplace(a, div_scalar(a, b));
}
at::Tensor& div_out(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  return copy_inplace(out, div_tensor(a, b));
}

at::Tensor& add_inplace(at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  return copy_inplace(a, gaudirt::generic::add(a, b, alpha));
}
at::Tensor& add_scalar_inplace(at::Tensor& a, const at::Scalar& b, const at::Scalar& alpha) {
  return copy_inplace(a, add_scalar(a, b, alpha));
}
at::Tensor& sub_inplace(at::Tensor& a, const at::Tensor& b, const at::Scalar& alpha) {
  return copy_inplace(a, gaudirt::generic::subtract(a, b, alpha));
}
at::Tensor& sub_scalar_inplace(at::Tensor& a, const at::Scalar& b, const at::Scalar& alpha) {
  return copy_inplace(a, subtract_scalar(a, b, alpha));
}

at::Tensor minimum_op(const at::Tensor& a, const at::Tensor& b) {
  return binary_ewise_op(a, b, FxOp::minimum);
}
at::Tensor maximum_op(const at::Tensor& a, const at::Tensor& b) {
  return binary_ewise_op(a, b, FxOp::maximum);
}
at::Tensor logical_and_op(const at::Tensor& a, const at::Tensor& b) {
  return binary_ewise_op(a, b, FxOp::logical_and);
}
at::Tensor logical_or_op(const at::Tensor& a, const at::Tensor& b) {
  return binary_ewise_op(a, b, FxOp::logical_or);
}
at::Tensor bitwise_and_op(const at::Tensor& a, const at::Tensor& b) {
  return binary_ewise_op(a, b, FxOp::bitwise_and);
}
at::Tensor bitwise_or_op(const at::Tensor& a, const at::Tensor& b) {
  return binary_ewise_op(a, b, FxOp::bitwise_or);
}

at::Tensor softmax_int_op(const at::Tensor& self, int64_t dim,
                          std::optional<at::ScalarType> dtype) {
  return softmax_impl(self, dim, false, dtype);
}
at::Tensor _softmax_op(const at::Tensor& self, int64_t dim, bool half_to_float) {
  return softmax_impl(self, dim, false,
                      half_to_float ? std::optional<at::ScalarType>(at::kFloat) : std::nullopt);
}
at::Tensor log_softmax_int_op(const at::Tensor& self, int64_t dim,
                              std::optional<at::ScalarType> dtype) {
  return softmax_impl(self, dim, true, dtype);
}
at::Tensor _log_softmax_op(const at::Tensor& self, int64_t dim, bool half_to_float) {
  return softmax_impl(self, dim, true,
                      half_to_float ? std::optional<at::ScalarType>(at::kFloat) : std::nullopt);
}

at::Tensor sum_all_op(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
  return reduce_dims_impl(self, FxOp::sum_dim, std::nullopt, false, dtype);
}
at::Tensor sum_dim_op(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                      std::optional<at::ScalarType> dtype) {
  return reduce_dims_impl(self, FxOp::sum_dim, dim, keepdim, dtype);
}
at::Tensor mean_all_op(const at::Tensor& self, std::optional<at::ScalarType> dtype) {
  return reduce_dims_impl(self, FxOp::mean_dim, std::nullopt, false, dtype);
}
at::Tensor mean_dim_op(const at::Tensor& self, at::OptionalIntArrayRef dim, bool keepdim,
                       std::optional<at::ScalarType> dtype) {
  return reduce_dims_impl(self, FxOp::mean_dim, dim, keepdim, dtype);
}
at::Tensor amax_op(const at::Tensor& self, at::IntArrayRef dim, bool keepdim) {
  return reduce_dims_impl(self, FxOp::amax_dim, dim, keepdim, std::nullopt);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> native_layer_norm_op(
    const at::Tensor& input, c10::SymIntArrayRef normalized_shape_sym,
    const std::optional<at::Tensor>& weight, const std::optional<at::Tensor>& bias,
    double epsilon) {
  std::vector<int64_t> normalized_shape;
  normalized_shape.reserve(normalized_shape_sym.size());
  for (const auto& s : normalized_shape_sym) normalized_shape.push_back(s.expect_int());
  const float eps = static_cast<float>(epsilon);
  TORCH_CHECK(std::isfinite(epsilon) && std::isfinite(eps) && eps >= 0.0f,
              "GaudiRT: layer_norm eps must be finite and non-negative");
  TORCH_CHECK(input.device().type() == c10::kPrivateUse1 &&
              (input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kFloat),
              "GaudiRT: layer_norm expects PrivateUse1 BF16 or FP32 tensor");
  const int64_t norm_ndim = static_cast<int64_t>(normalized_shape.size());
  TORCH_CHECK(norm_ndim >= 1 && norm_ndim <= input.dim(),
              "GaudiRT: layer_norm normalized_shape rank mismatch");
  int64_t norm_width = 1;
  for (int64_t i = 0; i < norm_ndim; ++i) {
    const int64_t dim_size = input.size(input.dim() - norm_ndim + i);
    TORCH_CHECK(dim_size == normalized_shape[i], "GaudiRT: layer_norm normalized_shape mismatch");
    norm_width *= dim_size;
  }
  const bool reshape_norm = norm_ndim > 1;
  std::vector<int64_t> flat_sizes;
  for (int64_t i = 0; i < input.dim() - norm_ndim; ++i) flat_sizes.push_back(input.size(i));
  flat_sizes.push_back(norm_width);
  if (flat_sizes.size() == 1) flat_sizes.insert(flat_sizes.begin(), 1);
  TORCH_CHECK(flat_sizes.size() <= 5, "GaudiRT: layer_norm rank exceeds 5");

  at::Tensor x = input;
  if (!x.is_contiguous()) x = x.contiguous();
  if (reshape_norm || input.dim() == 1) x = x.view(flat_sizes);

  std::vector<int64_t> stat_sizes(input.sizes().begin(), input.sizes().end());
  for (int64_t i = input.dim() - norm_ndim; i < input.dim(); ++i) stat_sizes[i] = 1;
  std::vector<int64_t> flat_stat_sizes = flat_sizes;
  flat_stat_sizes.back() = 1;

  auto out = at::empty(flat_sizes, input.options());
  auto mean = at::empty(flat_stat_sizes, input.options());
  auto rstd = at::empty(flat_stat_sizes, input.options());
  if (input.numel() == 0) {
    return {out.view(input.sizes()), mean.view(stat_sizes), rstd.view(stat_sizes)};
  }

  std::vector<at::Tensor> cold_inputs{x};
  int w_idx = -1;
  int b_idx = -1;
  auto prep_param = [&](const std::optional<at::Tensor>& param, int& out_idx) {
    if (!param.has_value() || !param->defined()) return;
    TORCH_CHECK(param->device() == input.device() && param->numel() == norm_width,
                "GaudiRT: layer_norm weight/bias shape or device mismatch");
    auto p = param->scalar_type() == input.scalar_type() ? *param : param->to(input.scalar_type());
    if (!p.is_contiguous()) p = p.contiguous();
    p = p.view({norm_width});
    out_idx = static_cast<int>(cold_inputs.size());
    cold_inputs.push_back(std::move(p));
  };
  prep_param(weight, w_idx);
  prep_param(bias, b_idx);

  const auto x_syn = to_syn_dims(x);
  std::ostringstream sig;
  sig << "ln_v1:" << cast_type(input.scalar_type()).name << ":" << std::bit_cast<uint32_t>(eps)
      << ":" << (w_idx >= 0 ? 1 : 0) << ":" << (b_idx >= 0 ? 1 : 0) << ":"
      << format_dims_key(x_syn);
  launch_cold_recipe(
      "cold_ln", sig.str(), std::move(cold_inputs), {out, mean, rstd},
      [&](SynapseGraphBuilder& builder, const std::vector<synTensor>& in,
          const std::vector<synTensor>& out_t) {
        emit_layer_norm(builder, in[0], w_idx >= 0 ? in[w_idx] : nullptr,
                        b_idx >= 0 ? in[b_idx] : nullptr, out_t[0], out_t[1], out_t[2],
                        input.scalar_type(), x_syn, eps);
      });
  return {out.view(input.sizes()), mean.view(stat_sizes), rstd.view(stat_sizes)};
}

at::Tensor layer_norm_op(const at::Tensor& input, c10::SymIntArrayRef normalized_shape,
                         const std::optional<at::Tensor>& weight,
                         const std::optional<at::Tensor>& bias, double epsilon,
                         bool /*cudnn_enable*/) {
  return std::get<0>(native_layer_norm_op(input, normalized_shape, weight, bias, epsilon));
}

}  // namespace

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
  m.impl("sub_.Tensor", TORCH_FN(gaudirt::generic::sub_inplace));
  m.impl("sub_.Scalar", TORCH_FN(gaudirt::generic::sub_scalar_inplace));
  m.impl("add.Tensor", TORCH_FN(gaudirt::generic::add));
  m.impl("add.Scalar", TORCH_FN(gaudirt::generic::add_scalar));
  m.impl("add.out", TORCH_FN(gaudirt::generic::add_out));
  m.impl("add.Scalar_out", TORCH_FN(gaudirt::generic::add_scalar_out));
  m.impl("add_.Tensor", TORCH_FN(gaudirt::generic::add_inplace));
  m.impl("add_.Scalar", TORCH_FN(gaudirt::generic::add_scalar_inplace));
  m.impl("mul.Tensor", TORCH_FN(gaudirt::generic::mul_tensor));
  m.impl("mul.Scalar", TORCH_FN(gaudirt::generic::mul_scalar));
  m.impl("mul_.Tensor", TORCH_FN(gaudirt::generic::mul_inplace));
  m.impl("mul_.Scalar", TORCH_FN(gaudirt::generic::mul_scalar_inplace));
  m.impl("mul.out", TORCH_FN(gaudirt::generic::mul_out));
  m.impl("div.Tensor", TORCH_FN(gaudirt::generic::div_tensor));
  m.impl("div.Scalar", TORCH_FN(gaudirt::generic::div_scalar));
  m.impl("div_.Tensor", TORCH_FN(gaudirt::generic::div_inplace));
  m.impl("div_.Scalar", TORCH_FN(gaudirt::generic::div_scalar_inplace));
  m.impl("div.out", TORCH_FN(gaudirt::generic::div_out));
  m.impl("minimum", TORCH_FN(gaudirt::generic::minimum_op));
  m.impl("maximum", TORCH_FN(gaudirt::generic::maximum_op));
  m.impl("logical_and", TORCH_FN(gaudirt::generic::logical_and_op));
  m.impl("logical_or", TORCH_FN(gaudirt::generic::logical_or_op));
  m.impl("logical_not", TORCH_FN(gaudirt::generic::logical_not_op));
  m.impl("bitwise_and.Tensor", TORCH_FN(gaudirt::generic::bitwise_and_op));
  m.impl("bitwise_or.Tensor", TORCH_FN(gaudirt::generic::bitwise_or_op));
  m.impl("tanh", TORCH_FN(gaudirt::generic::tanh_op));
  m.impl("tanh_", TORCH_FN(gaudirt::generic::tanh_inplace));
  m.impl("sigmoid", TORCH_FN(gaudirt::generic::sigmoid_op));
  m.impl("sigmoid_", TORCH_FN(gaudirt::generic::sigmoid_inplace));
  m.impl("exp", TORCH_FN(gaudirt::generic::exp_op));
  m.impl("exp_", TORCH_FN(gaudirt::generic::exp_inplace));
  m.impl("silu", TORCH_FN(gaudirt::generic::silu_op));
  m.impl("silu_", TORCH_FN(gaudirt::generic::silu_inplace));
  m.impl("rsqrt", TORCH_FN(gaudirt::generic::rsqrt_op));
  m.impl("rsqrt_", TORCH_FN(gaudirt::generic::rsqrt_inplace));
  m.impl("sqrt", TORCH_FN(gaudirt::generic::sqrt_op));
  m.impl("sqrt_", TORCH_FN(gaudirt::generic::sqrt_inplace));
  m.impl("neg", TORCH_FN(gaudirt::generic::neg_op));
  m.impl("neg_", TORCH_FN(gaudirt::generic::neg_inplace));
  m.impl("abs", TORCH_FN(gaudirt::generic::abs_op));
  m.impl("abs_", TORCH_FN(gaudirt::generic::abs_inplace));
  m.impl("gelu", TORCH_FN(gaudirt::generic::gelu_op));
  m.impl("isnan", TORCH_FN(gaudirt::generic::isnan_op));
  m.impl("clamp", TORCH_FN(gaudirt::generic::clamp_op));
  m.impl("clamp_", TORCH_FN(gaudirt::generic::clamp_inplace));
  m.impl("clamp_min", TORCH_FN(gaudirt::generic::clamp_min_op));
  m.impl("clamp_max", TORCH_FN(gaudirt::generic::clamp_max_op));
  m.impl("softmax.int", TORCH_FN(gaudirt::generic::softmax_int_op));
  m.impl("_softmax", TORCH_FN(gaudirt::generic::_softmax_op));
  m.impl("log_softmax.int", TORCH_FN(gaudirt::generic::log_softmax_int_op));
  m.impl("_log_softmax", TORCH_FN(gaudirt::generic::_log_softmax_op));
  m.impl("sum", TORCH_FN(gaudirt::generic::sum_all_op));
  m.impl("sum.dim_IntList", TORCH_FN(gaudirt::generic::sum_dim_op));
  m.impl("mean", TORCH_FN(gaudirt::generic::mean_all_op));
  m.impl("mean.dim", TORCH_FN(gaudirt::generic::mean_dim_op));
  m.impl("amax", TORCH_FN(gaudirt::generic::amax_op));
  m.impl("native_layer_norm", TORCH_FN(gaudirt::generic::native_layer_norm_op));
  m.impl("layer_norm", TORCH_FN(gaudirt::generic::layer_norm_op));
  m.impl("where.self", TORCH_FN(gaudirt::generic::where));
  m.impl("index_select", TORCH_FN(gaudirt::generic::index_select));
  m.impl("index.Tensor", TORCH_FN(gaudirt::generic::index_tensor));
  m.impl("_index_put_impl_", TORCH_FN(gaudirt::generic::index_put));
  m.impl("embedding", TORCH_FN(gaudirt::generic::embedding));
  m.impl("argmax", TORCH_FN(gaudirt::generic::argmax));
}
