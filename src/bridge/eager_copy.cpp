// SPDX-License-Identifier: Apache-2.0
#include "bridge_internal.hpp"
#include "eager_copy.hpp"
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

ColdRecipe& broadcast_recipe(Context& ctx, uint64_t count, uint64_t width) {
  TORCH_CHECK(width == 1 || width == 2 || width == 4 || width == 8,
              "GaudiRT: unsupported fill width");
  const std::string key = "cold_broadcast_" + std::to_string(width) + "_" + std::to_string(count);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>();
  {
    SynapseGraphBuilder builder(key, *recipe);
    const uint64_t lanes = width == 8 ? 2 : 1;
    const uint64_t output_columns = width == 8 ? count : (count * width + 3) / 4;
    auto in = builder.tensor(syn_type_uint32, {lanes, 1}, 0);
    auto out = builder.tensor(syn_type_uint32, {lanes, output_columns}, 1);
    builder.node(NodeSpec{"broadcast_non_fcd_u32", {in}, {out}, ns_BroadcastNonFcd::Params{1}});
    builder.compile(ctx.cache, key);
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate cold recipe");
  return *it->second;
}

CastType cast_type(at::ScalarType type) {
  switch (type) {
    case at::kBFloat16: return {syn_type_bf16, "bf16"};
    case at::kFloat: return {syn_type_float, "f32"};
    case at::kInt: return {syn_type_int32, "i32"};
    case at::kLong: return {syn_type_int64, "i64"};
    case at::kBool: return {syn_type_uint8, "u8"};
    default: TORCH_CHECK(false, "GaudiRT: unsupported cast dtype");
  }
}

bool supported_cast(at::ScalarType from, at::ScalarType to) {
  return
      (from == at::kBFloat16 && (to == at::kFloat || to == at::kInt || to == at::kLong)) ||
      (from == at::kFloat && (to == at::kBFloat16 || to == at::kInt || to == at::kLong)) ||
      (from == at::kInt && (to == at::kBFloat16 || to == at::kFloat || to == at::kLong)) ||
      (from == at::kLong && (to == at::kInt || to == at::kFloat || to == at::kBFloat16)) ||
      (from == at::kBool && (to == at::kBFloat16 || to == at::kFloat ||
                             to == at::kInt || to == at::kLong)) ||
      (to == at::kBool && (from == at::kBFloat16 || from == at::kFloat ||
                           from == at::kInt || from == at::kLong));
}
ColdRecipe& cast_recipe(Context& ctx, uint64_t max_count, at::ScalarType from, at::ScalarType to) {
  TORCH_CHECK(supported_cast(from, to), "GaudiRT: requested dtype conversion is not implemented");
  const auto input_type = cast_type(from);
  const auto output_type = cast_type(to);
  const std::string key = "cold_cast_v2_" + std::string(input_type.name) + "_" +
                          output_type.name + "_" + std::to_string(max_count);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(2);
  {
    SynapseGraphBuilder builder(key, *recipe);
    auto in = builder.tensor(input_type.syn, {max_count}, 0, -1, 0, true, {1ULL});
    auto out = builder.tensor(output_type.syn, {max_count}, 1, -1, 0, true, {1ULL});
    if (to == at::kBool) {
      if (from == at::kLong) {
        auto zero_i32 = builder.tensor(syn_type_int32, {1}, -1);
        auto zero_i64 = builder.tensor(syn_type_int64, {1}, -1);
        auto lt_neg = builder.tensor(syn_type_uint8, {max_count}, -1, -1, 0, true, {1ULL});
        auto lt_pos = builder.tensor(syn_type_uint8, {max_count}, -1, -1, 0, true, {1ULL});
        ns_ConstantKernel::ParamsV3 zero{};
        builder.node(NodeSpec{"constant_i32", {}, {zero_i32}, zero});
        ns_CastKernel::ParamsV3 cast{};
        cast.round_mode = CAST_ROUND_ZERO;
        builder.node(NodeSpec{"cast_i32_to_i64", {zero_i32}, {zero_i64}, cast});
        builder.node(NodeSpec{"less_fwd_i64", {in, zero_i64}, {lt_neg}, {}});
        builder.node(NodeSpec{"less_fwd_i64", {zero_i64, in}, {lt_pos}, {}});
        builder.node(NodeSpec{"or_fwd_u8", {lt_neg, lt_pos}, {out}, {}});
      } else {
        auto zero_val = builder.tensor(input_type.syn, {1}, -1);
        auto is_zero = builder.tensor(syn_type_uint8, {max_count}, -1, -1, 0, true, {1ULL});
        ns_ConstantKernel::ParamsV3 zero{};
        builder.node(NodeSpec{"constant_" + std::string(input_type.name), {}, {zero_val}, zero});
        builder.node(NodeSpec{"equal_fwd_" + std::string(input_type.name), {in, zero_val}, {is_zero}, {}});
        builder.node(NodeSpec{"not_fwd_u8", {is_zero}, {out}, {}});
      }
    } else if (to == at::kInt && (from == at::kFloat || from == at::kBFloat16)) {
      auto raw_i32 = builder.tensor(syn_type_int32, {max_count}, -1, -1, 0, true, {1ULL});
      auto pos_thr = builder.tensor(input_type.syn, {1}, -1);
      auto ovf_u8 = builder.tensor(syn_type_uint8, {max_count}, -1, -1, 0, true, {1ULL});
      auto nan_u8 = builder.tensor(syn_type_uint8, {max_count}, -1, -1, 0, true, {1ULL});
      auto bad_u8 = builder.tensor(syn_type_uint8, {max_count}, -1, -1, 0, true, {1ULL});
      auto neg_f32 = builder.tensor(syn_type_float, {1}, -1);
      auto sen_1 = builder.tensor(syn_type_int32, {1}, -1);
      auto zero_1 = builder.tensor(syn_type_int32, {1}, -1);
      auto z_dyn = builder.tensor(syn_type_int32, {max_count}, -1, -1, 0, true, {1ULL});
      auto sen_i32 = builder.tensor(syn_type_int32, {max_count}, -1, -1, 0, true, {1ULL});
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_ZERO;
      builder.node(NodeSpec{"cast_" + std::string(input_type.name) + "_to_i32", {in}, {raw_i32}, cast});
      ns_ConstantKernel::ParamsV3 positive{};
      positive.constant.f = 2147483648.0f;
      builder.node(NodeSpec{"constant_" + std::string(input_type.name), {}, {pos_thr}, positive});
      builder.node(NodeSpec{"greater_equal_fwd_" + std::string(input_type.name), {in, pos_thr}, {ovf_u8}, {}});
      builder.node(NodeSpec{"isnan_fwd_" + std::string(input_type.name), {in}, {nan_u8}, {}});
      builder.node(NodeSpec{"or_fwd_u8", {ovf_u8, nan_u8}, {bad_u8}, {}});
      ns_ConstantKernel::ParamsV3 negative{};
      negative.constant.f = -2147483648.0f;
      builder.node(NodeSpec{"constant_f32", {}, {neg_f32}, negative});
      builder.node(NodeSpec{"cast_f32_to_i32", {neg_f32}, {sen_1}, cast});
      ns_ConstantKernel::ParamsV3 zero{};
      builder.node(NodeSpec{"constant_i32", {}, {zero_1}, zero});
      builder.node(NodeSpec{"mult_fwd_i32", {raw_i32, zero_1}, {z_dyn}, {}});
      builder.node(NodeSpec{"add_fwd_i32", {z_dyn, sen_1}, {sen_i32}, {}});
      builder.node(NodeSpec{"where_fwd_i32", {bad_u8, sen_i32, raw_i32}, {out}, {}});
    } else {
      ns_CastKernel::ParamsV3 params{};
      params.round_mode = (to == at::kInt || to == at::kLong) ? CAST_ROUND_ZERO : CAST_ROUND_HALF_NE;
      builder.node(NodeSpec{"cast_" + std::string(input_type.name) + "_to_" + output_type.name,
                            {in}, {out}, params});
    }
    builder.compile(ctx.cache, key);
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate cold cast recipe");
  return *it->second;
}

void check_factory(const std::optional<at::Layout>& layout,
                   const std::optional<at::Device>& device,
                   const std::optional<bool>& pin_memory) {
  TORCH_CHECK(!layout || *layout == at::kStrided, "GaudiRT: only strided layout is supported");
  TORCH_CHECK(!pin_memory.value_or(false), "GaudiRT: pin_memory applies to CPU allocations");
  const auto active = current_device_index();
  TORCH_CHECK(!device || (device->type() == c10::kPrivateUse1 &&
              (device->index() < 0 || device->index() == active)), "GaudiRT: invalid device");
}
at::Tensor empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                 std::optional<at::Layout> layout, std::optional<at::Device> device,
                 std::optional<bool> pin_memory, std::optional<at::MemoryFormat> format) {
  check_factory(layout, device, pin_memory);
  return at::detail::empty_generic(size, c10::GetAllocator(c10::kPrivateUse1),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), dtype.value_or(at::kFloat), format);
}
at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                         std::optional<at::Device> device, std::optional<bool> pin_memory) {
  check_factory(layout, device, pin_memory);
  return at::detail::empty_strided_generic(size, stride, c10::GetAllocator(c10::kPrivateUse1),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), dtype.value_or(at::kFloat));
}
at::Tensor& set_storage(at::Tensor& tensor, at::Storage storage, int64_t storage_offset,
                        at::IntArrayRef size, at::IntArrayRef stride) {
  TORCH_CHECK(storage.device().type() == c10::kPrivateUse1 &&
              storage.device().index() == current_device_index(),
              "GaudiRT: invalid storage device");
  TORCH_CHECK(storage_offset >= 0 && (stride.empty() || stride.size() == size.size()),
              "GaudiRT: invalid storage view");
  std::vector<int64_t> contiguous;
  if (stride.empty()) {
    contiguous.resize(size.size());
    int64_t next = 1;
    for (size_t i = size.size(); i-- > 0;) {
      TORCH_CHECK(size[i] >= 0 && (!size[i] || next <= INT64_MAX / size[i]),
                  "GaudiRT: invalid storage size");
      contiguous[i] = next;
      next *= std::max<int64_t>(size[i], 1);
    }
    stride = contiguous;
  }
  uint64_t last = static_cast<uint64_t>(storage_offset);
  const bool empty = std::find(size.begin(), size.end(), 0) != size.end();
  for (size_t i = 0; i < size.size(); ++i) {
    TORCH_CHECK(size[i] >= 0 && stride[i] >= 0, "GaudiRT: invalid storage view");
    if (empty) continue;
    auto count = static_cast<uint64_t>(size[i] - 1);
    TORCH_CHECK(!count || static_cast<uint64_t>(stride[i]) <= (UINT64_MAX - last) / count,
                "GaudiRT: storage view overflow");
    last += count * static_cast<uint64_t>(stride[i]);
  }
  const auto item = tensor.element_size();
  TORCH_CHECK(last <= UINT64_MAX / item &&
              (empty ? last * item <= storage.nbytes() :
                       last < UINT64_MAX / item && (last + 1) * item <= storage.nbytes()),
              "GaudiRT: storage view is out of bounds");
  auto* impl = tensor.unsafeGetTensorImpl();
  impl->set_storage_keep_dtype(std::move(storage));
  impl->set_sizes_and_strides(size, stride, storage_offset);
  return tensor;
}

at::Tensor& fill(at::Tensor& tensor, const at::Scalar& scalar) {
  if (!tensor.numel()) return tensor;
  auto ctx = current();
  const auto address = address_of(tensor, *ctx);
  const uint64_t width = tensor.element_size();
  uint64_t pattern = 0;
  switch (tensor.scalar_type()) {
    case at::kBool: pattern = scalar.to<bool>(); break;
    case at::kBFloat16: pattern = scalar.to<c10::BFloat16>().x; break;
    case at::kFloat: pattern = std::bit_cast<uint32_t>(scalar.to<float>()); break;
    case at::kInt: pattern = std::bit_cast<uint32_t>(scalar.to<int32_t>()); break;
    case at::kLong: pattern = std::bit_cast<uint64_t>(scalar.to<int64_t>()); break;
    default: TORCH_CHECK(false, "GaudiRT: fill dtype is not implemented");
  }
  if (width == 1) pattern = (pattern & 0xff) * uint64_t{0x01010101};
  if (width == 2) pattern = (pattern & 0xffff) | ((pattern & 0xffff) << 16);
  const uint64_t bytes = tensor.nbytes();
  if (tensor.is_contiguous() && address % BlockCachingAllocator::kAlignment == 0 &&
      bytes % 4 == 0 && bytes / 4 <= UINT32_MAX &&
      (width != 8 || static_cast<uint32_t>(pattern) == static_cast<uint32_t>(pattern >> 32))) {
    const auto compute = ctx->stream();
    {
      std::lock_guard lock(ctx->cold_mutex);
      if (ctx->graph_pool && ctx->graph_pool->contains_live_range(address, 1))
        ctx->graph_pool->record_stream(address, compute);
      ctx->allocator.record_stream(address, compute);
    }
    TORCH_CHECK(synMemsetD32Async(address, static_cast<uint32_t>(pattern), bytes / 4, compute) == synSuccess,
                "GaudiRT: fill memset failed");
    return tensor;
  }
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = broadcast_recipe(*ctx, tensor.numel(), width);
  auto compute = ctx->stream();
  auto h2d = ctx->device.h2d_stream();
  const uint64_t scalar_bytes = std::max<uint64_t>(width, 4);
  TemporaryDevice value(*ctx, scalar_bytes);
  TemporaryHost host(*ctx, scalar_bytes);
  std::memcpy(host.address, &pattern, scalar_bytes);
  ctx->allocator.record_stream(value.address, h2d);
  auto& reusable = ctx->streams.state(active_stream_id).h2d_reusable;
  reusable.record_and_wait(compute, h2d);
  ctx->pinned.copy_h2d(value.address, host.address, scalar_bytes, h2d, true);
  auto& ready = ctx->streams.state(active_stream_id).h2d_ready;
  ready.record_and_wait(h2d, compute);
  std::unique_ptr<TemporaryDevice> output;
  const uint64_t graph_bytes = width == 8 ? tensor.nbytes() : ((tensor.nbytes() + 3) & ~uint64_t{3});
  if (!tensor.is_contiguous() || address % BlockCachingAllocator::kAlignment ||
      graph_bytes != tensor.nbytes())
    output = std::make_unique<TemporaryDevice>(*ctx, graph_bytes);
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  auto tensors = recipe.tensors;
  tensors[0].pTensorAddress = value.address;
  tensors[1].pTensorAddress = output ? output->address : address;
  ctx->allocator.record_stream(address, compute);
  TORCH_CHECK(synLaunch(compute, tensors.data(), tensors.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: fill launch failed");
  if (output) {
    const Layout layout(tensor);
    dma_copy(address, &layout, output->address, nullptr, 0, tensor.numel(), width,
             compute, DRAM_TO_DRAM);
  }
  return tensor;
}
at::Tensor& zero(at::Tensor& tensor) { return fill(tensor, at::Scalar(0)); }

void cast_direct(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  auto ctx = current();
  auto source = address_of(src, *ctx);
  auto target = address_of(dst, *ctx);
  at::assert_no_overlap(dst, src);
  const uint64_t count = static_cast<uint64_t>(src.numel());
  TORCH_CHECK(count <= (1ULL << 30), "GaudiRT: cast tensor exceeds max supported elements");
  const uint64_t max_count = std::max<uint64_t>(65536ULL, std::bit_ceil(count));
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = cast_recipe(*ctx, max_count, src.scalar_type(), dst.scalar_type());
  auto stream = ctx->stream();
  ctx->allocator.record_stream(source, stream);
  ctx->allocator.record_stream(target, stream);
  std::unique_ptr<TemporaryDevice> input, output, workspace;
  if (!src.is_contiguous() || source % BlockCachingAllocator::kAlignment) {
    input = std::make_unique<TemporaryDevice>(*ctx, src.nbytes());
    const Layout layout(src);
    dma_copy(input->address, nullptr, source, &layout, 0, src.numel(), src.element_size(),
             stream, DRAM_TO_DRAM);
  }
  if (!dst.is_contiguous() || target % BlockCachingAllocator::kAlignment)
    output = std::make_unique<TemporaryDevice>(*ctx, dst.nbytes());
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  auto tensors = recipe.tensors;
  tensors[0].pTensorAddress = input ? input->address : source;
  tensors[0].tensorSize[0] = count;
  tensors[1].pTensorAddress = output ? output->address : target;
  tensors[1].tensorSize[0] = count;
  TORCH_CHECK(synLaunch(stream, tensors.data(), tensors.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: cast launch failed");
  if (output) {
    const Layout layout(dst);
    dma_copy(target, &layout, output->address, nullptr, 0, dst.numel(), dst.element_size(),
             stream, DRAM_TO_DRAM);
  }
  if (!non_blocking) ctx->synchronize_stream();
}

at::Tensor compare(const at::Tensor& a, const at::Tensor& b, const char* op);
at::Tensor compare_scalar(const at::Tensor& a, const at::Scalar& scalar, const char* op);
at::Tensor pointwise(std::vector<at::Tensor> inputs, const std::vector<int64_t>& shape,
                     at::ScalarType output_type, const std::string& op,
                     const std::vector<synDataType>& types);
at::Tensor where(const at::Tensor& condition, const at::Tensor& a, const at::Tensor& b);
at::Tensor& copy(at::Tensor& dst, const at::Tensor& src, bool non_blocking);

void cast_device(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  if (src.scalar_type() == at::kLong && dst.scalar_type() == at::kBFloat16) {
    auto intermediate = at::empty(src.sizes(), src.options().dtype(at::kFloat));
    cast_device(intermediate, src, true);
    cast_direct(dst, intermediate, non_blocking);
    return;
  }
  if (src.scalar_type() == at::kInt && dst.scalar_type() == at::kBFloat16) {
    auto intermediate = at::empty(src.sizes(), src.options().dtype(at::kFloat));
    cast_direct(intermediate, src, true);
    cast_direct(dst, intermediate, non_blocking);
    return;
  }
  if (src.scalar_type() == at::kBool && dst.scalar_type() == at::kLong) {
    auto intermediate = at::empty(src.sizes(), src.options().dtype(at::kInt));
    cast_direct(intermediate, src, true);
    cast_direct(dst, intermediate, non_blocking);
    return;
  }
  if ((src.scalar_type() == at::kFloat || src.scalar_type() == at::kBFloat16) &&
      dst.scalar_type() == at::kLong) {
    const std::vector<int64_t> shape(src.sizes().begin(), src.sizes().end());
    auto scalar = [&](at::ScalarType dtype, const at::Scalar& value) {
      auto result = at::empty({}, src.options().dtype(dtype));
      fill(result, value);
      return result;
    };
    auto binary = [&](const at::Tensor& a, const at::Tensor& b,
                      at::ScalarType dtype, const char* guid) {
      const auto type = cast_type(dtype).syn;
      return pointwise({a, b}, shape, dtype, guid, {type, type, type});
    };
    auto convert = [&](const at::Tensor& value, at::ScalarType dtype) {
      auto result = at::empty(shape, src.options().dtype(dtype));
      cast_direct(result, value, true);
      return result;
    };
    auto value = src.scalar_type() == at::kFloat ? src : convert(src, at::kFloat);
    auto inverse_base = scalar(at::kFloat, at::Scalar(1.0 / 4294967296.0));
    auto base = scalar(at::kFloat, at::Scalar(4294967296.0));
    auto half = scalar(at::kFloat, at::Scalar(0.5));
    auto two = scalar(at::kFloat, at::Scalar(2.0));
    auto high = convert(binary(value, inverse_base, at::kFloat, "mult_fwd_f32"), at::kInt);
    auto high_float = convert(high, at::kFloat);
    auto remainder = binary(value, binary(high_float, base, at::kFloat, "mult_fwd_f32"),
                            at::kFloat, "sub_fwd_f32");
    auto half_remainder = convert(binary(remainder, half, at::kFloat, "mult_fwd_f32"), at::kInt);
    auto half_float = convert(half_remainder, at::kFloat);
    auto last = convert(binary(remainder, binary(half_float, two, at::kFloat, "mult_fwd_f32"),
                               at::kFloat, "sub_fwd_f32"), at::kInt);
    auto high64 = convert(high, at::kLong);
    auto half64 = convert(half_remainder, at::kLong);
    auto last64 = convert(last, at::kLong);
    auto base64 = scalar(at::kLong, at::Scalar(int64_t{4294967296}));
    auto two64 = scalar(at::kLong, at::Scalar(int64_t{2}));
    auto upper = binary(high64, base64, at::kLong, "mult_fwd_i64");
    auto lower = binary(half64, two64, at::kLong, "mult_fwd_i64");
    auto raw = binary(binary(upper, lower, at::kLong, "add_fwd_i64"), last64,
                      at::kLong, "add_fwd_i64");
    auto high_overflow = compare_scalar(value, at::Scalar(9223372036854775808.0), "greater_equal");
    auto low_overflow = compare_scalar(value, at::Scalar(-9223372036854775808.0), "less");
    auto nan = compare(value, value, "not_equal");
    auto outside = pointwise({high_overflow, low_overflow}, shape, at::kBool, "or_fwd_u8",
                             {syn_type_uint8, syn_type_uint8, syn_type_uint8});
    auto invalid = pointwise({outside, nan}, shape, at::kBool, "or_fwd_u8",
                             {syn_type_uint8, syn_type_uint8, syn_type_uint8});
    auto sentinel = scalar(at::kLong, at::Scalar(std::numeric_limits<int64_t>::min()));
    auto corrected = gaudirt::generic::where(invalid, sentinel, raw);
    copy(dst, corrected, non_blocking);
    return;
  }
  if (src.scalar_type() == at::kLong && dst.scalar_type() == at::kFloat) {
    const std::vector<int64_t> shape(src.sizes().begin(), src.sizes().end());
    auto scalar = [&](at::ScalarType dtype, const at::Scalar& value) {
      auto result = at::empty({}, src.options().dtype(dtype));
      fill(result, value);
      return result;
    };
    auto binary = [&](const at::Tensor& a, const at::Tensor& b,
                      at::ScalarType dtype, const char* guid) {
      const auto type = cast_type(dtype).syn;
      return pointwise({a, b}, shape, dtype, guid, {type, type, type});
    };
    auto boolean = [&](const at::Tensor& a, const at::Tensor& b, const char* guid) {
      return pointwise({a, b}, shape, at::kBool, guid,
                       {syn_type_uint8, syn_type_uint8, syn_type_uint8});
    };
    auto raw = at::empty(shape, src.options().dtype(at::kFloat));
    cast_direct(raw, src, true);
    auto raw_integer = at::empty(shape, src.options().dtype(at::kLong));
    cast_device(raw_integer, raw, true);
    auto bits = raw.view(at::kInt);
    auto one = scalar(at::kInt, at::Scalar(1));
    auto next_bits = binary(bits, one, at::kInt, "add_fwd_i32");
    auto next = next_bits.view(at::kFloat);
    auto delta = binary(next, raw, at::kFloat, "sub_fwd_f32");
    auto spacing_float = pointwise({delta}, shape, at::kFloat, "abs_fwd_f32",
                                   {syn_type_float, syn_type_float});
    auto spacing = at::empty(shape, src.options().dtype(at::kLong));
    cast_device(spacing, spacing_float, true);
    auto minus_one = scalar(at::kLong, at::Scalar(int64_t{-1}));
    auto negated_raw = binary(raw_integer, minus_one, at::kLong, "mult_fwd_i64");
    auto remainder = binary(src, negated_raw, at::kLong, "add_fwd_i64");
    auto negative = compare_scalar(remainder, at::Scalar(0), "less");
    auto negated = binary(remainder, minus_one, at::kLong, "mult_fwd_i64");
    auto magnitude = gaudirt::generic::where(negative, negated, remainder);
    auto two = scalar(at::kLong, at::Scalar(int64_t{2}));
    auto twice = binary(magnitude, two, at::kLong, "mult_fwd_i64");
    auto greater = compare(twice, spacing, "greater");
    auto tie = compare(twice, spacing, "equal");
    auto parity = binary(bits, one, at::kInt, "bitwise_and_fwd_i32");
    auto odd = compare_scalar(parity, at::Scalar(0), "not_equal");
    auto tie_odd = boolean(tie, odd, "and_fwd_u8");
    auto rounding = boolean(greater, tie_odd, "or_fwd_u8");
    auto nonzero = compare_scalar(remainder, at::Scalar(0), "not_equal");
    auto round_up = boolean(rounding, nonzero, "and_fwd_u8");
    auto rounded = gaudirt::generic::where(round_up, next, raw);
    auto too_high = compare_scalar(raw, at::Scalar(9223372036854775808.0), "greater_equal");
    auto too_low = compare_scalar(raw, at::Scalar(-9223372036854775808.0), "less_equal");
    auto boundary = boolean(too_high, too_low, "or_fwd_u8");
    auto corrected = gaudirt::generic::where(boundary, raw, rounded);
    copy(dst, corrected, non_blocking);
    return;
  }
  cast_direct(dst, src, non_blocking);
}

at::Tensor& copy(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  if (dst.sizes() != src.sizes()) {
    auto expanded = src.expand(dst.sizes());
    return copy(dst, expanded, non_blocking);
  }
  if (!dst.numel()) return dst;
  at::assert_no_internal_overlap(dst);
  auto ctx = current();
  const bool dst_device = dst.device().type() == c10::kPrivateUse1;
  const bool src_device = src.device().type() == c10::kPrivateUse1;
  TORCH_CHECK((dst_device || dst.device().is_cpu()) && (src_device || src.device().is_cpu()) &&
              (dst_device || src_device), "GaudiRT: unsupported copy devices");
  if (dst.scalar_type() != src.scalar_type()) {
    TORCH_CHECK(supported_cast(src.scalar_type(), dst.scalar_type()),
                "GaudiRT: requested dtype conversion is not implemented");
    if (!src_device) {
      auto temporary = at::empty(src.sizes(), dst.options().dtype(src.scalar_type()));
      copy(temporary, src, true);
      cast_device(dst, temporary, non_blocking);
    } else if (!dst_device) {
      auto temporary = at::empty(dst.sizes(), src.options().dtype(dst.scalar_type()));
      cast_device(temporary, src, true);
      copy(dst, temporary, non_blocking);
    } else {
      cast_device(dst, src, non_blocking);
    }
    return dst;
  }
  auto compute = ctx->stream();
  const uint64_t bytes = src.nbytes();
  if (dst_device && src_device) {
    auto d = address_of(dst, *ctx), s = address_of(src, *ctx);
    if (d == s && dst.strides() == src.strides()) return dst;
    at::assert_no_overlap(dst, src);
    ctx->allocator.record_stream(d, compute);
    ctx->allocator.record_stream(s, compute);
    const Layout dl(dst), sl(src);
    dma_copy(d, &dl, s, &sl, 0, src.numel(), src.element_size(), compute, DRAM_TO_DRAM);
    if (!non_blocking) ctx->synchronize_stream();
    return dst;
  }
  const uint64_t device = address_of(dst_device ? dst : src, *ctx);
  const uint64_t item = src.element_size();
  const uint64_t elements = src.numel();
  auto staging_elements = [&] {
    const auto available = std::min<uint64_t>(4 * 1024 * 1024, ctx->pinned.largest_free_block());
    TORCH_CHECK(available >= item, "GaudiRT: pinned pool has no staging space");
    return available / item;
  };
  auto& stream_state = ctx->streams.state(active_stream_id);
  if (dst_device) {
    auto stream = ctx->device.h2d_stream();
    auto& reusable = stream_state.h2d_reusable;
    auto& ready = stream_state.h2d_ready;
    const Layout dl(dst);
    if (ctx->pinned.contains_live_range(src.data_ptr(), span_bytes(src))) {
      const Layout sl(src);
      ctx->pinned.record_stream(src.data_ptr(), stream);
      ctx->allocator.record_stream(device, stream);
      reusable.record_and_wait(compute, stream);
      dma_copy(device, &dl, reinterpret_cast<uint64_t>(src.data_ptr()), &sl,
               0, elements, item, stream, HOST_TO_DRAM);
      ready.record_and_wait(stream, compute);
      if (!non_blocking) ready.synchronize();
      return dst;
    }
    const uint64_t chunk_elements = staging_elements();
    const uint64_t chunk_size = chunk_elements * item;
    std::unique_ptr<TemporaryDevice> temporary;
    if (!dst.is_contiguous()) {
      temporary = std::make_unique<TemporaryDevice>(*ctx, std::min(chunk_size, bytes));
      ctx->allocator.record_stream(temporary->address, stream);
    } else {
      ctx->allocator.record_stream(device, stream);
    }
    reusable.record_and_wait(compute, stream);
    for (uint64_t start = 0; start < elements; start += chunk_elements) {
      const auto count = std::min(chunk_elements, elements - start);
      TemporaryHost host(*ctx, count * item);
      host_copy(static_cast<uint8_t*>(host.address), src, start, count, true);
      if (temporary && start) {
        reusable.record_and_wait(compute, stream);
      }
      const auto target = temporary ? temporary->address : device + start * item;
      ctx->pinned.copy_h2d(target, host.address, count * item, stream, true);
      if (temporary) {
        ready.record_and_wait(stream, compute);
        dma_copy(device, &dl, temporary->address, nullptr, start, count, item, compute, DRAM_TO_DRAM);
      }
    }
    if (!temporary) {
      ready.record_and_wait(stream, compute);
    }
    if (!non_blocking) ctx->synchronize_stream();
  } else {
    auto stream = ctx->device.d2h_stream();
    auto& ready = stream_state.d2h_reusable;
    auto& done = stream_state.d2h_ready;
    const Layout sl(src);
    if (ctx->pinned.contains_live_range(dst.data_ptr(), span_bytes(dst))) {
      const Layout dl(dst);
      ctx->pinned.record_stream(dst.data_ptr(), stream);
      ctx->allocator.record_stream(device, stream);
      ready.record_and_wait(compute, stream);
      dma_copy(reinterpret_cast<uint64_t>(dst.data_ptr()), &dl, device, &sl,
               0, elements, item, stream, DRAM_TO_HOST);
      done.record_and_wait(stream, compute);
      if (!non_blocking) done.synchronize();
      return dst;
    }
    TORCH_CHECK(!non_blocking, "GaudiRT: nonblocking D2H requires pinned destination");
    const uint64_t chunk_elements = staging_elements();
    const uint64_t chunk_size = chunk_elements * item;
    std::unique_ptr<TemporaryDevice> temporary;
    if (!src.is_contiguous()) {
      temporary = std::make_unique<TemporaryDevice>(*ctx, std::min(chunk_size, bytes));
      ctx->allocator.record_stream(temporary->address, stream);
    } else {
      ctx->allocator.record_stream(device, stream);
    }
    for (uint64_t start = 0; start < elements; start += chunk_elements) {
      const auto count = std::min(chunk_elements, elements - start);
      TemporaryHost host(*ctx, count * item);
      if (temporary)
        dma_copy(temporary->address, nullptr, device, &sl, start, count, item, compute, DRAM_TO_DRAM);
      ready.record_and_wait(compute, stream);
      const auto source = temporary ? temporary->address : device + start * item;
      ctx->pinned.copy_d2h(host.address, source, count * item, stream, false);
      host_copy(static_cast<uint8_t*>(host.address), dst, start, count, false);
    }
  }
  return dst;
}

}  // namespace gaudirt::generic
