// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/block_caching_allocator.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/pinned_host_pool.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include "gaudirt/common/sha256.hpp"
#include "cold_recipe.hpp"
#include "bridge_internal.hpp"
#include "bucket_program.hpp"
#include "segmented_runner.hpp"
#include "context.hpp"
#include "fx_node.hpp"
#include "fx_geometry.hpp"
#include "layout_copy.hpp"
#include "eager_copy.hpp"
#include "paged_metadata.hpp"
#include "paged_attention.hpp"
#include "paged_kv_pool.hpp"
#include "paged_slot_allocator.hpp"
#include "temporaries.hpp"

#include <perf_lib_layer_params.h>
#include <c10/core/Allocator.h>
#include <torch/extension.h>
#include <torch/custom_class.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace gaudirt::generic {
using namespace runtime;

ColdRecipe& bucket_linear_recipe(Context& ctx, at::ScalarType dtype, uint64_t k,
                                 uint64_t n, uint32_t bucket, bool has_bias) {
  const auto type = cast_type(dtype);
  const std::string structure = "fx_mm_transpose_weight_v1_" + std::string(type.name) +
                                "_" + std::to_string(k) + "_" + std::to_string(n) +
                                (has_bias ? "_bias" : "");
  const auto digest = gaudirt::common::sha256_bytes(
      reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const auto prefix = "g" + digest.substr(0, 12) + "_b" + std::to_string(bucket);
  const auto key = "fx_" + digest + "_b" + std::to_string(bucket);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  const size_t external_count = has_bias ? 4 : 3;
  auto recipe = std::make_unique<ColdRecipe>(external_count);
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 4> sections{};
  std::array<synTensor, 7> tensors{};
  std::array<std::string, 3> intermediate_names{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "bucket graph creation failed");
    const std::array<std::array<uint64_t, 2>, 4> shapes = has_bias ?
        std::array<std::array<uint64_t, 2>, 4>{{{k, bucket}, {k, n}, {n, 1}, {n, bucket}}} :
        std::array<std::array<uint64_t, 2>, 4>{{{k, bucket}, {k, n}, {n, bucket}, {0, 0}}};
    for (size_t i = 0; i < external_count; ++i) {
      recipe->names[i] = prefix + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "bucket section creation failed");
      check(synSectionSetPersistent(sections[i], true), "bucket section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = type.syn;
      desc.m_dims = 2;
      for (size_t dim = 0; dim < 2; ++dim)
        desc.m_sizes[dim] = desc.m_minSizes[dim] = shapes[i][dim];
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "bucket tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
      recipe->tensors[i].tensorSize[0] = shapes[i][0];
      recipe->tensors[i].tensorSize[1] = shapes[i][1];
    }
    synTensor inputs[] = {tensors[0], tensors[1]};
    synGEMMParams params{false, true};
    const auto node = prefix + "_n0";
    if (has_bias) {
      const bool bf16 = dtype == at::kBFloat16;
      const size_t intermediate_count = bf16 ? 3 : 1;
      for (size_t i = 0; i < intermediate_count; ++i) {
        intermediate_names[i] = prefix + "_t" + std::to_string(i + external_count);
        synTensorDescriptor desc{};
        desc.m_name = intermediate_names[i].c_str();
        desc.m_dataType = syn_type_float;
        desc.m_dims = 2;
        desc.m_sizes[0] = desc.m_minSizes[0] = n;
        desc.m_sizes[1] = desc.m_minSizes[1] = bf16 && i == 1 ? 1 : bucket;
        check(synTensorCreate(&tensors[external_count + i], &desc, nullptr, 0),
              "bucket intermediate creation failed");
      }
    }
    auto* gemm_output = has_bias ? &tensors[external_count] : &tensors[2];
    check(synNodeCreate(graph, inputs, gemm_output, 2, 1, &params, sizeof(params),
                        "gemm", node.c_str(), nullptr, nullptr), "bucket GEMM node creation failed");
    if (has_bias) {
      const bool bf16 = dtype == at::kBFloat16;
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_HALF_NE;
      if (bf16) {
        const auto cast_node = prefix + "_n1";
        check(synNodeCreate(graph, &tensors[2], &tensors[5], 1, 1, &cast, sizeof(cast),
                            "cast_bf16_to_f32", cast_node.c_str(), nullptr, nullptr),
              "bucket bias cast failed");
      }
      synTensor add_inputs[] = {tensors[4], bf16 ? tensors[5] : tensors[2]};
      auto* add_output = bf16 ? &tensors[6] : &tensors[3];
      const auto add_node = prefix + "_n2";
      check(synNodeCreate(graph, add_inputs, add_output, 2, 1, nullptr, 0,
                          "add_fwd_f32", add_node.c_str(), nullptr, nullptr), "bucket bias add failed");
      if (bf16) {
        const auto output_node = prefix + "_n3";
        check(synNodeCreate(graph, &tensors[6], &tensors[3], 1, 1, &cast, sizeof(cast),
                            "cast_f32_to_bf16", output_node.c_str(), nullptr, nullptr),
              "bucket output cast failed");
      }
    }
    recipe->recipe = compile_cached(graph, prefix, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "bucket workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate bucket recipe");
  return *it->second;
}

std::optional<at::Tensor> prepare_strided_input(const at::Tensor& input) {
  TORCH_CHECK(input.device().type() == c10::kPrivateUse1 && input.device().index() == 0 &&
              input.dim() >= 1 && input.dim() <= 5 && input.size(0) > 0 && input.size(0) <= 4096,
              "GaudiRT FX: invalid dynamic input");
  if (input.is_contiguous()) {
    std::vector<int64_t> strides(input.dim());
    int64_t expected = 1;
    bool canonical = true;
    for (int64_t dim = input.dim() - 1; dim >= 0; --dim) {
      strides[dim] = expected;
      canonical &= input.stride(dim) == expected;
      TORCH_CHECK(input.size(dim) > 0 && expected <= INT64_MAX / input.size(dim),
                  "GaudiRT FX: invalid input geometry");
      expected *= input.size(dim);
    }
    if (canonical) return std::nullopt;
    return input.as_strided(input.sizes(), strides, input.storage_offset());
  }
  const auto rows = input.size(0);
  const auto bucket = std::bit_ceil(static_cast<uint32_t>(rows));
  uint64_t width = 1;
  for (int64_t dim = 1; dim < input.dim(); ++dim) {
    TORCH_CHECK(input.size(dim) > 0 && width <= UINT64_MAX / static_cast<uint64_t>(input.size(dim)),
                "GaudiRT FX: invalid input geometry");
    width *= static_cast<uint64_t>(input.size(dim));
  }
  const uint64_t item = input.element_size();
  TORCH_CHECK(width <= UINT64_MAX / bucket / item, "GaudiRT FX: input allocation size overflow");
  const uint64_t bytes = width * bucket * item;
  auto ctx = current();
  auto stream = ctx->stream();
  const auto source = address_of(input, *ctx);
  c10::DataPtr allocation;
  uint64_t destination = 0;
  {
    std::lock_guard lock(ctx->cold_mutex);
    if (!ctx->graph_pool) {
      const uint64_t budget = std::min<uint64_t>(ctx->allocator.capacity_bytes() / 8, uint64_t{1} << 30);
      TORCH_CHECK(budget >= 256, "GaudiRT FX: HBM budget is too small for graph temporaries");
      ctx->graph_pool = std::make_unique<GraphPool>(ctx->allocator, budget, stream);
    }
    destination = ctx->graph_pool->allocate(bytes, stream);
    allocation = allocation_data_ptr(ctx, destination, bytes, stream, AllocationKind::graph);
    if (ctx->graph_pool->contains_live_range(source, 1)) ctx->graph_pool->record_stream(source, stream);
    ctx->allocator.record_stream(source, stream);
  }
  if (input.scalar_type() == at::kInt || input.scalar_type() == at::kLong) {
    TORCH_CHECK(bytes <= UINT64_MAX - 3 && (bytes + 3) / 4 <= UINT32_MAX &&
                synMemsetD32Async(destination, 0, (bytes + 3) / 4, stream) == synSuccess,
                "GaudiRT FX: index tail zeroing failed");
  }
  const Layout layout(input);
  dma_copy(destination, nullptr, source, &layout, 0, input.numel(), item, stream, DRAM_TO_DRAM);
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  shape[0] = bucket;
  auto staged = tensor_from_allocation(std::move(allocation), bytes, shape, input.options());
  return staged.narrow(0, 0, rows);
}

std::optional<at::Tensor> restore_staged_alias(const at::Tensor& original,
                                               const at::Tensor& staged,
                                               const at::Tensor& output) {
  if (!output.storage().is_alias_of(staged.storage())) return std::nullopt;
  if (original.storage().is_alias_of(staged.storage())) return output;
  TORCH_CHECK(output.scalar_type() == original.scalar_type() && staged.is_contiguous(),
              "GaudiRT FX: unsupported staged output alias");
  if (output.sizes() == staged.sizes() && output.strides() == staged.strides() &&
      output.storage_offset() == staged.storage_offset()) return original;
  TORCH_CHECK(output.storage_offset() >= staged.storage_offset(),
              "GaudiRT FX: staged output alias is outside the logical input");
  const auto delta = output.storage_offset() - staged.storage_offset();
  int64_t maximum = delta;
  for (int64_t dim = 0; dim < output.dim(); ++dim) {
    const auto count = std::max<int64_t>(0, output.size(dim) - 1);
    TORCH_CHECK(output.stride(dim) >= 0 &&
                (!count || output.stride(dim) <= (INT64_MAX - maximum) / count),
                "GaudiRT FX: invalid staged output alias stride");
    maximum += count * output.stride(dim);
  }
  TORCH_CHECK(!output.numel() || maximum < original.numel(),
              "GaudiRT FX: staged output alias is outside the logical input");
  int64_t scale = 1, expected = 1;
  bool have_scale = false, uniformly_strided = true;
  for (int64_t dim = original.dim() - 1; dim >= 0; --dim) {
    if (original.size(dim) == 1) continue;
    if (!have_scale) {
      scale = expected = original.stride(dim);
      have_scale = true;
    }
    if (original.stride(dim) != expected) uniformly_strided = false;
    TORCH_CHECK(!expected || expected <= INT64_MAX / original.size(dim),
                "GaudiRT FX: staged output alias stride overflow");
    expected *= original.size(dim);
  }
  if (uniformly_strided) {
    std::vector<int64_t> strides(output.dim());
    for (int64_t dim = 0; dim < output.dim(); ++dim) {
      TORCH_CHECK(!scale || output.stride(dim) <= INT64_MAX / scale,
                  "GaudiRT FX: staged output alias stride overflow");
      strides[dim] = output.stride(dim) * scale;
    }
    TORCH_CHECK(!scale || delta <= (INT64_MAX - original.storage_offset()) / scale,
                "GaudiRT FX: staged output alias offset overflow");
    return original.as_strided(output.sizes(), strides, original.storage_offset() + delta * scale);
  }
  std::vector<int64_t> canonical(original.dim()), position(original.dim()), maxima(original.dim());
  int64_t width = 1;
  for (int64_t dim = original.dim() - 1; dim >= 0; --dim) {
    canonical[dim] = width;
    width *= original.size(dim);
  }
  int64_t offset = original.storage_offset();
  for (int64_t dim = 0; dim < original.dim(); ++dim) {
    position[dim] = (delta / canonical[dim]) % original.size(dim);
    maxima[dim] = position[dim];
    offset += position[dim] * original.stride(dim);
  }
  std::vector<int64_t> strides(output.dim(), 0);
  for (int64_t dim = 0; dim < output.dim(); ++dim) {
    for (int64_t src = 0; src < original.dim(); ++src) {
      const auto step = (output.stride(dim) / canonical[src]) % original.size(src);
      maxima[src] += std::max<int64_t>(0, output.size(dim) - 1) * step;
      strides[dim] += step * original.stride(src);
    }
  }
  for (int64_t dim = 0; dim < original.dim(); ++dim)
    TORCH_CHECK(maxima[dim] < original.size(dim),
                "GaudiRT FX: staged output alias cannot preserve the original layout");
  return original.as_strided(output.sizes(), strides, offset);
}

at::Tensor feature_view(const at::Tensor& tensor, int64_t start, int64_t length) {
  TORCH_CHECK(tensor.dim() == 2 && start >= 0 && length > 0 &&
              start <= tensor.size(1) && length <= tensor.size(1) - start,
              "GaudiRT FX: invalid feature view");
  return tensor.narrow(1, start, length);
}

at::Tensor reshape_view(const at::Tensor& tensor, const std::vector<int64_t>& shape) {
  TORCH_CHECK(!shape.empty() && shape.size() <= 5 && shape[0] == -1 &&
              std::all_of(shape.begin() + 1, shape.end(), [](int64_t size) { return size > 0; }),
              "GaudiRT FX: invalid reshape view");
  return tensor.view(shape);
}

at::Tensor transpose_feature_view(const at::Tensor& tensor, int64_t dim0, int64_t dim1) {
  TORCH_CHECK(tensor.dim() >= 3 && tensor.dim() <= 5 && dim0 > 0 && dim1 > 0 &&
              dim0 < tensor.dim() && dim1 < tensor.dim() && dim0 != dim1,
              "GaudiRT FX: invalid feature transpose view");
  return tensor.transpose(dim0, dim1);
}

at::Tensor bucket_output(const at::Tensor& reference, at::ScalarType dtype,
                         uint32_t bucket, int64_t rows, int64_t width) {
  const uint64_t item = c10::elementSize(dtype);
  TORCH_CHECK(width > 0 && static_cast<uint64_t>(width) <=
              (static_cast<uint64_t>(INT64_MAX) - BlockCachingAllocator::kAlignment) / bucket / item,
              "GaudiRT: bucket output is too large");
  const uint64_t elements = static_cast<uint64_t>(bucket) * width;
  const uint64_t bytes = (elements * item + BlockCachingAllocator::kAlignment - 1) &
                         ~(BlockCachingAllocator::kAlignment - 1);
  auto output = at::empty({static_cast<int64_t>(bytes / item)}, reference.options().dtype(dtype));
  output.unsafeGetTensorImpl()->set_sizes_contiguous({rows, width});
  return output;
}

at::Tensor bucket_linear(const at::Tensor& input, const at::Tensor& weight,
                         const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(input.device() == weight.device() && input.device().type() == c10::kPrivateUse1 &&
              input.dim() == 2 && weight.dim() == 2 && input.size(1) == weight.size(1),
              "GaudiRT: bucket GEMM expects same-device [M,K] and [N,K]");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type() &&
              (input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kFloat),
              "GaudiRT: bucket GEMM supports matching BF16 or FP32 tensors");
  const auto m = input.size(0), k = input.size(1), n = weight.size(0);
  TORCH_CHECK(m > 0 && m <= 4096 && k > 0 && n > 0, "GaudiRT: invalid bucket GEMM shape");
  if (bias) {
    TORCH_CHECK(bias->device() == input.device() && bias->scalar_type() == input.scalar_type() &&
                bias->dim() == 1 && bias->size(0) == n,
                "GaudiRT: bucket GEMM bias must be a same-device [N] tensor of matching dtype");
  }
  const uint32_t bucket = std::bit_ceil(static_cast<uint32_t>(m));
  const uint64_t item = input.element_size();
  TORCH_CHECK(static_cast<uint64_t>(k) <= UINT64_MAX / bucket / item,
              "GaudiRT: bucket GEMM input is too large");
  const uint64_t bucket_bytes = static_cast<uint64_t>(bucket) * k * item;
  auto ctx = current();
  auto stream = ctx->stream();
  const auto input_address = address_of(input, *ctx);
  auto weight_value = weight;
  auto weight_address = address_of(weight_value, *ctx);
  const uint64_t offset = static_cast<uint64_t>(input.storage_offset()) * item;
  const uint64_t storage_bytes = input.storage().nbytes();
  const bool direct = input.is_contiguous() && input_address % BlockCachingAllocator::kAlignment == 0 &&
                      offset <= storage_bytes && bucket_bytes <= storage_bytes - offset;
  const bool overlaps = direct && input_address < weight_address + weight.nbytes() &&
                        weight_address < input_address + bucket_bytes;
  if (!weight_value.is_contiguous() || weight_address % BlockCachingAllocator::kAlignment || overlaps) {
    weight_value = at::empty(weight.sizes(), weight.options());
    copy(weight_value, weight, true);
    weight_address = address_of(weight_value, *ctx);
  }
  at::Tensor bias_value;
  uint64_t bias_address = 0;
  if (bias) {
    bias_value = *bias;
    bias_address = address_of(bias_value, *ctx);
    const auto bytes = bias_value.nbytes();
    const bool input_overlap = direct && input_address < bias_address + bytes &&
                               bias_address < input_address + bucket_bytes;
    const bool weight_overlap = weight_address < bias_address + bytes &&
                                bias_address < weight_address + weight_value.nbytes();
    if (!bias_value.is_contiguous() || bias_address % BlockCachingAllocator::kAlignment ||
        input_overlap || weight_overlap) {
      bias_value = at::empty(bias->sizes(), bias->options());
      copy(bias_value, *bias, true);
      bias_address = address_of(bias_value, *ctx);
    }
  }
  auto result = bucket_output(input, input.scalar_type(), bucket, m, n);
  const auto output_address = address_of(result, *ctx);
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = bucket_linear_recipe(*ctx, input.scalar_type(), k, n, bucket, bias.has_value());
  if ((!direct || recipe.workspace) && !ctx->graph_pool) {
    const uint64_t budget = std::min<uint64_t>(ctx->allocator.capacity_bytes() / 8, uint64_t{1} << 30);
    TORCH_CHECK(budget >= 256, "GaudiRT: HBM budget is too small for graph temporaries");
    ctx->graph_pool = std::make_unique<GraphPool>(ctx->allocator, budget, stream);
  }
  std::unique_ptr<GraphTemporary> staged, workspace;
  ctx->allocator.record_stream(input_address, stream);
  ctx->allocator.record_stream(weight_address, stream);
  if (bias) ctx->allocator.record_stream(bias_address, stream);
  ctx->allocator.record_stream(output_address, stream);
  if (!direct) {
    staged = std::make_unique<GraphTemporary>(*ctx->graph_pool, stream, bucket_bytes);
    const Layout layout(input);
    dma_copy(staged->address, nullptr, input_address, &layout, 0, input.numel(), item, stream, DRAM_TO_DRAM);
  }
  if (recipe.workspace)
    workspace = std::make_unique<GraphTemporary>(*ctx->graph_pool, stream, recipe.workspace);
  auto bindings = recipe.tensors;
  bindings[0].pTensorAddress = staged ? staged->address : input_address;
  bindings[1].pTensorAddress = weight_address;
  if (bias) bindings[2].pTensorAddress = bias_address;
  bindings.back().pTensorAddress = output_address;
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(), workspace ? workspace->address : 0,
                        recipe.recipe, 0) == synSuccess, "GaudiRT: bucket GEMM launch failed");
  return result;
}

at::Tensor bucket_rms_norm(const at::Tensor& input, const at::Tensor& weight, double epsilon) {
  const float eps = static_cast<float>(epsilon);
  TORCH_CHECK(std::isfinite(epsilon) && std::isfinite(eps) && eps > 0,
              "GaudiRT: RMSNorm epsilon must be a positive finite FP32 value");
  BucketProgram program({1, 0}, {FxNode::norm(0, 1, eps)});
  return program.run({input, weight})[0];
}

std::tuple<at::Tensor, at::Tensor> bucket_fused_add_rms_norm(
    const at::Tensor& input, const at::Tensor& residual, const at::Tensor& weight, double epsilon) {
  const float eps = static_cast<float>(epsilon);
  TORCH_CHECK(std::isfinite(epsilon) && std::isfinite(eps) && eps > 0,
              "GaudiRT: fused RMSNorm epsilon must be a positive finite FP32 value");
  BucketProgram program({1, 1, 0}, {FxNode::fused_norm(0, 1, 2, eps), FxNode::unary(FxOp::fused_residual, 3)},
                        {3, 4});
  auto result = program.run({input, residual, weight});
  return {result[0], result[1]};
}

at::Tensor bucket_silu_and_mul(const at::Tensor& input) {
  BucketProgram program({1}, {FxNode::unary(FxOp::silu_and_mul, 0)});
  return program.run({input})[0];
}

at::Tensor paged_attn(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
                      const at::Tensor& key_pool, const at::Tensor& value_pool,
                      c10::intrusive_ptr<PagedMetadata> metadata, int64_t head_dim, double scale) {
  TORCH_CHECK(metadata, "GaudiRT paged attention: metadata is required");
  TORCH_CHECK(query.dim() == 2 && key_pool.dim() == 3 && (head_dim == 64 || head_dim == 128),
              "GaudiRT paged attention: invalid tensor ranks or head dimension");
  return paged_attn_validated(query, key, value, key_pool, value_pool,
                              metadata->row_locations(), metadata->row_requests(), metadata->rotary_positions(),
                              metadata->attention_plan(query.size(1) / head_dim, key_pool.size(1), head_dim),
                              metadata->rotary_header(), head_dim, scale);
}

ColdRecipe& rotary_recipe(Context& ctx, uint32_t bucket, uint64_t q_width,
                          uint64_t k_width, uint64_t cache_rows, int32_t head_dim,
                          int32_t rotary_dim, bool is_neox) {
  std::ostringstream signature;
  signature << "rotary_v1;" << bucket << ":" << q_width << ":" << k_width << ":"
            << cache_rows << ":" << head_dim << ":" << rotary_dim << ":" << is_neox;
  const auto structure = signature.str();
  const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const auto prefix = "g" + digest.substr(0, 12) + "_b" + std::to_string(bucket);
  const auto key = "rotary_" + digest;
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  struct TensorShape { synDataType type; std::vector<uint64_t> dims; };
  const std::array<TensorShape, 7> shapes{{
      {syn_type_bf16, {q_width, bucket}}, {syn_type_bf16, {k_width, bucket}},
      {syn_type_int32, {bucket}}, {syn_type_bf16, {static_cast<uint64_t>(rotary_dim), cache_rows}},
      {syn_type_int32, {1}}, {syn_type_bf16, {q_width, bucket}}, {syn_type_bf16, {k_width, bucket}}}};
  auto compiled = std::make_unique<ColdRecipe>(shapes.size());
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 7> sections{};
  std::array<synTensor, 7> tensors{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT rotary: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "graph creation failed");
    for (size_t i = 0; i < shapes.size(); ++i) {
      compiled->names[i] = prefix + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "section creation failed");
      check(synSectionSetPersistent(sections[i], true), "persistent section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = compiled->names[i].c_str();
      desc.m_dataType = shapes[i].type;
      desc.m_dims = shapes[i].dims.size();
      for (size_t dim = 0; dim < shapes[i].dims.size(); ++dim) {
        desc.m_sizes[dim] = desc.m_minSizes[dim] = shapes[i].dims[dim];
        compiled->tensors[i].tensorSize[dim] = shapes[i].dims[dim];
      }
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "tensor creation failed");
      compiled->tensors[i].tensorName = compiled->names[i].c_str();
      compiled->tensors[i].tensorType = DATA_TENSOR;
    }
    struct Params { int32_t head_dim, rotary_dim, is_neox; } params{head_dim, rotary_dim, is_neox};
    const auto node = prefix + "_n0";
    check(synNodeCreate(graph, tensors.data(), tensors.data() + 5, 5, 2, &params, sizeof(params),
                        "gk_rotary_bf16_v1", node.c_str(), nullptr, nullptr), "node creation failed");
    compiled->recipe = compile_cached(graph, prefix, key, &ctx.cache);
    check(synWorkspaceGetSize(&compiled->workspace, compiled->recipe), "workspace query failed");
    initialize_launch_tensor_ids(compiled->recipe, compiled->tensors.data(), compiled->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(compiled));
  TORCH_CHECK(inserted, "GaudiRT rotary: duplicate recipe");
  return *it->second;
}

std::tuple<at::Tensor, at::Tensor> rotary_embedding_validated(const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& cache, const at::Tensor& positions, const at::Tensor& header, int64_t context,
    int64_t head_dim, int64_t rotary_dim, bool is_neox) {
  TORCH_CHECK(head_dim > 0 && head_dim <= INT32_MAX && rotary_dim > 0 && rotary_dim <= head_dim &&
              rotary_dim % 2 == 0, "GaudiRT rotary: invalid head or rotary dimension");
  for (const auto& tensor : {query, key, cache})
    TORCH_CHECK(tensor.device().type() == c10::kPrivateUse1 && tensor.device().index() == 0 &&
                tensor.scalar_type() == at::kBFloat16, "GaudiRT rotary: expected gaudirt BF16 tensors");
  TORCH_CHECK(query.dim() == 2 && key.dim() == 2 && query.size(0) > 0 && query.size(0) <= 4096 &&
              key.size(0) == query.size(0) && query.size(1) > 0 && key.size(1) > 0 &&
              query.size(1) <= INT32_MAX && key.size(1) <= INT32_MAX &&
              query.size(1) % head_dim == 0 && key.size(1) % head_dim == 0,
              "GaudiRT rotary: invalid query or key geometry");
  TORCH_CHECK(cache.dim() == 2 && context > 0 && cache.size(0) >= context && cache.size(0) <= INT32_MAX &&
              cache.size(1) == rotary_dim, "GaudiRT rotary: cache does not cover validated positions");
  const auto rows = query.size(0);
  const auto bucket = std::bit_ceil(static_cast<uint32_t>(rows));
  TORCH_CHECK(positions.device() == query.device() && positions.scalar_type() == at::kInt &&
              positions.dim() == 1 && positions.size(0) == rows && positions.is_contiguous() &&
              positions.storage_offset() >= 0 &&
              static_cast<uint64_t>(positions.storage_offset()) * sizeof(int32_t) <= positions.storage().nbytes() &&
              static_cast<uint64_t>(bucket) * sizeof(int32_t) <= positions.storage().nbytes() -
                  static_cast<uint64_t>(positions.storage_offset()) * sizeof(int32_t) &&
              header.device() == query.device() && header.scalar_type() == at::kInt &&
              header.dim() == 1 && header.size(0) == 1 && header.is_contiguous(),
              "GaudiRT rotary: invalid validated metadata tensors");
  auto ctx = current();
  auto stream = ctx->stream();
  auto cache_value = cache;
  auto cache_address = address_of(cache_value, *ctx);
  if (!cache_value.is_contiguous() || cache_address % BlockCachingAllocator::kAlignment) {
    cache_value = at::empty(cache.sizes(), cache.options());
    copy(cache_value, cache, true);
    cache_address = address_of(cache_value, *ctx);
  }
  auto q_out = bucket_output(query, at::kBFloat16, bucket, rows, query.size(1));
  auto k_out = bucket_output(key, at::kBFloat16, bucket, rows, key.size(1));
  const auto q_out_address = address_of(q_out, *ctx), k_out_address = address_of(k_out, *ctx);
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = rotary_recipe(*ctx, bucket, query.size(1), key.size(1), cache.size(0),
                               static_cast<int32_t>(head_dim), static_cast<int32_t>(rotary_dim), is_neox);
  auto graph_pool = [&]() -> GraphPool& {
    if (!ctx->graph_pool) {
      const uint64_t budget = std::min<uint64_t>(ctx->allocator.capacity_bytes() / 8, uint64_t{1} << 30);
      TORCH_CHECK(budget >= 256, "GaudiRT rotary: HBM budget is too small for graph temporaries");
      ctx->graph_pool = std::make_unique<GraphPool>(ctx->allocator, budget, stream);
    }
    return *ctx->graph_pool;
  };
  std::vector<std::unique_ptr<GraphTemporary>> staged;
  std::vector<std::pair<uint64_t, uint64_t>> reserved{{cache_address, cache_value.nbytes()}};
  auto bind_input = [&](const at::Tensor& input) {
    const auto address = address_of(input, *ctx);
    ctx->allocator.record_stream(address, stream);
    const uint64_t bytes = static_cast<uint64_t>(bucket) * input.size(1) * sizeof(uint16_t);
    const uint64_t offset = static_cast<uint64_t>(input.storage_offset()) * input.element_size();
    const uint64_t storage = input.storage().nbytes();
    bool direct = input.is_contiguous() && address % BlockCachingAllocator::kAlignment == 0 &&
                  offset <= storage && bytes <= storage - offset;
    for (const auto& [other, length] : reserved)
      if (address < other + length && other < address + bytes) direct = false;
    auto binding = address;
    if (!direct) {
      auto temporary = std::make_unique<GraphTemporary>(graph_pool(), stream, bytes);
      binding = temporary->address;
      const Layout layout(input);
      dma_copy(binding, nullptr, address, &layout, 0, input.numel(), input.element_size(), stream, DRAM_TO_DRAM);
      staged.push_back(std::move(temporary));
    }
    reserved.emplace_back(binding, bytes);
    return binding;
  };
  const auto q_address = bind_input(query), k_address = bind_input(key);
  const auto positions_address = address_of(positions, *ctx);
  const auto header_address = address_of(header, *ctx);
  TORCH_CHECK(positions_address % BlockCachingAllocator::kAlignment == 0 &&
              header_address % BlockCachingAllocator::kAlignment == 0,
              "GaudiRT rotary: validated metadata must be aligned");
  for (auto address : {cache_address, positions_address, header_address, q_out_address, k_out_address})
    ctx->allocator.record_stream(address, stream);
  std::unique_ptr<GraphTemporary> workspace;
  if (recipe.workspace) workspace = std::make_unique<GraphTemporary>(graph_pool(), stream, recipe.workspace);
  auto bindings = recipe.tensors;
  const uint64_t addresses[] = {q_address, k_address, positions_address, cache_address,
                                header_address, q_out_address, k_out_address};
  for (size_t i = 0; i < bindings.size(); ++i) bindings[i].pTensorAddress = addresses[i];
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(), workspace ? workspace->address : 0,
                        recipe.recipe, 0) == synSuccess, "GaudiRT rotary: launch failed");
  return {q_out, k_out};
}

std::tuple<at::Tensor, at::Tensor> rotary_embedding(const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& cache, c10::intrusive_ptr<PagedMetadata> metadata,
    int64_t head_dim, int64_t rotary_dim, bool is_neox) {
  TORCH_CHECK(metadata, "GaudiRT rotary: metadata is required");
  return rotary_embedding_validated(query, key, cache, metadata->rotary_positions(),
                                    metadata->rotary_header(), metadata->context_,
                                    head_dim, rotary_dim, is_neox);
}
}  // namespace gaudirt::generic

TORCH_LIBRARY(gaudirt, m) {
  m.class_<gaudirt::generic::PagedMetadata>("PagedMetadata")
      .def(torch::init<at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t>())
      .def_static("from_device", &gaudirt::generic::PagedMetadata::from_device)
      .def_static("from_device_with_tokens", &gaudirt::generic::PagedMetadata::from_device_with_tokens)
      .def_static("_from_scheduler", &gaudirt::generic::PagedMetadata::from_scheduler)
      .def("rows", &gaudirt::generic::PagedMetadata::rows)
      .def("slots", &gaudirt::generic::PagedMetadata::slots)
      .def("context", &gaudirt::generic::PagedMetadata::context)
      .def("forward_tensors", &gaudirt::generic::PagedMetadata::forward_tensors)
      .def("rotary_positions", &gaudirt::generic::PagedMetadata::rotary_positions)
      .def("rotary_header", &gaudirt::generic::PagedMetadata::rotary_header)
      .def("row_locations", &gaudirt::generic::PagedMetadata::row_locations)
      .def("row_requests", &gaudirt::generic::PagedMetadata::row_requests)
      .def("request_mapping", &gaudirt::generic::PagedMetadata::request_mapping)
      .def("attention_plan", &gaudirt::generic::PagedMetadata::attention_plan)
      .def("last_token_indices", &gaudirt::generic::PagedMetadata::last_token_indices)
      .def("token_ids", &gaudirt::generic::PagedMetadata::token_ids)
      .def("vocabulary", &gaudirt::generic::PagedMetadata::vocabulary);
  m.def("rms_norm(Tensor input, Tensor weight, float epsilon) -> Tensor");
  m.def("fused_add_rms_norm(Tensor input, Tensor residual, Tensor weight, float epsilon) -> (Tensor, Tensor)");
  m.def("silu_and_mul(Tensor input) -> Tensor");
  m.def("paged_attn(Tensor query, Tensor key, Tensor value, Tensor(a!) key_pool, Tensor(b!) value_pool, "
        "__torch__.torch.classes.gaudirt.PagedMetadata metadata, int head_dim, float scale) -> Tensor");
  m.def("_paged_attn_validated(Tensor query, Tensor key, Tensor value, Tensor(a!) key_pool, Tensor(b!) value_pool, "
        "Tensor locations, Tensor requests, Tensor positions, Tensor mapping, Tensor header, "
        "int head_dim, float scale) -> Tensor");
  m.def("rotary_embedding(Tensor query, Tensor key, Tensor cache, "
        "__torch__.torch.classes.gaudirt.PagedMetadata metadata, int head_dim, int rotary_dim, bool is_neox) "
        "-> (Tensor, Tensor)");
  m.def("_rotary_embedding_validated(Tensor query, Tensor key, Tensor cache, Tensor positions, "
        "Tensor header, int context, int head_dim, int rotary_dim, bool is_neox) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gaudirt, PrivateUse1, m) {
  m.impl("rms_norm", TORCH_FN(gaudirt::generic::bucket_rms_norm));
  m.impl("fused_add_rms_norm", TORCH_FN(gaudirt::generic::bucket_fused_add_rms_norm));
  m.impl("silu_and_mul", TORCH_FN(gaudirt::generic::bucket_silu_and_mul));
  m.impl("paged_attn", TORCH_FN(gaudirt::generic::paged_attn));
  m.impl("_paged_attn_validated", TORCH_FN(gaudirt::generic::paged_attn_validated));
  m.impl("rotary_embedding", TORCH_FN(gaudirt::generic::rotary_embedding));
  m.impl("_rotary_embedding_validated", TORCH_FN(gaudirt::generic::rotary_embedding_validated));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  gaudirt::generic::register_backend();
  m.def("initialize", &gaudirt::generic::initialize);
  m.def("is_initialized", &gaudirt::generic::is_initialized);
  m.def("memory_info", &gaudirt::generic::memory_info);
  m.def("physical_memory_info", &gaudirt::generic::physical_memory_info);
  m.def("shutdown", &gaudirt::generic::shutdown);
  m.def("_shutdown_at_exit", &gaudirt::generic::shutdown_at_exit);
  m.def("synchronize", &gaudirt::generic::synchronize);
  m.def("empty_cache", &gaudirt::generic::empty_cache);
  m.def("memory_allocated", &gaudirt::generic::memory_allocated);
  m.def("max_memory_allocated", &gaudirt::generic::max_memory_allocated);
  m.def("recipe_cache_stats", &gaudirt::generic::recipe_cache_stats);
  m.def("_recipe_cache_keys", &gaudirt::generic::recipe_cache_keys);
  m.def("_prepare_strided_input", &gaudirt::generic::prepare_strided_input);
  m.def("_allocate_paged_kv", &gaudirt::generic::allocate_paged_kv);
  m.def("_allocate_paged_slots", &gaudirt::generic::allocate_paged_slots);
  m.def("_release_paged_slots", &gaudirt::generic::release_paged_slots);
  m.def("_restore_staged_alias", &gaudirt::generic::restore_staged_alias);
  m.def("_feature_view", &gaudirt::generic::feature_view);
  m.def("_reshape_view", &gaudirt::generic::reshape_view);
  m.def("_transpose_feature_view", &gaudirt::generic::transpose_feature_view);
  m.attr("_rotary_static_rows") = gaudirt::generic::kRotaryStaticRows;
  m.def("bucket_linear", &gaudirt::generic::bucket_linear,
        pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias") = pybind11::none());
  using gaudirt::generic::FxNode;
  using gaudirt::generic::FxOp;
  pybind11::enum_<FxOp>(m, "FxOp")
      .value("linear", FxOp::linear).value("add", FxOp::add).value("rms_norm", FxOp::rms_norm)
      .value("silu_and_mul", FxOp::silu_and_mul).value("feature_slice", FxOp::feature_slice)
      .value("fused_add_rms_norm", FxOp::fused_add_rms_norm).value("fused_residual", FxOp::fused_residual)
      .value("rotary", FxOp::rotary).value("rotary_key", FxOp::rotary_key)
      .value("embedding", FxOp::embedding).value("gather_rows", FxOp::gather_rows).value("cast", FxOp::cast)
      .value("sub", FxOp::sub).value("mul", FxOp::mul).value("div", FxOp::div)
      .value("concat", FxOp::concat).value("bmm", FxOp::bmm).value("reshape", FxOp::reshape)
      .value("transpose", FxOp::transpose).value("copy", FxOp::copy)
      .value("scalar_add", FxOp::scalar_add).value("scalar_sub", FxOp::scalar_sub)
      .value("scalar_mul", FxOp::scalar_mul).value("scalar_div", FxOp::scalar_div);
  pybind11::class_<FxNode>(m, "FxNode")
      .def_readonly("op", &FxNode::op)
      .def_static("unary", &FxNode::unary).def_static("binary", &FxNode::binary)
      .def_static("linear", &FxNode::linear).def_static("slice", &FxNode::slice)
      .def_static("norm", &FxNode::norm).def_static("fused_norm", &FxNode::fused_norm)
      .def_static("rotary", &FxNode::rotary).def_static("cast", &FxNode::cast)
      .def_static("concat", &FxNode::concat).def_static("reshape", &FxNode::reshape)
      .def_static("transpose", &FxNode::transpose).def_static("scalar", &FxNode::scalar);
  pybind11::class_<gaudirt::generic::BucketProgram, std::shared_ptr<gaudirt::generic::BucketProgram>>(m, "BucketProgram")
      .def(pybind11::init<std::vector<int64_t>, std::vector<FxNode>, std::vector<int64_t>>())
      .def("run", &gaudirt::generic::BucketProgram::run,
           pybind11::call_guard<pybind11::gil_scoped_release>());
  using gaudirt::generic::SegmentViewKind;
  using gaudirt::generic::SegmentView;
  using gaudirt::generic::SegmentRoute;
  using gaudirt::generic::SegmentedRunner;
  pybind11::enum_<SegmentViewKind>(m, "SegmentViewKind")
      .value("feature", SegmentViewKind::feature).value("reshape", SegmentViewKind::reshape)
      .value("transpose", SegmentViewKind::transpose);
  pybind11::class_<SegmentView>(m, "SegmentView")
      .def(pybind11::init<SegmentViewKind, std::vector<int64_t>>());
  pybind11::class_<SegmentRoute>(m, "SegmentRoute")
      .def(pybind11::init<bool, size_t, std::vector<SegmentView>>());
  pybind11::class_<SegmentedRunner>(m, "SegmentedRunner")
      .def(pybind11::init<size_t>())
      .def("add_program", &SegmentedRunner::add_program)
      .def("add_attention", &SegmentedRunner::add_attention)
      .def("set_outputs", &SegmentedRunner::set_outputs)
      .def("run", &SegmentedRunner::run, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("run_traced", &SegmentedRunner::run_traced,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("run_profiled", &SegmentedRunner::run_profiled,
           pybind11::call_guard<pybind11::gil_scoped_release>());
}
