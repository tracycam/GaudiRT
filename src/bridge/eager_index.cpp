// SPDX-License-Identifier: Apache-2.0
#include "bridge_internal.hpp"
#include "eager_copy.hpp"
#include "eager_ops.hpp"
#include "eager_index.hpp"
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
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 2> sections{};
  std::array<synTensor, 2> tensors{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "index check graph creation failed");
    for (size_t i = 0; i < 2; ++i) {
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "index check section creation failed");
      check(synSectionSetPersistent(sections[i], true), "index check section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = syn_type_float;
      desc.m_dims = 1;
      desc.m_sizes[0] = desc.m_minSizes[0] = i == 0 ? count : 1;
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "index check tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
      recipe->tensors[i].tensorSize[0] = i == 0 ? count : 1;
    }
    ns_Reduction::Params params{};
    params.reductionDimension = 0;
    const auto node = key + "_n0";
    check(synNodeCreate(graph, &tensors[0], &tensors[1], 1, 1, &params, sizeof(params),
                        "reduce_sum_f32", node.c_str(), nullptr, nullptr), "index check reduction failed");
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "index check workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
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
                          uint32_t indices, int64_t axis) {
  std::string signature = std::to_string(static_cast<int>(input.scalar_type())) + ":" +
                          std::to_string(axis) + ":" + std::to_string(indices);
  for (auto size : input.sizes()) signature += ":" + std::to_string(size);
  const auto hash = common::sha256_bytes(reinterpret_cast<const uint8_t*>(signature.data()),
                                          signature.size());
  const std::string key = "cold_gather_" + hash;
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(3);
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 3> sections{};
  std::array<synTensor, 3> tensors{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "gather graph creation failed");
    for (size_t i = 0; i < 3; ++i) {
      const auto& shaped = i == 0 ? input : output;
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "gather section creation failed");
      check(synSectionSetPersistent(sections[i], true), "gather section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = i == 1 ? syn_type_int32 : cast_type(input.scalar_type()).syn;
      desc.m_dims = i == 1 ? 1 : shaped.dim();
      for (unsigned dim = 0; dim < desc.m_dims; ++dim) {
        const auto size = i == 1 ? indices : shaped.size(shaped.dim() - dim - 1);
        TORCH_CHECK(size <= UINT32_MAX, "GaudiRT: gather dimension too large");
        desc.m_sizes[dim] = desc.m_minSizes[dim] = size;
        recipe->tensors[i].tensorSize[dim] = size;
      }
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "gather tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
    }
    ns_GatherKernel::ParamsV2 params{};
    params.axis = input.dim() - axis - 1;
    params.batchDims = 0;
    const auto guid = "gather_fwd_" + std::string(cast_type(input.scalar_type()).name);
    const auto node = key + "_n0";
    check(synNodeCreate(graph, tensors.data(), &tensors[2], 2, 1, &params, sizeof(params),
                        guid.c_str(), node.c_str(), nullptr, nullptr), "gather node creation failed");
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "gather workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate gather recipe");
  return *it->second;
}

at::Tensor index_select(const at::Tensor& input, int64_t axis, const at::Tensor& index) {
  TORCH_CHECK(input.device() == index.device() && input.device().type() == c10::kPrivateUse1 &&
              (index.scalar_type() == at::kInt || index.scalar_type() == at::kLong),
              "GaudiRT: index_select requires device int32 or int64 indices");
  TORCH_CHECK(input.dim() > 0 && input.dim() <= 5 && index.dim() <= 1,
              "GaudiRT: index_select rank is not supported");
  cast_type(input.scalar_type());
  axis = at::maybe_wrap_dim(axis, input.dim());
  const auto count = index.numel();
  validate_indices(index, input.size(axis));
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
  auto ids = index.scalar_type() == at::kInt ? index : index.to(at::kInt);
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
  auto& recipe = gather_recipe(*ctx, source, result, count, axis);
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

ColdRecipe& index_put_recipe(Context& ctx, uint32_t count, uint32_t slots) {
  const auto key = "cold_index_put_i64_" + std::to_string(count) + "_" + std::to_string(slots);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto compiled = std::make_unique<ColdRecipe>(3);
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 3> sections{};
  std::array<synTensor, 3> tensors{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT index put: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "graph creation failed");
    for (size_t i = 0; i < tensors.size(); ++i) {
      compiled->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "section creation failed");
      check(synSectionSetPersistent(sections[i], true), "section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = compiled->names[i].c_str();
      desc.m_dataType = syn_type_int32;
      desc.m_dims = i == 0 ? 1 : 2;
      desc.m_sizes[0] = desc.m_minSizes[0] = i == 0 ? count : 2;
      if (i != 0) desc.m_sizes[1] = desc.m_minSizes[1] = i == 1 ? count : slots;
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "tensor creation failed");
      compiled->tensors[i].tensorName = compiled->names[i].c_str();
      compiled->tensors[i].tensorType = DATA_TENSOR;
      for (unsigned dim = 0; dim < desc.m_dims; ++dim)
        compiled->tensors[i].tensorSize[dim] = desc.m_sizes[dim];
    }
    const auto node = key + "_n0";
    check(synNodeCreate(graph, tensors.data(), tensors.data() + 2, 2, 1, nullptr, 0,
                        "gk_index_put_i64_v1", node.c_str(), nullptr, nullptr), "node creation failed");
    compiled->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&compiled->workspace, compiled->recipe), "workspace query failed");
    initialize_launch_tensor_ids(compiled->recipe, compiled->tensors.data(), compiled->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
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

at::Tensor& index_put(at::Tensor& self, const c10::List<std::optional<at::Tensor>>& indices,
                      const at::Tensor& values, bool accumulate, bool unsafe) {
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
  validate_indices(index, self.numel());
  if (!index.numel()) return self;
  auto ctx = current();
  auto output = self;
  auto output_address = address_of(output, *ctx);
  if (!output.is_contiguous() || output_address % BlockCachingAllocator::kAlignment) {
    output = at::empty(self.sizes(), self.options());
    copy(output, self, true);
    output_address = address_of(output, *ctx);
  }
  auto ids = index.scalar_type() == at::kInt ? index : index.to(at::kInt);
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
    auto& recipe = index_put_recipe(*ctx, index.numel(), self.numel());
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
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 2> sections{};
  std::array<synTensor, 17> tensors{};
  std::array<std::string, 14> internal_names{};
  const std::string index_name = key + "_index_i32";
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "argmax graph creation failed");
    for (size_t i = 0; i < 2; ++i) {
      const auto& shaped = i == 0 ? input : reduced;
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "argmax section creation failed");
      check(synSectionSetPersistent(sections[i], true), "argmax section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = i == 0 ? cast_type(input.scalar_type()).syn :
                        (values_only ? syn_type_int32 : syn_type_int64);
      desc.m_dims = shaped.dim();
      for (unsigned dim = 0; dim < desc.m_dims; ++dim) {
        const auto size = shaped.size(shaped.dim() - dim - 1);
        TORCH_CHECK(size <= UINT32_MAX, "GaudiRT: argmax dimension too large");
        desc.m_sizes[dim] = desc.m_minSizes[dim] = size;
        recipe->tensors[i].tensorSize[dim] = size;
      }
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "argmax tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
    }
    ns_Reduction::Params params{};
    params.reductionDimension = input.dim() - axis - 1;
    const bool floating = !values_only &&
                          (input.scalar_type() == at::kFloat || input.scalar_type() == at::kBFloat16);
    if (!values_only) {
      synTensorDescriptor desc{};
      desc.m_name = index_name.c_str();
      desc.m_dataType = syn_type_int32;
      desc.m_dims = reduced.dim();
      for (unsigned dim = 0; dim < desc.m_dims; ++dim)
        desc.m_sizes[dim] = desc.m_minSizes[dim] = reduced.size(reduced.dim() - dim - 1);
      check(synTensorCreate(&tensors[16], &desc, nullptr, 0), "argmax index creation failed");
    }
    if (floating) {
      const synDataType dtypes[] = {syn_type_uint8, syn_type_float, syn_type_float,
                                   syn_type_float, syn_type_uint8, syn_type_int32, syn_type_int32,
                                   cast_type(input.scalar_type()).syn, syn_type_uint8, syn_type_float,
                                   syn_type_float, syn_type_uint8, syn_type_int32, syn_type_int32};
      for (size_t i = 0; i < internal_names.size(); ++i) {
        const auto& shaped = (i < 2 || (i >= 7 && i <= 9)) ? input : reduced;
        internal_names[i] = key + "_t" + std::to_string(i + 2);
        synTensorDescriptor desc{};
        desc.m_name = internal_names[i].c_str();
        desc.m_dataType = dtypes[i];
        desc.m_dims = shaped.dim();
        for (unsigned dim = 0; dim < desc.m_dims; ++dim)
          desc.m_sizes[dim] = desc.m_minSizes[dim] = shaped.size(shaped.dim() - dim - 1);
        check(synTensorCreate(&tensors[i + 2], &desc, nullptr, 0), "argmax intermediate creation failed");
      }
    }
    const auto guid = values_only ? std::string("reduce_max_i32") :
                                    "argmax_fwd_" + std::string(cast_type(input.scalar_type()).name);
    const auto node = key + "_n0";
    auto* ordinary = floating ? &tensors[8] : (values_only ? &tensors[1] : &tensors[16]);
    check(synNodeCreate(graph, &tensors[0], ordinary, 1, 1, &params, sizeof(params),
                        guid.c_str(), node.c_str(), nullptr, nullptr), "argmax node creation failed");
    if (floating) {
      const auto isnan_guid = "isnan_fwd_" + std::string(cast_type(input.scalar_type()).name);
      const auto isnan_node = key + "_n1";
      check(synNodeCreate(graph, &tensors[0], &tensors[2], 1, 1, nullptr, 0,
                          isnan_guid.c_str(), isnan_node.c_str(), nullptr, nullptr), "argmax isnan failed");
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_HALF_NE;
      const auto cast_node = key + "_n2";
      check(synNodeCreate(graph, &tensors[2], &tensors[3], 1, 1, &cast, sizeof(cast),
                          "cast_u8_to_f32", cast_node.c_str(), nullptr, nullptr), "argmax mask cast failed");
      const auto nan_argmax_node = key + "_n3";
      check(synNodeCreate(graph, &tensors[3], &tensors[7], 1, 1, &params, sizeof(params),
                          "argmax_fwd_f32", nan_argmax_node.c_str(), nullptr, nullptr), "argmax NaN index failed");
      const auto nan_reduce_node = key + "_n4";
      check(synNodeCreate(graph, &tensors[3], &tensors[4], 1, 1, &params, sizeof(params),
                          "reduce_max_f32", nan_reduce_node.c_str(), nullptr, nullptr), "argmax NaN reduction failed");
      ns_ConstantKernel::ParamsV3 zero{};
      const auto zero_node = key + "_n5";
      check(synNodeCreate(graph, nullptr, &tensors[5], 0, 1, &zero, sizeof(zero),
                          "constant_f32", zero_node.c_str(), nullptr, nullptr), "argmax zero creation failed");
      synTensor compare[] = {tensors[4], tensors[5]};
      const auto compare_node = key + "_n6";
      check(synNodeCreate(graph, compare, &tensors[6], 2, 1, nullptr, 0,
                          "greater_fwd_f32", compare_node.c_str(), nullptr, nullptr), "argmax NaN comparison failed");
      ns_ConstantKernel::ParamsV3 negative_inf{};
      negative_inf.constant.f = -std::numeric_limits<float>::infinity();
      const auto constant_guid = "constant_" + std::string(cast_type(input.scalar_type()).name);
      const auto inf_node = key + "_n7";
      check(synNodeCreate(graph, nullptr, &tensors[9], 0, 1, &negative_inf, sizeof(negative_inf),
                          constant_guid.c_str(), inf_node.c_str(), nullptr, nullptr), "argmax infinity creation failed");
      synTensor equal_inputs[] = {tensors[0], tensors[9]};
      const auto equal_guid = "equal_fwd_" + std::string(cast_type(input.scalar_type()).name);
      const auto equal_node = key + "_n8";
      check(synNodeCreate(graph, equal_inputs, &tensors[10], 2, 1, nullptr, 0,
                          equal_guid.c_str(), equal_node.c_str(), nullptr, nullptr), "argmax infinity check failed");
      const auto inf_cast_node = key + "_n9";
      check(synNodeCreate(graph, &tensors[10], &tensors[11], 1, 1, &cast, sizeof(cast),
                          "cast_u8_to_f32", inf_cast_node.c_str(), nullptr, nullptr), "argmax infinity cast failed");
      const auto inf_reduce_node = key + "_n10";
      check(synNodeCreate(graph, &tensors[11], &tensors[12], 1, 1, &params, sizeof(params),
                          "reduce_min_f32", inf_reduce_node.c_str(), nullptr, nullptr), "argmax infinity reduction failed");
      synTensor all_inf_inputs[] = {tensors[12], tensors[5]};
      const auto all_inf_node = key + "_n11";
      check(synNodeCreate(graph, all_inf_inputs, &tensors[13], 2, 1, nullptr, 0,
                          "greater_fwd_f32", all_inf_node.c_str(), nullptr, nullptr), "argmax infinity comparison failed");
      const auto zero_index_node = key + "_n12";
      check(synNodeCreate(graph, &tensors[5], &tensors[14], 1, 1, &cast, sizeof(cast),
                          "cast_f32_to_i32", zero_index_node.c_str(), nullptr, nullptr), "argmax zero index failed");
      synTensor safe_choices[] = {tensors[13], tensors[14], tensors[8]};
      const auto safe_node = key + "_n13";
      check(synNodeCreate(graph, safe_choices, &tensors[15], 3, 1, nullptr, 0,
                          "where_fwd_i32", safe_node.c_str(), nullptr, nullptr), "argmax infinity selection failed");
      synTensor choices[] = {tensors[6], tensors[7], tensors[15]};
      const auto select_node = key + "_n14";
      check(synNodeCreate(graph, choices, &tensors[16], 3, 1, nullptr, 0,
                          "where_fwd_i32", select_node.c_str(), nullptr, nullptr), "argmax NaN selection failed");
    }
    if (!values_only) {
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_HALF_NE;
      const auto cast_node = key + "_to_i64";
      check(synNodeCreate(graph, &tensors[16], &tensors[1], 1, 1, &cast, sizeof(cast),
                          "cast_i32_to_i64", cast_node.c_str(), nullptr, nullptr), "argmax int64 cast failed");
    }
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "argmax workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
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
    auto& recipe = argmax_recipe(*ctx, source, reduced, axis);
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
