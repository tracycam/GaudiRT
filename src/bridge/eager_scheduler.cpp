// SPDX-License-Identifier: Apache-2.0
#include "eager_scheduler.hpp"

#include "bridge_internal.hpp"
#include "cold_recipe.hpp"
#include "context.hpp"
#include "temporaries.hpp"

#include <ATen/ATen.h>
#include <perf_lib_layer_params.h>

#include <array>
#include <climits>
#include <memory>
#include <string>

namespace gaudirt::generic {
using namespace runtime;

static ColdRecipe& decode_lengths_recipe(Context& ctx, uint32_t count) {
  const auto key = "cold_decode_lengths_v1_" + std::to_string(count);
  if (auto found = ctx.cold_recipes.find(key); found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(4);
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 4> sections{};
  std::array<synTensor, 6> tensors{};
  const std::array<std::string, 2> inner_names{key + "_one_i32", key + "_one_i64"};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT decode lengths: ", message,
                " status=", static_cast<int>(status));
  };
  const auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "graph creation failed");
    for (size_t i = 0; i < sections.size(); ++i) {
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "section creation failed");
      check(synSectionSetPersistent(sections[i], true), "section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = i % 2 == 0 ? syn_type_int64 : syn_type_int32;
      desc.m_dims = 1;
      desc.m_sizes[0] = desc.m_minSizes[0] = count;
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
      recipe->tensors[i].tensorSize[0] = count;
    }
    for (size_t i = 0; i < inner_names.size(); ++i) {
      synTensorDescriptor desc{};
      desc.m_name = inner_names[i].c_str();
      desc.m_dataType = i ? syn_type_int64 : syn_type_int32;
      desc.m_dims = 1;
      desc.m_sizes[0] = desc.m_minSizes[0] = 1;
      check(synTensorCreate(&tensors[4 + i], &desc, nullptr, 0), "constant tensor creation failed");
    }
    ns_ConstantKernel::ParamsV3 one{};
    one.constant.i = 1;
    const auto constant_node = key + "_n0";
    check(synNodeCreate(graph, nullptr, &tensors[4], 0, 1, &one, sizeof(one),
                        "constant_i32", constant_node.c_str(), nullptr, nullptr), "constant failed");
    ns_CastKernel::ParamsV3 cast{};
    cast.round_mode = CAST_ROUND_HALF_NE;
    const auto cast_node = key + "_n1";
    check(synNodeCreate(graph, &tensors[4], &tensors[5], 1, 1, &cast, sizeof(cast),
                        "cast_i32_to_i64", cast_node.c_str(), nullptr, nullptr), "cast failed");
    synTensor wide_inputs[]{tensors[0], tensors[5]};
    synTensor narrow_inputs[]{tensors[1], tensors[4]};
    const auto wide_node = key + "_n2", narrow_node = key + "_n3";
    check(synNodeCreate(graph, wide_inputs, &tensors[2], 2, 1, nullptr, 0,
                        "add_fwd_i64", wide_node.c_str(), nullptr, nullptr), "int64 addition failed");
    check(synNodeCreate(graph, narrow_inputs, &tensors[3], 2, 1, nullptr, 0,
                        "add_fwd_i32", narrow_node.c_str(), nullptr, nullptr), "int32 addition failed");
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT decode lengths: duplicate recipe");
  return *it->second;
}

std::tuple<at::Tensor, at::Tensor> advance_decode_lengths(const at::Tensor& seq_lens,
                                                         const at::Tensor& orig_seq_lens) {
  TORCH_CHECK(seq_lens.device().type() == c10::kPrivateUse1 &&
              orig_seq_lens.device() == seq_lens.device() && seq_lens.scalar_type() == at::kLong &&
              orig_seq_lens.scalar_type() == at::kInt && seq_lens.dim() == 1 &&
              orig_seq_lens.sizes() == seq_lens.sizes() && seq_lens.numel() <= INT_MAX &&
              seq_lens.is_contiguous() && orig_seq_lens.is_contiguous(),
              "GaudiRT decode lengths: expected matching contiguous int64 and int32 vectors");
  auto next = at::empty_like(seq_lens), orig_next = at::empty_like(orig_seq_lens);
  if (!seq_lens.numel()) return {next, orig_next};
  auto ctx = current();
  const std::array<at::Tensor, 4> values{seq_lens, orig_seq_lens, next, orig_next};
  auto stream = ctx->stream();
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = decode_lengths_recipe(*ctx, seq_lens.numel());
  auto bindings = recipe.tensors;
  for (size_t i = 0; i < values.size(); ++i) {
    const auto address = address_of(values[i], *ctx);
    TORCH_CHECK(address % BlockCachingAllocator::kAlignment == 0,
                "GaudiRT decode lengths: inputs must be 256-byte aligned");
    bindings[i].pTensorAddress = address;
    ctx->allocator.record_stream(address, stream);
  }
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(), workspace ? workspace->address : 0,
                        recipe.recipe, 0) == synSuccess, "GaudiRT decode lengths: launch failed");
  return {next, orig_next};
}
}  // namespace gaudirt::generic
