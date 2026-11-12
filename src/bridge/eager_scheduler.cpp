// SPDX-License-Identifier: Apache-2.0
#include "eager_scheduler.hpp"

#include "bridge_internal.hpp"
#include "cold_recipe.hpp"
#include "context.hpp"
#include "graph_builder.hpp"
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
  {
    SynapseGraphBuilder builder(key, *recipe);
    std::array<synTensor, 4> ext{};
    for (size_t i = 0; i < ext.size(); ++i)
      ext[i] = builder.tensor(i % 2 == 0 ? syn_type_int64 : syn_type_int32, {count}, static_cast<int>(i));
    auto one_i32 = builder.tensor(syn_type_int32, {1}, -1);
    auto one_i64 = builder.tensor(syn_type_int64, {1}, -1);
    ns_ConstantKernel::ParamsV3 one{};
    one.constant.i = 1;
    builder.node(NodeSpec{"constant_i32", {}, {one_i32}, one});
    ns_CastKernel::ParamsV3 cast{};
    cast.round_mode = CAST_ROUND_HALF_NE;
    builder.node(NodeSpec{"cast_i32_to_i64", {one_i32}, {one_i64}, cast});
    builder.node(NodeSpec{"add_fwd_i64", {ext[0], one_i64}, {ext[2]}, {}});
    builder.node(NodeSpec{"add_fwd_i32", {ext[1], one_i32}, {ext[3]}, {}});
    builder.compile(ctx.cache, key);
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
