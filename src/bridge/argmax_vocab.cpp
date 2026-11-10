// SPDX-License-Identifier: Apache-2.0
#include "argmax_vocab.hpp"

#include "context.hpp"
#include "graph_builder.hpp"

#include <algorithm>
#include <memory>
#include <string>

namespace gaudirt::generic {

ColdRecipe& vocabulary_argmax_recipe(Context& ctx, const at::Tensor& input) {
  const auto width = input.size(1), rows = input.size(0);
  const int64_t total_blocks = (width + 127) / 128;
  const int32_t target_parts = std::min<int64_t>(24, total_blocks);
  const int32_t blocks = (total_blocks + target_parts - 1) / target_parts;
  const int32_t parts = (total_blocks + blocks - 1) / blocks;
  const bool bf16 = input.scalar_type() == at::kBFloat16;
  const auto key = "cold_vocab_argmax_v1_" + std::string(bf16 ? "bf16_" : "f32_") +
                   std::to_string(width) + "_" + std::to_string(rows);
  if (auto found = ctx.cold_recipes.find(key); found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(2);
  SynapseGraphBuilder graph(key, *recipe);
  auto source = graph.tensor(bf16 ? syn_type_bf16 : syn_type_float,
                             {static_cast<uint64_t>(width), static_cast<uint64_t>(rows)}, 0);
  auto output = graph.tensor(syn_type_int64, {1, static_cast<uint64_t>(rows)}, 1);
  auto maxima = graph.tensor(syn_type_float, {64, static_cast<uint64_t>(parts), static_cast<uint64_t>(rows)}, -1);
  auto indices = graph.tensor(syn_type_int32, {64, static_cast<uint64_t>(parts), static_cast<uint64_t>(rows)}, -1);
  auto nan_indices = graph.tensor(syn_type_int32, {64, static_cast<uint64_t>(parts), static_cast<uint64_t>(rows)}, -1);
  auto result = graph.tensor(syn_type_int32, {1, static_cast<uint64_t>(rows)}, -1);
  graph.node({bf16 ? "gk_argmax_parts_bf16_v1" : "gk_argmax_parts_f32_v1", {source},
              {maxima, indices, nan_indices}, ArgmaxParameters{static_cast<int32_t>(width), blocks}});
  graph.node({"gk_argmax_finish_i32_v1", {maxima, indices, nan_indices}, {result}, parts});
  ns_CastKernel::ParamsV3 cast{};
  cast.round_mode = CAST_ROUND_HALF_NE;
  graph.node({"cast_i32_to_i64", {result}, {output}, cast});
  graph.compile(ctx.cache, key);
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT argmax: duplicate vocabulary recipe");
  return *it->second;
}
}  // namespace gaudirt::generic
