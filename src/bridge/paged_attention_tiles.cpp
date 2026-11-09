// SPDX-License-Identifier: Apache-2.0
#include "paged_attention_tiles.hpp"

#include "graph_builder.hpp"
#include "paged_kv_pool.hpp"
#include "temporaries.hpp"
#include "gaudirt/common/sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <memory>
#include <sstream>

namespace gaudirt::generic {
using namespace runtime;
namespace {
uint64_t align(uint64_t bytes) { return (bytes + 255) & ~uint64_t{255}; }

struct GraphKey {
  std::string key, prefix;
  explicit GraphKey(const std::string& structure, uint32_t bucket) {
    const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
    key = "paged_tile_" + digest;
    prefix = "g" + digest.substr(0, 12) + "_b" + std::to_string(bucket);
  }
};

ColdRecipe& store_recipe(Context& ctx, const AttentionPlan& plan, uint32_t rows) {
  const uint64_t d = plan.head_dim, kv_heads = plan.kv_heads, width = d * kv_heads;
  GraphKey graph_key("store:" + std::to_string(rows) + ':' + std::to_string(d) + ':' +
                     std::to_string(kv_heads), rows);
  if (auto found = ctx.cold_recipes.find(graph_key.key); found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(6);
  SynapseGraphBuilder graph(graph_key.prefix, *recipe);
  auto key = graph.tensor(syn_type_bf16, {width, rows}, 0);
  auto value = graph.tensor(syn_type_bf16, {width, rows}, 1);
  auto locations = graph.tensor(syn_type_int32, {rows}, 2);
  auto header = graph.tensor(syn_type_int32, {3}, 3);
  auto old_pool = graph.tensor(syn_type_bf16, {d, 2 * kPagedKvDescriptorSlots * kv_heads}, 4);
  auto pool = graph.tensor(syn_type_bf16, {d, 2 * kPagedKvDescriptorSlots * kv_heads}, 5, 4);
  graph.node({"gk_kv_store_head_major_bf16_v1", {key, value, locations, header, old_pool}, {pool}, {}});
  graph.compile(ctx.cache, graph_key.key);
  auto [it, inserted] = ctx.cold_recipes.emplace(graph_key.key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT attention: duplicate KV store recipe");
  return *it->second;
}

ColdRecipe& step_recipe(Context& ctx, const AttentionPlan& plan, const AttentionBlock& block, float scale) {
  const uint64_t rows = block.query_bucket, kv = block.kv_bucket;
  const uint64_t d = plan.head_dim, heads = plan.query_heads, kv_heads = plan.kv_heads;
  const uint64_t group = heads / kv_heads, stats_rows = (rows + 63) / 64 * 64;
  std::ostringstream signature;
  signature << "step:" << rows << ':' << kv << ':' << d << ':' << heads << ':' << kv_heads << ':'
            << std::bit_cast<uint32_t>(scale);
  GraphKey graph_key(signature.str(), rows);
  if (auto found = ctx.cold_recipes.find(graph_key.key); found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(12);
  SynapseGraphBuilder graph(graph_key.prefix, *recipe);
  auto query = graph.tensor(syn_type_bf16, {d, rows, group, kv_heads}, 0);
  synTensorPermutation q_perm{};
  q_perm.dims = 4;
  q_perm.permutation[0] = 0; q_perm.permutation[1] = 2;
  q_perm.permutation[2] = 3; q_perm.permutation[3] = 1;
  if (rows > 1) graph.permutation(query, q_perm);
  auto pool = graph.tensor(syn_type_bf16, {d, 2 * kPagedKvDescriptorSlots * kv_heads}, 1);
  auto key_indices = graph.tensor(syn_type_int32, {kv * kv_heads}, 2);
  auto value_indices = graph.tensor(syn_type_int32, {kv * kv_heads}, 3, 2, kv * kv_heads * 4);
  auto positions = graph.tensor(syn_type_int32, {rows}, 4);
  auto header = graph.tensor(syn_type_int32, {3}, 5);
  auto old_m = graph.tensor(syn_type_float, {stats_rows, heads}, 6);
  auto old_l = graph.tensor(syn_type_float, {stats_rows, heads}, 7);
  auto old_o = graph.tensor(syn_type_float, {d, rows, heads}, 8);
  auto new_m = graph.tensor(syn_type_float, {stats_rows, heads}, 9);
  auto new_l = graph.tensor(syn_type_float, {stats_rows, heads}, 10);
  auto new_o = graph.tensor(syn_type_float, {d, rows, heads}, 11);
  ns_GatherKernel::ParamsV2 gather{};
  gather.axis = 1;
  std::array<synTensor, 2> tiles{};
  for (unsigned i = 0; i < 2; ++i) {
    auto gathered = graph.tensor(syn_type_bf16, {d, kv * kv_heads}, -1);
    tiles[i] = graph.tensor(syn_type_bf16, {d, kv, 1, kv_heads}, -1);
    graph.node({"gather_fwd_bf16", {pool, i ? value_indices : key_indices}, {gathered}, gather});
    graph.node({"reshape", {gathered}, {tiles[i]}, {}});
  }
  auto score4 = graph.tensor(syn_type_float, {kv, rows, group, kv_heads}, -1);
  auto score3 = graph.tensor(syn_type_float, {kv, rows, heads}, -1);
  auto probability3 = graph.tensor(syn_type_bf16, {kv, rows, heads}, -1);
  auto probability4 = graph.tensor(syn_type_bf16, {kv, rows, group, kv_heads}, -1);
  auto tile_m = graph.tensor(syn_type_float, {stats_rows, heads}, -1);
  auto tile_l = graph.tensor(syn_type_float, {stats_rows, heads}, -1);
  auto expanded_m = graph.tensor(syn_type_float, {64, rows, heads}, -1);
  auto expanded_l = graph.tensor(syn_type_float, {64, rows, heads}, -1);
  auto tile_o4 = graph.tensor(syn_type_float, {d, rows, group, kv_heads}, -1);
  auto tile_o = graph.tensor(syn_type_float, {d, rows, heads}, -1);
  graph.node({"batch_gemm", {query, tiles[0]}, {score4}, synGEMMParams{false, true}});
  graph.node({"reshape", {score4}, {score3}, {}});
  graph.node({"gk_attn_tile_softmax_bf16_v1", {score3, positions, header},
              {probability3, expanded_m, expanded_l}, scale});
  graph.node({"gk_attn_pack_stats_f32_v1", {expanded_m, expanded_l}, {tile_m, tile_l}, {}});
  graph.node({"reshape", {probability3}, {probability4}, {}});
  graph.node({"batch_gemm", {probability4, tiles[1]}, {tile_o4}, synGEMMParams{false, false}});
  graph.node({"reshape", {tile_o4}, {tile_o}, {}});
  graph.node({"gk_attn_online_merge_f32_v1", {old_m, old_l, old_o, tile_m, tile_l, tile_o},
              {new_m, new_l, new_o}, {}});
  graph.compile(ctx.cache, graph_key.key);
  auto [it, inserted] = ctx.cold_recipes.emplace(graph_key.key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT attention: duplicate tile recipe");
  return *it->second;
}

ColdRecipe& finish_recipe(Context& ctx, const AttentionPlan& plan, const AttentionBlock& block) {
  const uint64_t rows = block.query_bucket, heads = plan.query_heads, d = plan.head_dim;
  const uint64_t stats_rows = (rows + 63) / 64 * 64;
  GraphKey graph_key("finish:" + std::to_string(rows) + ':' + std::to_string(heads) + ':' +
                     std::to_string(d), rows);
  if (auto found = ctx.cold_recipes.find(graph_key.key); found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(3);
  SynapseGraphBuilder graph(graph_key.prefix, *recipe);
  auto sums = graph.tensor(syn_type_float, {stats_rows, heads}, 0);
  auto accumulator = graph.tensor(syn_type_float, {d, rows, heads}, 1);
  auto output = graph.tensor(syn_type_bf16, {d, rows, heads}, 2);
  synTensorPermutation permutation{};
  permutation.dims = 3;
  permutation.permutation[0] = 0; permutation.permutation[1] = 2; permutation.permutation[2] = 1;
  graph.permutation(output, permutation);
  graph.node({"gk_attn_online_finalize_bf16_v1", {sums, accumulator}, {output}, {}});
  graph.compile(ctx.cache, graph_key.key);
  auto [it, inserted] = ctx.cold_recipes.emplace(graph_key.key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT attention: duplicate finish recipe");
  return *it->second;
}

uint64_t zero_state(Context& ctx, uint64_t bytes, synStreamHandle stream) {
  const auto key = std::to_string(bytes) + ':' + std::to_string(reinterpret_cast<uintptr_t>(stream));
  if (auto found = ctx.attention_zero_states.find(key); found != ctx.attention_zero_states.end())
    return found->second.first;
  const auto address = ctx.allocator.allocate(bytes, stream);
  if (synMemsetD32Async(address, 0, bytes / 4, stream) != synSuccess) {
    ctx.allocator.free(address);
    TORCH_CHECK(false, "GaudiRT attention: zero-state initialization failed");
  }
  try {
    ctx.attention_zero_states.emplace(key, std::make_pair(address, bytes));
  } catch (...) {
    ctx.allocator.free(address);
    throw;
  }
  return address;
}

struct TiledRecipes {
  uint64_t generation = 0, zero = 0;
  synStreamHandle stream = nullptr;
  uint32_t rows = 0, kv = 0, scale = 0;
  int64_t heads = 0, kv_heads = 0, head_dim = 0;
  ColdRecipe *step = nullptr, *finish = nullptr;
};

const TiledRecipes& cached_tiled_recipes(Context& ctx, const AttentionPlan& plan,
                                          const AttentionBlock& block, float scale, uint64_t stride) {
  thread_local TiledRecipes cache;
  const auto stream = ctx.stream();
  const auto scale_bits = std::bit_cast<uint32_t>(scale);
  if (cache.step && cache.generation == ctx.generation && cache.stream == stream &&
      cache.rows == block.query_bucket && cache.kv == block.kv_bucket && cache.scale == scale_bits &&
      cache.heads == plan.query_heads && cache.kv_heads == plan.kv_heads && cache.head_dim == plan.head_dim)
    return cache;
  std::lock_guard lock(ctx.cold_mutex);
  auto* step = &step_recipe(ctx, plan, block, scale);
  auto* finish = &finish_recipe(ctx, plan, block);
  const auto zero = zero_state(ctx, stride, stream);
  cache = {ctx.generation, zero, stream, block.query_bucket, block.kv_bucket, scale_bits,
           plan.query_heads, plan.kv_heads, plan.head_dim, step, finish};
  return cache;
}
}  // namespace

void launch_layer_kv_store(Context& ctx, const AttentionPlan& plan, uint64_t key, uint64_t value,
                           uint64_t locations, uint64_t pool) {
  struct Cache {
    uint64_t generation = 0;
    uint32_t rows = 0;
    int64_t kv_heads = 0, head_dim = 0;
    ColdRecipe* recipe = nullptr;
  };
  thread_local Cache cached;
  const auto rows = std::bit_ceil(static_cast<uint32_t>(plan.rows));
  if (!cached.recipe || cached.generation != ctx.generation || cached.rows != rows ||
      cached.kv_heads != plan.kv_heads || cached.head_dim != plan.head_dim) {
    std::lock_guard lock(ctx.cold_mutex);
    cached = {ctx.generation, rows, plan.kv_heads, plan.head_dim, &store_recipe(ctx, plan, rows)};
  }
  auto& recipe = *cached.recipe;
  auto bindings = recipe.tensors;
  const std::array<uint64_t, 6> addresses{key, value, locations,
      plan.address + plan.store_header_offset, pool, pool};
  for (size_t i = 0; i < addresses.size(); ++i) bindings[i].pTensorAddress = addresses[i];
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(ctx, recipe.workspace);
  TORCH_CHECK(synLaunch(ctx.stream(), bindings.data(), bindings.size(), workspace ? workspace->address : 0,
                        recipe.recipe, 0) == synSuccess, "GaudiRT attention: KV store launch failed");
}

void launch_tiled_attention(Context& ctx, const AttentionPlan& plan, const AttentionBlock& block,
                            const TiledAttentionInputs& input, float scale) {
  const auto stream = ctx.stream();
  const uint64_t rows = block.query_bucket, heads = plan.query_heads, d = plan.head_dim;
  const uint64_t stats = align((rows + 63) / 64 * 64 * heads * sizeof(float));
  const uint64_t output_bytes = align(rows * heads * d * sizeof(float));
  const uint64_t stride = 2 * stats + output_bytes;
  TORCH_CHECK(plan.scratch_bytes >= 2 * stride, "GaudiRT attention: insufficient tile scratch");
  const auto& cached = cached_tiled_recipes(ctx, plan, block, scale, stride);
  auto* step = cached.step;
  auto* finish = cached.finish;
  const auto zero = cached.zero;
  const auto workspace_size = std::max(step->workspace, finish->workspace);
  std::unique_ptr<TemporaryDevice> workspace;
  if (workspace_size) workspace = std::make_unique<TemporaryDevice>(ctx, workspace_size);
  const auto work = workspace ? workspace->address : 0;
  auto bindings = step->tensors;
  const auto scratch = plan.address + plan.scratch_offset;
  const auto value_indices_offset = static_cast<uint64_t>(block.kv_bucket) * plan.kv_heads * sizeof(int32_t);
  for (size_t tile_index = 0; tile_index < block.tiles.size(); ++tile_index) {
    const auto& tile = block.tiles[tile_index];
    const uint64_t old = tile_index ? scratch + ((tile_index - 1) % 2) * stride : zero;
    const uint64_t next = scratch + (tile_index % 2) * stride;
    const std::array<uint64_t, 12> addresses{input.query, input.pool, plan.address + tile.indices_offset,
        plan.address + tile.indices_offset + value_indices_offset, plan.address + block.positions_offset,
        plan.address + tile.header_offset, old, old + stats, old + 2 * stats,
        next, next + stats, next + 2 * stats};
    for (size_t i = 0; i < addresses.size(); ++i) bindings[i].pTensorAddress = addresses[i];
    TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(), work, step->recipe, 0) == synSuccess,
                "GaudiRT attention: tile launch failed");
  }
  const uint64_t last = scratch + ((block.tiles.size() - 1) % 2) * stride;
  auto finish_bindings = finish->tensors;
  finish_bindings[0].pTensorAddress = last + stats;
  finish_bindings[1].pTensorAddress = last + 2 * stats;
  finish_bindings[2].pTensorAddress = input.output;
  TORCH_CHECK(synLaunch(stream, finish_bindings.data(), finish_bindings.size(), work, finish->recipe, 0) == synSuccess,
              "GaudiRT attention: finalization launch failed");
}

}  // namespace gaudirt::generic
