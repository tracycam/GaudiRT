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
#include <vector>

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

ColdRecipe& chunk_recipe(Context& ctx, const AttentionPlan& plan, const AttentionBlock& block, float scale,
                         uint32_t chunk_tiles, bool first_chunk, bool finalize_last, bool direct_kv) {
  TORCH_CHECK(chunk_tiles >= 1 && chunk_tiles <= kMaxFusedAttentionTiles &&
              (!first_chunk || chunk_tiles >= 2),
              "GaudiRT attention: invalid fused tile chunk");
  const uint64_t rows = block.query_bucket, kv = block.kv_bucket;
  const uint64_t d = plan.head_dim, heads = plan.query_heads, kv_heads = plan.kv_heads;
  const uint64_t group = heads / kv_heads, stats_rows = (rows + 63) / 64 * 64;
  std::ostringstream signature;
  signature << (direct_kv ? "chunk_direct_v6:" : "chunk_paged_v6:")
            << rows << ':' << kv << ':' << d << ':' << heads << ':' << kv_heads << ':'
            << std::bit_cast<uint32_t>(scale) << ':' << chunk_tiles << ':'
            << (first_chunk ? 1 : 0) << ':' << (finalize_last ? 1 : 0);
  GraphKey graph_key(signature.str(), rows);
  if (auto found = ctx.cold_recipes.find(graph_key.key); found != ctx.cold_recipes.end()) return *found->second;

  const size_t slot_count = (direct_kv ? 2 : 3) + 3 * static_cast<size_t>(chunk_tiles) +
                            (first_chunk ? 0 : 3) + (finalize_last ? 1 : 3);
  auto recipe = std::make_unique<ColdRecipe>(slot_count);
  SynapseGraphBuilder graph(graph_key.prefix, *recipe);
  synTransposeParams perm3d{};
  perm3d.tensorDim = 3;
  perm3d.permutation[0] = TPD_Channel;
  perm3d.permutation[1] = TPD_Height;
  perm3d.permutation[2] = TPD_Width;

  int slot = 0;
  synTensor query = nullptr;
  if (rows > 1) {
    auto query_in = graph.tensor(syn_type_bf16, {d, heads, rows}, slot++);
    auto query_t = graph.tensor(syn_type_bf16, {d, rows, heads}, -1);
    query = graph.tensor(syn_type_bf16, {d, rows, group, kv_heads}, -1);
    graph.node({"transpose", {query_in}, {query_t}, perm3d});
    graph.node({"reshape", {query_t}, {query}, {}});
  } else {
    query = graph.tensor(syn_type_bf16, {d, rows, group, kv_heads}, slot++);
  }
  synTensor pool = nullptr;
  if (!direct_kv) {
    pool = graph.tensor(syn_type_bf16, {d, 2 * kPagedKvDescriptorSlots * kv_heads}, slot++);
  }
  auto positions = graph.tensor(syn_type_int32, {rows}, slot++);

  std::vector<synTensor> key_sources(chunk_tiles), value_sources(chunk_tiles), headers(chunk_tiles);
  const uint64_t value_indices_offset = kv * kv_heads * sizeof(int32_t);
  for (uint32_t t = 0; t < chunk_tiles; ++t) {
    if (direct_kv) {
      key_sources[t] = graph.tensor(syn_type_bf16, {d, kv_heads, kv}, slot++);
      value_sources[t] = graph.tensor(syn_type_bf16, {d, kv_heads, kv}, slot++);
    } else {
      const int key_slot = slot++;
      key_sources[t] = graph.tensor(syn_type_int32, {kv * kv_heads}, key_slot);
      value_sources[t] = graph.tensor(syn_type_int32, {kv * kv_heads}, slot++, key_slot, value_indices_offset);
    }
    headers[t] = graph.tensor(syn_type_int32, {3}, slot++);
  }

  synTensor cur_m = nullptr, cur_l = nullptr, cur_o = nullptr;
  if (!first_chunk) {
    cur_m = graph.tensor(syn_type_float, {stats_rows, group, kv_heads}, slot++);
    cur_l = graph.tensor(syn_type_float, {stats_rows, group, kv_heads}, slot++);
    cur_o = graph.tensor(syn_type_float, {d, rows, group, kv_heads}, slot++);
  }

  synTensor final_out = nullptr, out_m = nullptr, out_l = nullptr, out_o = nullptr;
  if (finalize_last) {
    final_out = rows > 1 ? graph.tensor(syn_type_bf16, {d, heads, rows}, slot++)
                         : graph.tensor(syn_type_bf16, {d, rows, group, kv_heads}, slot++);
  } else {
    out_m = graph.tensor(syn_type_float, {stats_rows, group, kv_heads}, slot++);
    out_l = graph.tensor(syn_type_float, {stats_rows, group, kv_heads}, slot++);
    out_o = graph.tensor(syn_type_float, {d, rows, group, kv_heads}, slot++);
  }
  TORCH_CHECK(static_cast<size_t>(slot) == slot_count, "GaudiRT attention: chunk slot count mismatch");

  ns_GatherKernel::ParamsV2 gather{};
  gather.axis = 1;
  for (uint32_t t = 0; t < chunk_tiles; ++t) {
    std::array<synTensor, 2> kv_tiles{};
    for (unsigned i = 0; i < 2; ++i) {
      auto src_tensor = i ? value_sources[t] : key_sources[t];
      kv_tiles[i] = graph.tensor(syn_type_bf16, {d, kv, 1, kv_heads}, -1);
      if (direct_kv) {
        auto transposed = graph.tensor(syn_type_bf16, {d, kv, kv_heads}, -1);
        graph.node({"transpose", {src_tensor}, {transposed}, perm3d});
        graph.node({"reshape", {transposed}, {kv_tiles[i]}, {}});
      } else {
        auto gathered = graph.tensor(syn_type_bf16, {d, kv * kv_heads}, -1);
        graph.node({"gather_fwd_bf16", {pool, src_tensor}, {gathered}, gather});
        graph.node({"reshape", {gathered}, {kv_tiles[i]}, {}});
      }
    }
    auto score4 = graph.tensor(syn_type_float, {kv, rows, group, kv_heads}, -1);
    auto probability4 = graph.tensor(syn_type_bf16, {kv, rows, group, kv_heads}, -1);
    const bool write_persistent_state = (t + 1 == chunk_tiles) && !finalize_last;
    auto next_m = write_persistent_state ? out_m : graph.tensor(syn_type_float, {stats_rows, group, kv_heads}, -1);
    auto next_l = write_persistent_state ? out_l : graph.tensor(syn_type_float, {stats_rows, group, kv_heads}, -1);
    auto next_o = write_persistent_state ? out_o : graph.tensor(syn_type_float, {d, rows, group, kv_heads}, -1);
    if (first_chunk && t == 0) {
      graph.node({"batch_gemm", {query, kv_tiles[0]}, {score4}, synGEMMParams{false, true}});
      graph.node({"gk_attn_tile_softmax_bf16_v1", {score4, positions, headers[t]},
                  {probability4, next_m, next_l}, scale});
      graph.node({"batch_gemm", {probability4, kv_tiles[1]}, {next_o}, synGEMMParams{false, false}});
    } else {
      auto tile_m = graph.tensor(syn_type_float, {stats_rows, group, kv_heads}, -1);
      auto tile_l = graph.tensor(syn_type_float, {stats_rows, group, kv_heads}, -1);
      auto tile_o4 = graph.tensor(syn_type_float, {d, rows, group, kv_heads}, -1);
      const int32_t prev_unscaled = (first_chunk && t == 1) ? 1 : 0;
      graph.node({"batch_gemm", {query, kv_tiles[0]}, {score4}, synGEMMParams{false, true}});
      graph.node({"gk_attn_tile_softmax_bf16_v1", {score4, positions, headers[t]},
                  {probability4, tile_m, tile_l}, scale});
      graph.node({"batch_gemm", {probability4, kv_tiles[1]}, {tile_o4}, synGEMMParams{false, false}});
      graph.node({"gk_attn_online_merge_f32_v1", {cur_m, cur_l, cur_o, tile_m, tile_l, tile_o4},
                  {next_m, next_l, next_o}, prev_unscaled});
    }
    cur_m = next_m;
    cur_l = next_l;
    cur_o = next_o;
  }
  if (finalize_last) {
    if (rows > 1) {
      auto final_o4 = graph.tensor(syn_type_bf16, {d, rows, group, kv_heads}, -1);
      auto final_o3 = graph.tensor(syn_type_bf16, {d, rows, heads}, -1);
      graph.node({"gk_attn_online_finalize_bf16_v1", {cur_l, cur_o}, {final_o4}, {}});
      graph.node({"reshape", {final_o4}, {final_o3}, {}});
      graph.node({"transpose", {final_o3}, {final_out}, perm3d});
    } else {
      graph.node({"gk_attn_online_finalize_bf16_v1", {cur_l, cur_o}, {final_out}, {}});
    }
  }
  graph.compile(ctx.cache, graph_key.key);
  auto [it, inserted] = ctx.cold_recipes.emplace(graph_key.key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT attention: duplicate chunk recipe");
  return *it->second;
}

ColdRecipe& cached_chunk_recipe(Context& ctx, const AttentionPlan& plan, const AttentionBlock& block, float scale,
                                uint32_t chunk_tiles, bool first_chunk, bool finalize_last, bool direct_kv) {
  struct TiledRecipeCache {
    uint64_t generation = 0;
    uint32_t rows = 0, kv = 0, scale = 0;
    int64_t heads = 0, kv_heads = 0, head_dim = 0;
    std::array<ColdRecipe*, kMaxFusedAttentionTiles * 8> entries{};
  };
  thread_local TiledRecipeCache cache;
  const auto scale_bits = std::bit_cast<uint32_t>(scale);
  if (cache.generation != ctx.generation || cache.rows != block.query_bucket || cache.kv != block.kv_bucket ||
      cache.scale != scale_bits || cache.heads != plan.query_heads || cache.kv_heads != plan.kv_heads ||
      cache.head_dim != plan.head_dim) {
    cache.generation = ctx.generation;
    cache.rows = block.query_bucket;
    cache.kv = block.kv_bucket;
    cache.scale = scale_bits;
    cache.heads = plan.query_heads;
    cache.kv_heads = plan.kv_heads;
    cache.head_dim = plan.head_dim;
    cache.entries.fill(nullptr);
  }
  const size_t idx = (static_cast<size_t>(chunk_tiles - 1) * 8) +
                     (direct_kv ? 4 : 0) + (first_chunk ? 2 : 0) + (finalize_last ? 1 : 0);
  if (auto* found = cache.entries[idx]) return *found;
  std::lock_guard lock(ctx.cold_mutex);
  auto& recipe = chunk_recipe(ctx, plan, block, scale, chunk_tiles, first_chunk, finalize_last, direct_kv);
  cache.entries[idx] = &recipe;
  return recipe;
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
  const size_t total_tiles = block.tiles.size();
  TORCH_CHECK(total_tiles > 1, "GaudiRT attention: multi-tile launch requires at least two KV tiles");
  const bool direct_kv = input.direct_key != 0;
  TORCH_CHECK(!direct_kv || input.direct_value != 0, "GaudiRT attention: missing direct value binding");
  const auto stream = ctx.stream();
  const uint64_t rows = block.query_bucket, heads = plan.query_heads, d = plan.head_dim;
  const uint64_t kv_row_bytes = static_cast<uint64_t>(plan.kv_heads) * d * sizeof(uint16_t);
  const uint64_t stats = align((rows + 63) / 64 * 64 * heads * sizeof(float));
  const uint64_t output_bytes = align(rows * heads * d * sizeof(float));
  const uint64_t stride = 2 * stats + output_bytes;
  if (total_tiles > kMaxFusedAttentionTiles) {
    TORCH_CHECK(plan.scratch_bytes >= 2 * stride, "GaudiRT attention: insufficient tile scratch");
  }

  const size_t num_chunks = (total_tiles + kMaxFusedAttentionTiles - 1) / kMaxFusedAttentionTiles;
  uint64_t workspace_size = 0;
  ColdRecipe* first_recipe = nullptr;
  ColdRecipe* middle_recipe = nullptr;
  ColdRecipe* last_recipe = nullptr;
  if (num_chunks == 1) {
    first_recipe = &cached_chunk_recipe(ctx, plan, block, scale,
                                        static_cast<uint32_t>(total_tiles), true, true, direct_kv);
    workspace_size = first_recipe->workspace;
  } else {
    first_recipe = &cached_chunk_recipe(ctx, plan, block, scale,
                                        static_cast<uint32_t>(kMaxFusedAttentionTiles), true, false, direct_kv);
    workspace_size = first_recipe->workspace;
    if (num_chunks > 2) {
      middle_recipe = &cached_chunk_recipe(ctx, plan, block, scale,
                                           static_cast<uint32_t>(kMaxFusedAttentionTiles), false, false, direct_kv);
      workspace_size = std::max(workspace_size, middle_recipe->workspace);
    }
    const uint32_t tail_tiles = static_cast<uint32_t>(
        total_tiles - (num_chunks - 1) * kMaxFusedAttentionTiles);
    last_recipe = &cached_chunk_recipe(ctx, plan, block, scale, tail_tiles, false, true, direct_kv);
    workspace_size = std::max(workspace_size, last_recipe->workspace);
  }

  std::unique_ptr<TemporaryDevice> workspace;
  if (workspace_size) workspace = std::make_unique<TemporaryDevice>(ctx, workspace_size);
  const auto work = workspace ? workspace->address : 0;
  const auto scratch = plan.address + plan.scratch_offset;
  const uint64_t value_indices_offset = static_cast<uint64_t>(block.kv_bucket) * plan.kv_heads * sizeof(int32_t);
  constexpr size_t kMaxChunkBindings = 3 + 3 * kMaxFusedAttentionTiles + 3 + 3;
  std::array<synLaunchTensorInfo, kMaxChunkBindings> bindings{};

  for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    const size_t tile_offset = chunk_idx * kMaxFusedAttentionTiles;
    const bool first_chunk = (chunk_idx == 0);
    const bool finalize_last = (chunk_idx + 1 == num_chunks);
    const uint32_t chunk_tiles = finalize_last
        ? static_cast<uint32_t>(total_tiles - tile_offset)
        : static_cast<uint32_t>(kMaxFusedAttentionTiles);
    ColdRecipe* recipe = first_chunk ? first_recipe : (finalize_last ? last_recipe : middle_recipe);
    const size_t count = recipe->tensors.size();
    std::copy(recipe->tensors.begin(), recipe->tensors.end(), bindings.begin());
    size_t b = 0;
    bindings[b++].pTensorAddress = input.query;
    if (!direct_kv) bindings[b++].pTensorAddress = input.pool;
    bindings[b++].pTensorAddress = plan.address + block.positions_offset;
    for (uint32_t t = 0; t < chunk_tiles; ++t) {
      const size_t global_t = tile_offset + t;
      const auto& tile = block.tiles[global_t];
      if (direct_kv) {
        const bool use_tail = (global_t + 1 == total_tiles) && (input.tail_key != 0);
        const uint64_t off = static_cast<uint64_t>(tile.first_key) * kv_row_bytes;
        bindings[b++].pTensorAddress = use_tail ? input.tail_key : (input.direct_key + off);
        bindings[b++].pTensorAddress = use_tail ? input.tail_value : (input.direct_value + off);
        bindings[b++].pTensorAddress = plan.address + tile.header_offset;
      } else {
        bindings[b++].pTensorAddress = plan.address + tile.indices_offset;
        bindings[b++].pTensorAddress = plan.address + tile.indices_offset + value_indices_offset;
        bindings[b++].pTensorAddress = plan.address + tile.header_offset;
      }
    }
    if (!first_chunk) {
      const uint64_t in_state = scratch + ((chunk_idx - 1) % 2) * stride;
      bindings[b++].pTensorAddress = in_state;
      bindings[b++].pTensorAddress = in_state + stats;
      bindings[b++].pTensorAddress = in_state + 2 * stats;
    }
    if (finalize_last) {
      bindings[b++].pTensorAddress = input.output;
    } else {
      const uint64_t out_state = scratch + (chunk_idx % 2) * stride;
      bindings[b++].pTensorAddress = out_state;
      bindings[b++].pTensorAddress = out_state + stats;
      bindings[b++].pTensorAddress = out_state + 2 * stats;
    }
    TORCH_CHECK(b == count, "GaudiRT attention: launch binding count mismatch");
    TORCH_CHECK(synLaunch(stream, bindings.data(), count, work, recipe->recipe, 0) == synSuccess,
                "GaudiRT attention: fused tile chunk launch failed");
  }
}

}  // namespace gaudirt::generic
