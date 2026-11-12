// SPDX-License-Identifier: Apache-2.0
#include "paged_attention.hpp"

#include "attention_plan.hpp"
#include "bridge_internal.hpp"
#include "graph_builder.hpp"
#include "layout_copy.hpp"
#include "paged_kv_pool.hpp"
#include "paged_attention_tiles.hpp"
#include "temporaries.hpp"
#include "gaudirt/common/sha256.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <vector>

namespace gaudirt::generic {
using namespace runtime;
namespace {
enum FusedSlot {
  kQuery, kKey, kValue, kLocations, kStoreHeader, kOldPool, kPool,
  kKeyIndices, kValueIndices, kPositions, kTileHeader, kResult, kFusedSlots
};

ColdRecipe& fused_recipe(Context& ctx, const AttentionPlan& plan, const AttentionBlock& block, float scale,
                         bool write_kv) {
  std::ostringstream signature;
  signature << (write_kv ? "paged_fused_v2:" : "paged_fused_read_v2:")
            << block.query_bucket << ':' << block.kv_bucket << ':'
            << plan.query_heads << ':' << plan.kv_heads << ':' << plan.head_dim << ':'
            << std::bit_cast<uint32_t>(scale);
  const auto structure = signature.str();
  const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const auto prefix = "g" + digest.substr(0, 12) + "_b" + std::to_string(block.query_bucket);
  const auto key = "paged_fused_" + digest;
  if (auto found = ctx.cold_recipes.find(key); found != ctx.cold_recipes.end()) return *found->second;

  auto recipe = std::make_unique<ColdRecipe>(write_kv ? kFusedSlots : 7);
  SynapseGraphBuilder graph(prefix, *recipe);
  const uint64_t d = plan.head_dim, rows = block.query_bucket, kv = block.kv_bucket;
  const uint64_t heads = plan.query_heads, kv_heads = plan.kv_heads, group = heads / kv_heads;
  const uint64_t kv_width = kv_heads * d;
  int slot = 0;
  const auto query = graph.tensor(syn_type_bf16, {d, rows, group, kv_heads}, slot++);
  synTensorPermutation permutation{};
  permutation.dims = 4;
  permutation.permutation[0] = 0; permutation.permutation[1] = 2;
  permutation.permutation[2] = 3; permutation.permutation[3] = 1;
  if (rows > 1) graph.permutation(query, permutation);
  synTensor pool = nullptr;
  if (write_kv) {
    const auto key_input = graph.tensor(syn_type_bf16, {kv_width, rows}, slot++);
    const auto value_input = graph.tensor(syn_type_bf16, {kv_width, rows}, slot++);
    const auto locations = graph.tensor(syn_type_int32, {rows}, slot++);
    const auto store_header = graph.tensor(syn_type_int32, {3}, slot++);
    const auto old_pool_slot = slot++;
    const auto pool_input = graph.tensor(syn_type_bf16, {d, 2 * kPagedKvDescriptorSlots * kv_heads}, old_pool_slot);
    pool = graph.tensor(syn_type_bf16, {d, 2 * kPagedKvDescriptorSlots * kv_heads}, slot++, old_pool_slot);
    graph.node({"gk_kv_store_head_major_bf16_v1", {key_input, value_input, locations, store_header, pool_input},
                {pool}, {}});
  } else {
    pool = graph.tensor(syn_type_bf16, {d, 2 * kPagedKvDescriptorSlots * kv_heads}, slot++);
  }
  const auto key_indices_slot = slot++;
  const auto key_indices = graph.tensor(syn_type_int32, {kv * kv_heads}, key_indices_slot);
  const auto value_indices = graph.tensor(syn_type_int32, {kv * kv_heads}, slot++,
                                          key_indices_slot, kv * kv_heads * sizeof(int32_t));
  const auto positions_slot = slot++;
  const auto positions = graph.tensor(syn_type_int32, {rows}, positions_slot);
  const auto header_offset = (rows * sizeof(int32_t) + 255) & ~uint64_t{255};
  const auto header = graph.tensor(syn_type_int32, {3}, slot++, positions_slot, header_offset);
  const auto result = rows == 1 ? graph.tensor(syn_type_bf16, {d, heads, rows}, slot++) :
                                  graph.tensor(syn_type_bf16, {d, rows, group, kv_heads}, slot++);
  if (rows > 1) graph.permutation(result, permutation);
  ns_GatherKernel::ParamsV2 gather{};
  gather.axis = 1;
  std::array<synTensor, 2> tiles{};
  for (unsigned i = 0; i < 2; ++i) {
    const auto gathered = graph.tensor(syn_type_bf16, {d, kv * kv_heads}, -1);
    tiles[i] = graph.tensor(syn_type_bf16, {d, kv, 1, kv_heads}, -1);
    graph.node({"gather_fwd_bf16", {pool, i ? value_indices : key_indices}, {gathered}, gather});
    graph.node({"reshape", {gathered}, {tiles[i]}, {}});
  }
  const auto score4 = graph.tensor(syn_type_float, {kv, rows, group, kv_heads}, -1);
  const auto probability4 = graph.tensor(syn_type_bf16, {kv, rows, group, kv_heads}, -1);
  graph.node({"batch_gemm", {query, tiles[0]}, {score4}, synGEMMParams{false, true}});
  if (kv <= 512) {
    graph.node({"gk_attn_direct_softmax_bf16_v1", {score4, positions, header}, {probability4}, scale});
  } else {
    const auto score3 = graph.tensor(syn_type_float, {kv, rows, heads}, -1);
    const auto probability3 = graph.tensor(syn_type_bf16, {kv, rows, heads}, -1);
    const auto exponent = graph.tensor(syn_type_float, {kv, rows, heads}, -1);
    const auto sum = graph.tensor(syn_type_float, {64, heads}, -1);
    graph.node({"reshape", {score4}, {score3}, {}});
    graph.node({"gk_attn_decode_exp_f32_v1", {score3, positions, header}, {exponent, sum}, scale});
    graph.node({"gk_attn_decode_normalize_bf16_v1", {exponent, sum}, {probability3}, {}});
    graph.node({"reshape", {probability3}, {probability4}, {}});
  }
  if (rows == 1) {
    const auto output4 = graph.tensor(syn_type_bf16, {d, rows, group, kv_heads}, -1);
    graph.node({"batch_gemm", {probability4, tiles[1]}, {output4}, synGEMMParams{false, false}});
    graph.node({"reshape", {output4}, {result}, {}});
  } else {
    graph.node({"batch_gemm", {probability4, tiles[1]}, {result}, synGEMMParams{false, false}});
  }
  graph.compile(ctx.cache, key);
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT attention: duplicate fused recipe");
  return *it->second;
}

ColdRecipe& cached_fused_recipe(Context& ctx, const AttentionPlan& plan, const AttentionBlock& block, float scale,
                                bool write_kv) {
  struct Cache {
    uint64_t generation = 0;
    uint32_t rows = 0, kv = 0, scale = 0;
    int64_t heads = 0, kv_heads = 0, head_dim = 0;
    bool write_kv = false;
    ColdRecipe* recipe = nullptr;
  };
  thread_local Cache cache;
  const auto scale_bits = std::bit_cast<uint32_t>(scale);
  if (cache.recipe && cache.generation == ctx.generation && cache.rows == block.query_bucket &&
      cache.kv == block.kv_bucket && cache.scale == scale_bits && cache.heads == plan.query_heads &&
      cache.kv_heads == plan.kv_heads && cache.head_dim == plan.head_dim && cache.write_kv == write_kv)
    return *cache.recipe;
  std::lock_guard lock(ctx.cold_mutex);
  auto& recipe = fused_recipe(ctx, plan, block, scale, write_kv);
  cache = {ctx.generation, block.query_bucket, block.kv_bucket, scale_bits,
           plan.query_heads, plan.kv_heads, plan.head_dim, write_kv, &recipe};
  return recipe;
}

void record_if_needed(Context& ctx, const at::Tensor& tensor, synStreamHandle stream) {
  const auto& allocation = validated_program_input(tensor, ctx);
  if (allocation.stream != stream) {
    const auto address = reinterpret_cast<uint64_t>(tensor.data_ptr());
    if (allocation.kind == AllocationKind::graph) ctx.graph_pool->record_stream(address, stream);
    else ctx.allocator.record_stream(address, stream);
  }
}
}  // namespace

at::Tensor paged_attn_validated(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& key_pool, const at::Tensor& value_pool, const at::Tensor& locations,
    const at::Tensor& requests, const at::Tensor& positions, const at::Tensor& plan_tensor, const at::Tensor& header,
    int64_t head_dim, double scale) {
  return paged_attn_internal(query, key, value, key_pool, value_pool, locations, requests, positions,
                             plan_tensor, header, head_dim, scale, false, false, false);
}

at::Tensor paged_attn_internal(const at::Tensor& query, const at::Tensor& key, const at::Tensor& value,
    const at::Tensor& key_pool, const at::Tensor& value_pool, const at::Tensor& locations,
    const at::Tensor&, const at::Tensor&, const at::Tensor& plan_tensor, const at::Tensor&,
    int64_t head_dim, double scale, bool query_trusted, bool key_trusted, bool value_trusted) {
  auto ctx = current();
  const auto stream = ctx->stream();
  const auto& plan = attention_plan_data(plan_tensor, *ctx);
  const float scale_f = static_cast<float>(scale);
  TORCH_CHECK(plan.stream == stream && std::isfinite(scale) && std::isfinite(scale_f) && scale_f > 0 &&
              head_dim == plan.head_dim, "GaudiRT attention: invalid scale, stream, or head dimension");
  for (const auto& input : {query, key, value})
    TORCH_CHECK(input.device().type() == c10::kPrivateUse1 && input.device().index() == ctx->device_index &&
                input.scalar_type() == at::kBFloat16 && input.dim() == 2 && input.size(0) == plan.rows,
                "GaudiRT attention: Q/K/V must be BF16 device matrices with matching rows");
  TORCH_CHECK(query.size(1) == plan.query_heads * head_dim && key.size(1) == plan.kv_heads * head_dim &&
              value.size(1) == key.size(1) && locations.dim() == 1 && locations.numel() == plan.rows &&
              locations.scalar_type() == at::kInt,
              "GaudiRT attention: Q/K/V or slot dimensions do not match metadata");
  for (const auto& pool : {key_pool, value_pool})
    TORCH_CHECK(pool.device().type() == c10::kPrivateUse1 && pool.device().index() == ctx->device_index &&
                pool.scalar_type() == at::kBFloat16 && pool.dim() == 3 &&
                pool.size(0) == plan.slots && pool.size(1) == plan.kv_heads && pool.size(2) == head_dim &&
                pool.stride(0) == head_dim && pool.stride(1) == plan.slots * head_dim && pool.stride(2) == 1,
                "GaudiRT attention: head-major combined KV pool is required");
  const auto pool_address = reinterpret_cast<uint64_t>(key_pool.data_ptr());
  const auto value_address = reinterpret_cast<uint64_t>(value_pool.data_ptr());
  const uint64_t value_offset = static_cast<uint64_t>(plan.slots) * plan.kv_heads * head_dim * sizeof(uint16_t);
  const uint64_t descriptor_bytes = 2 * kPagedKvDescriptorSlots * plan.kv_heads * head_dim * sizeof(uint16_t);
  const auto pool_storage = key_pool.storage();
  const auto storage_base = reinterpret_cast<uint64_t>(pool_storage.data());
  TORCH_CHECK(pool_storage.unsafeGetStorageImpl() == value_pool.storage().unsafeGetStorageImpl() &&
              value_address == pool_address + value_offset && pool_address >= storage_base &&
              pool_address % BlockCachingAllocator::kAlignment == 0 &&
              pool_address - storage_base <= pool_storage.nbytes() &&
              descriptor_bytes <= pool_storage.nbytes() - (pool_address - storage_base),
              "GaudiRT attention: KV pool descriptor is not backed by combined storage");
  if (!query_trusted) record_if_needed(*ctx, query, stream);
  if (!key_trusted) record_if_needed(*ctx, key, stream);
  if (!value_trusted) record_if_needed(*ctx, value, stream);
  for (const auto& input : {locations, key_pool}) record_if_needed(*ctx, input, stream);
  const auto output_bucket = std::bit_ceil(static_cast<uint32_t>(plan.rows));
  auto output = bucket_output(query, at::kBFloat16, output_bucket, plan.rows, query.size(1));
  auto input_address = [&](const at::Tensor& input, int64_t first_row, int64_t valid_rows, uint32_t bucket,
                           std::vector<std::pair<uint64_t, uint64_t>>& regions,
                           std::vector<std::unique_ptr<TemporaryDevice>>& temporary) {
    const uint64_t item = input.element_size();
    const uint64_t width = input.dim() == 1 ? 1 : input.size(1);
    const uint64_t source = reinterpret_cast<uint64_t>(input.data_ptr()) +
                            static_cast<uint64_t>(first_row) * input.stride(0) * item;
    const uint64_t bytes = static_cast<uint64_t>(bucket) * width * item;
    const uint64_t base = reinterpret_cast<uint64_t>(input.storage().data());
    const uint64_t storage = input.storage().nbytes();
    bool direct = input.is_contiguous() && source % BlockCachingAllocator::kAlignment == 0 &&
                  source >= base && source - base <= storage && bytes <= storage - (source - base);
    for (const auto& [other, length] : regions)
      if (source < other + length && other < source + bytes) direct = false;
    uint64_t result = source;
    if (!direct) {
      auto staged = std::make_unique<TemporaryDevice>(*ctx, bytes);
      result = staged->address;
      const auto slice = input.narrow(0, first_row, valid_rows);
      if (!try_pack_strided_2d(*ctx, result, bytes, slice, stream)) {
        const Layout layout(slice);
        dma_copy(result, nullptr, source, &layout, 0, slice.numel(), item, stream, DRAM_TO_DRAM);
      }
      temporary.push_back(std::move(staged));
    }
    regions.emplace_back(result, bytes);
    return result;
  };
  // Multiple query blocks share one state update. The metadata contract makes
  // their target slots unique, and causal masking hides future rows.
  const bool separate_store = plan.blocks.size() != 1 || plan.blocks.front().tiles.size() != 1;
  std::vector<std::unique_ptr<TemporaryDevice>> layer_temporary;
  uint64_t layer_k = 0, layer_v = 0;
  const uint64_t kv_row_bytes = static_cast<uint64_t>(plan.kv_heads) * head_dim * sizeof(uint16_t);
  const uint64_t layer_kv_bytes = static_cast<uint64_t>(output_bucket) * kv_row_bytes;
  if (separate_store) {
    std::vector<std::pair<uint64_t, uint64_t>> regions{{pool_address, descriptor_bytes},
                                                       {plan.address, plan.bytes}};
    layer_k = input_address(key, 0, plan.rows, output_bucket, regions, layer_temporary);
    layer_v = input_address(value, 0, plan.rows, output_bucket, regions, layer_temporary);
    const auto loc = input_address(locations, 0, plan.rows, output_bucket, regions, layer_temporary);
    launch_layer_kv_store(*ctx, plan, layer_k, layer_v, loc, pool_address);
  }
  for (const auto& block : plan.blocks) {
    const auto& tile = block.tiles.front();
    const bool direct_kv = separate_store && block.tiles.size() > 1 && block.kv_source_row >= 0;
    std::vector<std::unique_ptr<TemporaryDevice>> temporary;
    std::vector<std::pair<uint64_t, uint64_t>> regions{{pool_address, descriptor_bytes},
                                                       {plan.address, plan.bytes}};
    if (direct_kv) {
      regions.emplace_back(layer_k, layer_kv_bytes);
      regions.emplace_back(layer_v, layer_kv_bytes);
    }
    const auto block_input = [&](const at::Tensor& input) {
      return input_address(input, block.first_row, block.valid_rows, block.query_bucket, regions, temporary);
    };
    const auto q_address = block_input(query);
    const auto k_address = separate_store ? 0 : block_input(key);
    const auto v_address = separate_store ? 0 : block_input(value);
    const auto locations_address = separate_store ? 0 : block_input(locations);
    const uint64_t out = reinterpret_cast<uint64_t>(output.data_ptr()) +
                         static_cast<uint64_t>(block.first_row) * query.size(1) * sizeof(uint16_t);
    const uint64_t out_base = reinterpret_cast<uint64_t>(output.storage().data());
    const uint64_t out_bytes = static_cast<uint64_t>(block.query_bucket) * query.size(1) * sizeof(uint16_t);
    const bool direct_out = out % BlockCachingAllocator::kAlignment == 0 && out >= out_base &&
        out - out_base <= output.storage().nbytes() && out_bytes <= output.storage().nbytes() - (out - out_base);
    uint64_t out_binding = out;
    if (!direct_out) {
      auto staged = std::make_unique<TemporaryDevice>(*ctx, out_bytes);
      out_binding = staged->address;
      temporary.push_back(std::move(staged));
    }
    if (block.tiles.size() > 1) {
      TiledAttentionInputs tiled_input{q_address, pool_address, out_binding};
      if (direct_kv) {
        const uint64_t seq_offset = static_cast<uint64_t>(block.kv_source_row) * kv_row_bytes;
        tiled_input.direct_key = layer_k + seq_offset;
        tiled_input.direct_value = layer_v + seq_offset;
        const auto& last_tile = block.tiles.back();
        if (last_tile.valid_keys < static_cast<int32_t>(block.kv_bucket)) {
          const uint64_t tile_bytes = static_cast<uint64_t>(block.kv_bucket) * kv_row_bytes;
          const uint64_t valid_bytes = static_cast<uint64_t>(last_tile.valid_keys) * kv_row_bytes;
          const uint64_t pad_bytes = tile_bytes - valid_bytes;
          const uint64_t tail_src_off = static_cast<uint64_t>(last_tile.first_key) * kv_row_bytes;
          auto tail_kv = std::make_unique<TemporaryDevice>(*ctx, 2 * tile_bytes);
          tiled_input.tail_key = tail_kv->address;
          tiled_input.tail_value = tail_kv->address + tile_bytes;
          TORCH_CHECK(synMemCopyAsync(stream, tiled_input.direct_key + tail_src_off, valid_bytes,
                                      tiled_input.tail_key, DRAM_TO_DRAM) == synSuccess &&
                      synMemCopyAsync(stream, tiled_input.direct_value + tail_src_off, valid_bytes,
                                      tiled_input.tail_value, DRAM_TO_DRAM) == synSuccess &&
                      synMemsetD32Async(tiled_input.tail_key + valid_bytes, 0,
                                        pad_bytes / sizeof(uint32_t), stream) == synSuccess &&
                      synMemsetD32Async(tiled_input.tail_value + valid_bytes, 0,
                                        pad_bytes / sizeof(uint32_t), stream) == synSuccess,
                      "GaudiRT attention: partial KV tile staging failed");
          temporary.push_back(std::move(tail_kv));
        }
      }
      launch_tiled_attention(*ctx, plan, block, tiled_input, scale_f);
    } else {
      auto* recipe = &cached_fused_recipe(*ctx, plan, block, scale_f, !separate_store);
      auto bindings = recipe->tensors;
      const uint64_t value_indices_offset = static_cast<uint64_t>(block.kv_bucket) * plan.kv_heads * sizeof(int32_t);
      std::vector<uint64_t> addresses{q_address};
      if (!separate_store) {
        addresses.insert(addresses.end(), {k_address, v_address, locations_address,
                         plan.address + block.store_header_offset, pool_address});
      }
      addresses.insert(addresses.end(), {pool_address, plan.address + tile.indices_offset,
          plan.address + tile.indices_offset + value_indices_offset, plan.address + block.positions_offset,
          plan.address + tile.header_offset, out_binding});
      for (size_t i = 0; i < addresses.size(); ++i) bindings[i].pTensorAddress = addresses[i];
      std::unique_ptr<TemporaryDevice> workspace;
      if (recipe->workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe->workspace);
      TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(), workspace ? workspace->address : 0,
                            recipe->recipe, 0) == synSuccess, "GaudiRT attention: fused launch failed");
    }
    if (!direct_out)
      TORCH_CHECK(synMemCopyAsync(stream, out_binding,
          static_cast<uint64_t>(block.valid_rows) * query.size(1) * sizeof(uint16_t), out, DRAM_TO_DRAM) == synSuccess,
          "GaudiRT attention: output copy failed");
  }
  return output;
}

}  // namespace gaudirt::generic
