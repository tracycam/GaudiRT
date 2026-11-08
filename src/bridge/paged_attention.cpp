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

ColdRecipe& fused_recipe(Context& ctx, const AttentionPlan& plan, const AttentionBlock& block, float scale) {
  std::ostringstream signature;
  signature << "paged_fused_v1:" << block.query_bucket << ':' << block.kv_bucket << ':'
            << plan.query_heads << ':' << plan.kv_heads << ':' << plan.head_dim << ':'
            << std::bit_cast<uint32_t>(scale);
  const auto structure = signature.str();
  const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const auto prefix = "g" + digest.substr(0, 12) + "_b" + std::to_string(block.query_bucket);
  const auto key = "paged_fused_" + digest;
  if (auto found = ctx.cold_recipes.find(key); found != ctx.cold_recipes.end()) return *found->second;

  auto recipe = std::make_unique<ColdRecipe>(kFusedSlots);
  SynapseGraphBuilder graph(prefix, *recipe);
  const uint64_t d = plan.head_dim, rows = block.query_bucket, kv = block.kv_bucket;
  const uint64_t heads = plan.query_heads, kv_heads = plan.kv_heads, group = heads / kv_heads;
  const uint64_t kv_width = kv_heads * d;
  const auto query = graph.tensor(syn_type_bf16, {d, rows, group, kv_heads}, kQuery);
  synTensorPermutation permutation{};
  permutation.dims = 4;
  permutation.permutation[0] = 0; permutation.permutation[1] = 2;
  permutation.permutation[2] = 3; permutation.permutation[3] = 1;
  if (rows > 1) graph.permutation(query, permutation);
  const auto key_input = graph.tensor(syn_type_bf16, {kv_width, rows}, kKey);
  const auto value_input = graph.tensor(syn_type_bf16, {kv_width, rows}, kValue);
  const auto locations = graph.tensor(syn_type_int32, {rows}, kLocations);
  const auto store_header = graph.tensor(syn_type_int32, {3}, kStoreHeader);
  const auto pool_input = graph.tensor(syn_type_bf16, {d, 2 * kPagedKvDescriptorSlots * kv_heads}, kOldPool);
  const auto pool = graph.tensor(syn_type_bf16, {d, 2 * kPagedKvDescriptorSlots * kv_heads}, kPool, kOldPool);
  const auto key_indices = graph.tensor(syn_type_int32, {kv * kv_heads}, kKeyIndices);
  const auto value_indices = graph.tensor(syn_type_int32, {kv * kv_heads}, kValueIndices,
                                          kKeyIndices, kv * kv_heads * sizeof(int32_t));
  const auto positions = graph.tensor(syn_type_int32, {rows}, kPositions);
  const auto header_offset = (rows * sizeof(int32_t) + 255) & ~uint64_t{255};
  const auto header = graph.tensor(syn_type_int32, {3}, kTileHeader, kPositions, header_offset);
  const auto result = rows == 1 ? graph.tensor(syn_type_bf16, {d, heads, rows}, kResult) :
                                  graph.tensor(syn_type_bf16, {d, rows, group, kv_heads}, kResult);
  if (rows > 1) graph.permutation(result, permutation);
  graph.node({"gk_kv_store_head_major_bf16_v1", {key_input, value_input, locations, store_header, pool_input},
              {pool}, {}});
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
  const auto score3 = graph.tensor(syn_type_float, {kv, rows, heads}, -1);
  const auto probability3 = graph.tensor(syn_type_bf16, {kv, rows, heads}, -1);
  const auto probability4 = graph.tensor(syn_type_bf16, {kv, rows, group, kv_heads}, -1);
  graph.node({"batch_gemm", {query, tiles[0]}, {score4}, synGEMMParams{false, true}});
  graph.node({"reshape", {score4}, {score3}, {}});
  if (kv <= 512) {
    graph.node({"gk_attn_direct_softmax_bf16_v1", {score3, positions, header}, {probability3}, scale});
  } else {
    const auto exponent = graph.tensor(syn_type_float, {kv, rows, heads}, -1);
    const auto sum = graph.tensor(syn_type_float, {64, heads}, -1);
    graph.node({"gk_attn_decode_exp_f32_v1", {score3, positions, header}, {exponent, sum}, scale});
    graph.node({"gk_attn_decode_normalize_bf16_v1", {exponent, sum}, {probability3}, {}});
  }
  graph.node({"reshape", {probability3}, {probability4}, {}});
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
    const at::Tensor&, const at::Tensor&, const at::Tensor& plan_tensor, const at::Tensor&,
    int64_t head_dim, double scale) {
  auto ctx = current();
  const auto stream = ctx->stream();
  const auto& plan = attention_plan_data(plan_tensor, *ctx);
  const float scale_f = static_cast<float>(scale);
  TORCH_CHECK(plan.stream == stream && std::isfinite(scale) && std::isfinite(scale_f) && scale_f > 0 &&
              head_dim == plan.head_dim, "GaudiRT attention: invalid scale, stream, or head dimension");
  for (const auto& input : {query, key, value})
    TORCH_CHECK(input.device().type() == c10::kPrivateUse1 && input.device().index() == 0 &&
                input.scalar_type() == at::kBFloat16 && input.dim() == 2 && input.size(0) == plan.rows,
                "GaudiRT attention: Q/K/V must be BF16 device matrices with matching rows");
  TORCH_CHECK(query.size(1) == plan.query_heads * head_dim && key.size(1) == plan.kv_heads * head_dim &&
              value.size(1) == key.size(1) && locations.dim() == 1 && locations.numel() == plan.rows &&
              locations.scalar_type() == at::kInt,
              "GaudiRT attention: Q/K/V or slot dimensions do not match metadata");
  for (const auto& pool : {key_pool, value_pool})
    TORCH_CHECK(pool.device().type() == c10::kPrivateUse1 && pool.device().index() == 0 &&
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
  for (const auto& input : {query, key, value, locations, key_pool}) record_if_needed(*ctx, input, stream);
  const auto output_bucket = std::bit_ceil(static_cast<uint32_t>(plan.rows));
  auto output = bucket_output(query, at::kBFloat16, output_bucket, plan.rows, query.size(1));
  for (const auto& block : plan.blocks) {
    const auto& tile = block.tiles.front();
    std::vector<std::unique_ptr<TemporaryDevice>> temporary;
    std::vector<std::pair<uint64_t, uint64_t>> regions{{pool_address, descriptor_bytes},
                                                       {plan.address, plan.bytes}};
    auto input_address = [&](const at::Tensor& input) {
      const uint64_t item = input.element_size();
      const uint64_t width = input.dim() == 1 ? 1 : input.size(1);
      const uint64_t source = reinterpret_cast<uint64_t>(input.data_ptr()) +
                              static_cast<uint64_t>(block.first_row) * input.stride(0) * item;
      const uint64_t bytes = static_cast<uint64_t>(block.query_bucket) * width * item;
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
        const auto slice = input.narrow(0, block.first_row, block.valid_rows);
        const Layout layout(slice);
        dma_copy(result, nullptr, source, &layout, 0, slice.numel(), item, stream, DRAM_TO_DRAM);
        temporary.push_back(std::move(staged));
      }
      regions.emplace_back(result, bytes);
      return result;
    };
    const auto q_address = input_address(query);
    const auto k_address = input_address(key);
    const auto v_address = input_address(value);
    const auto locations_address = input_address(locations);
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
      launch_tiled_attention(*ctx, plan, block,
                             {q_address, k_address, v_address, locations_address, pool_address, out_binding},
                             scale_f);
    } else {
      ColdRecipe* recipe;
      {
        std::lock_guard lock(ctx->cold_mutex);
        recipe = &fused_recipe(*ctx, plan, block, scale_f);
      }
      auto bindings = recipe->tensors;
      const uint64_t value_indices_offset = static_cast<uint64_t>(block.kv_bucket) * plan.kv_heads * sizeof(int32_t);
      const std::array<uint64_t, kFusedSlots> addresses{
          q_address, k_address, v_address, locations_address, plan.address + block.store_header_offset,
          pool_address, pool_address, plan.address + tile.indices_offset,
          plan.address + tile.indices_offset + value_indices_offset, plan.address + block.positions_offset,
          plan.address + tile.header_offset, out_binding};
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
