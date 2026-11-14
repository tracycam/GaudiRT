// SPDX-License-Identifier: Apache-2.0
#include "paged_slot_allocator.hpp"

#include "bridge_internal.hpp"
#include "graph_builder.hpp"
#include "temporaries.hpp"
#include "gaudirt/common/sha256.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace gaudirt::generic {
namespace {
using namespace runtime;

ColdRecipe& alloc_recipe(Context& ctx, uint32_t requests, uint32_t bucket, uint32_t pages_bucket, int32_t page_size) {
  const auto structure = std::string("paged_alloc_v2;") + std::to_string(requests) + ";" +
      std::to_string(bucket) + ";" + std::to_string(pages_bucket) + ";" + std::to_string(page_size);
  const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const auto key = "paged_slots_" + digest;
  if (auto found = ctx.cold_recipes.find(key); found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(4);
  SynapseGraphBuilder builder("g" + digest.substr(0, 12) + "_b" + std::to_string(bucket), *recipe);
  auto last = builder.tensor(syn_type_int32, {2, requests}, 0);
  auto pages = builder.tensor(syn_type_int32, {2, pages_bucket}, 1);
  auto metadata = builder.tensor(syn_type_int32, {4, requests}, 2);
  auto slots = builder.tensor(syn_type_int32, {2, bucket}, 3);
  builder.node(NodeSpec{"gk_paged_alloc_i32_v1", {last, pages, metadata}, {slots}, page_size});
  builder.compile(ctx.cache, key);
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT paged allocator: duplicate alloc recipe");
  return *it->second;
}

ColdRecipe& free_recipe(Context& ctx, uint32_t bitset_words, uint32_t bucket, int32_t page_size) {
  const auto structure = std::string("paged_free_v2;") + std::to_string(bitset_words) + ";" +
      std::to_string(bucket) + ";" + std::to_string(page_size);
  const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const auto key = "paged_slots_" + digest;
  if (auto found = ctx.cold_recipes.find(key); found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(5);
  SynapseGraphBuilder builder("g" + digest.substr(0, 12) + "_b" + std::to_string(bucket), *recipe);
  auto slots = builder.tensor(syn_type_int32, {2, bucket}, 0);
  auto metadata = builder.tensor(syn_type_int32, {3}, 1);
  auto seen_bitset = builder.tensor(syn_type_int32, {bitset_words}, 2);
  auto pages = builder.tensor(syn_type_int32, {2, bucket}, 3);
  auto count = builder.tensor(syn_type_int32, {1}, 4);
  builder.node(NodeSpec{"gk_paged_free_i32_v1", {slots, metadata, seen_bitset}, {pages, count}, page_size});
  builder.compile(ctx.cache, key);
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT paged allocator: duplicate free recipe");
  return *it->second;
}

void require_indices(const at::Tensor& tensor, const Context& ctx) {
  TORCH_CHECK(tensor.device().type() == c10::kPrivateUse1 && tensor.device().index() == ctx.device_index &&
              tensor.scalar_type() == at::kLong && tensor.dim() == 1 && tensor.is_contiguous(),
              "GaudiRT paged allocator: expected contiguous device int64 indices");
}

void upload(Context& ctx, uint64_t destination, const void* data, uint64_t bytes) {
  TemporaryHost host(ctx, bytes);
  std::memcpy(host.address, data, bytes);
  const auto stream = ctx.stream(), h2d = ctx.device.h2d_stream();
  auto& state = ctx.streams.state(active_stream_id);
  state.metadata_reusable.record_and_wait(stream, h2d);
  ctx.allocator.record_stream(destination, h2d);
  ctx.pinned.copy_h2d(destination, host.address, bytes, h2d, true);
  state.metadata_ready.record_and_wait(h2d, stream);
}

template <size_t N>
void launch(Context& ctx, ColdRecipe& recipe, const std::array<uint64_t, N>& addresses) {
  auto bindings = recipe.tensors;
  for (size_t i = 0; i < addresses.size(); ++i) bindings[i].pTensorAddress = addresses[i];
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(ctx, recipe.workspace);
  TORCH_CHECK(synLaunch(ctx.stream(), bindings.data(), bindings.size(), workspace ? workspace->address : 0,
                        recipe.recipe, 0) == synSuccess, "GaudiRT paged allocator: launch failed");
}
}  // namespace

std::tuple<at::Tensor, int64_t> allocate_paged_slots(
    const at::Tensor& last_slots, const at::Tensor& free_pages,
    const std::vector<int64_t>& prefixes, const std::vector<int64_t>& lengths, int64_t page_size) {
  auto ctx = current();
  require_indices(last_slots, *ctx);
  require_indices(free_pages, *ctx);
  TORCH_CHECK(page_size >= 1 && page_size <= INT32_MAX && !prefixes.empty() &&
              prefixes.size() <= (size_t{1} << 30) && prefixes.size() == lengths.size() &&
              last_slots.numel() == static_cast<int64_t>(prefixes.size()),
              "GaudiRT paged allocator: unsupported page size or request count");
  int64_t total = 0, num_pages = 0;
  for (size_t i = 0; i < prefixes.size(); ++i) {
    const auto prefix = prefixes[i], length = lengths[i];
    TORCH_CHECK(prefix >= 0 && length >= 0 && prefix + length <= INT32_MAX,
                "GaudiRT paged allocator: invalid request lengths");
    total += length;
    num_pages += (prefix + length + page_size - 1) / page_size - (prefix + page_size - 1) / page_size;
  }
  TORCH_CHECK(total > 0 && total <= (int64_t{1} << 30) && num_pages <= (int64_t{1} << 30) &&
              num_pages <= free_pages.numel(),
              "GaudiRT paged allocator: invalid token count or insufficient free pages");
  if (page_size == 1) {
    return {free_pages.narrow(0, 0, total).clone(), total};
  }
  const auto bucket = std::max<uint32_t>({
      std::bit_ceil(static_cast<uint32_t>(total)),
      std::bit_ceil(static_cast<uint32_t>(prefixes.size())),
      std::bit_ceil(static_cast<uint32_t>(std::max<int64_t>(1, num_pages)))});
  std::vector<int32_t> metadata(static_cast<size_t>(bucket) * 4, 0);
  int64_t offset = 0, page_offset = 0;
  for (size_t i = 0; i < prefixes.size(); ++i) {
    const auto prefix = prefixes[i], length = lengths[i];
    metadata[4 * i] = static_cast<int32_t>(prefix);
    metadata[4 * i + 1] = static_cast<int32_t>(length);
    metadata[4 * i + 2] = static_cast<int32_t>(offset);
    metadata[4 * i + 3] = static_cast<int32_t>(page_offset);
    offset += length;
    page_offset += (prefix + length + page_size - 1) / page_size - (prefix + page_size - 1) / page_size;
  }
  const auto stream = ctx->stream();
  TemporaryDevice last(*ctx, bucket * sizeof(int64_t)), pages(*ctx, bucket * sizeof(int64_t));
  TemporaryDevice meta(*ctx, metadata.size() * sizeof(int32_t));
  const auto last_address = address_of(last_slots, *ctx);
  ctx->allocator.record_stream(last_address, stream);
  PinnedHostPool::copy_d2d(last.address, last_address, prefixes.size() * sizeof(int64_t), stream, true);
  if (num_pages) {
    const auto page_address = address_of(free_pages, *ctx);
    ctx->allocator.record_stream(page_address, stream);
    PinnedHostPool::copy_d2d(pages.address, page_address, num_pages * sizeof(int64_t), stream, true);
  }
  upload(*ctx, meta.address, metadata.data(), metadata.size() * sizeof(int32_t));
  auto result = at::empty({static_cast<int64_t>(bucket)}, last_slots.options());
  const auto output = address_of(result, *ctx);
  ctx->allocator.record_stream(output, stream);
  {
    std::lock_guard lock(ctx->cold_mutex);
    auto& recipe = alloc_recipe(*ctx, bucket, bucket, bucket, static_cast<int32_t>(page_size));
    launch(*ctx, recipe, std::array<uint64_t, 4>{last.address, pages.address, meta.address, output});
  }
  return {result.narrow(0, 0, total), num_pages};
}

at::Tensor release_paged_slots(const at::Tensor& slots, int64_t page_size, int64_t max_pages) {
  auto ctx = current();
  require_indices(slots, *ctx);
  TORCH_CHECK(page_size >= 1 && page_size <= INT32_MAX && max_pages > 0 && max_pages <= INT32_MAX &&
              slots.numel() > 0 && slots.numel() <= (int64_t{1} << 30),
              "GaudiRT paged allocator: invalid release geometry");
  const auto bucket = std::bit_ceil(static_cast<uint32_t>(slots.numel()));
  const uint32_t bitset_words = static_cast<uint32_t>((max_pages / 32) + 2);
  const auto compute = ctx->stream();
  TemporaryDevice staged(*ctx, bucket * sizeof(int64_t)), meta(*ctx, 256), count(*ctx, 256);
  TemporaryDevice seen(*ctx, std::max<uint64_t>(256, static_cast<uint64_t>(bitset_words) * sizeof(int32_t)));
  TORCH_CHECK(synMemsetD32Async(seen.address, 0, bitset_words, compute) == synSuccess,
              "GaudiRT paged allocator: failed to initialize seen bitset");
  const auto input = address_of(slots, *ctx);
  ctx->allocator.record_stream(input, compute);
  PinnedHostPool::copy_d2d(staged.address, input, slots.numel() * sizeof(int64_t), compute, true);
  const std::array<int32_t, 3> metadata{static_cast<int32_t>(slots.numel()), static_cast<int32_t>(max_pages),
                                        static_cast<int32_t>(bitset_words)};
  upload(*ctx, meta.address, metadata.data(), sizeof(metadata));
  auto result = at::empty({static_cast<int64_t>(bucket)}, slots.options());
  const auto output = address_of(result, *ctx);
  ctx->allocator.record_stream(output, compute);
  {
    std::lock_guard lock(ctx->cold_mutex);
    auto& recipe = free_recipe(*ctx, bitset_words, bucket, static_cast<int32_t>(page_size));
    launch(*ctx, recipe, std::array<uint64_t, 5>{staged.address, meta.address, seen.address, output, count.address});
  }
  TemporaryHost host(*ctx, 256);
  const auto d2h = ctx->device.d2h_stream();
  ctx->streams.state(active_stream_id).d2h_reusable.record_and_wait(compute, d2h);
  ctx->allocator.record_stream(count.address, d2h);
  ctx->pinned.copy_d2h(host.address, count.address, sizeof(int32_t), d2h, false);
  int32_t released = 0;
  std::memcpy(&released, host.address, sizeof(released));
  TORCH_CHECK(released > 0 && static_cast<uint32_t>(released) <= bucket,
              "GaudiRT paged allocator: released slot is outside the KV pool");
  return result.narrow(0, 0, released);
}

}  // namespace gaudirt::generic
