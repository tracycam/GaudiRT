// SPDX-License-Identifier: Apache-2.0
#include "attention_plan.hpp"

#include "graph_builder.hpp"
#include "layout_copy.hpp"
#include "paged_kv_pool.hpp"
#include "paged_metadata.hpp"
#include "temporaries.hpp"
#include "gaudirt/common/sha256.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

namespace gaudirt::generic {
using namespace runtime;
namespace {
constexpr uint64_t kAlignment = BlockCachingAllocator::kAlignment;
constexpr uint32_t kTileKvBucket = 512;
// Keep this limit consistent with the 4D generic attention provider's Glue (kMaxScoreElements4d).
constexpr uint64_t kMaxTileScoreElements = 16ULL * 1024 * 1024;
uint64_t align(uint64_t size) { return (size + kAlignment - 1) & ~(kAlignment - 1); }

void release_attention_plan(void* pointer) {
  std::unique_ptr<AttentionPlan> plan(static_cast<AttentionPlan*>(pointer));
  if (auto ctx = plan->owner.lock()) ctx->attention_staging().release(plan->staging_slot, plan->stream);
}

ColdRecipe& indices_recipe(Context& ctx, uint32_t rows, uint32_t tasks) {
  const std::string structure = "attention_indices_v1:" + std::to_string(rows) + ":" + std::to_string(tasks);
  const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const auto key = "attention_indices_" + digest;
  if (auto found = ctx.cold_recipes.find(key); found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(5);
  const auto prefix = "g" + digest.substr(0, 12) + "_b" + std::to_string(rows);
  SynapseGraphBuilder graph(prefix, *recipe);
  auto map = graph.tensor(syn_type_int32, {256 * kPagedKvDescriptorSlots}, 0, -1, 0, true);
  auto requests = graph.tensor(syn_type_int32, {rows}, 1);
  auto work = graph.tensor(syn_type_int32, {4, tasks}, 2);
  auto context = graph.tensor(syn_type_int32, {1}, 3);
  auto output = graph.tensor(syn_type_int32, {64, tasks}, 4);
  graph.node({"gk_attn_tile_indices_i32_v1", {map, requests, work, context}, {output}, {}});
  graph.compile(ctx.cache, key);
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT attention: duplicate index recipe");
  return *it->second;
}
}  // namespace

AttentionStagingPool::AttentionStagingPool(Context& context)
    : context_(context), events_(context.device.device_id()) {}

AttentionStagingPool::~AttentionStagingPool() {
  release_all();
}

void AttentionStagingPool::prepare_stream(synStreamHandle stream) {
  std::lock_guard lock(mutex_);
  events_.prepare(stream);
}

void AttentionStagingPool::release_all() {
  for (auto& slot : slots_) {
    if (slot.recorded)
      TORCH_CHECK(events_.synchronize(slot.completed) == synSuccess,
                  "GaudiRT attention: staging completion wait failed");
    if (slot.host) context_.pinned.free(slot.host);
    if (slot.address) context_.allocator.free(slot.address);
  }
  slots_.clear();
}

bool AttentionStagingPool::trim() {
  std::lock_guard lock(mutex_);
  for (const auto& slot : slots_) if (slot.in_use) return false;
  release_all();
  return true;
}

AttentionStagingPool::Lease AttentionStagingPool::acquire(uint64_t device_bytes, uint64_t host_bytes) {
  std::lock_guard lock(mutex_);
  for (size_t i = 0; i < slots_.size(); ++i) {
    auto& slot = slots_[i];
    if (slot.in_use || slot.device_bytes < device_bytes || slot.host_bytes < host_bytes) continue;
    if (slot.recorded) {
      const auto status = events_.query(slot.completed);
      TORCH_CHECK(status == synSuccess || status == synBusy, "GaudiRT attention: staging event query failed");
      if (status == synBusy) continue;
      slot.recorded = false;
    }
    slot.in_use = true;
    return {i, slot.address, slot.host};
  }
  Slot slot;
  slot.device_bytes = device_bytes;
  slot.host_bytes = host_bytes;
  slot.in_use = true;
  try {
    slot.address = context_.allocator.allocate(device_bytes, context_.device.h2d_stream());
    slot.host = context_.pinned.allocate(host_bytes);
    slots_.push_back(slot);
  } catch (...) {
    if (slot.host) context_.pinned.free(slot.host);
    if (slot.address) context_.allocator.free(slot.address);
    throw;
  }
  return {slots_.size() - 1, slot.address, slot.host};
}

void AttentionStagingPool::release(size_t index, synStreamHandle compute) {
  std::lock_guard lock(mutex_);
  TORCH_CHECK(index < slots_.size() && slots_[index].in_use, "GaudiRT attention: invalid staging lease");
  auto& slot = slots_[index];
  const auto completion = events_.prepare(compute);
  const auto status = events_.query(completion);
  TORCH_CHECK(status == synSuccess || status == synBusy, "GaudiRT attention: staging event query failed");
  slot.completed = events_.record(compute);
  slot.recorded = true;
  slot.in_use = false;
}

const AttentionPlan& attention_plan_data(const at::Tensor& tensor, const Context& ctx) {
  TORCH_CHECK(tensor.device().type() == c10::kPrivateUse1 && tensor.device().index() == ctx.device_index,
              "GaudiRT attention: expected a device metadata plan");
  const auto* plan = tensor.storage().data_ptr().cast_context<AttentionPlan>(release_attention_plan);
  TORCH_CHECK(plan && plan->owner.lock().get() == &ctx &&
              reinterpret_cast<uint64_t>(tensor.data_ptr()) == plan->address,
              "GaudiRT attention: invalid or expired metadata plan");
  return *plan;
}

at::Tensor PagedMetadata::attention_plan(int64_t query_heads, int64_t kv_heads, int64_t head_dim) {
  auto ctx = current();
  if (attention_plan_.defined()) {
    const auto& plan = attention_plan_data(attention_plan_, *ctx);
    TORCH_CHECK(plan.query_heads == query_heads && plan.kv_heads == kv_heads && plan.head_dim == head_dim,
                "GaudiRT attention: metadata already bound to different head geometry");
    return attention_plan_;
  }
  TORCH_CHECK(query_heads > 0 && query_heads <= 256 && kv_heads > 0 && kv_heads <= 256 &&
              query_heads % kv_heads == 0 && (head_dim == 64 || head_dim == 128) &&
              (kv_heads * head_dim) % 128 == 0 && slots_ > 0 &&
              static_cast<uint64_t>(slots_) <= kPagedKvDescriptorSlots && request_count_ <= 256 &&
              context_ <= static_cast<int64_t>(kPagedKvDescriptorSlots),
              "GaudiRT attention: unsupported head or request-map geometry");

  auto plan = std::make_unique<AttentionPlan>();
  plan->owner = ctx;
  plan->rows = rows_;
  plan->slots = slots_;
  plan->query_heads = query_heads;
  plan->kv_heads = kv_heads;
  plan->head_dim = head_dim;
  std::vector<std::array<int32_t, 4>> tasks;
  for (const auto& sequence : sequences_) {
    for (int32_t done = 0; done < sequence.length;) {
      const int32_t remaining = sequence.length - done;
      const int32_t end = sequence.prefix + done + remaining;
      int32_t count = std::min<int32_t>(remaining, 256);
      uint32_t q_bucket = std::bit_ceil(static_cast<uint32_t>(count));
      const uint32_t direct_kv = (remaining == 1 && end > 512 && end <= 8192)
          ? static_cast<uint32_t>((end + 511) & ~511)
          : std::max<uint32_t>(128, std::bit_ceil(static_cast<uint32_t>(end)));
      // Account for BF16 K/V, FP32 scores and exponentials, and BF16 probabilities.
      // Leave SRAM headroom for graph scheduling instead of tying the choice to a model geometry.
      const uint64_t fused_bytes = uint64_t{direct_kv} *
          (4 * static_cast<uint64_t>(kv_heads) * static_cast<uint64_t>(head_dim) +
           10 * static_cast<uint64_t>(query_heads));
      constexpr uint64_t kFusedDecodeBudget = 40ULL * 1024 * 1024;
      const bool single_decode = remaining == 1 &&
          (end <= 4096 || (end <= 8192 && fused_bytes <= kFusedDecodeBudget));
      const bool short_prefill = end <= 512 &&
          uint64_t{q_bucket} * direct_kv * static_cast<uint64_t>(query_heads) <= 4 * 1024 * 1024;
      if (!single_decode && !short_prefill) {
        const uint64_t row_cap = sequence.length >= 2048 ? 512ULL : 256ULL;
        const auto max_rows = std::bit_floor(static_cast<uint32_t>(std::min<uint64_t>(
            row_cap, kMaxTileScoreElements / (kTileKvBucket * static_cast<uint64_t>(query_heads)))));
        count = std::min<int32_t>(remaining, static_cast<int32_t>(max_rows));
        q_bucket = std::bit_ceil(static_cast<uint32_t>(count));
      }
      const int32_t visible = sequence.prefix + done + count;
      const uint32_t kv_bucket = (single_decode || short_prefill) ? direct_kv : kTileKvBucket;
      AttentionBlock block{};
      block.first_row = sequence.row + done;
      block.valid_rows = count;
      block.query_bucket = q_bucket;
      block.kv_bucket = kv_bucket;
      block.kv_source_row = sequence.prefix == 0 ? sequence.row : -1;
      const bool emit_tile_indices = block.kv_source_row < 0 || visible <= static_cast<int32_t>(kv_bucket);
      for (int32_t first = 0; first < visible; first += kv_bucket) {
        const int32_t valid = std::min<int32_t>(kv_bucket, visible - first);
        AttentionTile tile{};
        tile.indices_offset = static_cast<uint64_t>(tasks.size()) * 64 * sizeof(int32_t);
        tile.first_key = first;
        tile.valid_keys = valid;
        if (emit_tile_indices) {
          for (int32_t kind = 0; kind < 2; ++kind) {
            for (int32_t head = 0; head < kv_heads; ++head) {
              for (int32_t key = 0; key < static_cast<int32_t>(kv_bucket); key += 64) {
                const auto lanes = std::clamp(valid - key, 0, 64);
                tasks.push_back({block.first_row, first + key, lanes,
                                 static_cast<int32_t>((kind * kv_heads + head) * slots_)});
              }
            }
          }
        }
        block.tiles.push_back(tile);
      }
      plan->blocks.push_back(std::move(block));
      done += count;
    }
  }
  TORCH_CHECK(!tasks.empty() && tasks.size() <= (uint64_t{1} << 21),
              "GaudiRT attention: too many tile-index tasks");
  const auto task_bucket = std::bit_ceil(static_cast<uint32_t>(tasks.size()));
  const uint64_t index_bytes = static_cast<uint64_t>(task_bucket) * 64 * sizeof(int32_t);
  std::vector<int32_t> host(static_cast<size_t>(task_bucket) * 4, 0);
  for (size_t i = 0; i < tasks.size(); ++i)
    std::copy(tasks[i].begin(), tasks[i].end(), host.begin() + i * 4);
  auto append_aligned = [&](const std::vector<int32_t>& values) {
    host.resize(align(host.size() * sizeof(int32_t)) / sizeof(int32_t), 0);
    const uint64_t offset = index_bytes + host.size() * sizeof(int32_t);
    host.insert(host.end(), values.begin(), values.end());
    return offset;
  };
  const uint64_t context_offset = append_aligned({static_cast<int32_t>(context_)});
  plan->store_header_offset = append_aligned({static_cast<int32_t>(rows_), static_cast<int32_t>(slots_),
                                               static_cast<int32_t>(kv_heads)});
  size_t sequence_index = 0;
  for (auto& block : plan->blocks) {
    while (sequence_index + 1 < sequences_.size() &&
           block.first_row >= sequences_[sequence_index].row + sequences_[sequence_index].length)
      ++sequence_index;
    const auto& sequence = sequences_[sequence_index];
    std::vector<int32_t> positions(block.query_bucket, 0);
    for (int32_t i = 0; i < block.valid_rows; ++i)
      positions[i] = sequence.prefix + block.first_row - sequence.row + i;
    block.positions_offset = append_aligned(positions);
    for (auto& tile : block.tiles)
      tile.header_offset = append_aligned({block.valid_rows, tile.valid_keys, tile.first_key});
    block.store_header_offset = append_aligned({block.valid_rows, static_cast<int32_t>(slots_),
                                                static_cast<int32_t>(kv_heads)});
  }
  const bool pending_metadata = !pending_host_.empty();
  const uint64_t pending_offset = pending_metadata ? append_aligned(pending_host_) : 0;
  host.resize(align(host.size() * sizeof(int32_t)) / sizeof(int32_t), 0);
  const uint64_t host_bytes = host.size() * sizeof(int32_t);
  plan->scratch_offset = index_bytes + host_bytes;
  plan->scratch_bytes = 0;
  for (const auto& block : plan->blocks) {
    if (block.tiles.size() <= kMaxFusedAttentionTiles) continue;
    const uint64_t stats = align(((block.query_bucket + 63) / 64 * 64) * query_heads * sizeof(float));
    const uint64_t output = align(static_cast<uint64_t>(block.query_bucket) * query_heads * head_dim * sizeof(float));
    plan->scratch_bytes = std::max(plan->scratch_bytes, 2 * (2 * stats + output));
  }
  plan->bytes = plan->scratch_offset + plan->scratch_bytes;
  const auto compute = ctx->stream();
  plan->stream = compute;
  const auto staging = ctx->attention_staging().acquire(plan->bytes, host_bytes);
  plan->address = staging.address;
  plan->staging_slot = staging.index;
  const auto address = plan->address;
  const auto bytes = plan->bytes;
  c10::DataPtr data(reinterpret_cast<void*>(address), plan.release(), release_attention_plan,
                    c10::Device(c10::kPrivateUse1, ctx->device_index));
  auto tensor = tensor_from_allocation(std::move(data), bytes, {static_cast<int64_t>(bytes / 4)},
      at::TensorOptions().device(c10::Device(c10::kPrivateUse1, ctx->device_index)).dtype(at::kInt));
  auto& stream_state = ctx->streams.state(active_stream_id);
  const auto h2d = ctx->device.h2d_stream();
  {
    std::memcpy(staging.host, host.data(), host_bytes);
    ctx->pinned.copy_h2d(address + index_bytes, staging.host, host_bytes, h2d, true);
    stream_state.metadata_ready.record_and_wait(h2d, compute);
  }
  if (pending_metadata) {
    const auto destination = address_of(header_, *ctx);
    ctx->allocator.record_stream(destination, compute);
    dma_copy(destination, nullptr, address + pending_offset, nullptr, 0, pending_host_.size(),
             sizeof(int32_t), compute, DRAM_TO_DRAM);
    finish_pending_metadata();
  }
  const auto map_address = address_of(mapping_, *ctx);
  const auto request_address = address_of(requests_, *ctx);
  ctx->allocator.record_stream(map_address, compute);
  ctx->allocator.record_stream(request_address, compute);
  {
    std::lock_guard lock(ctx->cold_mutex);
    auto& recipe = indices_recipe(*ctx, bucket_, task_bucket);
    std::unique_ptr<TemporaryDevice> workspace;
    if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
    auto bindings = recipe.tensors;
    const std::array<uint64_t, 5> addresses{map_address, request_address, address + index_bytes,
                                            address + context_offset, address};
    for (size_t i = 0; i < addresses.size(); ++i) bindings[i].pTensorAddress = addresses[i];
    bindings[0].tensorSize[0] = mapping_.numel();
    TORCH_CHECK(synLaunch(compute, bindings.data(), bindings.size(), workspace ? workspace->address : 0,
                          recipe.recipe, 0) == synSuccess, "GaudiRT attention: tile-index launch failed");
  }
  attention_plan_ = tensor;
  return tensor;
}

}  // namespace gaudirt::generic
