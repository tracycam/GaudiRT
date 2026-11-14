// SPDX-License-Identifier: Apache-2.0
#include "sampler_vocab.hpp"

#include "bridge_internal.hpp"
#include "eager_copy.hpp"
#include "graph_builder.hpp"
#include "layout_copy.hpp"
#include "temporaries.hpp"
#include "gaudirt/common/sha256.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace gaudirt::generic {
using namespace runtime;
namespace {

struct SampleVocabParameters {
  int32_t vocab_size;
  int32_t nan_to_neg_inf;
};

struct VocabBitmaskParameters {
  int32_t active_cols;
  int32_t mask_tail;
};

constexpr int64_t kMaxBitmaskChunkRows = 8192;

ColdRecipe& vocab_bitmask_recipe(Context& ctx, bool bf16, int64_t vocab_cols, int64_t mask_words,
                                 uint32_t bucket_rows, int32_t active_cols, bool mask_tail) {
  std::ostringstream signature;
  signature << "vocab_bitmask_v1;" << (bf16 ? "bf16;" : "f32;") << vocab_cols << ';' << mask_words
            << ';' << bucket_rows << ';' << active_cols << ';' << (mask_tail ? 1 : 0);
  const auto structure = signature.str();
  const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const auto key = "vocab_bitmask_" + digest;
  if (auto found = ctx.cold_recipes.find(key); found != ctx.cold_recipes.end()) return *found->second;

  auto recipe = std::make_unique<ColdRecipe>(3);
  const auto prefix = "vbm_" + digest.substr(0, 12) + "_b" + std::to_string(bucket_rows);
  SynapseGraphBuilder graph(prefix, *recipe);
  const auto logits_type = bf16 ? syn_type_bf16 : syn_type_float;
  const uint64_t rows = static_cast<uint64_t>(bucket_rows);
  auto logits_in = graph.tensor(logits_type, {static_cast<uint64_t>(vocab_cols), rows}, 0);
  auto bitmask = graph.tensor(syn_type_int32, {static_cast<uint64_t>(mask_words), rows}, 1);
  auto logits_out = graph.tensor(logits_type, {static_cast<uint64_t>(vocab_cols), rows}, 2);
  graph.node({bf16 ? "gk_apply_vocab_bitmask_bf16_v1" : "gk_apply_vocab_bitmask_f32_v1",
              {logits_in, bitmask},
              {logits_out},
              pack_node_params(VocabBitmaskParameters{active_cols, mask_tail ? 1 : 0})});
  graph.compile(ctx.cache, key);
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT bitmask: duplicate recipe");
  return *it->second;
}

ColdRecipe& cached_vocab_bitmask_recipe(Context& ctx, bool bf16, int64_t vocab_cols, int64_t mask_words,
                                        uint32_t bucket_rows, int32_t active_cols, bool mask_tail) {
  std::lock_guard lock(ctx.cold_mutex);
  return vocab_bitmask_recipe(ctx, bf16, vocab_cols, mask_words, bucket_rows, active_cols, mask_tail);
}

ColdRecipe& sample_vocab_recipe(Context& ctx, bool bf16, int64_t vocab_size, uint32_t bucket_rows,
                                bool nan_to_neg_inf) {
  std::ostringstream signature;
  signature << "sample_vocab_v1;" << (bf16 ? "bf16;" : "f32;") << vocab_size << ';'
            << bucket_rows << ';' << (nan_to_neg_inf ? 1 : 0);
  const auto structure = signature.str();
  const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const auto key = "sample_vocab_" + digest;
  if (auto found = ctx.cold_recipes.find(key); found != ctx.cold_recipes.end()) return *found->second;

  auto recipe = std::make_unique<ColdRecipe>(7);
  const auto prefix = "g" + digest.substr(0, 12) + "_b" + std::to_string(bucket_rows);
  SynapseGraphBuilder graph(prefix, *recipe);
  const uint64_t width = static_cast<uint64_t>(vocab_size);
  const uint64_t rows = static_cast<uint64_t>(bucket_rows);
  auto logits = graph.tensor(bf16 ? syn_type_bf16 : syn_type_float, {width, rows}, 0);
  auto temperatures = graph.tensor(syn_type_float, {rows}, 1);
  auto top_ps = graph.tensor(syn_type_float, {rows}, 2);
  auto top_ks = graph.tensor(syn_type_int32, {rows}, 3);
  auto min_ps = graph.tensor(syn_type_float, {rows}, 4);
  auto seeds = graph.tensor(syn_type_int32, {2, rows}, 5);
  auto output_ids = graph.tensor(syn_type_int32, {rows}, 6);
  graph.node({bf16 ? "gk_sample_vocab_bf16_v1" : "gk_sample_vocab_f32_v1",
              {logits, temperatures, top_ps, top_ks, min_ps, seeds},
              {output_ids},
              pack_node_params(SampleVocabParameters{static_cast<int32_t>(vocab_size),
                                                     nan_to_neg_inf ? 1 : 0})});
  graph.compile(ctx.cache, key);
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT sampler: duplicate vocabulary sampling recipe");
  return *it->second;
}

ColdRecipe& cached_sample_vocab_recipe(Context& ctx, bool bf16, int64_t vocab_size, uint32_t bucket_rows,
                                       bool nan_to_neg_inf) {
  std::lock_guard lock(ctx.cold_mutex);
  return sample_vocab_recipe(ctx, bf16, vocab_size, bucket_rows, nan_to_neg_inf);
}

void record_if_needed(Context& ctx, const at::Tensor& tensor, synStreamHandle stream) {
  const auto& allocation = validated_program_input(tensor, ctx);
  if (allocation.stream != stream) {
    const auto address = reinterpret_cast<uint64_t>(tensor.data_ptr());
    if (allocation.kind == AllocationKind::graph) ctx.graph_pool->record_stream(address, stream);
    else ctx.allocator.record_stream(address, stream);
  }
}

at::Tensor normalize_batch_vector(const at::Tensor& tensor, int64_t batch, at::ScalarType dtype,
                                  const char* name) {
  TORCH_CHECK(tensor.defined(), "GaudiRT sampler: missing required ", name);
  TORCH_CHECK(tensor.scalar_type() == dtype, "GaudiRT sampler: unexpected dtype for ", name);
  if (tensor.dim() == 1 && tensor.size(0) == batch) return tensor;
  if (tensor.dim() == 2 && tensor.size(0) == batch && tensor.size(1) == 1) return tensor.reshape({batch});
  if ((tensor.dim() == 1 || tensor.dim() == 0) && tensor.numel() == 1) return tensor.reshape({1}).expand({batch});
  if (tensor.dim() == 2 && tensor.size(0) == 1 && tensor.size(1) == 1) return tensor.reshape({1}).expand({batch});
  TORCH_CHECK(false, "GaudiRT sampler: invalid shape for ", name);
}

void h2d_upload(Context& ctx, uint64_t dst, const void* src, uint64_t bytes) {
  TemporaryHost host(ctx, bytes);
  std::memcpy(host.address, src, bytes);
  const auto stream = ctx.stream(), h2d = ctx.device.h2d_stream();
  auto& state = ctx.streams.state(active_stream_id);
  state.metadata_reusable.record_and_wait(stream, h2d);
  ctx.allocator.record_stream(dst, h2d);
  ctx.pinned.copy_h2d(dst, host.address, bytes, h2d, true);
  state.metadata_ready.record_and_wait(h2d, stream);
}

void d2h_download(Context& ctx, void* dst, uint64_t src, uint64_t bytes) {
  TemporaryHost host(ctx, bytes);
  const auto compute = ctx.stream(), d2h = ctx.device.d2h_stream();
  ctx.streams.state(active_stream_id).d2h_reusable.record_and_wait(compute, d2h);
  ctx.allocator.record_stream(src, d2h);
  ctx.pinned.copy_d2h(host.address, src, bytes, d2h, false);
  std::memcpy(dst, host.address, bytes);
}

uint64_t stage_device_or_host_tensor(
    Context& ctx, const at::Tensor& tensor, uint32_t bucket_rows, uint64_t width,
    std::vector<std::pair<uint64_t, uint64_t>>& regions,
    std::vector<std::unique_ptr<TemporaryDevice>>& temporaries) {
  const auto stream = ctx.stream();
  const uint64_t item = static_cast<uint64_t>(tensor.element_size());
  const uint64_t bytes = static_cast<uint64_t>(bucket_rows) * width * item;
  if (tensor.device().is_cpu()) {
    const auto contig = tensor.contiguous();
    auto staged = std::make_unique<TemporaryDevice>(ctx, std::max<uint64_t>(256, bytes));
    if (contig.numel() < static_cast<int64_t>(bucket_rows * width) && bytes >= 4) {
      TORCH_CHECK(synMemsetD32Async(staged->address, 0, bytes / sizeof(uint32_t), stream) == synSuccess,
                  "GaudiRT sampler: failed to clear staged host buffer");
    }
    h2d_upload(ctx, staged->address, contig.data_ptr(), static_cast<uint64_t>(contig.numel()) * item);
    const uint64_t result = staged->address;
    regions.emplace_back(result, bytes);
    temporaries.push_back(std::move(staged));
    return result;
  }
  TORCH_CHECK(tensor.device().type() == c10::kPrivateUse1 && tensor.device().index() == ctx.device_index,
              "GaudiRT sampler: tensor must reside on the active GaudiRT device or CPU");
  record_if_needed(ctx, tensor, stream);
  const uint64_t source = reinterpret_cast<uint64_t>(tensor.data_ptr());
  const uint64_t base = reinterpret_cast<uint64_t>(tensor.storage().data());
  const uint64_t storage = tensor.storage().nbytes();
  bool direct = tensor.is_contiguous() && source % BlockCachingAllocator::kAlignment == 0 &&
                source >= base && source - base <= storage && bytes <= storage - (source - base);
  for (const auto& [other, length] : regions) {
    if (source < other + length && other < source + bytes) direct = false;
  }
  uint64_t result = source;
  if (!direct) {
    auto staged = std::make_unique<TemporaryDevice>(ctx, std::max<uint64_t>(256, bytes));
    result = staged->address;
    if (tensor.numel() < static_cast<int64_t>(bucket_rows * width) && bytes >= 4) {
      TORCH_CHECK(synMemsetD32Async(result, 0, bytes / sizeof(uint32_t), stream) == synSuccess,
                  "GaudiRT sampler: failed to clear staged device buffer");
    }
    if (tensor.dim() != 2 || !try_pack_strided_2d(ctx, result, bytes, tensor, stream)) {
      const Layout layout(tensor);
      dma_copy(result, nullptr, source, &layout, 0, static_cast<uint64_t>(tensor.numel()),
               item, stream, DRAM_TO_DRAM);
    }
    temporaries.push_back(std::move(staged));
  }
  regions.emplace_back(result, bytes);
  return result;
}

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

std::vector<int64_t> read_int64_values(Context& ctx, const at::Tensor& tensor, int64_t batch) {
  std::vector<int64_t> values(static_cast<size_t>(batch));
  const auto contig = tensor.contiguous();
  const uint64_t bytes = static_cast<uint64_t>(batch) * static_cast<uint64_t>(contig.element_size());
  if (contig.device().is_cpu()) {
    if (contig.scalar_type() == at::kLong) {
      std::memcpy(values.data(), contig.data_ptr<int64_t>(), bytes);
    } else {
      const auto* ptr = contig.data_ptr<int32_t>();
      for (int64_t i = 0; i < batch; ++i) values[static_cast<size_t>(i)] = static_cast<int64_t>(ptr[i]);
    }
    return values;
  }
  TORCH_CHECK(contig.device().type() == c10::kPrivateUse1 && contig.device().index() == ctx.device_index,
              "GaudiRT sampler: seed/position tensor on unexpected device");
  const auto addr = reinterpret_cast<uint64_t>(contig.data_ptr());
  ctx.allocator.record_stream(addr, ctx.stream());
  if (contig.scalar_type() == at::kLong) {
    d2h_download(ctx, values.data(), addr, bytes);
  } else {
    std::vector<int32_t> tmp(static_cast<size_t>(batch));
    d2h_download(ctx, tmp.data(), addr, bytes);
    for (int64_t i = 0; i < batch; ++i) values[static_cast<size_t>(i)] = static_cast<int64_t>(tmp[i]);
  }
  return values;
}

uint64_t prepare_seeds(
    Context& ctx, int64_t batch, uint32_t bucket_rows,
    const c10::optional<at::Tensor>& seeds, const c10::optional<at::Tensor>& positions,
    std::vector<std::pair<uint64_t, uint64_t>>& regions,
    std::vector<std::unique_ptr<TemporaryDevice>>& temporaries) {
  const bool has_seeds = seeds.has_value() && seeds->defined();
  const bool has_positions = positions.has_value() && positions->defined();
  if (!has_seeds) {
    thread_local std::mt19937_64 rng([]() {
      std::random_device rd;
      return (static_cast<uint64_t>(rd()) << 32) ^ static_cast<uint64_t>(rd());
    }());
    std::vector<uint64_t> host_seeds(static_cast<size_t>(bucket_rows), 0);
    for (int64_t i = 0; i < batch; ++i) host_seeds[static_cast<size_t>(i)] = rng();
    const uint64_t bytes = static_cast<uint64_t>(bucket_rows) * sizeof(uint64_t);
    auto staged = std::make_unique<TemporaryDevice>(ctx, std::max<uint64_t>(256, bytes));
    h2d_upload(ctx, staged->address, host_seeds.data(), bytes);
    const uint64_t addr = staged->address;
    regions.emplace_back(addr, bytes);
    temporaries.push_back(std::move(staged));
    return addr;
  }

  const auto seed_vec = normalize_batch_vector(*seeds, batch, at::kLong, "seeds");
  if (!has_positions) {
    return stage_device_or_host_tensor(ctx, seed_vec, bucket_rows, 1, regions, temporaries);
  }

  TORCH_CHECK(positions->scalar_type() == at::kInt || positions->scalar_type() == at::kLong,
              "GaudiRT sampler: positions must be int32 or int64");
  const auto pos_vec = normalize_batch_vector(*positions, batch, positions->scalar_type(), "positions");
  const auto seed_vals = read_int64_values(ctx, seed_vec, batch);
  const auto pos_vals = read_int64_values(ctx, pos_vec, batch);
  std::vector<uint64_t> host_seeds(static_cast<size_t>(bucket_rows), 0);
  for (int64_t i = 0; i < batch; ++i) {
    const uint64_t s = static_cast<uint64_t>(seed_vals[static_cast<size_t>(i)]);
    const uint64_t p = static_cast<uint64_t>(pos_vals[static_cast<size_t>(i)]);
    host_seeds[static_cast<size_t>(i)] = splitmix64(s ^ splitmix64(p));
  }
  const uint64_t bytes = static_cast<uint64_t>(bucket_rows) * sizeof(uint64_t);
  auto staged = std::make_unique<TemporaryDevice>(ctx, std::max<uint64_t>(256, bytes));
  h2d_upload(ctx, staged->address, host_seeds.data(), bytes);
  const uint64_t addr = staged->address;
  regions.emplace_back(addr, bytes);
  temporaries.push_back(std::move(staged));
  return addr;
}

void apply_token_bitmask_dense(Context& ctx, at::Tensor& logits_2d, const at::Tensor& bitmask_2d,
                               int32_t active_cols, bool mask_tail) {
  const int64_t batch = logits_2d.size(0);
  const int64_t vocab_cols = logits_2d.size(1);
  const int64_t mask_words = bitmask_2d.size(1);
  const bool bf16 = logits_2d.scalar_type() == at::kBFloat16;
  const auto stream = ctx.stream();

  for (int64_t offset = 0; offset < batch; offset += kMaxBitmaskChunkRows) {
    const int64_t chunk_rows = std::min<int64_t>(kMaxBitmaskChunkRows, batch - offset);
    auto logits_chunk = logits_2d.narrow(0, offset, chunk_rows);
    auto mask_chunk = bitmask_2d.narrow(0, offset, chunk_rows);
    const uint32_t bucket_rows = std::bit_ceil(static_cast<uint32_t>(chunk_rows));

    auto output = at::empty({static_cast<int64_t>(bucket_rows), vocab_cols}, logits_2d.options());
    const uint64_t output_addr = address_of(output, ctx);
    ctx.allocator.record_stream(output_addr, stream);

    std::vector<std::unique_ptr<TemporaryDevice>> temporaries;
    std::vector<std::pair<uint64_t, uint64_t>> regions{
        {output_addr, static_cast<uint64_t>(bucket_rows) * static_cast<uint64_t>(vocab_cols) *
                          static_cast<uint64_t>(logits_2d.element_size())}};

    const uint64_t logits_addr = stage_device_or_host_tensor(
        ctx, logits_chunk, bucket_rows, static_cast<uint64_t>(vocab_cols), regions, temporaries);
    const uint64_t mask_addr = stage_device_or_host_tensor(
        ctx, mask_chunk, bucket_rows, static_cast<uint64_t>(mask_words), regions, temporaries);

    auto& recipe = cached_vocab_bitmask_recipe(
        ctx, bf16, vocab_cols, mask_words, bucket_rows, active_cols, mask_tail);
    std::unique_ptr<TemporaryDevice> workspace;
    if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(ctx, recipe.workspace);
    auto bindings = recipe.tensors;
    const std::array<uint64_t, 3> addresses{logits_addr, mask_addr, output_addr};
    for (size_t i = 0; i < addresses.size(); ++i) bindings[i].pTensorAddress = addresses[i];
    TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                          workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
                "GaudiRT bitmask: kernel launch failed");
    copy(logits_chunk, output.narrow(0, 0, chunk_rows), true);
  }
}

}  // namespace

at::Tensor sample_from_logits(
    const at::Tensor& logits,
    const at::Tensor& temperatures,
    const at::Tensor& top_ps,
    const at::Tensor& top_ks,
    const at::Tensor& min_ps,
    const c10::optional<at::Tensor>& seeds,
    const c10::optional<at::Tensor>& positions,
    bool nan_to_neg_inf) {
  auto ctx = current();
  const auto stream = ctx->stream();
  TORCH_CHECK(logits.device().type() == c10::kPrivateUse1 && logits.device().index() == ctx->device_index &&
              logits.dim() == 2 &&
              (logits.scalar_type() == at::kBFloat16 || logits.scalar_type() == at::kFloat),
              "GaudiRT sampler: logits must be a 2D bf16 or fp32 tensor on the active device");
  const int64_t batch = logits.size(0);
  const int64_t vocab_size = logits.size(1);
  TORCH_CHECK(batch > 0 && batch <= (int64_t{1} << 30) && vocab_size > 0 && vocab_size <= INT32_MAX,
              "GaudiRT sampler: logits shape outside supported range");

  const uint32_t bucket_rows = std::bit_ceil(static_cast<uint32_t>(batch));
  const bool bf16 = logits.scalar_type() == at::kBFloat16;

  const auto temp_vec = normalize_batch_vector(temperatures, batch, at::kFloat, "temperatures");
  const auto top_p_vec = normalize_batch_vector(top_ps, batch, at::kFloat, "top_ps");
  const auto top_k_vec = normalize_batch_vector(top_ks, batch, at::kInt, "top_ks");
  const auto min_p_vec = normalize_batch_vector(min_ps, batch, at::kFloat, "min_ps");

  auto output = at::empty({static_cast<int64_t>(bucket_rows)}, logits.options().dtype(at::kInt));
  const uint64_t output_addr = address_of(output, *ctx);
  ctx->allocator.record_stream(output_addr, stream);

  std::vector<std::unique_ptr<TemporaryDevice>> temporaries;
  std::vector<std::pair<uint64_t, uint64_t>> regions{
      {output_addr, static_cast<uint64_t>(bucket_rows) * sizeof(int32_t)}};

  const uint64_t logits_addr = stage_device_or_host_tensor(
      *ctx, logits, bucket_rows, static_cast<uint64_t>(vocab_size), regions, temporaries);
  const uint64_t temp_addr = stage_device_or_host_tensor(*ctx, temp_vec, bucket_rows, 1, regions, temporaries);
  const uint64_t top_p_addr = stage_device_or_host_tensor(*ctx, top_p_vec, bucket_rows, 1, regions, temporaries);
  const uint64_t top_k_addr = stage_device_or_host_tensor(*ctx, top_k_vec, bucket_rows, 1, regions, temporaries);
  const uint64_t min_p_addr = stage_device_or_host_tensor(*ctx, min_p_vec, bucket_rows, 1, regions, temporaries);
  const uint64_t seeds_addr = prepare_seeds(*ctx, batch, bucket_rows, seeds, positions, regions, temporaries);

  auto& recipe = cached_sample_vocab_recipe(*ctx, bf16, vocab_size, bucket_rows, nan_to_neg_inf);
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  auto bindings = recipe.tensors;
  const std::array<uint64_t, 7> addresses{
      logits_addr, temp_addr, top_p_addr, top_k_addr, min_p_addr, seeds_addr, output_addr};
  for (size_t i = 0; i < addresses.size(); ++i) bindings[i].pTensorAddress = addresses[i];
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(), workspace ? workspace->address : 0,
                        recipe.recipe, 0) == synSuccess,
              "GaudiRT sampler: vocabulary sampling launch failed");
  return output.narrow(0, 0, batch);
}

at::Tensor& apply_token_bitmask_inplace_(
    at::Tensor& logits,
    const at::Tensor& bitmask,
    const c10::optional<at::Tensor>& indices,
    int64_t vocab_size,
    bool mask_tail) {
  auto ctx = current();
  TORCH_CHECK(logits.defined() && logits.device().type() == c10::kPrivateUse1 &&
                  logits.device().index() == ctx->device_index &&
                  (logits.dim() == 1 || logits.dim() == 2) &&
                  (logits.scalar_type() == at::kBFloat16 || logits.scalar_type() == at::kFloat),
              "GaudiRT bitmask: logits must be a 1D or 2D bf16/fp32 tensor on the active device");
  TORCH_CHECK(bitmask.defined() && bitmask.scalar_type() == at::kInt &&
                  (bitmask.dim() == 1 || bitmask.dim() == 2) &&
                  (bitmask.device().is_cpu() ||
                   (bitmask.device().type() == c10::kPrivateUse1 &&
                    bitmask.device().index() == ctx->device_index)),
              "GaudiRT bitmask: bitmask must be a 1D or 2D int32 tensor on CPU or active device");

  auto work_logits = logits.dim() == 1 ? logits.unsqueeze(0) : logits;
  auto work_bitmask = bitmask.dim() == 1 ? bitmask.unsqueeze(0) : bitmask;
  const int64_t batch = work_logits.size(0);
  const int64_t vocab_cols = work_logits.size(1);
  const int64_t mask_rows = work_bitmask.size(0);
  const int64_t mask_words = work_bitmask.size(1);
  TORCH_CHECK(vocab_cols > 0 && vocab_cols <= INT32_MAX && mask_words > 0 && mask_words <= INT32_MAX,
              "GaudiRT bitmask: invalid tensor dimensions");

  const int64_t mask_bits = mask_words * 32LL;
  int64_t active_cols_i64 = 0;
  if (vocab_size >= 0) {
    TORCH_CHECK(vocab_size <= vocab_cols, "GaudiRT bitmask: vocab_size exceeds logits width");
    TORCH_CHECK(mask_tail || vocab_size <= mask_bits,
                "GaudiRT bitmask: bitmask is smaller than vocab_size");
    active_cols_i64 = std::min(vocab_size, mask_bits);
  } else {
    active_cols_i64 = std::min(vocab_cols, mask_bits);
  }
  const int32_t active_cols = static_cast<int32_t>(active_cols_i64);

  if (indices.has_value() && indices->defined()) {
    const auto& idx = *indices;
    TORCH_CHECK(idx.dim() <= 1 && (idx.scalar_type() == at::kInt || idx.scalar_type() == at::kLong) &&
                    (idx.device().is_cpu() ||
                     (idx.device().type() == c10::kPrivateUse1 &&
                      idx.device().index() == ctx->device_index)),
                "GaudiRT bitmask: indices must be an int32 or int64 tensor on CPU or active device");
    const auto idx_1d = idx.dim() == 0 ? idx.reshape({1}) : idx;
    const int64_t count = idx_1d.size(0);
    if (count == 0) return logits;
    const auto raw_rows = read_int64_values(*ctx, idx_1d, count);
    std::vector<int64_t> unique_rows;
    unique_rows.reserve(static_cast<size_t>(count));
    std::unordered_set<int64_t> seen;
    seen.reserve(static_cast<size_t>(count));
    for (int64_t row : raw_rows) {
      TORCH_CHECK(row >= 0 && row < batch && row < mask_rows,
                  "GaudiRT bitmask: index ", row, " out of bounds for logits/bitmask rows (",
                  batch, "/", mask_rows, ")");
      if (seen.insert(row).second) unique_rows.push_back(row);
    }
    const int64_t unique_count = static_cast<int64_t>(unique_rows.size());
    auto sub_logits = at::empty({unique_count, vocab_cols}, work_logits.options());
    at::Tensor sub_mask;
    if (work_bitmask.device().is_cpu()) {
      sub_mask = at::empty({unique_count, mask_words}, work_bitmask.options());
      for (int64_t j = 0; j < unique_count; ++j) {
        sub_mask[j].copy_(work_bitmask[unique_rows[static_cast<size_t>(j)]]);
      }
    } else {
      sub_mask = at::empty({unique_count, mask_words}, work_bitmask.options());
      for (int64_t j = 0; j < unique_count; ++j) {
        auto dst_m = sub_mask.narrow(0, j, 1);
        copy(dst_m, work_bitmask.narrow(0, unique_rows[static_cast<size_t>(j)], 1), true);
      }
    }
    for (int64_t j = 0; j < unique_count; ++j) {
      auto dst_l = sub_logits.narrow(0, j, 1);
      copy(dst_l, work_logits.narrow(0, unique_rows[static_cast<size_t>(j)], 1), true);
    }
    apply_token_bitmask_dense(*ctx, sub_logits, sub_mask, active_cols, mask_tail);
    for (int64_t j = 0; j < unique_count; ++j) {
      auto dst_l = work_logits.narrow(0, unique_rows[static_cast<size_t>(j)], 1);
      copy(dst_l, sub_logits.narrow(0, j, 1), true);
    }
    return logits;
  }

  TORCH_CHECK(mask_rows == batch,
              "GaudiRT bitmask: bitmask row count (", mask_rows, ") must match logits batch (", batch, ")");
  if (batch == 0) return logits;
  apply_token_bitmask_dense(*ctx, work_logits, work_bitmask, active_cols, mask_tail);
  return logits;
}

}  // namespace gaudirt::generic
