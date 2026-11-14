// SPDX-License-Identifier: Apache-2.0
#include "layout_copy.hpp"

#include "bridge_internal.hpp"
#include "graph_builder.hpp"
#include "gaudirt/common/sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace gaudirt::generic {
using runtime::BlockCachingAllocator;
namespace {

ColdRecipe& strided_2d_pack_recipe(Context& ctx, uint32_t rows, uint64_t u_stride,
                                   uint64_t u_col, uint64_t u_width) {
  const std::string structure = "strided_2d_pack_v1:" + std::to_string(rows) + ':' +
      std::to_string(u_stride) + ':' + std::to_string(u_col) + ':' + std::to_string(u_width);
  const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(structure.data()),
                                           structure.size());
  const auto key = "strided_2d_pack_" + digest;
  if (auto found = ctx.cold_recipes.find(key); found != ctx.cold_recipes.end()) return *found->second;

  auto recipe = std::make_unique<ColdRecipe>(2);
  const auto prefix = "g" + digest.substr(0, 12) + "_b" + std::to_string(rows);
  SynapseGraphBuilder graph(prefix, *recipe);
  auto in = graph.tensor(syn_type_bf16, {u_stride, rows}, 0);
  auto out = graph.tensor(syn_type_bf16, {u_width, rows}, 1);
  std::vector<synTensor> parts;
  if (u_col > 0) parts.push_back(graph.tensor(syn_type_bf16, {u_col, rows}, -1));
  parts.push_back(out);
  const uint64_t u_tail = u_stride - u_col - u_width;
  if (u_tail > 0) parts.push_back(graph.tensor(syn_type_bf16, {u_tail, rows}, -1));
  synSplitParams split{0};
  graph.node({"split", {in}, std::move(parts), split});
  graph.compile(ctx.cache, key);
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate 2D strided pack recipe");
  return *it->second;
}

void launch_strided_2d_pack(Context& ctx, ColdRecipe& recipe, uint64_t src_addr,
                            uint64_t dst_addr, synStreamHandle stream) {
  std::array<synLaunchTensorInfo, 2> bindings{recipe.tensors[0], recipe.tensors[1]};
  bindings[0].pTensorAddress = src_addr;
  bindings[1].pTensorAddress = dst_addr;
  uint64_t ws = 0;
  if (recipe.workspace) ws = ctx.allocator.allocate(recipe.workspace, stream);
  const auto status = synLaunch(stream, bindings.data(), bindings.size(), ws, recipe.recipe, 0);
  if (ws) ctx.allocator.free(ws);
  TORCH_CHECK(status == synSuccess, "GaudiRT: 2D strided pack launch failed");
}

}  // namespace

// Addresses come from host-visible tensor metadata; D2D data stays in HBM.
Layout::Layout(const at::Tensor& value) : tensor(value), run(1) {
  for (int64_t i = tensor.dim() - 1; i >= 0; --i) {
    const auto size = tensor.size(i);
    if (size != 1 && tensor.stride(i) != static_cast<int64_t>(run)) break;
    run *= static_cast<uint64_t>(size);
  }
}

uint64_t Layout::offset(uint64_t index) const {
  uint64_t result = 0;
  for (int64_t i = tensor.dim() - 1; i >= 0; --i) {
    const auto size = static_cast<uint64_t>(tensor.size(i));
    result += (index % size) * static_cast<uint64_t>(tensor.stride(i));
    index /= size;
  }
  return result;
}

void host_copy(uint8_t* packed, const at::Tensor& tensor, uint64_t start,
               uint64_t elements, bool pack) {
  const Layout layout(tensor);
  const uint64_t item = tensor.element_size();
  auto* base = static_cast<uint8_t*>(tensor.data_ptr());
  for (uint64_t done = 0; done < elements;) {
    const uint64_t index = start + done;
    const uint64_t count = std::min(elements - done, layout.run - index % layout.run);
    auto* slot = base + layout.offset(index) * item;
    if (pack) std::memcpy(packed + done * item, slot, count * item);
    else std::memcpy(slot, packed + done * item, count * item);
    done += count;
  }
}

void dma_copy(uint64_t dst, const Layout* dst_layout, uint64_t src,
              const Layout* src_layout, uint64_t start, uint64_t elements,
              uint64_t item, synStreamHandle stream, synDmaDir direction) {
  std::array<uint64_t, 128> sources{}, sizes{}, destinations{};
  size_t used = 0;
  bool had_multi_d2d = false;
  auto flush = [&] {
    if (!used) return;
    if (used > 1 && direction == DRAM_TO_DRAM) had_multi_d2d = true;
    const auto status = used == 1
        ? synMemCopyAsync(stream, sources[0], sizes[0], destinations[0], direction)
        : synMemCopyAsyncMultiple(stream, sources.data(), sizes.data(), destinations.data(),
                                  direction, used);
    if (status != synSuccess) throw std::runtime_error("GaudiRT: strided DMA failed");
    used = 0;
  };
  for (uint64_t done = 0; done < elements;) {
    const uint64_t index = start + done;
    uint64_t count = elements - done;
    if (src_layout) count = std::min(count, src_layout->run - index % src_layout->run);
    if (dst_layout) count = std::min(count, dst_layout->run - index % dst_layout->run);
    sources[used] = src + (src_layout ? src_layout->offset(index) : done) * item;
    destinations[used] = dst + (dst_layout ? dst_layout->offset(index) : done) * item;
    sizes[used] = count * item;
    ++used;
    if (used == sources.size()) flush();
    done += count;
  }
  flush();
  // Synapse 1.24.2 synMemCopyAsyncMultiple(DRAM_TO_DRAM) can race with an immediately
  // following synMemCopyAsync(DRAM_TO_DRAM) on the same stream (P-RT-95).
  if (had_multi_d2d && synStreamSynchronize(stream) != synSuccess) {
    throw std::runtime_error("GaudiRT: strided D2D DMA stream sync failed");
  }
}

bool try_pack_strided_2d(Context& ctx, uint64_t dst, uint64_t dst_capacity_bytes,
                         const at::Tensor& src, synStreamHandle stream) {
  if (src.dim() != 2 || dst == 0 || dst % BlockCachingAllocator::kAlignment != 0) return false;
  const int64_t rows_i = src.size(0);
  const int64_t width_i = src.size(1);
  const int64_t stride0_i = src.stride(0);
  if (rows_i < 32 || width_i <= 0 || src.stride(1) != 1 || stride0_i <= width_i) return false;

  const uint64_t item = static_cast<uint64_t>(src.element_size());
  const uint64_t rows = static_cast<uint64_t>(rows_i);
  const uint64_t width_bytes = static_cast<uint64_t>(width_i) * item;
  const uint64_t stride_bytes = static_cast<uint64_t>(stride0_i) * item;
  if (width_bytes % BlockCachingAllocator::kAlignment != 0 ||
      stride_bytes % BlockCachingAllocator::kAlignment != 0) {
    return false;
  }

  const uint64_t storage_base = reinterpret_cast<uint64_t>(src.storage().data());
  const uint64_t storage_bytes = src.storage().nbytes();
  const uint64_t data_addr = reinterpret_cast<uint64_t>(src.data_ptr());
  if (storage_base == 0 || storage_base % BlockCachingAllocator::kAlignment != 0 ||
      data_addr < storage_base || data_addr - storage_base > storage_bytes) {
    return false;
  }

  const uint64_t byte_offset = data_addr - storage_base;
  const uint64_t row0_offset = (byte_offset / stride_bytes) * stride_bytes;
  const uint64_t col_bytes = byte_offset - row0_offset;
  if (col_bytes % BlockCachingAllocator::kAlignment != 0 ||
      col_bytes + width_bytes > stride_bytes) {
    return false;
  }
  if (rows > (storage_bytes - row0_offset) / stride_bytes ||
      rows * width_bytes > dst_capacity_bytes) {
    return false;
  }

  const uint64_t src_base = storage_base + row0_offset;
  const uint64_t src_span = rows * stride_bytes;
  if (src_base < dst + dst_capacity_bytes && dst < src_base + src_span) return false;

  const uint64_t u_stride = stride_bytes / sizeof(uint16_t);
  const uint64_t u_col = col_bytes / sizeof(uint16_t);
  const uint64_t u_width = width_bytes / sizeof(uint16_t);
  const uint32_t bucket = std::bit_ceil(static_cast<uint32_t>(rows));
  const uint64_t bucket_src_span = static_cast<uint64_t>(bucket) * stride_bytes;
  const bool single_bucket =
      rows == bucket ||
      (bucket <= (storage_bytes - row0_offset) / stride_bytes &&
       static_cast<uint64_t>(bucket) * width_bytes <= dst_capacity_bytes &&
       !(src_base < dst + dst_capacity_bytes && dst < src_base + bucket_src_span));

  if (single_bucket) {
    ColdRecipe* recipe = nullptr;
    {
      std::lock_guard lock(ctx.cold_mutex);
      recipe = &strided_2d_pack_recipe(ctx, bucket, u_stride, u_col, u_width);
    }
    launch_strided_2d_pack(ctx, *recipe, src_base, dst, stream);
    return true;
  }

  const uint32_t p2 = std::bit_floor(static_cast<uint32_t>(rows));
  const uint64_t tail = rows - p2;
  ColdRecipe* recipe = nullptr;
  {
    std::lock_guard lock(ctx.cold_mutex);
    recipe = &strided_2d_pack_recipe(ctx, p2, u_stride, u_col, u_width);
  }
  launch_strided_2d_pack(ctx, *recipe, src_base, dst, stream);
  launch_strided_2d_pack(ctx, *recipe, src_base + tail * stride_bytes,
                         dst + tail * width_bytes, stream);
  return true;
}

}  // namespace gaudirt::generic
