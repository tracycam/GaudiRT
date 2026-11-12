// SPDX-License-Identifier: Apache-2.0
// Verify multi-tile fused prefill paged attention graph DAG, launch counts,
// non-aliasing per-tile bindings, bit-exact CPU reference equivalence, and
// zero steady-state Synapse event creation under mock_synapse.

#include "attention_plan.hpp"
#include "bridge_internal.hpp"
#include "paged_attention.hpp"
#include "paged_attention_tiles.hpp"
#include "paged_kv_pool.hpp"
#include "paged_metadata.hpp"
#include "segmented_runner.hpp"

#include <hccl.h>
#include <perf_lib_layer_params.h>
#include <ATen/ATen.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

uint16_t f32_to_bf16(float value) {
  const auto bits = std::bit_cast<uint32_t>(value);
  if ((bits & 0x7fffffffU) > 0x7f800000U) return static_cast<uint16_t>((bits >> 16) | 0x0040U);
  return static_cast<uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

float bf16_to_f32(uint16_t value) {
  return std::bit_cast<float>(static_cast<uint32_t>(value) << 16);
}

size_t dtype_bytes(synDataType dt) {
  switch (dt) {
    case syn_type_bf16:
    case syn_type_fp16:
    case syn_type_int16:
    case syn_type_uint16:
      return 2;
    case syn_type_float:
    case syn_type_int32:
    case syn_type_uint32:
      return 4;
    case syn_type_int64:
    case syn_type_uint64:
      return 8;
    default:
      return 1;
  }
}

struct MockSection {
  bool persistent = false;
  bool rmw = false;
  uint64_t id = 0;
};

struct MockTensor {
  std::string name;
  synDataType dtype = syn_type_na;
  synTensorType tensor_type = DATA_TENSOR;
  uint32_t dims = 0;
  std::array<uint64_t, HABANA_DIM_MAX> sizes{};
  std::array<uint64_t, HABANA_DIM_MAX> min_sizes{};
  MockSection* section = nullptr;
  uint64_t section_offset = 0;
  bool has_perm = false;
  synTensorPermutation perm{};
  uint64_t id = 0;

  uint64_t elements() const {
    if (dims == 0) return 0;
    uint64_t n = 1;
    for (uint32_t i = 0; i < dims; ++i) n *= sizes[i];
    return n;
  }
  uint64_t byte_size() const { return elements() * dtype_bytes(dtype); }
};

struct MockNode {
  std::string guid;
  std::string name;
  std::vector<MockTensor*> inputs;
  std::vector<MockTensor*> outputs;
  std::vector<uint8_t> params;
};

struct MockGraph {
  std::vector<MockNode> nodes;
};

struct MockRecipeTensor {
  MockTensor spec;
  uint64_t section_id = 0;
  bool is_input = false;
};

struct MockRecipe {
  std::string name;
  std::vector<MockNode> nodes;
  std::unordered_map<MockTensor*, MockTensor> tensor_snapshots;
  std::vector<MockRecipeTensor> launch_tensors;
  uint64_t workspace = 4096;
};

struct LaunchRecord {
  const MockRecipe* recipe = nullptr;
  uint64_t workspace = 0;
  std::vector<synLaunchTensorInfo> bindings;
  size_t persistent_fp32_state_bindings = 0;
};

struct MockEvent {
  bool recorded = false;
};

struct MockArena {
  // Sparse 4 KiB page map so 80 GiB virtual HBM & 512 MiB fixed KV descriptors
  // can be addressed at zero physical RAM cost unless touched.
  static constexpr uint64_t kPageBits = 12;
  static constexpr uint64_t kPageSize = uint64_t{1} << kPageBits;
  uint64_t next_base = 0x10000000ULL;
  std::unordered_map<uint64_t, std::unique_ptr<std::array<uint8_t, kPageSize>>> pages;

  uint64_t alloc(uint64_t bytes) {
    const uint64_t aligned = (bytes + 4095ULL) & ~uint64_t{4095};
    const uint64_t addr = next_base;
    next_base += aligned + 4096ULL;
    return addr;
  }

  uint8_t* byte_ptr(uint64_t addr, bool create) {
    const uint64_t page = addr >> kPageBits;
    const uint64_t off = addr & (kPageSize - 1);
    auto it = pages.find(page);
    if (it == pages.end()) {
      if (!create) return nullptr;
      auto buf = std::make_unique<std::array<uint8_t, kPageSize>>();
      buf->fill(0);
      it = pages.emplace(page, std::move(buf)).first;
    }
    return it->second->data() + off;
  }

  void write(uint64_t dst, const void* src, uint64_t bytes) {
    const auto* p = static_cast<const uint8_t*>(src);
    for (uint64_t i = 0; i < bytes; ++i) *byte_ptr(dst + i, true) = p[i];
  }

  void read(void* dst, uint64_t src, uint64_t bytes) {
    auto* p = static_cast<uint8_t*>(dst);
    for (uint64_t i = 0; i < bytes; ++i) {
      const uint8_t* b = byte_ptr(src + i, false);
      p[i] = b ? *b : 0;
    }
  }

  void copy_d2d(uint64_t dst, uint64_t src, uint64_t bytes) {
    std::vector<uint8_t> tmp(bytes);
    read(tmp.data(), src, bytes);
    write(dst, tmp.data(), bytes);
  }

  void memset_d32(uint64_t dst, uint32_t value, size_t count) {
    if (value == 0) {
      // Only zero already-materialized pages so huge pool clears stay instant!
      const uint64_t end = dst + count * 4;
      for (auto& [page, buf] : pages) {
        const uint64_t p_start = page << kPageBits;
        const uint64_t p_end = p_start + kPageSize;
        const uint64_t lo = std::max(dst, p_start);
        const uint64_t hi = std::min(end, p_end);
        if (lo < hi) std::memset(buf->data() + (lo - p_start), 0, hi - lo);
      }
      return;
    }
    for (size_t i = 0; i < count; ++i) write(dst + i * 4, &value, 4);
  }
};

struct MockState {
  MockArena arena;
  uint64_t next_section_id = 1;
  uint64_t next_tensor_id = 1;
  uint64_t next_stream_id = 1;
  uint64_t event_creates = 0;
  uint64_t event_destroys = 0;
  uint64_t node_creates = 0;
  uint64_t node_with_id_creates = 0;
  uint64_t multi_dma_calls = 0;
  bool execute_math = false;
  std::vector<LaunchRecord> launches;
  std::vector<const MockRecipe*> compiled_recipes;

  void reset_counters() {
    event_creates = 0;
    event_destroys = 0;
    node_creates = 0;
    node_with_id_creates = 0;
    multi_dma_calls = 0;
    launches.clear();
  }
};

MockState& mock() {
  static MockState state;
  return state;
}

void verify_recipe_structure(const MockRecipe& recipe) {
  bool has_tile_softmax = false;
  for (const auto& n : recipe.nodes) {
    if (n.guid == "gk_attn_tile_softmax_bf16_v1") has_tile_softmax = true;
  }
  if (!has_tile_softmax) return;

  // Verify sequential multi-tile 4D zero-reshape online-softmax DAG invariants.
  std::vector<const MockNode*> inits;
  std::vector<const MockNode*> merges;
  std::vector<const MockNode*> softmaxes;
  std::vector<const MockNode*> packs;
  std::vector<const MockNode*> finalizes;
  std::vector<const MockNode*> constants;
  std::vector<const MockNode*> reshapes;
  std::vector<const MockNode*> transposes;
  std::vector<const MockNode*> gathers;
  std::set<uint64_t> header_sections;
  std::set<uint64_t> key_index_sections;
  std::set<uint64_t> direct_kv_sections;

  for (const auto& n : recipe.nodes) {
    if (n.guid == "gk_attn_online_init_f32_v1") inits.push_back(&n);
    else if (n.guid == "gk_attn_online_merge_f32_v1") merges.push_back(&n);
    else if (n.guid == "gk_attn_tile_softmax_bf16_v1") softmaxes.push_back(&n);
    else if (n.guid == "gk_attn_pack_stats_f32_v1") packs.push_back(&n);
    else if (n.guid == "gk_attn_online_finalize_bf16_v1") finalizes.push_back(&n);
    else if (n.guid == "constant_f32") constants.push_back(&n);
    else if (n.guid == "reshape") reshapes.push_back(&n);
    else if (n.guid == "transpose") transposes.push_back(&n);
    else if (n.guid == "gather_fwd_bf16") {
      gathers.push_back(&n);
      assert(n.inputs.size() == 2);
      const auto& idx_t = recipe.tensor_snapshots.at(n.inputs[1]);
      assert(idx_t.section && idx_t.section->persistent);
      if (idx_t.section_offset == 0) key_index_sections.insert(idx_t.section->id);
    }
  }

  const size_t chunk_tiles = softmaxes.size();
  const bool first_chunk = !inits.empty();
  const bool finalize_last = !finalizes.empty();
  const bool direct_kv = gathers.empty();
  const uint64_t rows = recipe.tensor_snapshots.at(softmaxes[0]->inputs[0]).sizes[1];
  const size_t io_transforms = (rows > 1 ? 1U : 0U) + (finalize_last && rows > 1 ? 1U : 0U);
  assert(chunk_tiles >= 1 && chunk_tiles <= gaudirt::generic::kMaxFusedAttentionTiles);
  assert(packs.empty() && "packed 3D tile stats eliminate redundant gk_attn_pack_stats_f32_v1 nodes");
  assert(constants.empty() && "gk_attn_online_init_f32_v1 eliminates redundant constant_f32 zero nodes");
  assert(reshapes.size() == 2 * chunk_tiles + io_transforms &&
         "only 2 KV reshapes per tile plus query/output 3D<->4D reshapes; zero score/prob reshapes");
  assert(transposes.size() == (direct_kv ? 2 * chunk_tiles : 0) + io_transforms &&
         "explicit 3D transpose nodes avoid per-tile query re-transposition");
  if (direct_kv) {
    for (const auto* tn : transposes) {
      assert(tn->inputs.size() == 1 && tn->outputs.size() == 1);
      const auto& in_snap = recipe.tensor_snapshots.at(tn->inputs[0]);
      if (in_snap.section && in_snap.section->persistent) {
        direct_kv_sections.insert(in_snap.section->id);
      }
    }
    assert(direct_kv_sections.size() == 2 * chunk_tiles + (rows > 1 ? 1U : 0U) &&
           "each direct KV tile + query must bind its own persistent input section");
  } else {
    assert(gathers.size() == 2 * chunk_tiles);
    assert(key_index_sections.size() == chunk_tiles && "each tile must use a distinct persistent index section");
  }
  if (first_chunk) {
    assert(inits.size() == 1 && merges.size() + 1 == chunk_tiles);
  } else {
    assert(inits.empty() && merges.size() == chunk_tiles);
  }

  const MockTensor* shared_positions = softmaxes[0]->inputs[1];
  for (size_t t = 0; t < chunk_tiles; ++t) {
    assert(softmaxes[t]->inputs.size() == 3 && softmaxes[t]->outputs.size() == 3);
    assert(recipe.tensor_snapshots.at(softmaxes[t]->inputs[0]).dims == 4);
    assert(recipe.tensor_snapshots.at(softmaxes[t]->outputs[0]).dims == 4);
    assert(softmaxes[t]->inputs[1] == shared_positions && "all tiles in chunk share block positions tensor");
    const auto& hdr = recipe.tensor_snapshots.at(softmaxes[t]->inputs[2]);
    assert(hdr.section && hdr.section->persistent);
    assert(header_sections.insert(hdr.section->id).second && "each tile must bind its own persistent header section");
    assert(recipe.tensor_snapshots.at(softmaxes[t]->outputs[1]).dims == 3);
    assert(recipe.tensor_snapshots.at(softmaxes[t]->outputs[2]).dims == 3);

    if (first_chunk && t == 0) {
      assert(inits[0]->inputs.size() == 3 && inits[0]->outputs.size() == 3);
      assert(inits[0]->inputs[0] == softmaxes[0]->outputs[1]);
      assert(inits[0]->inputs[1] == softmaxes[0]->outputs[2]);
    } else {
      const size_t mi = first_chunk ? (t - 1) : t;
      assert(merges[mi]->inputs.size() == 6 && merges[mi]->outputs.size() == 3);
      assert(merges[mi]->inputs[3] == softmaxes[t]->outputs[1]);
      assert(merges[mi]->inputs[4] == softmaxes[t]->outputs[2]);
      const MockNode* prev = (first_chunk && t == 1) ? inits[0] : (t > 0 ? merges[mi - 1] : nullptr);
      if (prev) {
        assert(merges[mi]->inputs[0] == prev->outputs[0] && "tile k new_m must feed tile k+1 old_m");
        assert(merges[mi]->inputs[1] == prev->outputs[1] && "tile k new_l must feed tile k+1 old_l");
        assert(merges[mi]->inputs[2] == prev->outputs[2] && "tile k new_o must feed tile k+1 old_o");
        for (int s = 0; s < 3; ++s) {
          assert(recipe.tensor_snapshots.at(prev->outputs[s]).section == nullptr &&
                 "intermediate running state (m, l, o) inside fused chunk must be virtual (non-persistent)");
        }
      } else {
        for (int s = 0; s < 3; ++s) {
          assert(recipe.tensor_snapshots.at(merges[0]->inputs[s]).section != nullptr &&
                 recipe.tensor_snapshots.at(merges[0]->inputs[s]).section->persistent);
        }
      }
    }
  }

  const MockNode* last_state_node = (first_chunk && chunk_tiles == 1) ? inits[0] : merges.back();
  if (!finalizes.empty()) {
    assert(finalizes.size() == 1);
    const auto* fin = finalizes[0];
    assert(fin->inputs.size() == 2 && fin->outputs.size() == 1);
    assert(fin->inputs[0] == last_state_node->outputs[1] && "finalize sums must come from last tile state new_l");
    assert(fin->inputs[1] == last_state_node->outputs[2] && "finalize acc must come from last tile state new_o");
    const auto& out_spec = recipe.tensor_snapshots.at(fin->outputs[0]);
    assert(out_spec.dtype == syn_type_bf16);
    if (rows == 1) {
      assert(out_spec.section && out_spec.section->persistent);
    } else {
      assert(out_spec.section == nullptr);
      const auto& out_t_spec = recipe.tensor_snapshots.at(transposes.back()->outputs[0]);
      assert(out_t_spec.section && out_t_spec.section->persistent && out_t_spec.dtype == syn_type_bf16);
    }
    for (int s = 0; s < 3; ++s) {
      assert(recipe.tensor_snapshots.at(last_state_node->outputs[s]).section == nullptr &&
             "finalized chunk must not write running (m, l, o) to persistent HBM");
    }
  } else {
    for (int s = 0; s < 3; ++s) {
      assert(recipe.tensor_snapshots.at(last_state_node->outputs[s]).section != nullptr &&
             recipe.tensor_snapshots.at(last_state_node->outputs[s]).section->persistent);
    }
  }
}

void execute_recipe_on_cpu(const MockRecipe& recipe, const synLaunchTensorInfo* launch_infos, uint32_t count) {
  std::unordered_map<uint64_t, uint64_t> id_to_addr;
  for (uint32_t i = 0; i < count; ++i) {
    id_to_addr[launch_infos[i].tensorId] = launch_infos[i].pTensorAddress;
  }

  std::unordered_map<MockTensor*, std::vector<uint8_t>> buffers;
  // Load persistent input tensors from mock HBM (inverting memory permutation if set!).
  auto load_tensor = [&](MockTensor* handle) -> std::vector<uint8_t>& {
    auto it = buffers.find(handle);
    if (it != buffers.end()) return it->second;
    const auto& spec = recipe.tensor_snapshots.at(handle);
    std::vector<uint8_t> logical(spec.byte_size(), 0);
    if (spec.section && spec.section->persistent) {
      const uint64_t addr = id_to_addr.at(spec.id);
      std::vector<uint8_t> raw(spec.byte_size(), 0);
      mock().arena.read(raw.data(), addr, raw.size());
      if (!spec.has_perm) {
        logical = std::move(raw);
      } else if (spec.dims == 4 && spec.perm.permutation[0] == 0 && spec.perm.permutation[1] == 2 &&
                 spec.perm.permutation[2] == 3 && spec.perm.permutation[3] == 1) {
        // Physical memory layout is [d, group, kv_heads, rows] (i.e. row-major [rows, kv_heads, group, d]).
        // Logical layout is [d, rows, group, kv_heads] (FCD = d).
        const uint64_t d = spec.sizes[0], rows = spec.sizes[1], group = spec.sizes[2], kvh = spec.sizes[3];
        const size_t elem_sz = dtype_bytes(spec.dtype);
        for (uint64_t kh = 0; kh < kvh; ++kh)
          for (uint64_t g = 0; g < group; ++g)
            for (uint64_t r = 0; r < rows; ++r) {
              const uint64_t phys_idx = ((r * kvh + kh) * group + g) * d;
              const uint64_t log_idx = ((kh * group + g) * rows + r) * d;
              std::memcpy(logical.data() + log_idx * elem_sz, raw.data() + phys_idx * elem_sz, d * elem_sz);
            }
      } else {
        assert(false && "unexpected input permutation in mock executor");
      }
    }
    return buffers.emplace(handle, std::move(logical)).first->second;
  };

  auto alloc_tensor = [&](MockTensor* handle) -> std::vector<uint8_t>& {
    const auto& spec = recipe.tensor_snapshots.at(handle);
    auto& buf = buffers[handle];
    buf.assign(spec.byte_size(), 0);
    return buf;
  };

  for (const auto& node : recipe.nodes) {
    if (node.guid == "gk_attn_tile_indices_i32_v1") {
      const auto& map_b = load_tensor(node.inputs[0]);
      const auto& req_b = load_tensor(node.inputs[1]);
      const auto& work_b = load_tensor(node.inputs[2]);
      const auto& ctx_b = load_tensor(node.inputs[3]);
      auto& out_b = alloc_tensor(node.outputs[0]);
      const auto* mapping = reinterpret_cast<const int32_t*>(map_b.data());
      const auto* requests = reinterpret_cast<const int32_t*>(req_b.data());
      const auto* work = reinterpret_cast<const int32_t*>(work_b.data());
      const int32_t context = *reinterpret_cast<const int32_t*>(ctx_b.data());
      auto* out = reinterpret_cast<int32_t*>(out_b.data());
      const uint64_t tasks = recipe.tensor_snapshots.at(node.inputs[2]).sizes[1];
      for (uint64_t t = 0; t < tasks; ++t) {
        const int32_t first_row = work[t * 4 + 0];
        const int32_t first_key = work[t * 4 + 1];
        const int32_t lanes = work[t * 4 + 2];
        const int32_t base = work[t * 4 + 3];
        const int32_t req = requests[first_row];
        for (int32_t l = 0; l < 64; ++l) {
          int32_t slot = 0;
          if (l < lanes) slot = mapping[req * context + first_key + l];
          out[t * 64 + l] = base + slot;
        }
      }
    } else if (node.guid == "gk_kv_store_head_major_bf16_v1") {
      const auto& k_b = load_tensor(node.inputs[0]);
      const auto& v_b = load_tensor(node.inputs[1]);
      const auto& loc_b = load_tensor(node.inputs[2]);
      const auto& hdr_b = load_tensor(node.inputs[3]);
      const auto* k_in = reinterpret_cast<const uint16_t*>(k_b.data());
      const auto* v_in = reinterpret_cast<const uint16_t*>(v_b.data());
      const auto* locs = reinterpret_cast<const int32_t*>(loc_b.data());
      const auto* hdr = reinterpret_cast<const int32_t*>(hdr_b.data());
      const int32_t valid_rows = hdr[0], slots = hdr[1], kvh = hdr[2];
      const uint64_t d = recipe.tensor_snapshots.at(node.outputs[0]).sizes[0];
      const uint64_t width = static_cast<uint64_t>(kvh) * d;
      const uint64_t pool_addr = id_to_addr.at(recipe.tensor_snapshots.at(node.outputs[0]).id);
      for (int32_t r = 0; r < valid_rows; ++r) {
        const int32_t slot = locs[r];
        for (int32_t h = 0; h < kvh; ++h) {
          const uint64_t k_dst = pool_addr + (static_cast<uint64_t>(h) * slots + slot) * d * 2;
          const uint64_t v_dst = pool_addr + (static_cast<uint64_t>(kvh + h) * slots + slot) * d * 2;
          mock().arena.write(k_dst, k_in + static_cast<uint64_t>(r) * width + h * d, d * 2);
          mock().arena.write(v_dst, v_in + static_cast<uint64_t>(r) * width + h * d, d * 2);
        }
      }
      // Mark pool handle if consumed later in fused_recipe.
      buffers.erase(node.outputs[0]);
    } else if (node.guid == "constant_f32") {
      auto& out_b = alloc_tensor(node.outputs[0]);
      ns_ConstantKernel::ParamsV3 p{};
      std::memcpy(&p, node.params.data(), sizeof(p));
      auto* out = reinterpret_cast<float*>(out_b.data());
      std::fill_n(out, out_b.size() / 4, p.constant.f);
    } else if (node.guid == "gather_fwd_bf16") {
      const auto& pool_spec = recipe.tensor_snapshots.at(node.inputs[0]);
      const uint64_t pool_addr = id_to_addr.at(pool_spec.id);
      const auto& idx_b = load_tensor(node.inputs[1]);
      auto& out_b = alloc_tensor(node.outputs[0]);
      const uint64_t d = pool_spec.sizes[0];
      const auto* indices = reinterpret_cast<const int32_t*>(idx_b.data());
      const uint64_t n = idx_b.size() / 4;
      for (uint64_t i = 0; i < n; ++i) {
        mock().arena.read(out_b.data() + i * d * 2, pool_addr + static_cast<uint64_t>(indices[i]) * d * 2, d * 2);
      }
    } else if (node.guid == "reshape") {
      const auto& in_b = load_tensor(node.inputs[0]);
      buffers[node.outputs[0]] = in_b;
    } else if (node.guid == "transpose") {
      synTransposeParams p{};
      std::memcpy(&p, node.params.data(), sizeof(p));
      const auto& in_spec = recipe.tensor_snapshots.at(node.inputs[0]);
      const auto& out_spec = recipe.tensor_snapshots.at(node.outputs[0]);
      assert(p.tensorDim == 3 && in_spec.dims == 3 && out_spec.dims == 3 &&
             p.permutation[0] == TPD_Channel && p.permutation[1] == TPD_Height &&
             p.permutation[2] == TPD_Width);
      const auto& in_b = load_tensor(node.inputs[0]);
      auto& out_b = alloc_tensor(node.outputs[0]);
      const uint64_t d0 = in_spec.sizes[0], d1 = in_spec.sizes[1], d2 = in_spec.sizes[2];
      const size_t elem_sz = dtype_bytes(in_spec.dtype);
      for (uint64_t i2 = 0; i2 < d2; ++i2)
        for (uint64_t i1 = 0; i1 < d1; ++i1) {
          const uint64_t src_off = (i2 * d1 + i1) * d0 * elem_sz;
          const uint64_t dst_off = (i1 * d2 + i2) * d0 * elem_sz;
          std::memcpy(out_b.data() + dst_off, in_b.data() + src_off, d0 * elem_sz);
        }
    } else if (node.guid == "split") {
      synSplitParams p{};
      std::memcpy(&p, node.params.data(), sizeof(p));
      assert(p.axis == 0);
      const auto& in_spec = recipe.tensor_snapshots.at(node.inputs[0]);
      assert(in_spec.dims == 2);
      const auto& in_b = load_tensor(node.inputs[0]);
      const uint64_t total_cols = in_spec.sizes[0], rows = in_spec.sizes[1];
      const size_t elem_sz = dtype_bytes(in_spec.dtype);
      uint64_t col_cursor = 0;
      for (MockTensor* out_h : node.outputs) {
        const auto& out_spec = recipe.tensor_snapshots.at(out_h);
        assert(out_spec.dims == 2 && out_spec.sizes[1] == rows);
        const uint64_t part_cols = out_spec.sizes[0];
        auto& out_b = alloc_tensor(out_h);
        for (uint64_t r = 0; r < rows; ++r) {
          std::memcpy(out_b.data() + r * part_cols * elem_sz,
                      in_b.data() + (r * total_cols + col_cursor) * elem_sz,
                      part_cols * elem_sz);
        }
        col_cursor += part_cols;
      }
      assert(col_cursor == total_cols);
    } else if (node.guid == "batch_gemm") {
      synGEMMParams p{};
      std::memcpy(&p, node.params.data(), sizeof(p));
      const auto& a_spec = recipe.tensor_snapshots.at(node.inputs[0]);
      const auto& b_spec = recipe.tensor_snapshots.at(node.inputs[1]);
      const auto& c_spec = recipe.tensor_snapshots.at(node.outputs[0]);
      const auto& a_b = load_tensor(node.inputs[0]);
      const auto& b_b = load_tensor(node.inputs[1]);
      auto& c_b = alloc_tensor(node.outputs[0]);
      const auto* a = reinterpret_cast<const uint16_t*>(a_b.data());
      const auto* b = reinterpret_cast<const uint16_t*>(b_b.data());
      const uint64_t b2 = c_spec.sizes[2], b3 = c_spec.sizes[3];
      const uint64_t a_b2 = a_spec.sizes[2], b_b2 = b_spec.sizes[2];
      if (!p.transpose_a && p.transpose_b) {
        // A: [K=d, M=rows, group, kvh], B: [K=d, N=kv, 1, kvh] -> C: [N=kv, M=rows, group, kvh] FP32
        const uint64_t K = a_spec.sizes[0], M = a_spec.sizes[1], N = b_spec.sizes[1];
        auto* c = reinterpret_cast<float*>(c_b.data());
        for (uint64_t i3 = 0; i3 < b3; ++i3)
          for (uint64_t i2 = 0; i2 < b2; ++i2) {
            const auto* a_slice = a + ((i3 * a_b2 + (a_b2 == 1 ? 0 : i2)) * M) * K;
            const auto* b_slice = b + ((i3 * b_b2 + (b_b2 == 1 ? 0 : i2)) * N) * K;
            auto* c_slice = c + ((i3 * b2 + i2) * M) * N;
            for (uint64_t m = 0; m < M; ++m)
              for (uint64_t n_idx = 0; n_idx < N; ++n_idx) {
                float acc = 0.f;
                for (uint64_t k = 0; k < K; ++k)
                  acc += bf16_to_f32(a_slice[m * K + k]) * bf16_to_f32(b_slice[n_idx * K + k]);
                c_slice[m * N + n_idx] = acc;
              }
          }
      } else if (!p.transpose_a && !p.transpose_b) {
        // A: [K=kv, M=rows, group, kvh], B: [N=d, K=kv, 1, kvh] -> C: [N=d, M=rows, group, kvh] (FP32 or BF16)
        const uint64_t K = a_spec.sizes[0], M = a_spec.sizes[1], N = b_spec.sizes[0];
        for (uint64_t i3 = 0; i3 < b3; ++i3)
          for (uint64_t i2 = 0; i2 < b2; ++i2) {
            const auto* a_slice = a + ((i3 * a_b2 + (a_b2 == 1 ? 0 : i2)) * M) * K;
            const auto* b_slice = b + ((i3 * b_b2 + (b_b2 == 1 ? 0 : i2)) * K) * N;
            for (uint64_t m = 0; m < M; ++m)
              for (uint64_t n_idx = 0; n_idx < N; ++n_idx) {
                float acc = 0.f;
                for (uint64_t k = 0; k < K; ++k)
                  acc += bf16_to_f32(a_slice[m * K + k]) * bf16_to_f32(b_slice[k * N + n_idx]);
                const uint64_t out_idx = (((i3 * b2 + i2) * M + m) * N + n_idx);
                if (c_spec.dtype == syn_type_float) {
                  reinterpret_cast<float*>(c_b.data())[out_idx] = acc;
                } else {
                  reinterpret_cast<uint16_t*>(c_b.data())[out_idx] = f32_to_bf16(acc);
                }
              }
          }
      } else {
        assert(false && "unsupported GEMM transpose flags");
      }
    } else if (node.guid == "gk_attn_direct_softmax_bf16_v1") {
      float scale = 0.f;
      std::memcpy(&scale, node.params.data(), sizeof(scale));
      const auto& s_spec = recipe.tensor_snapshots.at(node.inputs[0]);
      const uint64_t keys = s_spec.sizes[0], rows = s_spec.sizes[1];
      const uint64_t heads = s_spec.dims == 4 ? (s_spec.sizes[2] * s_spec.sizes[3]) : s_spec.sizes[2];
      const auto* scores = reinterpret_cast<const float*>(load_tensor(node.inputs[0]).data());
      const auto* positions = reinterpret_cast<const int32_t*>(load_tensor(node.inputs[1]).data());
      const auto* hdr = reinterpret_cast<const int32_t*>(load_tensor(node.inputs[2]).data());
      auto* probs = reinterpret_cast<uint16_t*>(alloc_tensor(node.outputs[0]).data());
      const int32_t valid_rows = hdr[0], valid_keys = hdr[1], key_start = hdr[2];
      for (uint64_t h = 0; h < heads; ++h)
        for (uint64_t r = 0; r < rows; ++r) {
          uint16_t* dst = probs + (h * rows + r) * keys;
          if (static_cast<int32_t>(r) >= valid_rows) {
            std::fill_n(dst, keys, uint16_t{0});
            continue;
          }
          const int32_t last_key = std::min(positions[r] - key_start, valid_keys - 1);
          float m = -std::numeric_limits<float>::infinity();
          const float* src = scores + (h * rows + r) * keys;
          for (uint64_t k = 0; k < keys; ++k)
            if (static_cast<int32_t>(k) <= last_key) m = std::max(m, src[k] * scale);
          const float origin = std::isinf(m) ? 0.f : m;
          float sum = 0.f;
          std::vector<float> exps(keys, 0.f);
          for (uint64_t k = 0; k < keys; ++k) {
            if (static_cast<int32_t>(k) <= last_key) {
              exps[k] = std::exp(src[k] * scale - origin);
              sum += exps[k];
            }
          }
          const float inv = sum > 0.f ? (1.f / sum) : 0.f;
          for (uint64_t k = 0; k < keys; ++k) dst[k] = f32_to_bf16(exps[k] * inv);
        }
    } else if (node.guid == "gk_attn_tile_softmax_bf16_v1") {
      float scale = 0.f;
      std::memcpy(&scale, node.params.data(), sizeof(scale));
      const auto& s_spec = recipe.tensor_snapshots.at(node.inputs[0]);
      const uint64_t keys = s_spec.sizes[0], rows = s_spec.sizes[1];
      const uint64_t heads = s_spec.dims == 4 ? (s_spec.sizes[2] * s_spec.sizes[3]) : s_spec.sizes[2];
      const auto* scores = reinterpret_cast<const float*>(load_tensor(node.inputs[0]).data());
      const auto* positions = reinterpret_cast<const int32_t*>(load_tensor(node.inputs[1]).data());
      const auto* hdr = reinterpret_cast<const int32_t*>(load_tensor(node.inputs[2]).data());
      auto* probs = reinterpret_cast<uint16_t*>(alloc_tensor(node.outputs[0]).data());
      const auto& stat_spec = recipe.tensor_snapshots.at(node.outputs[1]);
      const uint64_t padded = (rows + 63) / 64 * 64;
      const bool packed_stats = (stat_spec.dims == 2) ||
                                (stat_spec.dims == 3 && stat_spec.sizes[0] == padded &&
                                 stat_spec.sizes[1] * stat_spec.sizes[2] == heads);
      auto* out_m = reinterpret_cast<float*>(alloc_tensor(node.outputs[1]).data());
      auto* out_l = reinterpret_cast<float*>(alloc_tensor(node.outputs[2]).data());
      const int32_t valid_rows = hdr[0], valid_keys = hdr[1], key_start = hdr[2];
      for (uint64_t h = 0; h < heads; ++h) {
        if (packed_stats) {
          for (uint64_t r = rows; r < padded; ++r) {
            out_m[h * padded + r] = -std::numeric_limits<float>::infinity();
            out_l[h * padded + r] = 0.f;
          }
        }
        for (uint64_t r = 0; r < rows; ++r) {
          float m = -std::numeric_limits<float>::infinity();
          float sum = 0.f;
          uint16_t* dst = probs + (h * rows + r) * keys;
          if (static_cast<int32_t>(r) < valid_rows) {
            const int32_t last_key = std::min(positions[r] - key_start, valid_keys - 1);
            const float* src = scores + (h * rows + r) * keys;
            for (uint64_t k = 0; k < keys; ++k)
              if (static_cast<int32_t>(k) <= last_key) m = std::max(m, src[k] * scale);
            const float origin = std::isinf(m) ? 0.f : m;
            std::vector<float> exps(keys, 0.f);
            for (uint64_t k = 0; k < keys; ++k) {
              if (static_cast<int32_t>(k) <= last_key) {
                exps[k] = std::exp(src[k] * scale - origin);
                sum += exps[k];
              }
            }
            const float inv = sum > 0.f ? (1.f / sum) : 0.f;
            for (uint64_t k = 0; k < keys; ++k) dst[k] = f32_to_bf16(exps[k] * inv);
          } else {
            std::fill_n(dst, keys, uint16_t{0});
          }
          if (packed_stats) {
            out_m[h * padded + r] = m;
            out_l[h * padded + r] = sum;
          } else {
            std::fill_n(out_m + (h * rows + r) * 64, 64, m);
            std::fill_n(out_l + (h * rows + r) * 64, 64, sum);
          }
        }
      }
    } else if (node.guid == "gk_attn_pack_stats_f32_v1") {
      const auto& in_spec = recipe.tensor_snapshots.at(node.inputs[0]);
      const uint64_t rows = in_spec.sizes[1], heads = in_spec.sizes[2];
      const uint64_t padded = (rows + 63) / 64 * 64;
      const auto* exp_m = reinterpret_cast<const float*>(load_tensor(node.inputs[0]).data());
      const auto* exp_l = reinterpret_cast<const float*>(load_tensor(node.inputs[1]).data());
      auto* tile_m = reinterpret_cast<float*>(alloc_tensor(node.outputs[0]).data());
      auto* tile_l = reinterpret_cast<float*>(alloc_tensor(node.outputs[1]).data());
      for (uint64_t h = 0; h < heads; ++h)
        for (uint64_t r = 0; r < padded; ++r) {
          tile_m[h * padded + r] = r < rows ? exp_m[(h * rows + r) * 64] : -std::numeric_limits<float>::infinity();
          tile_l[h * padded + r] = r < rows ? exp_l[(h * rows + r) * 64] : 0.f;
        }
    } else if (node.guid == "gk_attn_online_init_f32_v1") {
      const auto& o_spec = recipe.tensor_snapshots.at(node.inputs[2]);
      const uint64_t d = o_spec.sizes[0], rows = o_spec.sizes[1];
      const uint64_t heads = o_spec.dims == 4 ? (o_spec.sizes[2] * o_spec.sizes[3]) : o_spec.sizes[2];
      const uint64_t padded = (rows + 63) / 64 * 64;
      const auto* tile_m = reinterpret_cast<const float*>(load_tensor(node.inputs[0]).data());
      const auto* tile_l = reinterpret_cast<const float*>(load_tensor(node.inputs[1]).data());
      const auto* tile_o = reinterpret_cast<const float*>(load_tensor(node.inputs[2]).data());
      auto* new_m = reinterpret_cast<float*>(alloc_tensor(node.outputs[0]).data());
      auto* new_l = reinterpret_cast<float*>(alloc_tensor(node.outputs[1]).data());
      auto* new_o = reinterpret_cast<float*>(alloc_tensor(node.outputs[2]).data());
      const float ni = -std::numeric_limits<float>::infinity();
      for (uint64_t h = 0; h < heads; ++h)
        for (uint64_t r = 0; r < padded; ++r) {
          const uint64_t idx = h * padded + r;
          const float m1 = tile_m[idx], l1 = tile_l[idx];
          new_m[idx] = l1 > 0.f ? m1 : ni;
          new_l[idx] = l1 > 0.f ? l1 : 0.f;
          if (r < rows) {
            const float bb = l1 > 0.f ? l1 : 0.f;
            const uint64_t base = (h * rows + r) * d;
            for (uint64_t c = 0; c < d; ++c) new_o[base + c] = bb * tile_o[base + c];
          }
        }
    } else if (node.guid == "gk_attn_online_merge_f32_v1") {
      const auto& o_spec = recipe.tensor_snapshots.at(node.inputs[2]);
      const uint64_t d = o_spec.sizes[0], rows = o_spec.sizes[1];
      const uint64_t heads = o_spec.dims == 4 ? (o_spec.sizes[2] * o_spec.sizes[3]) : o_spec.sizes[2];
      const uint64_t padded = (rows + 63) / 64 * 64;
      const auto* old_m = reinterpret_cast<const float*>(load_tensor(node.inputs[0]).data());
      const auto* old_l = reinterpret_cast<const float*>(load_tensor(node.inputs[1]).data());
      const auto* old_o = reinterpret_cast<const float*>(load_tensor(node.inputs[2]).data());
      const auto* tile_m = reinterpret_cast<const float*>(load_tensor(node.inputs[3]).data());
      const auto* tile_l = reinterpret_cast<const float*>(load_tensor(node.inputs[4]).data());
      const auto* tile_o = reinterpret_cast<const float*>(load_tensor(node.inputs[5]).data());
      auto* new_m = reinterpret_cast<float*>(alloc_tensor(node.outputs[0]).data());
      auto* new_l = reinterpret_cast<float*>(alloc_tensor(node.outputs[1]).data());
      auto* new_o = reinterpret_cast<float*>(alloc_tensor(node.outputs[2]).data());
      const float ni = -std::numeric_limits<float>::infinity();
      for (uint64_t h = 0; h < heads; ++h)
        for (uint64_t r = 0; r < padded; ++r) {
          const uint64_t idx = h * padded + r;
          const float m0 = old_m[idx], l0 = old_l[idx];
          const float m1 = tile_m[idx], l1 = tile_l[idx];
          const float v0 = l0 > 0.f ? m0 : ni;
          const float v1 = l1 > 0.f ? m1 : ni;
          const float m = std::max(v0, v1);
          const float origin = (m == ni) ? 0.f : m;
          const float a = l0 > 0.f ? std::exp(m0 - origin) : 0.f;
          const float b = l1 > 0.f ? std::exp(m1 - origin) : 0.f;
          new_m[idx] = m;
          new_l[idx] = a * l0 + b * l1;
          if (r < rows) {
            const float bb = b * l1;
            const uint64_t base = (h * rows + r) * d;
            for (uint64_t c = 0; c < d; ++c) new_o[base + c] = a * old_o[base + c] + bb * tile_o[base + c];
          }
        }
    } else if (node.guid == "gk_attn_online_finalize_bf16_v1") {
      const auto& o_spec = recipe.tensor_snapshots.at(node.inputs[1]);
      const uint64_t d = o_spec.sizes[0], rows = o_spec.sizes[1];
      const uint64_t heads = o_spec.dims == 4 ? (o_spec.sizes[2] * o_spec.sizes[3]) : o_spec.sizes[2];
      const uint64_t padded = (rows + 63) / 64 * 64;
      const auto* sums = reinterpret_cast<const float*>(load_tensor(node.inputs[0]).data());
      const auto* acc = reinterpret_cast<const float*>(load_tensor(node.inputs[1]).data());
      auto* out = reinterpret_cast<uint16_t*>(alloc_tensor(node.outputs[0]).data());
      for (uint64_t h = 0; h < heads; ++h)
        for (uint64_t r = 0; r < rows; ++r) {
          const float den = sums[h * padded + r];
          const float inv = den > 0.f ? (1.f / den) : 0.f;
          const uint64_t base = (h * rows + r) * d;
          for (uint64_t c = 0; c < d; ++c) out[base + c] = f32_to_bf16(acc[base + c] * inv);
        }
    }
  }

  // Write persistent output tensors back to mock HBM (applying output permutation if set!).
  for (const auto& rt : recipe.launch_tensors) {
    if (rt.is_input) continue;
    auto it = buffers.find(nullptr);
    // Find matching tensor handle in snapshots by id.
    for (const auto& [handle, spec] : recipe.tensor_snapshots) {
      if (spec.id != rt.spec.id) continue;
      auto bit = buffers.find(handle);
      if (bit == buffers.end()) continue;
      const auto& logical = bit->second;
      const uint64_t addr = id_to_addr.at(spec.id);
      if (!spec.has_perm) {
        mock().arena.write(addr, logical.data(), logical.size());
      } else if (spec.dims == 3 && spec.perm.permutation[0] == 0 && spec.perm.permutation[1] == 2 &&
                 spec.perm.permutation[2] == 1) {
        // Logical [d, rows, heads] -> physical [d, heads, rows] (row-major [rows, heads, d]).
        const uint64_t d = spec.sizes[0], rows = spec.sizes[1], heads = spec.sizes[2];
        const size_t elem_sz = dtype_bytes(spec.dtype);
        std::vector<uint8_t> raw(logical.size(), 0);
        for (uint64_t h = 0; h < heads; ++h)
          for (uint64_t r = 0; r < rows; ++r) {
            const uint64_t log_idx = (h * rows + r) * d;
            const uint64_t phys_idx = (r * heads + h) * d;
            std::memcpy(raw.data() + phys_idx * elem_sz, logical.data() + log_idx * elem_sz, d * elem_sz);
          }
        mock().arena.write(addr, raw.data(), raw.size());
      } else if (spec.dims == 4 && spec.perm.permutation[0] == 0 && spec.perm.permutation[1] == 2 &&
                 spec.perm.permutation[2] == 3 && spec.perm.permutation[3] == 1) {
        const uint64_t d = spec.sizes[0], rows = spec.sizes[1], group = spec.sizes[2], kvh = spec.sizes[3];
        const size_t elem_sz = dtype_bytes(spec.dtype);
        std::vector<uint8_t> raw(logical.size(), 0);
        for (uint64_t kh = 0; kh < kvh; ++kh)
          for (uint64_t g = 0; g < group; ++g)
            for (uint64_t r = 0; r < rows; ++r) {
              const uint64_t log_idx = ((kh * group + g) * rows + r) * d;
              const uint64_t phys_idx = ((r * kvh + kh) * group + g) * d;
              std::memcpy(raw.data() + phys_idx * elem_sz, logical.data() + log_idx * elem_sz, d * elem_sz);
            }
        mock().arena.write(addr, raw.data(), raw.size());
      } else {
        assert(false && "unexpected output permutation in mock executor");
      }
    }
    (void)it;
  }
}

}  // namespace

extern "C" {

synStatus synInitialize() { return synSuccess; }
synStatus synDestroy() { return synSuccess; }
synStatus synDeviceGetCount(uint32_t* pCount) { *pCount = 8; return synSuccess; }
synStatus synDeviceGetModuleIDs(uint32_t* ids, uint32_t* size) {
  const uint32_t n = std::min<uint32_t>(*size, 8);
  for (uint32_t i = 0; i < n; ++i) ids[i] = i;
  *size = n;
  return synSuccess;
}
synStatus synDeviceAcquireByModuleId(synDeviceId* pDeviceId, synModuleId moduleId) {
  *pDeviceId = moduleId;
  return synSuccess;
}
synStatus synDeviceRelease(synDeviceId) { return synSuccess; }
synStatus synDeviceSynchronize(synDeviceId) { return synSuccess; }
synStatus synDeviceGetMemoryInfo(synDeviceId, uint64_t* free_b, uint64_t* total_b) {
  *free_b = *total_b = 96ULL * 1024 * 1024 * 1024;
  return synSuccess;
}
synStatus synDeviceMalloc(synDeviceId, uint64_t size, uint64_t, uint32_t, uint64_t* addr) {
  *addr = mock().arena.alloc(size);
  return synSuccess;
}
synStatus synDeviceFree(synDeviceId, uint64_t, uint32_t) { return synSuccess; }
synStatus synHostMalloc(synDeviceId, uint64_t size, uint32_t, void** ptr) {
  *ptr = std::aligned_alloc(256, (size + 255ULL) & ~uint64_t{255});
  std::memset(*ptr, 0, size);
  return synSuccess;
}
synStatus synHostFree(synDeviceId, const void* ptr, uint32_t) {
  std::free(const_cast<void*>(ptr));
  return synSuccess;
}

synStatus synStreamCreateGeneric(synStreamHandle* s, synDeviceId, uint32_t) {
  *s = reinterpret_cast<synStreamHandle>(mock().next_stream_id++);
  return synSuccess;
}
synStatus synStreamDestroy(synStreamHandle) { return synSuccess; }
synStatus synStreamQuery(synStreamHandle) { return synSuccess; }
synStatus synStreamWaitEvent(synStreamHandle, synEventHandle, uint32_t) { return synSuccess; }
synStatus synStreamSynchronize(synStreamHandle) { return synSuccess; }

synStatus synEventCreate(synEventHandle* e, synDeviceId, uint32_t) {
  ++mock().event_creates;
  *e = new MockEvent{};
  return synSuccess;
}
synStatus synEventDestroy(synEventHandle e) {
  ++mock().event_destroys;
  delete static_cast<MockEvent*>(e);
  return synSuccess;
}
synStatus synEventRecord(synEventHandle e, synStreamHandle) {
  static_cast<MockEvent*>(e)->recorded = true;
  return synSuccess;
}
synStatus synEventQuery(synEventHandle) { return synSuccess; }
synStatus synEventSynchronize(synEventHandle) { return synSuccess; }
synStatus synEventElapsedTime(uint64_t* ns, synEventHandle, synEventHandle) { *ns = 0; return synSuccess; }

synStatus synMemCopyAsync(synStreamHandle, uint64_t src, uint64_t size, uint64_t dst, synDmaDir dir) {
  if (dir == HOST_TO_DRAM) mock().arena.write(dst, reinterpret_cast<const void*>(src), size);
  else if (dir == DRAM_TO_HOST) mock().arena.read(reinterpret_cast<void*>(dst), src, size);
  else mock().arena.copy_d2d(dst, src, size);
  return synSuccess;
}
synStatus synMemCopyAsyncMultiple(synStreamHandle s, const uint64_t* src, const uint64_t* size,
                                  const uint64_t* dst, synDmaDir dir, uint64_t n) {
  ++mock().multi_dma_calls;
  for (uint64_t i = 0; i < n; ++i) synMemCopyAsync(s, src[i], size[i], dst[i], dir);
  return synSuccess;
}
synStatus synMemsetD32Async(uint64_t dst, uint32_t val, size_t count, synStreamHandle) {
  mock().arena.memset_d32(dst, val, count);
  return synSuccess;
}

synStatus synGraphCreate(synGraphHandle* g, synDeviceType) {
  *g = new MockGraph{};
  return synSuccess;
}
synStatus synGraphDestroy(synGraphHandle g) {
  delete static_cast<MockGraph*>(g);
  return synSuccess;
}
synStatus synSectionCreate(synSectionHandle* s, uint64_t, synGraphHandle) {
  auto* sec = new MockSection{};
  sec->id = mock().next_section_id++;
  *s = sec;
  return synSuccess;
}
synStatus synSectionDestroy(synSectionHandle s) {
  delete static_cast<MockSection*>(s);
  return synSuccess;
}
synStatus synSectionSetPersistent(synSectionHandle s, bool persistent) {
  static_cast<MockSection*>(s)->persistent = persistent;
  return synSuccess;
}
synStatus synSectionSetRMW(synSectionHandle s, bool rmw) {
  static_cast<MockSection*>(s)->rmw = rmw;
  return synSuccess;
}

synStatus synTensorCreate(synTensor* out, const synTensorDescriptor* desc,
                          const synSectionHandle sec, uint64_t offset) {
  auto* t = new MockTensor{};
  t->name = desc->m_name ? desc->m_name : "";
  t->dtype = desc->m_dataType;
  t->tensor_type = desc->m_tensorType;
  t->dims = desc->m_dims;
  for (unsigned i = 0; i < desc->m_dims; ++i) {
    t->sizes[i] = desc->m_sizes[i];
    t->min_sizes[i] = desc->m_minSizes[i];
  }
  t->section = static_cast<MockSection*>(sec);
  t->section_offset = offset;
  t->id = mock().next_tensor_id++;
  *out = t;
  return synSuccess;
}
synStatus synTensorDestroy(synTensor t) {
  delete static_cast<MockTensor*>(t);
  return synSuccess;
}
synStatus synTensorSetPermutation(synTensor t, const synTensorPermutation* perm) {
  auto* mt = static_cast<MockTensor*>(t);
  mt->has_perm = true;
  mt->perm = *perm;
  return synSuccess;
}
synStatus synTensorSetQuantizationData(synTensor, synQuantMetadataType, void*, uint64_t) { return synSuccess; }

synStatus synNodeCreateWithId(synGraphHandle g, const synTensor* inputs, const synTensor* outputs,
                              uint32_t num_in, uint32_t num_out, const void* params, unsigned params_size,
                              const char* guid, const char* name, synNodeId* node_id,
                              const char**, const char**) {
  ++mock().node_with_id_creates;
  auto* graph = static_cast<MockGraph*>(g);
  MockNode node;
  node.guid = guid ? guid : "";
  node.name = name ? name : "";
  for (uint32_t i = 0; i < num_in; ++i) node.inputs.push_back(static_cast<MockTensor*>(inputs[i]));
  for (uint32_t i = 0; i < num_out; ++i) node.outputs.push_back(static_cast<MockTensor*>(outputs[i]));
  if (params && params_size) {
    const auto* p = static_cast<const uint8_t*>(params);
    node.params.assign(p, p + params_size);
  }
  graph->nodes.push_back(std::move(node));
  if (node_id) *node_id = graph->nodes.size();
  return synSuccess;
}

synStatus synNodeCreate(synGraphHandle g, const synTensor* inputs, const synTensor* outputs,
                        uint32_t num_in, uint32_t num_out, const void* params, unsigned params_size,
                        const char* guid, const char* name, const char** in_l, const char** out_l) {
  ++mock().node_creates;
  synNodeId id = 0;
  return synNodeCreateWithId(g, inputs, outputs, num_in, num_out, params, params_size, guid, name, &id, in_l, out_l);
}

synStatus synGraphCompile(synRecipeHandle* out, const synGraphHandle g, const char* name, const char*) {
  const auto* graph = static_cast<const MockGraph*>(g);
  auto* recipe = new MockRecipe{};
  recipe->name = name ? name : "";
  recipe->nodes = graph->nodes;
  std::set<MockTensor*> seen_inputs, seen_outputs;
  std::vector<MockTensor*> ordered;
  auto record_tensor = [&](MockTensor* t) {
    if (!t) return;
    if (recipe->tensor_snapshots.find(t) == recipe->tensor_snapshots.end()) {
      MockTensor snap = *t;
      if (t->section) {
        // Preserve persistent/id metadata even after SynapseGraphBuilder destroys temporary synSectionHandles!
        auto* sec_copy = new MockSection(*t->section);
        snap.section = sec_copy;
      }
      recipe->tensor_snapshots.emplace(t, snap);
      ordered.push_back(t);
    }
  };
  for (const auto& n : recipe->nodes) {
    for (auto* t : n.inputs) { record_tensor(t); seen_inputs.insert(t); }
    for (auto* t : n.outputs) { record_tensor(t); seen_outputs.insert(t); }
  }
  for (auto* t : ordered) {
    const auto& snap = recipe->tensor_snapshots.at(t);
    if (snap.section && snap.section->persistent) {
      const bool is_in = seen_inputs.count(t) > 0 && seen_outputs.count(t) == 0;
      recipe->launch_tensors.push_back({snap, snap.section->id, is_in});
    }
  }
  verify_recipe_structure(*recipe);
  mock().compiled_recipes.push_back(recipe);
  *out = recipe;
  return synSuccess;
}

synStatus synRecipeDestroy(synRecipeHandle r) {
  auto* recipe = static_cast<MockRecipe*>(r);
  for (auto& [k, snap] : recipe->tensor_snapshots) delete snap.section;
  delete recipe;
  return synSuccess;
}
synStatus synRecipeSerialize(const synRecipeHandle, const char*) { return synSuccess; }
synStatus synRecipeDeSerialize(synRecipeHandle*, const char*) { return synFail; }
synStatus synWorkspaceGetSize(uint64_t* ws, const synRecipeHandle r) {
  *ws = static_cast<const MockRecipe*>(r)->workspace;
  return synSuccess;
}

synStatus synTensorRetrieveLaunchAmount(const synRecipeHandle r, uint32_t* count) {
  *count = static_cast<uint32_t>(static_cast<const MockRecipe*>(r)->launch_tensors.size());
  return synSuccess;
}
synStatus synTensorRetrieveLaunchIds(const synRecipeHandle r, uint64_t* ids, uint32_t count) {
  const auto& lt = static_cast<const MockRecipe*>(r)->launch_tensors;
  for (uint32_t i = 0; i < count; ++i) ids[i] = lt[i].spec.id;
  return synSuccess;
}
synStatus synTensorRetrieveLaunchInfoByIdExt(const synRecipeHandle r, uint32_t count,
                                             synRetrievedLaunchTensorInfoExt* infos) {
  const auto& lt = static_cast<const MockRecipe*>(r)->launch_tensors;
  for (uint32_t i = 0; i < count; ++i) {
    for (const auto& entry : lt) {
      if (entry.spec.id == infos[i].tensorId) {
        std::snprintf(infos[i].tensorName, sizeof(infos[i].tensorName), "%s", entry.spec.name.c_str());
        infos[i].tensorType = entry.spec.tensor_type;
        infos[i].tensorDataType = entry.spec.dtype;
        infos[i].tensorDims = entry.spec.dims;
        for (uint32_t d = 0; d < entry.spec.dims; ++d) {
          infos[i].tensorMaxSize[d] = entry.spec.sizes[d];
          infos[i].tensorMinSize[d] = entry.spec.min_sizes[d];
        }
        infos[i].isInput = entry.is_input ? 1 : 0;
        break;
      }
    }
  }
  return synSuccess;
}

synStatus synLaunchWithExternalEvents(synStreamHandle, const synLaunchTensorInfo* infos,
                                      uint32_t count, uint64_t workspace, const synRecipeHandle r,
                                      synEventHandle*, uint32_t, uint32_t) {
  const auto* recipe = static_cast<const MockRecipe*>(r);
  assert(count == recipe->launch_tensors.size());

  // Check section address consistency and non-overlap across distinct persistent sections!
  struct SectionSpan { uint64_t base = 0, hi = 0; };
  std::map<uint64_t, SectionSpan> section_spans;
  size_t fp32_state_count = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const uint64_t addr = infos[i].pTensorAddress;
    assert(addr != 0 && addr % 256 == 0 && "every persistent launch binding must be non-null and 256B-aligned");
    const MockRecipeTensor* meta = nullptr;
    for (const auto& entry : recipe->launch_tensors) {
      if (entry.spec.id == infos[i].tensorId) { meta = &entry; break; }
    }
    assert(meta != nullptr);
    if (meta->spec.dtype == syn_type_float) ++fp32_state_count;
    const uint64_t sec_base = addr - meta->spec.section_offset;
    const uint64_t bytes = meta->spec.tensor_type == DATA_TENSOR_DYNAMIC
        ? infos[i].tensorSize[0] * dtype_bytes(meta->spec.dtype)
        : meta->spec.byte_size();
    const uint64_t sec_hi = addr + bytes;
    auto [it, inserted] = section_spans.emplace(meta->section_id, SectionSpan{sec_base, sec_hi});
    if (!inserted) {
      assert(it->second.base == sec_base && "shared persistent section offset mismatch");
      it->second.hi = std::max(it->second.hi, sec_hi);
    }
  }
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  for (const auto& [sec_id, span] : section_spans) {
    (void)sec_id;
    for (const auto& [other_lo, other_hi] : ranges) {
      assert((span.hi <= other_lo || other_hi <= span.base) &&
             "persistent sections within a single launch must not alias/overlap");
    }
    ranges.emplace_back(span.base, span.hi);
  }

  LaunchRecord rec;
  rec.recipe = recipe;
  rec.workspace = workspace;
  rec.bindings.assign(infos, infos + count);
  rec.persistent_fp32_state_bindings = fp32_state_count;
  mock().launches.push_back(std::move(rec));

  if (mock().execute_math) {
    execute_recipe_on_cpu(*recipe, infos, count);
  }
  return synSuccess;
}

synStatus synLaunch(synStreamHandle s, const synLaunchTensorInfo* infos, uint32_t count,
                    uint64_t ws, const synRecipeHandle r, uint32_t flags) {
  return synLaunchWithExternalEvents(s, infos, count, ws, r, nullptr, 0, flags);
}

hcclResult_t hcclCommInitRank(hcclComm_t* comm, int, hcclUniqueId, int) {
  *comm = reinterpret_cast<hcclComm_t>(uintptr_t{1});
  return hcclSuccess;
}
hcclResult_t hcclCommDestroy(hcclComm_t) { return hcclSuccess; }

}  // extern "C"

// Minimal PyTorch PrivateUse1 allocation glue for testing attention_plan + paged_attention
// across PyTorch versions without requiring version-specific DeviceGuardImpl hooks.
namespace gaudirt::generic {

namespace {
std::shared_ptr<Context> g_ctx;

void release_mock_alloc(void* ptr) {
  std::unique_ptr<AllocationContext> alloc(static_cast<AllocationContext*>(ptr));
  if (auto ctx = alloc->owner.lock()) {
    if (alloc->kind == AllocationKind::graph) ctx->graph_pool->free(alloc->address);
    else if (alloc->kind == AllocationKind::pinned) ctx->pinned.free(reinterpret_cast<void*>(alloc->address));
    else ctx->allocator.free(alloc->address);
  }
}
}  // namespace

bool SegmentOutputPool::trim() { return true; }

std::shared_ptr<Context> current() {
  TORCH_CHECK(g_ctx != nullptr, "mock context not initialized");
  return g_ctx;
}

c10::DataPtr allocation_data_ptr(const std::shared_ptr<Context>& ctx, uint64_t address,
                                 uint64_t bytes, synStreamHandle stream, AllocationKind kind) {
  auto* alloc = new AllocationContext{ctx, address, bytes, stream, kind};
  const auto dev = kind == AllocationKind::pinned ? c10::Device(c10::kCPU) : c10::Device(c10::kPrivateUse1, 0);
  return {reinterpret_cast<void*>(address), alloc, release_mock_alloc, dev};
}

at::Tensor tensor_from_allocation(c10::DataPtr data, uint64_t bytes, at::IntArrayRef shape,
                                  const at::TensorOptions& options) {
  auto storage = c10::make_intrusive<c10::StorageImpl>(
      c10::StorageImpl::use_byte_size_t{}, bytes, std::move(data), nullptr, false);
  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), options.dtype());
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
  return tensor;
}

const AllocationContext& validated_program_input(const at::Tensor& tensor, const Context& ctx) {
  const auto* alloc = tensor.storage().data_ptr().cast_context<AllocationContext>(release_mock_alloc);
  TORCH_CHECK(alloc && alloc->owner.lock().get() == &ctx, "invalid mock device tensor");
  return *alloc;
}

uint64_t address_of(const at::Tensor& tensor, const Context& ctx) {
  validated_program_input(tensor, ctx);
  return reinterpret_cast<uint64_t>(tensor.data_ptr()) +
         static_cast<uint64_t>(tensor.storage_offset()) * tensor.element_size();
}

at::Tensor bucket_output(const at::Tensor&, at::ScalarType dtype, uint32_t bucket, int64_t rows, int64_t width) {
  auto ctx = current();
  const auto stream = ctx->stream();
  const size_t item = c10::elementSize(dtype);
  const uint64_t bytes = static_cast<uint64_t>(bucket) * static_cast<uint64_t>(width) * item;
  const uint64_t addr = ctx->allocator.allocate(bytes, stream);
  auto data = allocation_data_ptr(ctx, addr, bytes, stream, AllocationKind::device);
  return tensor_from_allocation(
      std::move(data), bytes, {rows, width},
      at::TensorOptions().device(c10::Device(c10::kPrivateUse1, 0)).dtype(dtype));
}

namespace {
class MockTorchAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t bytes) override {
    auto ctx = current();
    if (!bytes) return {nullptr, c10::Device(c10::kPrivateUse1, 0)};
    const auto stream = ctx->stream();
    const uint64_t alloc_bytes = std::max<uint64_t>(256, (bytes + 255ULL) & ~uint64_t{255});
    const uint64_t addr = ctx->allocator.allocate(alloc_bytes, stream);
    return allocation_data_ptr(ctx, addr, alloc_bytes, stream, AllocationKind::device);
  }
  void copy_data(void* dst, const void* src, size_t bytes) const override {
    mock().arena.copy_d2d(reinterpret_cast<uint64_t>(dst), reinterpret_cast<uint64_t>(src), bytes);
  }
};

at::Tensor mock_empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                      std::optional<at::Layout>, std::optional<at::Device>,
                      std::optional<bool>, std::optional<at::MemoryFormat> format) {
  static MockTorchAllocator alloc;
  return at::detail::empty_generic(
      size, &alloc, c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), dtype.value_or(at::kFloat), format);
}

at::Tensor mock_empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                              std::optional<at::ScalarType> dtype, std::optional<at::Layout>,
                              std::optional<at::Device>, std::optional<bool>) {
  static MockTorchAllocator alloc;
  return at::detail::empty_strided_generic(
      size, stride, &alloc, c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), dtype.value_or(at::kFloat));
}
}  // namespace

}  // namespace gaudirt::generic

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(gaudirt::generic::mock_empty));
  m.impl("empty_strided", TORCH_FN(gaudirt::generic::mock_empty_strided));
  m.impl("as_strided", TORCH_FN(at::native::as_strided_tensorimpl));
  m.impl("view", TORCH_FN(at::native::view));
  m.impl("_reshape_alias", TORCH_FN(at::native::_reshape_alias));
}

namespace {

using namespace gaudirt::generic;

at::Tensor make_device_tensor(at::IntArrayRef shape, at::ScalarType dtype, const void* host_src = nullptr) {
  auto ctx = current();
  const auto stream = ctx->stream();
  int64_t elems = 1;
  for (auto s : shape) elems *= s;
  const uint64_t raw_bytes = static_cast<uint64_t>(elems) * c10::elementSize(dtype);
  const uint64_t alloc_bytes = std::max<uint64_t>(256, (raw_bytes + 255ULL) & ~uint64_t{255});
  const uint64_t addr = ctx->allocator.allocate(alloc_bytes, stream);
  if (host_src && raw_bytes) mock().arena.write(addr, host_src, raw_bytes);
  auto data = allocation_data_ptr(ctx, addr, alloc_bytes, stream, AllocationKind::device);
  return tensor_from_allocation(
      std::move(data), alloc_bytes, shape,
      at::TensorOptions().device(c10::Device(c10::kPrivateUse1, 0)).dtype(dtype));
}

// Build a PagedMetadata schedule directly without running the external metadata validation TPC kernel,
// then call the REAL production PagedMetadata::attention_plan() + paged_attn_validated()!
struct BuiltSchedule {
  c10::intrusive_ptr<PagedMetadata> meta;
  at::Tensor plan_tensor;
  at::Tensor locations;
  at::Tensor requests;
  at::Tensor positions;
  at::Tensor header;
};

BuiltSchedule build_schedule(int64_t prefix, int64_t length, int64_t slots, int64_t context_cap,
                             int64_t q_heads, int64_t kv_heads, int64_t head_dim,
                             const std::vector<int32_t>* custom_slot_table = nullptr) {
  const int64_t total = prefix + length;
  assert(total <= context_cap && total < slots);
  std::vector<int32_t> mapping_host(context_cap, 0);
  for (int64_t pos = 0; pos < total; ++pos) {
    mapping_host[pos] = custom_slot_table ? (*custom_slot_table)[pos] : static_cast<int32_t>(1 + pos);
  }
  std::vector<int32_t> loc_host(length), req_host(length, 0), pos_host(length);
  for (int64_t i = 0; i < length; ++i) {
    pos_host[i] = static_cast<int32_t>(prefix + i);
    loc_host[i] = mapping_host[prefix + i];
  }
  const uint32_t bucket = std::bit_ceil(static_cast<uint32_t>(length));
  std::vector<int32_t> loc_padded(bucket, 0), req_padded(bucket, 0), pos_padded(bucket, 0);
  std::copy(loc_host.begin(), loc_host.end(), loc_padded.begin());
  std::copy(req_host.begin(), req_host.end(), req_padded.begin());
  std::copy(pos_host.begin(), pos_host.end(), pos_padded.begin());
  const int32_t hdr_val = static_cast<int32_t>(length);

  // Use CPU constructor of PagedMetadata on small inputs? No, construct fields directly on an allocated
  // PagedMetadata or via CPU constructor? Wait: PagedMetadata has public members in paged_metadata.hpp!
  // Let's populate PagedMetadata directly so we test 100% real PagedMetadata::attention_plan()!
  auto meta = c10::make_intrusive<PagedMetadata>(
      at::tensor({loc_host[0]}, at::kInt),
      at::tensor({0}, at::kInt),
      at::tensor({pos_host[0]}, at::kInt),
      at::from_blob(mapping_host.data(), {1, context_cap}, at::kInt).clone(),
      slots);
  // Now override for arbitrary length (even >4096 context schedules if needed) cleanly:
  meta->rows_ = length;
  meta->slots_ = slots;
  meta->request_count_ = 1;
  meta->context_ = context_cap;
  meta->bucket_ = bucket;
  meta->locations_ = make_device_tensor({static_cast<int64_t>(bucket)}, at::kInt, loc_padded.data());
  meta->requests_ = make_device_tensor({static_cast<int64_t>(bucket)}, at::kInt, req_padded.data());
  meta->positions_ = make_device_tensor({static_cast<int64_t>(bucket)}, at::kInt, pos_padded.data());
  meta->mapping_ = make_device_tensor({1, context_cap}, at::kInt, mapping_host.data());
  meta->header_ = make_device_tensor({1}, at::kInt, &hdr_val);
  meta->sequences_ = {{0, static_cast<int32_t>(length), static_cast<int32_t>(prefix)}};
  meta->pending_host_.clear();
  meta->attention_plan_ = at::Tensor{};

  at::Tensor plan_t = meta->attention_plan(q_heads, kv_heads, head_dim);
  return {meta, plan_t, meta->row_locations(), meta->row_requests(), meta->rotary_positions(), meta->rotary_header()};
}

std::pair<at::Tensor, at::Tensor> make_kv_views(at::Tensor& pool_5d) {
  auto k_view = pool_5d.select(0, 0).select(0, 0).transpose(0, 1);
  auto v_view = pool_5d.select(0, 0).select(0, 1).transpose(0, 1);
  return {k_view, v_view};
}

// Synthetic block test helper for arbitrary num_tiles (e.g. 9, 11, 15, 17, 64) via launch_tiled_attention.
size_t run_synthetic_block(Context& ctx, uint32_t num_tiles, uint32_t q_bucket, uint32_t valid_rows,
                           uint32_t kv_bucket, int64_t q_heads, int64_t kv_heads, int64_t head_dim) {
  AttentionPlan plan{};
  plan.owner = current();
  plan.stream = ctx.stream();
  plan.rows = valid_rows;
  plan.slots = 65536;
  plan.query_heads = q_heads;
  plan.kv_heads = kv_heads;
  plan.head_dim = head_dim;

  AttentionBlock block{};
  block.first_row = 0;
  block.valid_rows = static_cast<int32_t>(valid_rows);
  block.query_bucket = q_bucket;
  block.kv_bucket = kv_bucket;

  const uint64_t idx_stride = 2ULL * kv_bucket * kv_heads * sizeof(int32_t);
  uint64_t cursor = num_tiles * idx_stride;
  block.positions_offset = cursor;
  cursor += (static_cast<uint64_t>(q_bucket) * sizeof(int32_t) + 255ULL) & ~uint64_t{255};
  for (uint32_t t = 0; t < num_tiles; ++t) {
    AttentionTile tile{};
    tile.indices_offset = t * idx_stride;
    tile.header_offset = cursor;
    cursor += 256;
    tile.first_key = static_cast<int32_t>(t * kv_bucket);
    tile.valid_keys = static_cast<int32_t>(kv_bucket);
    block.tiles.push_back(tile);
  }
  plan.scratch_offset = cursor;
  const uint64_t stats = (((q_bucket + 63) / 64 * 64) * q_heads * sizeof(float) + 255ULL) & ~uint64_t{255};
  const uint64_t out_f32 = (static_cast<uint64_t>(q_bucket) * q_heads * head_dim * sizeof(float) + 255ULL) & ~uint64_t{255};
  plan.scratch_bytes = num_tiles > kMaxFusedAttentionTiles ? 2 * (2 * stats + out_f32) : 0;
  plan.bytes = plan.scratch_offset + plan.scratch_bytes;
  plan.address = ctx.allocator.allocate(plan.bytes, ctx.stream());

  const uint64_t q_addr = ctx.allocator.allocate(static_cast<uint64_t>(q_bucket) * q_heads * head_dim * 2, ctx.stream());
  const uint64_t pool_addr = ctx.allocator.allocate(2ULL * kPagedKvDescriptorSlots * kv_heads * head_dim * 2, ctx.stream());
  const uint64_t out_addr = ctx.allocator.allocate(static_cast<uint64_t>(q_bucket) * q_heads * head_dim * 2, ctx.stream());

  mock().reset_counters();
  const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
  launch_tiled_attention(ctx, plan, block, {q_addr, pool_addr, out_addr}, scale);

  // Verify every tile's indices & header address was bound distinctly across the chunk launches!
  std::set<uint64_t> seen_key_indices, seen_headers;
  for (const auto& launch : mock().launches) {
    for (const auto& b : launch.bindings) {
      if (b.pTensorAddress >= plan.address && b.pTensorAddress < plan.address + num_tiles * idx_stride) {
        if ((b.pTensorAddress - plan.address) % idx_stride == 0)
          assert(seen_key_indices.insert(b.pTensorAddress).second && "duplicate tile key_indices address");
      } else if (b.pTensorAddress >= plan.address + block.positions_offset + 256 &&
                 b.pTensorAddress < plan.address + plan.scratch_offset) {
        assert(seen_headers.insert(b.pTensorAddress).second && "duplicate tile header address");
      }
    }
  }
  assert(seen_key_indices.size() == num_tiles && "all tiles must bind their own key_indices");
  assert(seen_headers.size() == num_tiles && "all tiles must bind their own header");
  assert(mock().launches.back().bindings.back().pTensorAddress == out_addr &&
         "final chunk must write BF16 output directly to input.output");

  const size_t launches = mock().launches.size();
  ctx.allocator.free(out_addr);
  ctx.allocator.free(pool_addr);
  ctx.allocator.free(q_addr);
  ctx.allocator.free(plan.address);
  return launches;
}

// Independent single-tile IEEE-754 CPU reference matching TPC tile_softmax + online_merge + online_finalize!
std::vector<uint16_t> cpu_reference_paged_attention(
    const std::vector<uint16_t>& q,          // [rows, q_heads * d]
    const std::vector<uint16_t>& k_new,      // [rows, kv_heads * d]
    const std::vector<uint16_t>& v_new,      // [rows, kv_heads * d]
    std::vector<uint16_t> k_pool,            // [slots, kv_heads * d]
    std::vector<uint16_t> v_pool,            // [slots, kv_heads * d]
    const std::vector<int32_t>& slot_table,  // [prefix + rows]
    int32_t prefix, int32_t rows, int32_t q_heads, int32_t kv_heads, int32_t d,
    uint32_t q_bucket_cap, uint32_t kv_tile_size, float scale) {
  const int32_t group = q_heads / kv_heads;
  const int32_t kv_width = kv_heads * d;
  for (int32_t r = 0; r < rows; ++r) {
    const int32_t slot = slot_table[prefix + r];
    std::copy_n(k_new.begin() + static_cast<size_t>(r) * kv_width, kv_width,
                k_pool.begin() + static_cast<size_t>(slot) * kv_width);
    std::copy_n(v_new.begin() + static_cast<size_t>(r) * kv_width, kv_width,
                v_pool.begin() + static_cast<size_t>(slot) * kv_width);
  }

  std::vector<uint16_t> out(static_cast<size_t>(rows) * q_heads * d, 0);
  for (int32_t done = 0; done < rows; done += static_cast<int32_t>(q_bucket_cap)) {
    const int32_t valid_rows = std::min<int32_t>(rows - done, q_bucket_cap);
    const int32_t visible = prefix + done + valid_rows;
    const uint32_t num_tiles = (visible + kv_tile_size - 1) / kv_tile_size;
    for (int32_t r = 0; r < valid_rows; ++r) {
      const int32_t global_row = done + r;
      const int32_t pos = prefix + global_row;
      for (int32_t h = 0; h < q_heads; ++h) {
        const int32_t kh = h / group;
        const uint16_t* q_vec = q.data() + (static_cast<size_t>(global_row) * q_heads + h) * d;
        if (num_tiles == 1) {
          // Single-tile direct softmax path
          float m = -std::numeric_limits<float>::infinity();
          std::vector<float> scores(visible, 0.f);
          for (int32_t k_idx = 0; k_idx <= pos; ++k_idx) {
            const int32_t slot = slot_table[k_idx];
            const uint16_t* k_vec = k_pool.data() + static_cast<size_t>(slot) * kv_width + kh * d;
            float dot = 0.f;
            for (int32_t c = 0; c < d; ++c) dot += bf16_to_f32(q_vec[c]) * bf16_to_f32(k_vec[c]);
            scores[k_idx] = dot * scale;
            m = std::max(m, scores[k_idx]);
          }
          const float origin = std::isinf(m) ? 0.f : m;
          float sum = 0.f;
          std::vector<float> exps(pos + 1, 0.f);
          for (int32_t k_idx = 0; k_idx <= pos; ++k_idx) {
            exps[k_idx] = std::exp(scores[k_idx] - origin);
            sum += exps[k_idx];
          }
          const float inv = sum > 0.f ? (1.f / sum) : 0.f;
          std::vector<uint16_t> probs(pos + 1, 0);
          for (int32_t k_idx = 0; k_idx <= pos; ++k_idx) probs[k_idx] = f32_to_bf16(exps[k_idx] * inv);
          for (int32_t c = 0; c < d; ++c) {
            float acc = 0.f;
            for (int32_t k_idx = 0; k_idx <= pos; ++k_idx) {
              const int32_t slot = slot_table[k_idx];
              acc += bf16_to_f32(probs[k_idx]) * bf16_to_f32(v_pool[static_cast<size_t>(slot) * kv_width + kh * d + c]);
            }
            out[(static_cast<size_t>(global_row) * q_heads + h) * d + c] = f32_to_bf16(acc);
          }
        } else {
          // Sequential single-tile online softmax + merge + finalize
          const float ni = -std::numeric_limits<float>::infinity();
          float run_m = 0.f, run_l = 0.f;
          std::vector<float> run_o(d, 0.f);
          for (uint32_t t = 0; t < num_tiles; ++t) {
            const int32_t key_start = static_cast<int32_t>(t * kv_tile_size);
            const int32_t valid_keys = std::min<int32_t>(kv_tile_size, visible - key_start);
            const int32_t last_key = std::min(pos - key_start, valid_keys - 1);
            float tm = ni, tl = 0.f;
            std::vector<uint16_t> probs(kv_tile_size, 0);
            if (last_key >= 0) {
              std::vector<float> scores(last_key + 1, 0.f);
              for (int32_t k = 0; k <= last_key; ++k) {
                const int32_t slot = slot_table[key_start + k];
                const uint16_t* k_vec = k_pool.data() + static_cast<size_t>(slot) * kv_width + kh * d;
                float dot = 0.f;
                for (int32_t c = 0; c < d; ++c) dot += bf16_to_f32(q_vec[c]) * bf16_to_f32(k_vec[c]);
                scores[k] = dot * scale;
                tm = std::max(tm, scores[k]);
              }
              const float origin = (tm == ni) ? 0.f : tm;
              std::vector<float> exps(last_key + 1, 0.f);
              for (int32_t k = 0; k <= last_key; ++k) {
                exps[k] = std::exp(scores[k] - origin);
                tl += exps[k];
              }
              const float inv = tl > 0.f ? (1.f / tl) : 0.f;
              for (int32_t k = 0; k <= last_key; ++k) probs[k] = f32_to_bf16(exps[k] * inv);
            }
            std::vector<float> tile_o(d, 0.f);
            for (int32_t c = 0; c < d; ++c) {
              float acc = 0.f;
              for (int32_t k = 0; k < valid_keys; ++k) {
                const int32_t slot = slot_table[key_start + k];
                acc += bf16_to_f32(probs[k]) * bf16_to_f32(v_pool[static_cast<size_t>(slot) * kv_width + kh * d + c]);
              }
              tile_o[c] = acc;
            }
            const float v0 = run_l > 0.f ? run_m : ni;
            const float v1 = tl > 0.f ? tm : ni;
            const float next_m = std::max(v0, v1);
            const float origin = (next_m == ni) ? 0.f : next_m;
            const float a = run_l > 0.f ? std::exp(run_m - origin) : 0.f;
            const float b = tl > 0.f ? std::exp(tm - origin) : 0.f;
            const float bb = b * tl;
            for (int32_t c = 0; c < d; ++c) run_o[c] = a * run_o[c] + bb * tile_o[c];
            run_l = a * run_l + b * tl;
            run_m = next_m;
          }
          const float inv = run_l > 0.f ? (1.f / run_l) : 0.f;
          for (int32_t c = 0; c < d; ++c)
            out[(static_cast<size_t>(global_row) * q_heads + h) * d + c] = f32_to_bf16(run_o[c] * inv);
        }
      }
    }
  }
  return out;
}

}  // namespace

int main() {
  using namespace gaudirt::generic;
  g_ctx = std::make_shared<Context>(0, 80ULL * 1024 * 1024 * 1024, 256ULL * 1024 * 1024);
  auto& ctx = *g_ctx;

  constexpr int64_t kQHeads = 56;
  constexpr int64_t kKvHeads = 8;
  constexpr int64_t kHeadDim = 128;
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));

  auto pool = allocate_paged_kv(1, 65536, kKvHeads, kHeadDim);
  auto [k_pool, v_pool] = make_kv_views(pool);

  // 1. Decode schedules: (q=1, kv=128) and (q=1, kv=4096)
  for (int64_t ctx_len : {128L, 4096L}) {
    auto sched = build_schedule(ctx_len - 1, 1, 65536, 32768, kQHeads, kKvHeads, kHeadDim);
    const auto& plan = attention_plan_data(sched.plan_tensor, ctx);
    assert(plan.blocks.size() == 1 && plan.blocks[0].tiles.size() == 1);
    assert(plan.scratch_bytes == 0);
    auto q = make_device_tensor({1, kQHeads * kHeadDim}, at::kBFloat16);
    auto k = make_device_tensor({1, kKvHeads * kHeadDim}, at::kBFloat16);
    auto v = make_device_tensor({1, kKvHeads * kHeadDim}, at::kBFloat16);
    mock().reset_counters();
    auto out = paged_attn_validated(q, k, v, k_pool, v_pool, sched.locations, sched.requests,
                                    sched.positions, sched.plan_tensor, sched.header, kHeadDim, scale);
    assert(out.sizes() == at::IntArrayRef({1, kQHeads * kHeadDim}));
    assert(mock().launches.size() == 1 && "decode must stay a single fused launch");
  }

  // 2. Prefill M=1024 (q_bucket=256, kv_bucket=512, num_tiles in {1, 2})
  {
    auto sched = build_schedule(0, 1024, 65536, 32768, kQHeads, kKvHeads, kHeadDim);
    const auto& plan = attention_plan_data(sched.plan_tensor, ctx);
    assert(plan.blocks.size() == 4);
    assert(plan.blocks[0].tiles.size() == 1 && plan.blocks[1].tiles.size() == 1);
    assert(plan.blocks[2].tiles.size() == 2 && plan.blocks[3].tiles.size() == 2);
    assert(plan.scratch_bytes == 0 && "M=1024 (num_tiles <= 8) must allocate 0 bytes of HBM tile scratch");
    auto q = make_device_tensor({1024, kQHeads * kHeadDim}, at::kBFloat16);
    auto k = make_device_tensor({1024, kKvHeads * kHeadDim}, at::kBFloat16);
    auto v = make_device_tensor({1024, kKvHeads * kHeadDim}, at::kBFloat16);
    mock().reset_counters();
    auto out = paged_attn_validated(q, k, v, k_pool, v_pool, sched.locations, sched.requests,
                                    sched.positions, sched.plan_tensor, sched.header, kHeadDim, scale);
    assert(out.sizes() == at::IntArrayRef({1024, kQHeads * kHeadDim}));
    assert(mock().launches.size() == 5 && "M=1024 must launch 1 kv_store + 4 query blocks = 5 launches");
    for (const auto& l : mock().launches) {
      assert(l.persistent_fp32_state_bindings == 0 && "M=1024 must have zero persistent FP32 (m,l,o) bindings");
    }
  }

  // 3. Prefill M=4096 (q_bucket=512, kv_bucket=512, num_tiles in {1..8})
  {
    auto sched = build_schedule(0, 4096, 65536, 32768, kQHeads, kKvHeads, kHeadDim);
    const auto& plan = attention_plan_data(sched.plan_tensor, ctx);
    assert(plan.blocks.size() == 8);
    for (size_t b = 0; b < 8; ++b) {
      assert(plan.blocks[b].query_bucket == 512 && plan.blocks[b].kv_bucket == 512);
      assert(plan.blocks[b].tiles.size() == b + 1);
    }
    assert(plan.scratch_bytes == 0 && "M=4096 (num_tiles <= 8) must allocate 0 bytes of HBM tile scratch");
    auto q = make_device_tensor({4096, kQHeads * kHeadDim}, at::kBFloat16);
    auto k = make_device_tensor({4096, kKvHeads * kHeadDim}, at::kBFloat16);
    auto v = make_device_tensor({4096, kKvHeads * kHeadDim}, at::kBFloat16);
    mock().reset_counters();
    auto out = paged_attn_validated(q, k, v, k_pool, v_pool, sched.locations, sched.requests,
                                    sched.positions, sched.plan_tensor, sched.header, kHeadDim, scale);
    assert(out.sizes() == at::IntArrayRef({4096, kQHeads * kHeadDim}));
    assert(mock().launches.size() == 9 && "M=4096 must launch exactly 9 recipes (1 kv_store + 8 blocks)");
    for (const auto& l : mock().launches) {
      assert(l.persistent_fp32_state_bindings == 0 &&
             "M=4096 (num_tiles <= 8) must have zero persistent HBM (m, l, o) tensor bindings");
    }
  }

  // 4. Long-context M=4096 with cached prefix up to 32768 (num_tiles = 64) + non-multiple tails {9, 11, 15, 17}
  {
    const size_t launches_64 = run_synthetic_block(ctx, 64, 512, 512, 512, kQHeads, kKvHeads, kHeadDim);
    assert(launches_64 == 8 && "64 KV tiles (32768 context) must fuse into 8 chunk launches");
    assert(mock().launches[0].persistent_fp32_state_bindings == 3);  // out_m, out_l, out_o only
    for (size_t i = 1; i < 7; ++i) assert(mock().launches[i].persistent_fp32_state_bindings == 6);  // old + out
    assert(mock().launches[7].persistent_fp32_state_bindings == 3);  // old_m, old_l, old_o only -> BF16 output

    assert(run_synthetic_block(ctx, 9, 256, 256, 512, kQHeads, kKvHeads, kHeadDim) == 2);
    assert(run_synthetic_block(ctx, 11, 512, 512, 512, kQHeads, kKvHeads, kHeadDim) == 2);
    assert(run_synthetic_block(ctx, 15, 256, 256, 512, kQHeads, kKvHeads, kHeadDim) == 2);
    assert(run_synthetic_block(ctx, 17, 512, 512, 512, kQHeads, kKvHeads, kHeadDim) == 3);

    // Full end-to-end 32768 context schedule (prefix=28672, M=4096 -> 8 blocks with 57..64 tiles, 8 launches/block)
    auto sched32k = build_schedule(28672, 4096, 65536, 32768, kQHeads, kKvHeads, kHeadDim);
    const auto& plan32k = attention_plan_data(sched32k.plan_tensor, ctx);
    assert(plan32k.blocks.size() == 8 && plan32k.blocks.back().tiles.size() == 64);
    assert(plan32k.scratch_bytes > 0);
    auto q = make_device_tensor({4096, kQHeads * kHeadDim}, at::kBFloat16);
    auto k = make_device_tensor({4096, kKvHeads * kHeadDim}, at::kBFloat16);
    auto v = make_device_tensor({4096, kKvHeads * kHeadDim}, at::kBFloat16);
    mock().reset_counters();
    auto out32k = paged_attn_validated(q, k, v, k_pool, v_pool, sched32k.locations, sched32k.requests,
                                       sched32k.positions, sched32k.plan_tensor, sched32k.header, kHeadDim, scale);
    assert(out32k.sizes() == at::IntArrayRef({4096, kQHeads * kHeadDim}));
    assert(mock().launches.size() == 1 + 8 * 8 && "28672+4096=32768 prefill must take 1 + 8*8 = 65 launches");
  }

  // 5. Steady-state zero synEventCreate / synEventDestroy check across repeated warm runs (P-RT-78)
  {
    auto q4k = make_device_tensor({4096, kQHeads * kHeadDim}, at::kBFloat16);
    auto k4k = make_device_tensor({4096, kKvHeads * kHeadDim}, at::kBFloat16);
    auto v4k = make_device_tensor({4096, kKvHeads * kHeadDim}, at::kBFloat16);
    auto q1 = make_device_tensor({1, kQHeads * kHeadDim}, at::kBFloat16);
    auto k1 = make_device_tensor({1, kKvHeads * kHeadDim}, at::kBFloat16);
    auto v1 = make_device_tensor({1, kKvHeads * kHeadDim}, at::kBFloat16);

    // Warmup once so staging slot + recipes are populated.
    {
      auto warm_p = build_schedule(0, 4096, 65536, 32768, kQHeads, kKvHeads, kHeadDim);
      (void)paged_attn_validated(q4k, k4k, v4k, k_pool, v_pool, warm_p.locations, warm_p.requests,
                                 warm_p.positions, warm_p.plan_tensor, warm_p.header, kHeadDim, scale);
      auto warm_d = build_schedule(4096, 1, 65536, 32768, kQHeads, kKvHeads, kHeadDim);
      (void)paged_attn_validated(q1, k1, v1, k_pool, v_pool, warm_d.locations, warm_d.requests,
                                 warm_d.positions, warm_d.plan_tensor, warm_d.header, kHeadDim, scale);
    }

    mock().reset_counters();
    const size_t recipes_before = mock().compiled_recipes.size();
    for (int iter = 0; iter < 6; ++iter) {
      auto sp = build_schedule(0, 4096, 65536, 32768, kQHeads, kKvHeads, kHeadDim);
      (void)paged_attn_validated(q4k, k4k, v4k, k_pool, v_pool, sp.locations, sp.requests,
                                 sp.positions, sp.plan_tensor, sp.header, kHeadDim, scale);
      auto sd = build_schedule(4096, 1, 65536, 32768, kQHeads, kKvHeads, kHeadDim);
      (void)paged_attn_validated(q1, k1, v1, k_pool, v_pool, sd.locations, sd.requests,
                                 sd.positions, sd.plan_tensor, sd.header, kHeadDim, scale);
    }
    assert(mock().event_creates == 0 && "zero steady-state synEventCreate calls allowed");
    assert(mock().event_destroys == 0 && "zero steady-state synEventDestroy calls allowed");
    assert(mock().compiled_recipes.size() == recipes_before && "zero steady-state recipe recompilations");
  }

  // 6. Bit-exact IEEE-754 CPU reference equivalence (max_abs == 0.0, 0 ULP) across single-chunk & multi-chunk!
  {
    constexpr int32_t kTestQHeads = 14;
    constexpr int32_t kTestKvHeads = 2;
    constexpr int32_t kTestD = 64;
    constexpr int32_t kTestSlots = 12288;
    const float test_scale = 1.f / std::sqrt(static_cast<float>(kTestD));
    std::mt19937 rng(20261110);
    std::normal_distribution<float> dist(0.f, 0.5f);

    auto math_pool = allocate_paged_kv(1, kTestSlots, kTestKvHeads, kTestD);
    auto [mk_pool, mv_pool] = make_kv_views(math_pool);
    const uint64_t pool_base = address_of(mk_pool, ctx);

    // Test both M=1024 empty-KV causal prefill (tiles 1..2) AND cached-prefix multi-chunk (9, 11, 17 tiles, partial rows)!
    struct MathCase { int32_t prefix; int32_t rows; };
    const std::vector<MathCase> cases = {
        {0, 1024},     // 4 blocks, tiles = [1, 1, 2, 2]
        {3840, 263},   // visible up to 4103 -> 9 tiles ([8, 1]) + partial tail block (7 valid_rows)
        {5300, 197},   // visible = 5497 -> 11 tiles ([8, 3]) + partial valid_rows & partial valid_keys
        {8300, 229},   // visible = 8529 -> 17 tiles ([8, 8, 1], 3-chunk ping-pong!)
    };

    mock().execute_math = true;
    for (const auto& tc : cases) {
      const int32_t total_ctx = tc.prefix + tc.rows;
      std::vector<int32_t> slot_table(total_ctx);
      for (int32_t i = 0; i < total_ctx; ++i) {
        // Non-identity paged slot permutation
        slot_table[i] = 1 + ((i * 17 + 5) % (kTestSlots - 2));
      }
      // Ensure unique slots for this test sequence
      for (int32_t i = 0; i < total_ctx; ++i) slot_table[i] = 1 + (total_ctx - 1 - i);

      const int32_t q_width = kTestQHeads * kTestD;
      const int32_t kv_width = kTestKvHeads * kTestD;
      std::vector<uint16_t> k_pool_init(static_cast<size_t>(kTestSlots) * kv_width, 0);
      std::vector<uint16_t> v_pool_init(static_cast<size_t>(kTestSlots) * kv_width, 0);
      for (int32_t i = 0; i < tc.prefix; ++i) {
        const int32_t s = slot_table[i];
        for (int32_t c = 0; c < kv_width; ++c) {
          k_pool_init[static_cast<size_t>(s) * kv_width + c] = f32_to_bf16(dist(rng));
          v_pool_init[static_cast<size_t>(s) * kv_width + c] = f32_to_bf16(dist(rng));
        }
      }
      // Write head-major physical pool into mock HBM
      for (int32_t h = 0; h < kTestKvHeads; ++h)
        for (int32_t s = 0; s < kTestSlots; ++s) {
          const uint64_t k_dst = pool_base + (static_cast<uint64_t>(h) * kTestSlots + s) * kTestD * 2;
          const uint64_t v_dst = pool_base + (static_cast<uint64_t>(kTestKvHeads + h) * kTestSlots + s) * kTestD * 2;
          mock().arena.write(k_dst, k_pool_init.data() + static_cast<size_t>(s) * kv_width + h * kTestD, kTestD * 2);
          mock().arena.write(v_dst, v_pool_init.data() + static_cast<size_t>(s) * kv_width + h * kTestD, kTestD * 2);
        }

      std::vector<uint16_t> q_host(static_cast<size_t>(tc.rows) * q_width);
      std::vector<uint16_t> k_host(static_cast<size_t>(tc.rows) * kv_width);
      std::vector<uint16_t> v_host(static_cast<size_t>(tc.rows) * kv_width);
      for (auto& x : q_host) x = f32_to_bf16(dist(rng));
      for (auto& x : k_host) x = f32_to_bf16(dist(rng));
      for (auto& x : v_host) x = f32_to_bf16(dist(rng));

      auto sched = build_schedule(tc.prefix, tc.rows, kTestSlots, 16384,
                                  kTestQHeads, kTestKvHeads, kTestD, &slot_table);
      auto q_dev = make_device_tensor({tc.rows, q_width}, at::kBFloat16, q_host.data());
      auto k_dev = make_device_tensor({tc.rows, kv_width}, at::kBFloat16, k_host.data());
      auto v_dev = make_device_tensor({tc.rows, kv_width}, at::kBFloat16, v_host.data());

      auto out_dev = paged_attn_validated(q_dev, k_dev, v_dev, mk_pool, mv_pool,
                                          sched.locations, sched.requests, sched.positions,
                                          sched.plan_tensor, sched.header, kTestD, test_scale);
      std::vector<uint16_t> actual(static_cast<size_t>(tc.rows) * q_width, 0);
      mock().arena.read(actual.data(), address_of(out_dev, ctx), actual.size() * 2);

      const auto expected = cpu_reference_paged_attention(
          q_host, k_host, v_host, k_pool_init, v_pool_init, slot_table,
          tc.prefix, tc.rows, kTestQHeads, kTestKvHeads, kTestD, 256, 512, test_scale);
      assert(actual.size() == expected.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
          std::cerr << "Bit mismatch at case prefix=" << tc.prefix << " rows=" << tc.rows
                    << " idx=" << i << " actual=" << bf16_to_f32(actual[i])
                    << " expected=" << bf16_to_f32(expected[i]) << '\n';
          return 1;
        }
      }
    }

    // 7a. Partial-tail direct_kv (prefix=0, rows=600 -> tail valid_keys=88 < 512) with NaN-poisoned
    // caller bucket padding in rows [600, 1024): must neither propagate NaNs nor mutate caller storage!
    {
      constexpr int32_t kRows = 600;
      constexpr int32_t kBucket = 1024;
      const int32_t q_width = kTestQHeads * kTestD;
      const int32_t kv_width = kTestKvHeads * kTestD;
      std::vector<int32_t> slot_table(kRows);
      for (int32_t i = 0; i < kRows; ++i) slot_table[i] = 1 + i;
      std::vector<uint16_t> k_pool_init(static_cast<size_t>(kTestSlots) * kv_width, 0);
      std::vector<uint16_t> v_pool_init(static_cast<size_t>(kTestSlots) * kv_width, 0);

      std::vector<uint16_t> q_host(static_cast<size_t>(kRows) * q_width);
      std::vector<uint16_t> k_padded(static_cast<size_t>(kBucket) * kv_width, uint16_t{0x7fc0U});
      std::vector<uint16_t> v_padded(static_cast<size_t>(kBucket) * kv_width, uint16_t{0x7fc0U});
      for (auto& x : q_host) x = f32_to_bf16(dist(rng));
      for (int32_t i = 0; i < kRows * kv_width; ++i) {
        k_padded[i] = f32_to_bf16(dist(rng));
        v_padded[i] = f32_to_bf16(dist(rng));
      }
      std::vector<uint16_t> k_valid(k_padded.begin(), k_padded.begin() + kRows * kv_width);
      std::vector<uint16_t> v_valid(v_padded.begin(), v_padded.begin() + kRows * kv_width);

      auto sched = build_schedule(0, kRows, kTestSlots, 16384, kTestQHeads, kTestKvHeads, kTestD, &slot_table);
      auto q_dev = make_device_tensor({kRows, q_width}, at::kBFloat16, q_host.data());
      auto k_full = make_device_tensor({kBucket, kv_width}, at::kBFloat16, k_padded.data());
      auto v_full = make_device_tensor({kBucket, kv_width}, at::kBFloat16, v_padded.data());
      auto k_dev = k_full.narrow(0, 0, kRows);
      auto v_dev = v_full.narrow(0, 0, kRows);

      auto out_dev = paged_attn_validated(q_dev, k_dev, v_dev, mk_pool, mv_pool,
                                          sched.locations, sched.requests, sched.positions,
                                          sched.plan_tensor, sched.header, kTestD, test_scale);
      std::vector<uint16_t> actual(static_cast<size_t>(kRows) * q_width, 0);
      mock().arena.read(actual.data(), address_of(out_dev, ctx), actual.size() * 2);
      const auto expected = cpu_reference_paged_attention(
          q_host, k_valid, v_valid, k_pool_init, v_pool_init, slot_table,
          0, kRows, kTestQHeads, kTestKvHeads, kTestD, 256, 512, test_scale);
      assert(actual == expected && "NaN-padded partial tail direct_kv must match CPU reference bit-for-bit");

      // Caller's backing storage beyond row 600 must remain untouched!
      std::vector<uint16_t> k_tail_check(kv_width, 0);
      mock().arena.read(k_tail_check.data(), address_of(k_full, ctx) + static_cast<uint64_t>(kRows) * kv_width * 2,
                        kv_width * 2);
      for (uint16_t w : k_tail_check) assert(w == uint16_t{0x7fc0U});
    }

    // 7b. Strided 2D packed_kv slices (exact-row 137 -> 2-launch split, and pow2 1024 -> 1-launch split):
    // zero synMemCopyAsyncMultiple descriptor storms and 0-ULP CPU reference match!
    for (int32_t s_rows : {137, 1024}) {
      const int32_t q_width = kTestQHeads * kTestD;
      const int32_t kv_width = kTestKvHeads * kTestD;
      std::vector<int32_t> slot_table(s_rows);
      for (int32_t i = 0; i < s_rows; ++i) slot_table[i] = 1 + i;
      std::vector<uint16_t> k_pool_init(static_cast<size_t>(kTestSlots) * kv_width, 0);
      std::vector<uint16_t> v_pool_init(static_cast<size_t>(kTestSlots) * kv_width, 0);
      std::vector<uint16_t> q_host(static_cast<size_t>(s_rows) * q_width);
      std::vector<uint16_t> k_host(static_cast<size_t>(s_rows) * kv_width);
      std::vector<uint16_t> v_host(static_cast<size_t>(s_rows) * kv_width);
      std::vector<uint16_t> kv_interleaved(static_cast<size_t>(s_rows) * 2 * kv_width);
      for (auto& x : q_host) x = f32_to_bf16(dist(rng));
      for (int32_t r = 0; r < s_rows; ++r) {
        for (int32_t c = 0; c < kv_width; ++c) {
          const uint16_t kv_k = f32_to_bf16(dist(rng));
          const uint16_t kv_v = f32_to_bf16(dist(rng));
          k_host[r * kv_width + c] = kv_k;
          v_host[r * kv_width + c] = kv_v;
          kv_interleaved[r * (2 * kv_width) + c] = kv_k;
          kv_interleaved[r * (2 * kv_width) + kv_width + c] = kv_v;
        }
      }
      auto sched = build_schedule(0, s_rows, kTestSlots, 16384, kTestQHeads, kTestKvHeads, kTestD, &slot_table);
      auto q_dev = make_device_tensor({s_rows, q_width}, at::kBFloat16, q_host.data());
      auto packed_kv = make_device_tensor({s_rows, 2 * kv_width}, at::kBFloat16, kv_interleaved.data());
      auto k_slice = packed_kv.narrow(1, 0, kv_width);
      auto v_slice = packed_kv.narrow(1, kv_width, kv_width);
      assert(!k_slice.is_contiguous() && !v_slice.is_contiguous());

      mock().reset_counters();
      auto out_dev = paged_attn_validated(q_dev, k_slice, v_slice, mk_pool, mv_pool,
                                          sched.locations, sched.requests, sched.positions,
                                          sched.plan_tensor, sched.header, kTestD, test_scale);
      assert(mock().multi_dma_calls == 0 && "256B-aligned 2D strided KV slices must bypass synMemCopyAsyncMultiple");
      std::vector<uint16_t> actual(static_cast<size_t>(s_rows) * q_width, 0);
      mock().arena.read(actual.data(), address_of(out_dev, ctx), actual.size() * 2);
      const auto expected = cpu_reference_paged_attention(
          q_host, k_host, v_host, k_pool_init, v_pool_init, slot_table,
          0, s_rows, kTestQHeads, kTestKvHeads, kTestD, 256, 512, test_scale);
      assert(actual == expected && "strided 2D packed_kv output must match CPU reference bit-for-bit");
    }

    mock().execute_math = false;
  }

  pool = at::Tensor{};
  g_ctx.reset();
  std::cout << "multitile_paged_attention_unit_test PASSED: "
            << "M=4096 launches=9 (0 HBM state bindings), 64-tile launches=8, "
            << "tails {9,11,15,17}->{2,2,2,3}, CPU ref max_abs=0.0 (0 ULP), steady_events=0\n";
  return 0;
}
