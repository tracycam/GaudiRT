// SPDX-License-Identifier: Apache-2.0
#include "communicator.hpp"

#include "graph_builder.hpp"
#include "gaudirt/common/sha256.hpp"

#include <c10/util/Exception.h>
#include <perf_lib_layer_params.h>

#include <algorithm>
#include <sstream>
#include <utility>

namespace gaudirt::generic {
using namespace runtime;

namespace {

ColdRecipe& rank_sum_recipe(Context& ctx, uint32_t bucket_elements, uint32_t world_size,
                            at::ScalarType dtype) {
  TORCH_CHECK(world_size >= 2 && world_size <= 8 &&
              (dtype == at::kBFloat16 || dtype == at::kFloat),
              "GaudiRT collective: unsupported rank_sum configuration");
  const std::string structure = "tp_rank_sum_v1:" + std::to_string(world_size) + ":" +
                                std::to_string(bucket_elements) + ":" +
                                std::to_string(static_cast<int>(dtype));
  const auto digest = common::sha256_bytes(
      reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const std::string key = "tp_rank_sum_" + digest;
  if (auto found = ctx.cold_recipes.find(key); found != ctx.cold_recipes.end()) {
    return *found->second;
  }

  auto recipe = std::make_unique<ColdRecipe>(world_size + 1);
  const std::string prefix = "grsum_" + digest.substr(0, 10) + "_b" + std::to_string(bucket_elements);
  SynapseGraphBuilder builder(prefix, *recipe);
  const synDataType io_type = dtype == at::kBFloat16 ? syn_type_bf16 : syn_type_float;

  std::vector<synTensor> wide(world_size, nullptr);
  ns_CastKernel::ParamsV3 cast_params{};
  cast_params.round_mode = CAST_ROUND_HALF_NE;

  for (uint32_t r = 0; r < world_size; ++r) {
    synTensor in = builder.tensor(io_type, {bucket_elements}, static_cast<int>(r));
    if (dtype == at::kBFloat16) {
      synTensor w = builder.tensor(syn_type_float, {bucket_elements}, -1);
      builder.node({"cast_bf16_to_f32", {in}, {w}, cast_params});
      wide[r] = w;
    } else {
      wide[r] = in;
    }
  }

  synTensor out = builder.tensor(io_type, {bucket_elements}, static_cast<int>(world_size));
  synTensor acc = wide[0];
  for (uint32_t r = 1; r < world_size; ++r) {
    const bool final_fp32_out = (dtype == at::kFloat && r + 1 == world_size);
    synTensor next = final_fp32_out ? out : builder.tensor(syn_type_float, {bucket_elements}, -1);
    builder.node({"add_fwd_f32", {acc, wide[r]}, {next}, std::monostate{}});
    acc = next;
  }
  if (dtype == at::kBFloat16) {
    builder.node({"cast_f32_to_bf16", {acc}, {out}, cast_params});
  }

  builder.compile(ctx.cache, key);
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT collective: duplicate rank_sum recipe");
  return *it->second;
}

ColdRecipe& gather_transpose_recipe(Context& ctx, uint64_t world_size, uint64_t batch_rows,
                                    uint64_t cols, at::ScalarType dtype) {
  uint64_t width_units = cols;
  synDataType syn_dtype = syn_type_bf16;
  switch (dtype) {
    case at::kBFloat16: syn_dtype = syn_type_bf16; break;
    case at::kHalf: syn_dtype = syn_type_fp16; break;
    case at::kFloat: syn_dtype = syn_type_float; break;
    case at::kInt: syn_dtype = syn_type_int32; break;
    case at::kLong:
      syn_dtype = syn_type_int32;
      width_units = cols * 2;
      break;
    case at::kByte:
    case at::kBool:
    case at::kFloat8_e4m3fn:
      syn_dtype = syn_type_uint8;
      break;
    default:
      TORCH_CHECK(false, "GaudiRT collective: unsupported gather-transpose dtype");
  }

  const std::string structure = "tp_gather_transpose_v1:" + std::to_string(world_size) + ":" +
                                std::to_string(batch_rows) + ":" + std::to_string(width_units) + ":" +
                                std::to_string(static_cast<int>(syn_dtype));
  const auto digest = common::sha256_bytes(
      reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const std::string key = "tp_gather_transpose_" + digest;
  if (auto found = ctx.cold_recipes.find(key); found != ctx.cold_recipes.end()) {
    return *found->second;
  }

  auto recipe = std::make_unique<ColdRecipe>(2);
  const std::string prefix = "ggtr_" + digest.substr(0, 10);
  SynapseGraphBuilder builder(prefix, *recipe);
  synTensor src = builder.tensor(syn_dtype, {width_units, batch_rows, world_size}, 0);
  synTensor dst = builder.tensor(syn_dtype, {width_units, world_size, batch_rows}, 1);

  synTransposeParams params{};
  params.tensorDim = 3;
  for (unsigned dim = 0; dim < 3; ++dim) {
    params.permutation[dim] = static_cast<TransposePermutationDim>(dim);
  }
  std::swap(params.permutation[1], params.permutation[2]);
  builder.node({"transpose", {src}, {dst}, params});

  builder.compile(ctx.cache, key);
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT collective: duplicate gather-transpose recipe");
  return *it->second;
}

}  // namespace

CommunicatorRegistry& communicator_registry(Context& ctx) {
  if (ctx.comm_state) {
    return *static_cast<CommunicatorRegistry*>(ctx.comm_state.get());
  }
  std::lock_guard lock(ctx.cold_mutex);
  if (!ctx.comm_state) {
    auto registry = std::make_shared<CommunicatorRegistry>();
    Context* raw_ctx = &ctx;
    ctx.comm_state = registry;
    ctx.comm_trim = [raw_ctx, registry]() { registry->trim_scratch(*raw_ctx); };
    ctx.comm_teardown = [raw_ctx, registry](bool abort_mode) {
      registry->teardown_all(*raw_ctx, abort_mode);
    };
  }
  return *static_cast<CommunicatorRegistry*>(ctx.comm_state.get());
}

uint32_t CommunicatorRegistry::register_group(Context& ctx, int rank, int world_size,
                                              std::string group_name) {
  TORCH_CHECK(world_size >= 1 && rank >= 0 && rank < world_size,
              "GaudiRT collective: invalid rank ", rank, " or world_size ", world_size);
  std::lock_guard lock(mutex_);
  const auto id = static_cast<uint32_t>(groups_.size());
  auto grp = std::make_unique<CommGroup>(
      id, rank, world_size, std::move(group_name), ctx.device.device_id());
  if (world_size == 1) {
    grp->initialized = true;
  }
  groups_.push_back(std::move(grp));
  return id;
}

void CommunicatorRegistry::init_group(Context& ctx, uint32_t comm_id, const hcclUniqueId& uid) {
  CommGroup* grp = nullptr;
  {
    std::lock_guard lock(mutex_);
    TORCH_CHECK(comm_id < groups_.size(), "GaudiRT collective: invalid comm_id ", comm_id);
    grp = groups_[comm_id].get();
    TORCH_CHECK(!grp->destroyed, "GaudiRT collective: communicator ", comm_id, " was destroyed");
    if (grp->initialized) return;
  }

  if (grp->world_size > 1) {
    synStreamHandle stream = ctx.streams.get(ctx.streams.create());
    hcclComm_t comm = nullptr;
    uint64_t barrier_addr = 0;
    try {
      ctx.prepare_stream(stream);
      barrier_addr = ctx.allocator.allocate(BlockCachingAllocator::kAlignment, stream);
      TORCH_CHECK(
          synMemsetD32Async(barrier_addr, 0, BlockCachingAllocator::kAlignment / 4, stream) == synSuccess,
          "GaudiRT collective: barrier slot zero-init failed");
      const hcclResult_t res = hcclCommInitRank(&comm, grp->world_size, uid, grp->rank);
      check_hccl(res, "hcclCommInitRank");
    } catch (...) {
      if (barrier_addr != 0) {
        ctx.allocator.record_stream(barrier_addr, stream);
        ctx.allocator.free(barrier_addr);
      }
      if (comm) hcclCommAbort(comm);
      throw;
    }
    std::lock_guard lock(mutex_);
    grp->comm_stream = stream;
    grp->comm = comm;
    grp->barrier_address = barrier_addr;
  }
  std::lock_guard lock(mutex_);
  grp->initialized = true;
}

CommGroup& CommunicatorRegistry::group(uint32_t comm_id) {
  TORCH_CHECK(comm_id < groups_.size(), "GaudiRT collective: unknown comm_id ", comm_id);
  auto* grp = groups_[comm_id].get();
  TORCH_CHECK(grp && grp->initialized && !grp->destroyed,
              "GaudiRT collective: communicator ", comm_id, " is not active");
  return *grp;
}

uint64_t CommunicatorRegistry::ensure_scratch(Context& ctx, CommGroup& grp,
                                              uint64_t bytes, synStreamHandle stream) {
  constexpr uint64_t kAlign = BlockCachingAllocator::kAlignment;
  const uint64_t aligned_bytes = (std::max<uint64_t>(bytes, kAlign) + kAlign - 1) & ~(kAlign - 1);
  if (grp.scratch_bytes < aligned_bytes) {
    if (grp.scratch_address != 0) {
      if (grp.scratch_stream) ctx.allocator.record_stream(grp.scratch_address, grp.scratch_stream);
      ctx.allocator.free(grp.scratch_address);
      grp.scratch_address = 0;
      grp.scratch_bytes = 0;
      grp.scratch_stream = nullptr;
    }
    grp.scratch_address = ctx.allocator.allocate(aligned_bytes, stream);
    grp.scratch_bytes = aligned_bytes;
  }
  grp.scratch_stream = stream;
  ctx.allocator.record_stream(grp.scratch_address, stream);
  return grp.scratch_address;
}

void CommunicatorRegistry::trim_scratch(Context& ctx) {
  std::lock_guard lock(mutex_);
  for (auto& grp : groups_) {
    if (grp && grp->scratch_address != 0) {
      if (grp->scratch_stream) {
        ctx.allocator.record_stream(grp->scratch_address, grp->scratch_stream);
      }
      ctx.allocator.free(grp->scratch_address);
      grp->scratch_address = 0;
      grp->scratch_bytes = 0;
      grp->scratch_stream = nullptr;
    }
  }
}

void CommunicatorRegistry::destroy_group(Context& ctx, uint32_t comm_id, bool abort_mode) {
  std::lock_guard lock(mutex_);
  if (comm_id >= groups_.size()) return;
  auto& grp = groups_[comm_id];
  if (!grp || grp->destroyed) return;
  if (grp->scratch_address != 0) {
    if (grp->scratch_stream) {
      ctx.allocator.record_stream(grp->scratch_address, grp->scratch_stream);
    }
    ctx.allocator.free(grp->scratch_address);
    grp->scratch_address = 0;
    grp->scratch_bytes = 0;
    grp->scratch_stream = nullptr;
  }
  if (grp->barrier_address != 0) {
    if (grp->comm_stream) {
      ctx.allocator.record_stream(grp->barrier_address, grp->comm_stream);
    }
    ctx.allocator.free(grp->barrier_address);
    grp->barrier_address = 0;
  }
  grp->destroyed = true;
  if (!grp->initialized) return;
  if (grp->comm_stream && !abort_mode) {
    synStreamSynchronize(grp->comm_stream);
  }
  if (grp->comm) {
    if (abort_mode) {
      hcclCommAbort(grp->comm);
    } else {
      hcclCommDestroy(grp->comm);
    }
    grp->comm = nullptr;
  }
  grp->comm_stream = nullptr;
}

void CommunicatorRegistry::teardown_all(Context& ctx, bool abort_mode) {
  std::lock_guard lock(mutex_);
  for (auto it = groups_.rbegin(); it != groups_.rend(); ++it) {
    auto& grp = *it;
    if (!grp || grp->destroyed) continue;
    if (grp->scratch_address != 0) {
      if (grp->scratch_stream) {
        ctx.allocator.record_stream(grp->scratch_address, grp->scratch_stream);
      }
      ctx.allocator.free(grp->scratch_address);
      grp->scratch_address = 0;
      grp->scratch_bytes = 0;
      grp->scratch_stream = nullptr;
    }
    if (grp->barrier_address != 0) {
      if (grp->comm_stream) {
        ctx.allocator.record_stream(grp->barrier_address, grp->comm_stream);
      }
      ctx.allocator.free(grp->barrier_address);
      grp->barrier_address = 0;
    }
    grp->destroyed = true;
    if (!grp->initialized) continue;
    if (grp->comm_stream && !abort_mode) {
      synStreamSynchronize(grp->comm_stream);
    }
    if (grp->comm) {
      if (abort_mode) {
        hcclCommAbort(grp->comm);
      } else {
        hcclCommDestroy(grp->comm);
      }
      grp->comm = nullptr;
    }
    grp->comm_stream = nullptr;
  }
}

void CommunicatorRegistry::record_tensor_on_stream(Context& ctx, const at::Tensor& tensor,
                                                   synStreamHandle stream) {
  if (!tensor.numel()) return;
  const auto& alloc = validated_program_input(tensor, ctx);
  if (alloc.stream != stream) {
    const auto addr = reinterpret_cast<uint64_t>(tensor.data_ptr());
    if (alloc.kind == AllocationKind::graph) {
      std::lock_guard lock(ctx.cold_mutex);
      if (ctx.graph_pool) ctx.graph_pool->record_stream(addr, stream);
    } else {
      ctx.allocator.record_stream(addr, stream);
    }
  }
}

HcclTransferSpec CommunicatorRegistry::resolve_hccl_transfer(
    uint64_t numel, at::ScalarType dtype, bool is_reduction) {
  switch (dtype) {
    case at::kBFloat16:
      return {static_cast<size_t>(numel), hcclBfloat16};
    case at::kHalf:
      return {static_cast<size_t>(numel), hcclFloat16};
    case at::kFloat:
      return {static_cast<size_t>(numel), hcclFloat32};
    default:
      break;
  }
  TORCH_CHECK(!is_reduction,
              "GaudiRT collective: Gaudi2 HCCL arithmetic reduction supports only BF16/FP16/FP32, got ",
              c10::toString(dtype));
  switch (dtype) {
    case at::kChar:
    case at::kByte:
    case at::kBool:
    case at::kFloat8_e4m3fn:
    case at::kFloat8_e5m2:
    case at::kShort:
    case at::kInt:
    case at::kLong:
    case at::kDouble: {
      const uint64_t bytes = numel * static_cast<uint64_t>(c10::elementSize(dtype));
      if (bytes % 4 == 0) {
        return {static_cast<size_t>(bytes / 4), hcclFloat32};
      }
      if (bytes % 2 == 0) {
        return {static_cast<size_t>(bytes / 2), hcclBfloat16};
      }
      TORCH_CHECK(false,
                  "GaudiRT collective: Gaudi2 HCCL raw byte transfer requires an even byte count, got ",
                  bytes, " bytes for ", c10::toString(dtype));
    }
    default:
      TORCH_CHECK(false, "GaudiRT collective: unsupported tensor dtype ", c10::toString(dtype));
  }
}

void CommunicatorRegistry::check_hccl(hcclResult_t status, const char* op) {
  if (status != hcclSuccess) {
    const char* msg = hcclGetErrorString(status);
    TORCH_CHECK(false, "GaudiRT collective: ", op, " failed with status=",
                static_cast<int>(status), " (", (msg ? msg : "unknown"), ")");
  }
}

bool CommunicatorRegistry::should_gather_reduce(uint64_t exact_elements, uint32_t bucket_elements,
                                                int world_size, at::ScalarType dtype) {
  if (world_size <= 2 || world_size > 8) return false;
  if (dtype != at::kBFloat16 && dtype != at::kFloat) return false;
  if (exact_elements == 0 || bucket_elements < exact_elements) return false;
  const uint64_t elem_size = c10::elementSize(dtype);
  const uint64_t exact_bytes = exact_elements * elem_size;
  if (exact_bytes % BlockCachingAllocator::kAlignment != 0) return false;
  const uint64_t total_bucket_bytes =
      static_cast<uint64_t>(world_size) * static_cast<uint64_t>(bucket_elements) * elem_size;
  return total_bucket_bytes <= kMaxGatherReduceBytes;
}

void CommunicatorRegistry::launch_rank_sum(Context& ctx, uint64_t gathered_addr, uint64_t dst_addr,
                                           uint64_t exact_elements, uint32_t bucket_elements,
                                           uint32_t world_size, at::ScalarType dtype,
                                           synStreamHandle stream) {
  const uint64_t elem_size = c10::elementSize(dtype);
  const uint64_t exact_bytes = exact_elements * elem_size;
  ColdRecipe* recipe = nullptr;
  {
    std::lock_guard lock(ctx.cold_mutex);
    recipe = &rank_sum_recipe(ctx, bucket_elements, world_size, dtype);
  }
  auto bindings = recipe->tensors;
  for (uint32_t r = 0; r < world_size; ++r) {
    bindings[r].pTensorAddress = gathered_addr + static_cast<uint64_t>(r) * exact_bytes;
  }
  bindings[world_size].pTensorAddress = dst_addr;
  uint64_t workspace = 0;
  if (recipe->workspace > 0) {
    workspace = ctx.allocator.allocate(recipe->workspace, stream);
  }
  const synStatus st = synLaunch(stream, bindings.data(), bindings.size(),
                                 workspace, recipe->recipe, 0);
  if (workspace != 0) {
    ctx.allocator.free(workspace);
  }
  TORCH_CHECK(st == synSuccess, "GaudiRT collective: rank_sum launch failed, status=",
              static_cast<int>(st));
}

void CommunicatorRegistry::launch_gather_transpose(Context& ctx, uint64_t gathered_addr,
                                                   uint64_t dst_addr, uint64_t world_size,
                                                   uint64_t batch_rows, uint64_t cols,
                                                   at::ScalarType dtype, synStreamHandle stream) {
  ColdRecipe* recipe = nullptr;
  {
    std::lock_guard lock(ctx.cold_mutex);
    recipe = &gather_transpose_recipe(ctx, world_size, batch_rows, cols, dtype);
  }
  auto bindings = recipe->tensors;
  bindings[0].pTensorAddress = gathered_addr;
  bindings[1].pTensorAddress = dst_addr;
  uint64_t workspace = 0;
  if (recipe->workspace > 0) {
    workspace = ctx.allocator.allocate(recipe->workspace, stream);
  }
  const synStatus st = synLaunch(stream, bindings.data(), bindings.size(),
                                 workspace, recipe->recipe, 0);
  if (workspace != 0) {
    ctx.allocator.free(workspace);
  }
  TORCH_CHECK(st == synSuccess, "GaudiRT collective: gather_transpose launch failed, status=",
              static_cast<int>(st));
}

}  // namespace gaudirt::generic
