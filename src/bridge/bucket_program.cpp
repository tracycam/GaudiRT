// SPDX-License-Identifier: Apache-2.0
#include "bucket_program.hpp"

#include "bridge_internal.hpp"
#include "fx_geometry.hpp"
#include "fx_op_lowering.hpp"
#include "graph_builder.hpp"
#include "layout_copy.hpp"

#include <ATen/Functions.h>
#include <c10/util/Exception.h>
#include <c10/util/SmallVector.h>
#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <memory>
#include <mutex>
#include <utility>
#include <variant>

namespace gaudirt::generic {
using namespace runtime;

BucketProgram::BucketProgram(std::vector<int64_t> roles, std::vector<FxNode> operations,
              std::vector<int64_t> outputs)
    : roles_(std::move(roles)), operations_(std::move(operations)), outputs_(std::move(outputs)) {
  TORCH_CHECK(!roles_.empty() && !operations_.empty(), "GaudiRT FX: empty program");
  bool has_dynamic = false;
  for (auto role : roles_) {
    TORCH_CHECK(role == 0 || role == 1, "GaudiRT FX: invalid input role");
    has_dynamic |= role == 1;
  }
  TORCH_CHECK(has_dynamic, "GaudiRT FX: no dynamic token input");
  std::vector<int64_t> residual_producers, rotary_producers;
  for (size_t i = 0; i < operations_.size(); ++i) {
    const auto& op = operations_[i];
    const int64_t upper = roles_.size() + i;
    TORCH_CHECK(!op.inputs.empty() && std::all_of(op.inputs.begin(), op.inputs.end(),
                [upper](int64_t id) { return id >= 0 && id < upper; }), "GaudiRT FX: invalid node inputs");
    const auto no_params = std::holds_alternative<std::monostate>(op.parameters);
    bool valid = false;
    switch (op.op) {
      case FxOp::linear:
        valid = (op.inputs.size() == 2 || op.inputs.size() == 3) &&
                std::holds_alternative<FxLinear>(op.parameters);
        break;
      case FxOp::block_fp8_linear: case FxOp::fp8_linear:
        valid = (op.inputs.size() == 3 || op.inputs.size() == 4) && no_params;
        break;
      case FxOp::add: case FxOp::sub: case FxOp::mul: case FxOp::div: case FxOp::bmm:
      case FxOp::minimum: case FxOp::maximum: case FxOp::eq: case FxOp::ne:
      case FxOp::lt: case FxOp::le: case FxOp::gt: case FxOp::ge:
      case FxOp::logical_and: case FxOp::logical_or:
      case FxOp::bitwise_and: case FxOp::bitwise_or:
        valid = op.inputs.size() == 2 && no_params;
        break;
      case FxOp::where:
        valid = op.inputs.size() == 3 && no_params;
        break;
      case FxOp::embedding:
        valid = op.inputs.size() == 2 && no_params && op.inputs[0] < static_cast<int64_t>(roles_.size()) &&
                op.inputs[1] < static_cast<int64_t>(roles_.size());
        if (valid) has_embedding_ = true;
        break;
      case FxOp::shard_embedding: {
        const auto* params = std::get_if<FxShardEmbedding>(&op.parameters);
        valid = op.inputs.size() == 2 && params &&
                op.inputs[0] < static_cast<int64_t>(roles_.size()) &&
                op.inputs[1] < static_cast<int64_t>(roles_.size()) &&
                params->org_start >= 0 && params->org_end >= params->org_start &&
                params->added_start >= params->org_end && params->added_end >= params->added_start &&
                params->added_end <= INT32_MAX &&
                params->added_offset >= INT32_MIN && params->added_offset <= INT32_MAX &&
                (params->added_end == params->added_start ||
                 params->added_start - params->added_offset >= params->org_end - params->org_start);
        if (valid) has_embedding_ = true;
        break;
      }
      case FxOp::gather_rows:
        valid = op.inputs.size() == 2 && no_params && op.inputs[1] < static_cast<int64_t>(roles_.size());
        break;
      case FxOp::rms_norm: case FxOp::gemma_rms_norm:
      case FxOp::fused_add_rms_norm: case FxOp::fused_add_gemma_rms_norm: {
        const auto* params = std::get_if<FxEpsilon>(&op.parameters);
        const bool is_fused = (op.op == FxOp::fused_add_rms_norm || op.op == FxOp::fused_add_gemma_rms_norm);
        valid = op.inputs.size() == (is_fused ? 3u : 2u) && params &&
                std::isfinite(params->value) && params->value > 0;
        break;
      }
      case FxOp::silu_and_mul: case FxOp::copy: case FxOp::isnan: case FxOp::logical_not:
        valid = op.inputs.size() == 1 && no_params;
        break;
      case FxOp::gelu_and_mul:
        valid = op.inputs.size() == 1 && std::holds_alternative<FxGeluMul>(op.parameters);
        break;
      case FxOp::ewise_unary: {
        const auto* params = std::get_if<FxUnary>(&op.parameters);
        valid = op.inputs.size() == 1 && params && params->op_code >= 0 && params->op_code <= 11;
        break;
      }
      case FxOp::fused_residual: case FxOp::rotary_key: {
        valid = op.inputs.size() == 1 && no_params && op.inputs[0] >= static_cast<int64_t>(roles_.size());
        if (valid) {
          auto& producers = op.op == FxOp::fused_residual ? residual_producers : rotary_producers;
          const auto producer_op = operations_[op.inputs[0] - roles_.size()].op;
          const bool ok_producer = op.op == FxOp::fused_residual
              ? (producer_op == FxOp::fused_add_rms_norm || producer_op == FxOp::fused_add_gemma_rms_norm)
              : (producer_op == FxOp::rotary || producer_op == FxOp::rotary_pos);
          valid = ok_producer &&
                  std::find(producers.begin(), producers.end(), op.inputs[0]) == producers.end();
          if (valid) producers.push_back(op.inputs[0]);
        }
        break;
      }
      case FxOp::feature_slice: {
        const auto* params = std::get_if<FxSlice>(&op.parameters);
        valid = op.inputs.size() == 1 && params && params->start >= 0 && params->width > 0 && params->total > 0;
        break;
      }
      case FxOp::slice_dim: {
        const auto* params = std::get_if<FxSliceDim>(&op.parameters);
        valid = op.inputs.size() == 1 && params && params->start >= 0 && params->length > 0;
        break;
      }
      case FxOp::rotary: {
        const auto* params = std::get_if<FxRotary>(&op.parameters);
        valid = op.inputs.size() == 5 && params && params->context > 0 &&
                params->head_dim > 0 && params->rotary_dim > 0;
        break;
      }
      case FxOp::rotary_pos: {
        const auto* params = std::get_if<FxRotary>(&op.parameters);
        valid = op.inputs.size() == 4 && params && params->head_dim > 0 && params->rotary_dim > 0;
        break;
      }
      case FxOp::cast:
        valid = op.inputs.size() == 1 &&
                (std::holds_alternative<FxCast>(op.parameters) ||
                 std::holds_alternative<FxCastDtype>(op.parameters));
        break;
      case FxOp::concat:
        valid = op.inputs.size() >= 2 &&
                (no_params || std::holds_alternative<FxAxis>(op.parameters));
        break;
      case FxOp::reshape: {
        const auto* params = std::get_if<FxShape>(&op.parameters);
        valid = op.inputs.size() == 1 && params && params->dimensions.size() <= 5 &&
                std::all_of(params->dimensions.begin(), params->dimensions.end(), [](int64_t d) { return d > 0; });
        break;
      }
      case FxOp::expand: {
        const auto* params = std::get_if<FxShape>(&op.parameters);
        valid = op.inputs.size() == 1 && params && !params->dimensions.empty() && params->dimensions.size() <= 5;
        break;
      }
      case FxOp::transpose: {
        const auto* params = std::get_if<FxTranspose>(&op.parameters);
        valid = op.inputs.size() == 1 && params && params->dim0 >= 0 && params->dim1 >= 0 &&
                params->dim0 != params->dim1;
        break;
      }
      case FxOp::scalar_add: case FxOp::scalar_sub: case FxOp::scalar_mul: case FxOp::scalar_div: {
        const auto* params = std::get_if<FxScalar>(&op.parameters);
        valid = op.inputs.size() == 1 && params && std::isfinite(params->value);
        break;
      }
      case FxOp::scalar_eq: case FxOp::scalar_ne: case FxOp::scalar_lt:
      case FxOp::scalar_le: case FxOp::scalar_gt: case FxOp::scalar_ge: {
        const auto* params = std::get_if<FxScalar>(&op.parameters);
        valid = op.inputs.size() == 1 && params && !std::isnan(params->value);
        break;
      }
      case FxOp::sum_dim: case FxOp::mean_dim: case FxOp::amax_dim:
        valid = op.inputs.size() == 1 && std::holds_alternative<FxReduce>(op.parameters);
        break;
      case FxOp::softmax: case FxOp::log_softmax:
        valid = op.inputs.size() == 1 && std::holds_alternative<FxAxis>(op.parameters);
        break;
      case FxOp::layer_norm: {
        const auto* params = std::get_if<FxLayerNorm>(&op.parameters);
        valid = params && op.inputs.size() ==
                (1 + static_cast<size_t>(params->has_weight) + static_cast<size_t>(params->has_bias)) &&
                std::isfinite(params->epsilon) && params->epsilon > 0;
        break;
      }
      case FxOp::full_like:
        valid = op.inputs.size() == 1 && std::holds_alternative<FxConstant>(op.parameters);
        break;
    }
    TORCH_CHECK(valid, "GaudiRT FX: invalid normalized node");
  }
  if (outputs_.empty()) outputs_.push_back(roles_.size() + operations_.size() - 1);
  std::vector<int64_t> unique_outputs = outputs_;
  std::sort(unique_outputs.begin(), unique_outputs.end());
  TORCH_CHECK(std::adjacent_find(unique_outputs.begin(), unique_outputs.end()) == unique_outputs.end(),
              "GaudiRT FX: duplicate program output");
  for (auto id : outputs_)
    TORCH_CHECK(id >= static_cast<int64_t>(roles_.size()) &&
                id < static_cast<int64_t>(roles_.size() + operations_.size()),
                "GaudiRT FX: output must be a program operation");
  rotary_cache_inputs_ = rotary_cache_inputs(roles_, operations_);
  rotary_cache_contexts_.resize(roles_.size());
  for (const auto& op : operations_) {
    if (op.op != FxOp::rotary) continue;
    const auto id = op.inputs[2];
    if (id < static_cast<int64_t>(roles_.size()) && rotary_cache_inputs_[id])
      rotary_cache_contexts_[id] = std::max(rotary_cache_contexts_[id], std::get<FxRotary>(op.parameters).context);
  }
  std::vector<bool> static_inputs(roles_.size(), false);
  const auto mark_weight = [&](int64_t id) {
    if (id >= 0 && id < static_cast<int64_t>(roles_.size()) && !roles_[id]) static_inputs[id] = true;
  };
  for (const auto& op : operations_) {
    switch (op.op) {
      case FxOp::linear:
        mark_weight(op.inputs[1]);
        if (op.inputs.size() == 3) mark_weight(op.inputs[2]);
        break;
      case FxOp::block_fp8_linear: case FxOp::fp8_linear:
        mark_weight(op.inputs[1]);
        mark_weight(op.inputs[2]);
        if (op.inputs.size() == 4) mark_weight(op.inputs[3]);
        break;
      case FxOp::rms_norm: case FxOp::gemma_rms_norm: case FxOp::embedding: case FxOp::shard_embedding:
        mark_weight(op.inputs[1]);
        break;
      case FxOp::fused_add_rms_norm: case FxOp::fused_add_gemma_rms_norm:
      case FxOp::rotary: case FxOp::rotary_pos:
        mark_weight(op.inputs[2]);
        break;
      case FxOp::layer_norm:
        for (size_t idx = 1; idx < op.inputs.size(); ++idx) mark_weight(op.inputs[idx]);
        break;
      default: break;
    }
  }
  for (size_t i = 0; i < static_inputs.size(); ++i)
    if (static_inputs[i]) static_input_indices_.push_back(i);
}

BucketProgram::~BucketProgram() {
  auto* cache = cached_contexts_.load(std::memory_order_relaxed);
  while (cache) {
    auto* next = cache->next;
    delete cache;
    cache = next;
  }
  auto* bindings = static_bindings_.load(std::memory_order_relaxed);
  while (bindings) {
    auto* next = bindings->next;
    delete bindings;
    bindings = next;
  }
}

void BucketProgram::initialize_geometry(const std::vector<at::Tensor>& inputs) {
  auto geometry = infer_fx_geometry(inputs, roles_, operations_, outputs_);
  specs_ = std::move(geometry.specs);
  digest_ = std::move(geometry.digest);
}

const BucketProgram::StaticBindings* BucketProgram::find_static_bindings(
    const std::vector<at::Tensor>& inputs, const Context& ctx, synStreamHandle stream) const {
  for (auto* binding = static_bindings_.load(std::memory_order_acquire); binding; binding = binding->next) {
    if (binding->generation != ctx.generation || binding->stream != stream) continue;
    bool matches = true;
    for (const auto& cached : binding->inputs) {
      const auto& value = inputs[cached.index];
      const auto* impl = value.unsafeGetTensorImpl();
      if (impl != cached.tensor._unsafe_get_target() ||
          reinterpret_cast<uint64_t>(value.data_ptr()) != cached.address) {
        matches = false;
        break;
      }
      const auto& version = impl->version_counter();
      if (version.enabled() != cached.versioned ||
          (cached.versioned && version.current_version() != cached.version) ||
          (!cached.versioned && (!value.sizes().equals(specs_[cached.index].sizes) ||
                                 !value.strides().equals(cached.strides)))) {
        matches = false;
        break;
      }
    }
    if (matches) return binding;
  }
  return nullptr;
}

const BucketProgram::StaticBindings* BucketProgram::bind_static_inputs(
    const std::vector<at::Tensor>& inputs, Context& ctx, synStreamHandle stream) {
  if (static_input_indices_.empty() || static_binding_count_.load(std::memory_order_relaxed) >= 64) return nullptr;
  auto binding = std::make_unique<StaticBindings>();
  binding->generation = ctx.generation;
  binding->stream = stream;
  binding->inputs.reserve(static_input_indices_.size());
  for (const auto index : static_input_indices_) {
    const auto& input = inputs[index];
    const auto& allocation = validated_program_input(input, ctx);
    const auto address = reinterpret_cast<uint64_t>(input.data_ptr());
    const auto base = reinterpret_cast<uint64_t>(input.storage().data());
    const uint64_t bytes = rotary_cache_inputs_[index]
        ? kRotaryStaticRows * static_cast<uint64_t>(input.size(1)) * input.element_size()
        : input.nbytes();
    const auto storage = input.storage().nbytes();
    if (!input.is_contiguous() || address % BlockCachingAllocator::kAlignment != 0 || address < base ||
        address - base > storage || bytes > storage - (address - base)) return nullptr;
    for (const auto& other : binding->inputs)
      if (address < other.address + other.bytes && other.address < address + bytes) return nullptr;
    if (allocation.stream != stream) {
      if (allocation.kind == AllocationKind::graph) ctx.graph_pool->record_stream(address, stream);
      else ctx.allocator.record_stream(address, stream);
    }
    const auto& version = input.unsafeGetTensorImpl()->version_counter();
    binding->inputs.push_back({index,
        c10::weak_intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl>(input.getIntrusivePtr()),
        address, bytes, version.enabled() ? version.current_version() : 0, version.enabled(),
        version.enabled() ? std::vector<int64_t>{} : input.strides().vec()});
  }
  auto* head = static_bindings_.load(std::memory_order_acquire);
  do {
    binding->next = head;
  } while (!static_bindings_.compare_exchange_weak(head, binding.get(), std::memory_order_acq_rel,
                                                    std::memory_order_acquire));
  static_binding_count_.fetch_add(1, std::memory_order_relaxed);
  return binding.release();
}

ColdRecipe& BucketProgram::cached_recipe(Context& ctx, uint32_t bucket) {
  auto* head = cached_contexts_.load(std::memory_order_acquire);
  CachedContext* cached = nullptr;
  while (!cached) {
    for (auto* item = head; item; item = item->next) {
      if (item->generation == ctx.generation) {
        cached = item;
        break;
      }
    }
    if (cached) break;
    auto fresh = std::make_unique<CachedContext>(ctx.generation);
    fresh->next = head;
    if (cached_contexts_.compare_exchange_weak(head, fresh.get(), std::memory_order_acq_rel,
                                                std::memory_order_acquire))
      cached = fresh.release();
  }
  auto& slot = cached->recipes[std::countr_zero(bucket)];
  auto* result = slot.load(std::memory_order_acquire);
  if (!result) {
    std::lock_guard lock(ctx.cold_mutex);
    result = slot.load(std::memory_order_relaxed);
    if (!result) {
      result = &recipe(ctx, bucket);
      slot.store(result, std::memory_order_release);
    }
  }
  return *result;
}

ColdRecipe& BucketProgram::recipe(Context& ctx, uint32_t bucket) {
  const auto prefix = "g" + digest_.substr(0, 12) + "_b" + std::to_string(bucket);
  const auto key = "fx_program_" + digest_ + "_b" + std::to_string(bucket);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  const auto row_count_slot = roles_.size() + outputs_.size();
  auto compiled = std::make_unique<ColdRecipe>(row_count_slot + static_cast<size_t>(has_embedding_));
  {
    SynapseGraphBuilder builder(prefix, *compiled);
    std::vector<synTensor> values(specs_.size(), nullptr);
    auto tensor_nd = [&](synDataType type, const std::vector<uint64_t>& dims, int slot) {
      return builder.tensor(type, dims, slot);
    };
    auto tensor = [&](synDataType type, uint64_t fcd, uint64_t rows, int slot,
                      uint32_t rank = 2, uint64_t depth = 1) {
      std::vector<uint64_t> dims{fcd};
      if (rank >= 2) dims.push_back(rows);
      if (rank == 3) dims.push_back(depth);
      return tensor_nd(type, dims, slot);
    };
    auto tensor_from_spec = [&](const TensorSpec& spec, int slot) {
      const auto type = spec.dtype == at::kFloat8_e4m3fn ? syn_type_fp8_143 : cast_type(spec.dtype).syn;
      return tensor_nd(type, spec.synapse_sizes(bucket), slot);
    };
    auto node = [&](std::vector<synTensor> in, synTensor out, NodeParameters params,
                    std::string guid) {
      builder.node(NodeSpec{std::move(guid), std::move(in), {out}, std::move(params)});
    };
    for (size_t i = 0; i < roles_.size(); ++i) {
      const auto& spec = specs_[i];
      if (rotary_cache_inputs_[i]) {
        const auto width = static_cast<uint64_t>(spec.sizes[1]);
        values[i] = builder.tensor(cast_type(spec.dtype).syn, {width, kRotaryStaticRows}, i);
      } else {
        values[i] = tensor_from_spec(spec, i);
      }
    }
    const auto row_count = has_embedding_ ?
        tensor_nd(syn_type_int32, {1}, static_cast<int>(row_count_slot)) : nullptr;
    for (size_t i = 0; i < operations_.size(); ++i) {
      const auto& op = operations_[i];
      const auto id = roles_.size() + i;
      if (op.op == FxOp::fused_residual || op.op == FxOp::rotary_key) {
        TORCH_CHECK(values[id], "GaudiRT FX: missing second result tensor");
        continue;
      }
      if (op.op == FxOp::feature_slice && values[id]) continue;
      const auto& spec = specs_[id];
      const auto type = cast_type(spec.dtype);
      const uint64_t width = spec.sizes.back();
      const auto output = std::find(outputs_.begin(), outputs_.end(), static_cast<int64_t>(id));
      const int slot = output == outputs_.end() ? -1 :
          static_cast<int>(roles_.size() + std::distance(outputs_.begin(), output));
      values[id] = tensor_from_spec(spec, slot);
      if (op.op == FxOp::transpose) {
        const auto& transpose = std::get<FxTranspose>(op.parameters);
        synTransposeParams params{};
        params.tensorDim = spec.sizes.size();
        for (unsigned dim = 0; dim < params.tensorDim; ++dim)
          params.permutation[dim] = static_cast<TransposePermutationDim>(dim);
        std::swap(params.permutation[params.tensorDim - 1 - transpose.dim0],
                  params.permutation[params.tensorDim - 1 - transpose.dim1]);
        node({values[op.inputs[0]]}, values[id], params, "transpose");
        continue;
      }
      if (op.op == FxOp::reshape || op.op == FxOp::copy) {
        node({values[op.inputs[0]]}, values[id], std::monostate{}, "reshape");
        continue;
      }
      if (op.op == FxOp::expand) {
        const auto src_dims = specs_[op.inputs[0]].synapse_sizes(bucket);
        const auto dst_dims = spec.synapse_sizes(bucket);
        if (src_dims == dst_dims) {
          emit_cast(builder, values[op.inputs[0]], dst_dims, spec.dtype, spec.dtype, values[id]);
        } else {
          auto cur = values[op.inputs[0]];
          auto cur_dims = src_dims;
          if (cur_dims.size() < dst_dims.size()) {
            while (cur_dims.size() < dst_dims.size()) cur_dims.push_back(1);
            auto padded = tensor_nd(type.syn, cur_dims, -1);
            node({cur}, cur_dims == dst_dims ? values[id] : padded, std::monostate{}, "reshape");
            cur = padded;
          }
          if (cur_dims != dst_dims) node({cur}, values[id], std::monostate{}, "broadcast");
        }
        continue;
      }
      if (op.op == FxOp::bmm) {
        synGEMMParams params{false, false};
        node({values[op.inputs[0]], values[op.inputs[1]]}, values[id], params, "batch_gemm");
        continue;
      }
      if (op.op == FxOp::gather_rows) {
        ns_GatherKernel::ParamsV2 params{};
        params.axis = 1;
        params.batchDims = 0;
        const auto guid = "gather_fwd_" + std::string(type.name);
        auto idx = values[op.inputs[1]];
        if (specs_[op.inputs[1]].dtype == at::kLong) {
          idx = emit_cast(builder, idx, specs_[op.inputs[1]].synapse_sizes(bucket), at::kLong, at::kInt);
        }
        node({values[op.inputs[0]], idx}, values[id], params, guid.c_str());
        continue;
      }
      if (op.op == FxOp::cast) {
        emit_cast(builder, values[op.inputs[0]], spec.synapse_sizes(bucket),
                  specs_[op.inputs[0]].dtype, spec.dtype, values[id]);
        continue;
      }
      if (op.op == FxOp::concat) {
        std::vector<synTensor> inputs;
        for (auto input : op.inputs) inputs.push_back(values[input]);
        const int64_t pt_dim = std::holds_alternative<FxAxis>(op.parameters)
                                   ? std::get<FxAxis>(op.parameters).dim
                                   : static_cast<int64_t>(spec.sizes.size()) - 1;
        const int64_t resolved = pt_dim < 0 ? pt_dim + static_cast<int64_t>(spec.sizes.size()) : pt_dim;
        synConcatenateParams params{static_cast<unsigned int>(spec.sizes.size() - 1 - resolved)};
        node(inputs, values[id], params, "concat");
        continue;
      }
      if (op.op == FxOp::embedding) {
        // Token values were validated at the FX boundary. Mask only the
        // unused bucket rows before gather; the external input stays bound.
        auto tok_src = values[op.inputs[0]];
        if (specs_[op.inputs[0]].dtype == at::kLong) {
          tok_src = emit_cast(builder, tok_src, {bucket}, at::kLong, at::kInt);
        }
        auto input = tensor_nd(syn_type_int32, {1, bucket}, -1);
        auto masked = tensor_nd(syn_type_int32, {bucket}, -1);
        auto status = tensor_nd(syn_type_int32, {bucket}, -1);
        node({tok_src}, input, std::monostate{}, "reshape");
        builder.node(NodeSpec{"gk_scheduler_tokens_i32_v1", {input, row_count},
                              {masked, status}, int32_t{1}});
        ns_GatherKernel::ParamsV2 params{};
        params.axis = 1;
        params.batchDims = 0;
        const auto guid = "gather_fwd_" + std::string(type.name);
        node({values[op.inputs[1]], masked}, values[id], params, guid.c_str());
        continue;
      }
      if (op.op == FxOp::shard_embedding) {
        lower_shard_embedding(builder, op, spec, bucket, row_count, values, values[id]);
        continue;
      }
      if (op.op == FxOp::rotary) {
        const auto& rotary = std::get<FxRotary>(op.parameters);
        size_t key_id = specs_.size();
        for (size_t j = i + 1; j < operations_.size(); ++j)
          if (operations_[j].op == FxOp::rotary_key && operations_[j].inputs[0] == static_cast<int64_t>(id))
            key_id = roles_.size() + j;
        const auto key_output = std::find(outputs_.begin(), outputs_.end(), static_cast<int64_t>(key_id));
        const int key_slot = key_output == outputs_.end() ? -1 :
            static_cast<int>(roles_.size() + std::distance(outputs_.begin(), key_output));
        auto key = tensor(type.syn, specs_[op.inputs[1]].sizes[1], bucket, key_slot);
        if (key_id < values.size()) values[key_id] = key;
        RotaryParameters params{
            static_cast<int32_t>(rotary.head_dim), static_cast<int32_t>(rotary.rotary_dim), rotary.neox};
        builder.node(NodeSpec{"gk_rotary_bf16_v1",
                              {values[op.inputs[0]], values[op.inputs[1]], values[op.inputs[3]],
                               values[op.inputs[2]], values[op.inputs[4]]},
                              {values[id], key}, params});
        continue;
      }
      if (op.op == FxOp::rotary_pos) {
        const auto& rotary = std::get<FxRotary>(op.parameters);
        size_t key_id = specs_.size();
        for (size_t j = i + 1; j < operations_.size(); ++j)
          if (operations_[j].op == FxOp::rotary_key && operations_[j].inputs[0] == static_cast<int64_t>(id))
            key_id = roles_.size() + j;
        const auto key_output = std::find(outputs_.begin(), outputs_.end(), static_cast<int64_t>(key_id));
        const int key_slot = key_output == outputs_.end() ? -1 :
            static_cast<int>(roles_.size() + std::distance(outputs_.begin(), key_output));
        const auto k_syn_dims = specs_[op.inputs[1]].synapse_sizes(bucket);
        auto key = tensor_nd(type.syn, k_syn_dims, key_slot);
        if (key_id < values.size()) values[key_id] = key;
        emit_rotary_pos(builder, values[op.inputs[0]], values[op.inputs[1]], values[op.inputs[2]],
                        values[op.inputs[3]], values[id], key, spec.synapse_sizes(bucket), k_syn_dims,
                        specs_[op.inputs[3]].dtype, bucket,
                        static_cast<int32_t>(rotary.head_dim),
                        static_cast<int32_t>(rotary.rotary_dim), rotary.neox);
        continue;
      }
      if (op.op == FxOp::fused_add_rms_norm || op.op == FxOp::fused_add_gemma_rms_norm) {
        size_t residual_id = specs_.size();
        for (size_t j = i + 1; j < operations_.size(); ++j)
          if (operations_[j].op == FxOp::fused_residual && operations_[j].inputs[0] == static_cast<int64_t>(id))
            residual_id = roles_.size() + j;
        const auto residual_output = std::find(outputs_.begin(), outputs_.end(), static_cast<int64_t>(residual_id));
        const int residual_slot = residual_output == outputs_.end() ? -1 :
            static_cast<int>(roles_.size() + std::distance(outputs_.begin(), residual_output));
        const auto syn_dims = spec.synapse_sizes(bucket);
        auto residual = tensor_nd(type.syn, syn_dims, residual_slot);
        if (residual_id < values.size()) values[residual_id] = residual;
        const float epsilon = std::get<FxEpsilon>(op.parameters).value;
        const float weight_offset = op.op == FxOp::fused_add_gemma_rms_norm ? 1.0f : 0.0f;
        emit_rms_norm(builder, values[op.inputs[0]], values[op.inputs[1]], values[op.inputs[2]],
                      values[id], residual, syn_dims, epsilon, weight_offset);
        continue;
      }
      if (op.op == FxOp::feature_slice) {
        lower_feature_slice(builder, i, roles_, operations_, outputs_, specs_, bucket, values);
        continue;
      }
      if (op.op == FxOp::slice_dim) {
        lower_slice_dim(builder, op, specs_[op.inputs[0]], spec, bucket, values[op.inputs[0]], values[id]);
        continue;
      }
      if (op.op == FxOp::scalar_add || op.op == FxOp::scalar_sub ||
          op.op == FxOp::scalar_mul || op.op == FxOp::scalar_div) {
        lower_scalar_arith(builder, op, spec, bucket, values, values[id]);
        continue;
      }
      if (op.op == FxOp::scalar_eq || op.op == FxOp::scalar_ne || op.op == FxOp::scalar_lt ||
          op.op == FxOp::scalar_le || op.op == FxOp::scalar_gt || op.op == FxOp::scalar_ge) {
        lower_scalar_cmp(builder, op, specs_[op.inputs[0]], spec, bucket, values, values[id]);
        continue;
      }
      if (op.op == FxOp::ewise_unary) {
        const auto& u = std::get<FxUnary>(op.parameters);
        emit_ewise_unary(builder, values[op.inputs[0]], values[id], spec.dtype,
                         spec.synapse_sizes(bucket), u.op_code, u.alpha, u.beta);
        continue;
      }
      if (op.op == FxOp::isnan) {
        emit_isnan(builder, values[op.inputs[0]], values[id], specs_[op.inputs[0]].dtype);
        continue;
      }
      if (op.op == FxOp::logical_not) {
        emit_logical_not(builder, values[op.inputs[0]], values[id]);
        continue;
      }
      if (op.op == FxOp::full_like) {
        const auto& c = std::get<FxConstant>(op.parameters);
        emit_constant(builder, spec.dtype, spec.synapse_sizes(bucket),
                      c.float_value, c.int_value, c.is_integral, values[id]);
        continue;
      }
      if (op.op == FxOp::add || op.op == FxOp::sub || op.op == FxOp::mul || op.op == FxOp::div ||
          op.op == FxOp::minimum || op.op == FxOp::maximum ||
          op.op == FxOp::eq || op.op == FxOp::ne || op.op == FxOp::lt ||
          op.op == FxOp::le || op.op == FxOp::gt || op.op == FxOp::ge ||
          op.op == FxOp::logical_and || op.op == FxOp::logical_or ||
          op.op == FxOp::bitwise_and || op.op == FxOp::bitwise_or) {
        lower_binary_arith(builder, op, spec, specs_, bucket, values, values[id]);
        continue;
      }
      if (op.op == FxOp::where) {
        emit_where(builder, values[op.inputs[0]], values[op.inputs[1]], values[op.inputs[2]],
                   values[id], spec.dtype, specs_[op.inputs[0]].synapse_sizes(bucket),
                   specs_[op.inputs[1]].synapse_sizes(bucket),
                   specs_[op.inputs[2]].synapse_sizes(bucket), spec.synapse_sizes(bucket));
        continue;
      }
      if (op.op == FxOp::rms_norm || op.op == FxOp::gemma_rms_norm) {
        const float epsilon = std::get<FxEpsilon>(op.parameters).value;
        const float weight_offset = op.op == FxOp::gemma_rms_norm ? 1.0f : 0.0f;
        emit_rms_norm(builder, values[op.inputs[0]], nullptr, values[op.inputs[1]],
                      values[id], nullptr, spec.synapse_sizes(bucket), epsilon, weight_offset);
        continue;
      }
      if (op.op == FxOp::silu_and_mul || op.op == FxOp::gelu_and_mul) {
        const bool is_gelu = op.op == FxOp::gelu_and_mul;
        const int32_t approx = is_gelu ? std::get<FxGeluMul>(op.parameters).approximate : 0;
        emit_gated_mul(builder, is_gelu, approx, values[op.inputs[0]], values[id],
                       specs_[op.inputs[0]].synapse_sizes(bucket), spec.synapse_sizes(bucket));
        continue;
      }
      if (op.op == FxOp::sum_dim || op.op == FxOp::mean_dim || op.op == FxOp::amax_dim) {
        const auto& r = std::get<FxReduce>(op.parameters);
        emit_reduction(builder, op.op, values[op.inputs[0]], values[id], spec.dtype,
                       specs_[op.inputs[0]].synapse_sizes(bucket), spec.synapse_sizes(bucket),
                       r.dim, r.keepdim);
        continue;
      }
      if (op.op == FxOp::softmax || op.op == FxOp::log_softmax) {
        const auto& ax = std::get<FxAxis>(op.parameters);
        emit_softmax(builder, op.op == FxOp::log_softmax, values[op.inputs[0]], values[id],
                     spec.dtype, spec.synapse_sizes(bucket), ax.dim);
        continue;
      }
      if (op.op == FxOp::layer_norm) {
        const auto& ln = std::get<FxLayerNorm>(op.parameters);
        size_t next_in = 1;
        synTensor weight = ln.has_weight ? values[op.inputs[next_in++]] : nullptr;
        synTensor bias = ln.has_bias ? values[op.inputs[next_in++]] : nullptr;
        emit_layer_norm(builder, values[op.inputs[0]], weight, bias, values[id], nullptr, nullptr,
                        spec.dtype, spec.synapse_sizes(bucket), ln.epsilon);
        continue;
      }
      if (op.op == FxOp::block_fp8_linear) {
        lower_block_fp8_linear(builder, ctx.device.device_id(), op, specs_, bucket, values, values[id]);
        continue;
      }
      if (op.op == FxOp::fp8_linear) {
        lower_fp8_linear(builder, ctx.device.device_id(), op, specs_, bucket, values, values[id]);
        continue;
      }
      TORCH_CHECK(op.op == FxOp::linear, "GaudiRT FX: unsupported graph operation");
      synGEMMParams gemm{false, std::get<FxLinear>(op.parameters).transpose_weight};
      const auto syn_dims = spec.synapse_sizes(bucket);
      const auto flat_dims = flatten_2d_syn_dims(syn_dims);
      const uint64_t flat_rows = flat_dims[1];
      auto act = values[op.inputs[0]];
      if (specs_[op.inputs[0]].sizes.size() != 2) {
        const auto in_flat = flatten_2d_syn_dims(specs_[op.inputs[0]].synapse_sizes(bucket));
        auto reshaped = tensor_nd(type.syn, in_flat, -1);
        node({act}, reshaped, std::monostate{}, "reshape");
        act = reshaped;
      }
      auto gemm_out = (op.inputs.size() == 2 && syn_dims.size() == 2)
                          ? values[id]
                          : tensor(op.inputs.size() == 2 ? type.syn : syn_type_float, width, flat_rows, -1);
      node({act, values[op.inputs[1]]}, gemm_out, gemm, "gemm");
      if (op.inputs.size() == 2) {
        if (syn_dims.size() != 2) node({gemm_out}, values[id], std::monostate{}, "reshape");
        continue;
      }
      const bool bf16 = spec.dtype == at::kBFloat16;
      auto bias = values[op.inputs[2]];
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_HALF_NE;
      if (bf16) {
        auto wide_bias = tensor(syn_type_float, width, 1, -1, 1);
        node({bias}, wide_bias, cast, "cast_bf16_to_f32");
        bias = wide_bias;
      }
      auto bias_b = emit_broadcast_to(builder, bias, syn_type_float, {width}, flat_dims);
      auto sum = (bf16 || syn_dims.size() != 2) ? tensor(syn_type_float, width, flat_rows, -1) : values[id];
      node({gemm_out, bias_b}, sum, std::monostate{}, "add_fwd_f32");
      auto final_2d = sum;
      if (bf16) {
        final_2d = syn_dims.size() == 2 ? values[id] : tensor(syn_type_bf16, width, flat_rows, -1);
        emit_cast(builder, sum, flat_dims, at::kFloat, at::kBFloat16, final_2d);
      }
      if (syn_dims.size() != 2) node({final_2d}, values[id], std::monostate{}, "reshape");
    }
    builder.compile(ctx.cache, key);
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(compiled));
  TORCH_CHECK(inserted, "GaudiRT FX: duplicate program recipe");
  return *it->second;
}

// Temporary storage remains live until launch. Holding the cold lock while
// acquiring and returning it also keeps empty_cache() from destroying its pool.
class ProgramTemporaries {
 public:
  ProgramTemporaries(Context& ctx, synStreamHandle stream) : ctx_(ctx), stream_(stream) {}
  ~ProgramTemporaries() {
    if (addresses_.empty()) return;
    std::lock_guard lock(ctx_.cold_mutex);
    for (auto address : addresses_) ctx_.graph_pool->free(address);
  }
  uint64_t allocate(uint64_t bytes) {
    std::lock_guard lock(ctx_.cold_mutex);
    if (!ctx_.graph_pool) {
      const uint64_t budget = std::min<uint64_t>(ctx_.allocator.capacity_bytes() / 8, uint64_t{1} << 30);
      TORCH_CHECK(budget >= 256, "GaudiRT FX: HBM budget is too small for graph temporaries");
      ctx_.ensure_graph_pool(budget, stream_);
    }
    const auto address = ctx_.graph_pool->allocate(bytes, stream_);
    try {
      addresses_.push_back(address);
    } catch (...) {
      ctx_.graph_pool->free(address);
      throw;
    }
    return address;
  }
 private:
  Context& ctx_;
  synStreamHandle stream_;
  c10::SmallVector<uint64_t, 4> addresses_;
};

std::vector<at::Tensor> BucketProgram::run(const std::vector<at::Tensor>& inputs) {
  return run_internal(inputs, {});
}

void BucketProgram::ReusableOutputs::prepare(uint32_t new_bucket) {
  if (bucket == new_bucket) return;
  available.clear();
  bucket = new_bucket;
}

void BucketProgram::ReusableOutputs::reclaim() {
  for (size_t i = 0; i < active.size();) {
    auto& block = active[i];
    // A route may leave another TensorImpl pointing at this storage. Trace
    // captures can also retain a Tensor after its graph slot is consumed.
    if (block.tensor.use_count() != 1 || !block.tensor.storage().unique()) {
      ++i;
      continue;
    }
    if (block.bucket == bucket) available.push_back(std::move(block));
    if (i + 1 != active.size()) active[i] = std::move(active.back());
    active.pop_back();
  }
}

at::Tensor BucketProgram::ReusableOutputs::acquire(const at::Tensor& reference, at::ScalarType dtype,
                                                   at::IntArrayRef sizes, bool dynamic, int64_t width) {
  for (size_t i = 0; i < available.size(); ++i) {
    auto& block = available[i];
    if (block.dynamic != dynamic || block.tensor.scalar_type() != dtype || block.tensor.sizes() != sizes) continue;
    active.push_back(std::move(block));
    if (i + 1 != available.size()) available[i] = std::move(available.back());
    available.pop_back();
    return active.back().tensor;
  }
  auto tensor = dynamic ? bucket_output(reference, dtype, bucket, bucket, width)
                        : at::empty(sizes, reference.options().dtype(dtype));
  if (dynamic) tensor.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
  active.push_back({bucket, dynamic, std::move(tensor)});
  return active.back().tensor;
}

std::vector<at::Tensor> BucketProgram::run_internal(const std::vector<at::Tensor>& inputs,
                                                    const std::vector<bool>& trusted,
                                                    ReusableOutputs* reusable,
                                                    const std::vector<bool>* reusable_mask) {
  TORCH_CHECK(inputs.size() == roles_.size(), "GaudiRT FX: wrong input count");
  TORCH_CHECK(trusted.empty() || trusted.size() == inputs.size(), "GaudiRT FX: invalid trusted input plan");
  int64_t m = -1;
  size_t dynamic_input = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!roles_[i]) continue;
    TORCH_CHECK(inputs[i].dim() >= 1 && inputs[i].dim() <= 5,
                "GaudiRT FX: dynamic input must have rank one through five");
    if (m < 0) {
      m = inputs[i].size(0);
      dynamic_input = i;
    } else {
      TORCH_CHECK(inputs[i].size(0) == m, "GaudiRT FX: inconsistent token dimensions");
    }
  }
  TORCH_CHECK(m > 0 && m <= kMaxBucketRows, "GaudiRT FX: token count outside 1..8192");
  std::call_once(geometry_once_, [&] { initialize_geometry(inputs); });
  auto ctx = current();
  auto stream = ctx->stream();
  auto* static_binding = find_static_bindings(inputs, *ctx, stream);
  size_t next_static = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (static_binding && next_static < static_binding->inputs.size() &&
        static_binding->inputs[next_static].index == i) {
      ++next_static;
      continue;
    }
    if (!trusted.empty() && trusted[i]) continue;
    const auto& input = inputs[i];
    const auto& spec = specs_[i];
    TORCH_CHECK(input.device().type() == c10::kPrivateUse1 && input.device().index() == ctx->device_index &&
                input.scalar_type() == spec.dtype && input.dim() == static_cast<int64_t>(spec.sizes.size()),
                "GaudiRT FX: input device, dtype or rank changed");
    for (int64_t dim = 0; dim < input.dim(); ++dim) {
      if (rotary_cache_inputs_[i] && dim == 0) {
        TORCH_CHECK(input.size(0) >= rotary_cache_contexts_[i], "GaudiRT FX: rotary cache is too short");
        const uint64_t offset = static_cast<uint64_t>(input.storage_offset()) * input.element_size();
        const uint64_t required = kRotaryStaticRows * static_cast<uint64_t>(input.size(1)) * input.element_size();
        TORCH_CHECK(input.is_contiguous() && offset <= input.storage().nbytes() &&
                    required <= input.storage().nbytes() - offset,
                    "GaudiRT FX: rotary cache requires reserved static storage");
        continue;
      }
      const auto expected = spec.dynamic() && dim == 0 ? m : spec.sizes[dim];
      TORCH_CHECK(input.size(dim) == expected, "GaudiRT FX: input geometry changed");
    }
  }
  if (!static_binding) static_binding = bind_static_inputs(inputs, *ctx, stream);
  const uint32_t bucket = std::bit_ceil(static_cast<uint32_t>(m));
  if (reusable) reusable->prepare(bucket);
  std::vector<at::Tensor> results;
  results.reserve(outputs_.size());
  for (size_t output_index = 0; output_index < outputs_.size(); ++output_index) {
    const auto id = outputs_[output_index];
    const auto& spec = specs_[id];
    const bool cache = reusable && reusable_mask && output_index < reusable_mask->size() &&
                       (*reusable_mask)[output_index];
    if (spec.dynamic()) {
      int64_t width = 1;
      auto sizes = spec.sizes;
      sizes[0] = m;
      for (size_t dim = 1; dim < sizes.size(); ++dim) {
        TORCH_CHECK(width <= INT64_MAX / sizes[dim], "GaudiRT FX: output size overflow");
        width *= sizes[dim];
      }
      at::Tensor result;
      if (cache) {
        auto full_sizes = sizes;
        full_sizes[0] = bucket;
        auto stored = reusable->acquire(inputs[dynamic_input], spec.dtype, full_sizes, true, width);
        result = m == bucket ? stored : stored.narrow(0, 0, m);
      } else {
        result = bucket_output(inputs[dynamic_input], spec.dtype, bucket, m, width);
      }
      results.push_back(cache || sizes.size() == 2 ? std::move(result) : result.view(sizes));
    } else {
      if (cache) {
        results.push_back(reusable->acquire(inputs[dynamic_input], spec.dtype, spec.sizes, false, 0));
      } else {
        results.push_back(at::empty(spec.sizes, inputs[dynamic_input].options().dtype(spec.dtype)));
      }
    }
  }
  auto& compiled = cached_recipe(*ctx, bucket);
  ProgramTemporaries temporaries(*ctx, stream);
  c10::SmallVector<std::pair<uint64_t, uint64_t>, 8> regions;
  c10::SmallVector<synLaunchTensorInfo, 16> bindings(compiled.tensors.begin(), compiled.tensors.end());
  if (static_binding) {
    for (const auto& cached : static_binding->inputs) {
      regions.emplace_back(cached.address, cached.bytes);
      bindings[cached.index].pTensorAddress = cached.address;
    }
  }
  next_static = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (static_binding && next_static < static_binding->inputs.size() &&
        static_binding->inputs[next_static].index == i) {
      ++next_static;
      continue;
    }
    const auto& input = inputs[i];
    const auto& spec = specs_[i];
    const uint64_t item = input.element_size();
    uint64_t dynamic_width = 1;
    for (size_t dim = 1; dim < spec.sizes.size(); ++dim) {
      TORCH_CHECK(dynamic_width <= UINT64_MAX / static_cast<uint64_t>(spec.sizes[dim]),
                  "GaudiRT FX: input allocation size overflow");
      dynamic_width *= static_cast<uint64_t>(spec.sizes[dim]);
    }
    TORCH_CHECK(!spec.dynamic() || dynamic_width <= UINT64_MAX / bucket / item,
                "GaudiRT FX: input allocation size overflow");
    const uint64_t bytes = rotary_cache_inputs_[i] ? kRotaryStaticRows * dynamic_width * item
        : spec.dynamic() ? static_cast<uint64_t>(bucket) * dynamic_width * item : input.nbytes();
    const uint64_t address = reinterpret_cast<uint64_t>(input.data_ptr());
    if (trusted.empty() || !trusted[i]) {
      const auto& allocation = validated_program_input(input, *ctx);
      if (allocation.stream != stream) {
        if (allocation.kind == AllocationKind::graph) ctx->graph_pool->record_stream(address, stream);
        else ctx->allocator.record_stream(address, stream);
      }
    }
    const uint64_t offset = static_cast<uint64_t>(input.storage_offset()) * item;
    const uint64_t storage = input.storage().nbytes();
    bool direct = input.is_contiguous() && address % BlockCachingAllocator::kAlignment == 0 &&
                  offset <= storage && bytes <= storage - offset;
    for (const auto& [other, length] : regions)
      if (address < other + length && other < address + bytes) direct = false;
    uint64_t binding = address;
    if (!direct) {
      const auto temp = temporaries.allocate(bytes);
      if (!try_pack_strided_2d(*ctx, temp, bytes, input, stream)) {
        const Layout layout(input);
        dma_copy(temp, nullptr, address, &layout, 0, input.numel(), item, stream, DRAM_TO_DRAM);
      }
      binding = temp;
    }
    regions.emplace_back(binding, bytes);
    bindings[i].pTensorAddress = binding;
  }
  for (size_t i = 0; i < results.size(); ++i) {
    bindings[roles_.size() + i].pTensorAddress = reinterpret_cast<uint64_t>(results[i].data_ptr());
  }
  if (has_embedding_)
    bindings[roles_.size() + outputs_.size()].pTensorAddress = ctx->row_count_address(static_cast<uint32_t>(m));
  const uint64_t workspace = compiled.workspace ? temporaries.allocate(compiled.workspace) : 0;
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(), workspace,
                        compiled.recipe, 0) == synSuccess, "GaudiRT FX: program launch failed");
  return results;
}

}  // namespace gaudirt::generic
