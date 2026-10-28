// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/mimo_attention.hpp"
#include "gaudirt/runtime/mimo_full_attention.hpp"
#include "gaudirt/runtime/mimo_moe.hpp"
#include "gaudirt/runtime/tp_reduce.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"

#include <torch/extension.h>
#include <torch/library.h>
#include "backend/habana_device/HPUDevice.h"
#include "backend/synapse_helpers/stream.h"
#include "habana_lazy/aten_lazy_bridge.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace gaudirt::bridge {
namespace {

using runtime::DeviceRuntime;
using runtime::HbmArena;
using runtime::MimoAttention;
using runtime::MimoAttentionIo;
using runtime::MimoFullAttention;
using runtime::MimoMoeLayer1;
using runtime::TpReduce;
using runtime::WeightManifest;
using runtime::WeightTensor;

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  const uint64_t bytes = std::filesystem::file_size(path);
  std::vector<uint8_t> data(bytes);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(data.data()), data.size())) {
    throw std::runtime_error("cannot read " + path.string());
  }
  return data;
}

void write_file(const std::filesystem::path& path,
                const std::vector<uint8_t>& data) {
  std::ofstream output(path, std::ios::binary);
  if (!output.write(reinterpret_cast<const char*>(data.data()), data.size())) {
    throw std::runtime_error("cannot write " + path.string());
  }
}

hcclUniqueId exchange_unique_id(uint32_t rank,
                                const std::filesystem::path& path) {
  hcclUniqueId id;
  if (rank == 0) {
    if (hcclGetUniqueId(&id) != hcclSuccess) {
      throw std::runtime_error("hcclGetUniqueId failed");
    }
    auto temporary = path;
    temporary += ".tmp";
    write_file(temporary,
               std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&id),
                                    reinterpret_cast<uint8_t*>(&id) + sizeof(id)));
    std::filesystem::rename(temporary, path);
    return id;
  }
  for (unsigned attempt = 0; attempt < 2400; ++attempt) {
    if (std::filesystem::is_regular_file(path) &&
        std::filesystem::file_size(path) == sizeof(id)) {
      const auto data = read_file(path);
      std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t*>(&id));
      return id;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  throw std::runtime_error("timed out waiting for HCCL unique ID");
}

uint64_t address(const at::Tensor& tensor, at::ScalarType dtype,
                 int64_t elements, int64_t device, const char* label = "") {
  void* data = tensor.data_ptr();
  if (data == nullptr && tensor.device().type() == c10::DeviceType::HPU) {
    data = habana_lazy::GetLazyTensorDataPtr(tensor);
  }
  if (tensor.device().type() != c10::DeviceType::HPU ||
      tensor.device().index() != device || tensor.scalar_type() != dtype ||
      !tensor.is_contiguous() || tensor.numel() != elements || data == nullptr) {
    throw std::invalid_argument(
        std::string("BridgeLayer: ") + label + " device=" + tensor.device().str() +
        ", dtype=" + std::to_string(static_cast<int>(tensor.scalar_type())) +
        ", expected_dtype=" + std::to_string(static_cast<int>(dtype)) +
        ", contiguous=" + std::to_string(tensor.is_contiguous()) +
        ", numel=" + std::to_string(tensor.numel()) +
        ", expected_numel=" + std::to_string(elements) +
        ", device_index=" + std::to_string(device) +
        ", null_data_ptr=" + std::to_string(data == nullptr));
  }
  return habana::HPUDeviceContext::get_device(device).get_fixed_address(data);
}

WeightTensor weight(const at::Tensor& tensor, const char* name,
                    synDataType dtype, at::ScalarType at_dtype,
                    std::vector<uint64_t> shape, int64_t device) {
  uint64_t elements = 1;
  for (const uint64_t size : shape) elements *= size;
  WeightTensor result;
  result.name = name;
  result.dtype = dtype;
  result.shape = std::move(shape);
  result.elements = elements;
  result.bytes = tensor.nbytes();
  result.device_address = address(tensor, at_dtype, elements, device, name);
  if (result.bytes != tensor.nbytes()) {
    throw std::invalid_argument("BridgeLayer: weight byte count mismatch");
  }
  return result;
}

std::unique_ptr<WeightManifest> make_attention_weights(
    const std::vector<at::Tensor>& tensors, int64_t device, bool has_sink = true) {
  const size_t required = has_sink ? 15 : 14;
  if (tensors.size() < required) {
    throw std::invalid_argument("BridgeLayer: insufficient layer weights");
  }
  std::vector<WeightTensor> manifest;
  manifest.reserve(has_sink ? 6 : 5);
  manifest.push_back(weight(tensors[0], "input_layernorm", syn_type_bf16,
                            at::kBFloat16, {6144}, device));
  manifest.push_back(weight(tensors[1], "qkv_weight", syn_type_fp8_143,
                            at::kFloat8_e4m3fn, {128, 3392, 48}, device));
  manifest.push_back(weight(tensors[2], "qkv_scales", syn_type_float,
                            at::kFloat, {48, 27}, device));
  manifest.push_back(weight(tensors[3], "qkv_bias", syn_type_float,
                            at::kFloat, {3392}, device));
  if (has_sink) {
    manifest.push_back(weight(tensors[4], "attention_sink", syn_type_bf16,
                              at::kBFloat16, {16}, device));
  }
  manifest.push_back(weight(tensors[has_sink ? 5 : 4], "oproj_weight",
                            syn_type_bf16, at::kBFloat16, {2048, 6144}, device));
  return std::make_unique<WeightManifest>(std::move(manifest));
}

MimoMoeLayer1::Weights make_moe_weights(
    const std::vector<at::Tensor>& tensors, int64_t device, bool has_sink = true) {
  const size_t required = has_sink ? 15 : 14;
  if (tensors.size() < required) {
    throw std::invalid_argument("BridgeLayer: insufficient layer weights");
  }
  const size_t base = has_sink ? 6 : 5;
  MimoMoeLayer1::Weights result{};
  result.post_attention_gamma_bf16 =
      address(tensors[base], at::kBFloat16, 6144, device);
  result.router_weight_bf16 =
      address(tensors[base + 1], at::kBFloat16, 384 * 6144, device);
  result.router_correction_bias_fp32 =
      address(tensors[base + 2], at::kFloat, 384, device);
  result.gate_up_packed_u8 =
      address(tensors[base + 3], at::kByte, 603979776, device);
  result.gate_up_scales_u8 =
      address(tensors[base + 4], at::kByte, 37748736, device);
  result.down_packed_u8 =
      address(tensors[base + 5], at::kByte, 301989888, device);
  result.down_scales_u8 =
      address(tensors[base + 6], at::kByte, 18874368, device);
  result.mxfp4_lut_bf16 = address(tensors[base + 7], at::kBFloat16, 512, device);
  result.down_combine_directions_u8 =
      address(tensors[base + 8], at::kByte, 512, device);
  return result;
}

class LayerProgram {
 public:
  LayerProgram(uint32_t rank, synDeviceId device, uint64_t stream,
               std::vector<at::Tensor> weights, const at::Tensor& key_cache,
               const at::Tensor& value_cache,
               std::vector<at::Tensor> static_inputs,
               const at::Tensor& arena_slab,
               const std::filesystem::path& unique_id_path, bool full_attention)
      : rank_(rank), device_(device), keep_alive_(std::move(weights)),
        input_owners_(std::move(static_inputs)), full_attention_(full_attention) {
    const size_t expected_weights = full_attention_ ? 14 : 15;
    if (rank_ >= 8 || keep_alive_.size() != expected_weights ||
        input_owners_.size() != 6) {
      throw std::invalid_argument("BridgeLayer: invalid initialization contract");
    }
    const auto cache_shape = key_cache.sizes();
    const auto value_shape = value_cache.sizes();
    if (cache_shape.size() != 3 || value_shape.size() != 3 ||
        cache_shape[1] != 1 || value_shape[1] != 1 || cache_shape[2] != 192 ||
        value_shape[2] != 128 || cache_shape[0] != value_shape[0] ||
        cache_shape[0] <= 0 || cache_shape[0] > UINT32_MAX) {
      throw std::invalid_argument("BridgeLayer: invalid KV geometry");
    }
    cache_slots_ = static_cast<uint32_t>(cache_shape[0]);
    key_cache_address_ = address(key_cache, at::kBFloat16, cache_slots_ * 192, device_);
    value_cache_address_ = address(value_cache, at::kBFloat16, cache_slots_ * 128, device_);
    keep_alive_.push_back(key_cache);
    keep_alive_.push_back(value_cache);
    const at::ScalarType metadata_types[] = {at::kInt, at::kInt, at::kInt,
                                             at::kInt, at::kBFloat16, at::kBFloat16};
    const int64_t metadata_elements[] = {1, 1, 2, 2, 64, 64};
    for (size_t i = 0; i < input_owners_.size(); ++i) {
      metadata_addresses_[i] =
          address(input_owners_[i], metadata_types[i], metadata_elements[i], device_);
    }

    const auto runtime_device = habana::HPUDeviceContext::get_device(device_).id();
    DeviceRuntime::Config config;
    config.module_id = rank_;
    config.rank = rank_;
    config.world_size = 8;
    config.pin_cpu = rank_ * 4;
    config.borrowed_device_id = runtime_device;
    runtime_ = std::make_unique<DeviceRuntime>(config);
    runtime_->init_comm(exchange_unique_id(rank_, unique_id_path));
    auto& production_device = habana::HPUDeviceContext::get_device(runtime_device);
    auto& production_stream = production_device.get_stream(stream);
    runtime_->use_compute_stream(static_cast<synStreamHandle>(production_stream));
    arena_owner_ = arena_slab;
    constexpr uint64_t kSlabBytes = 16 * 1024 * 1024;
    const uint64_t slab_address =
        address(arena_owner_, at::kByte, kSlabBytes, device_);
    const uint64_t aligned_slab_address =
        (slab_address + HbmArena::kDefaultAlignment - 1) &
        ~(HbmArena::kDefaultAlignment - 1);
    arena_ = std::make_unique<HbmArena>(
        runtime_->device_id(), aligned_slab_address,
        kSlabBytes - (aligned_slab_address - slab_address),
        HbmArena::BorrowedSlab{});
    constexpr uint64_t kRequiredWorkspaceBytes = 651520;
    const uint64_t workspace = production_device.get_workspace_buffer(kRequiredWorkspaceBytes);
    const uint64_t workspace_bytes = production_device.get_workspace_size();
    if (workspace == 0 || workspace_bytes < kRequiredWorkspaceBytes) {
      throw std::runtime_error("BridgeLayer: invalid production recipe workspace");
    }
    arena_->set_borrowed_workspace(workspace, workspace_bytes);

    weights_ = make_attention_weights(keep_alive_, device_, !full_attention_);
    if (full_attention_) {
      full_attention_graph_ =
          std::make_unique<MimoFullAttention>(*arena_, *weights_, cache_slots_);
    } else {
      attention_ = std::make_unique<MimoAttention>(*arena_, *weights_, cache_slots_);
    }
    attention_reduce_ = std::make_unique<TpReduce>(*arena_, 6144, syn_type_bf16);
    moe_reduce_ = std::make_unique<TpReduce>(*arena_, 6144, syn_type_float);
    moe_ = std::make_unique<MimoMoeLayer1>(
        rank_, make_moe_weights(keep_alive_, device_, !full_attention_));

    auto& io = attention_io_;
    io.residual_out = arena_->allocate(12288);
    io.normalized = arena_->allocate(12288);
    io.qkv = arena_->allocate(3392 * 2);
    io.query = arena_->allocate(16 * 192 * 2);
    io.new_key = arena_->allocate(192 * 2);
    io.new_value = arena_->allocate(128 * 2);
    io.context = arena_->allocate(16 * 128 * 2);
    io.output_local = arena_->allocate(12288);
    attention_output_ = arena_->allocate(12288);
    attention_gathered_ = arena_->allocate(8 * 12288);
    moe_gathered_ = arena_->allocate(8 * 24576);
    moe_inputs_.attention_bf16 = attention_output_;
    moe_inputs_.residual_in_bf16 = io.residual_out;
    moe_outputs_.normalized_bf16 = arena_->allocate(12288);
    moe_outputs_.topk_ids_i32 = arena_->allocate(32);
    moe_outputs_.topk_weights_fp32 = arena_->allocate(32);
    moe_outputs_.moe_local_fp32 = arena_->allocate(24576);
    const auto output_options = at::TensorOptions()
                                    .dtype(at::kBFloat16)
                                    .device(c10::DeviceType::HPU, device_);
    layer_output_owner_ = at::empty({1, 6144}, output_options);
    residual_output_owner_ = at::empty({1, 6144}, output_options);
    if (full_attention_) {
      query_owner_ = at::empty({1, 1, 3072}, output_options);
      io.query = address(query_owner_, at::kBFloat16, 3072, device_);
    }
    layer_output_ = address(layer_output_owner_, at::kBFloat16, 6144, device_);
    moe_outputs_.residual_out_bf16 =
        address(residual_output_owner_, at::kBFloat16, 6144, device_);
    moe_workspace_ = arena_->ensure_workspace(moe_->workspace_bytes());
  }

  ~LayerProgram() {
    if (runtime_) {
      try {
        runtime_->synchronize_compute();
      } catch (...) {
      }
    }
  }

  at::Tensor prepare_full(
      const at::Tensor& hidden, const at::Tensor& residual,
      const at::Tensor& key_cache, const at::Tensor& value_cache,
      const at::Tensor& position, const at::Tensor& slot_mapping,
      const at::Tensor& window_pages, const at::Tensor& window_groups,
      const at::Tensor& rope_cos, const at::Tensor& rope_sin) {
    if (!full_attention_) {
      throw std::invalid_argument("BridgeLayer: prepare_full requires a full layer");
    }
    if (hidden.dim() != 2 || hidden.size(0) != 1 || hidden.size(1) != 6144 ||
        residual.dim() != 2 || residual.size(0) != 1 || residual.size(1) != 6144) {
      throw std::invalid_argument("BridgeLayer: expected BF16 [1,6144] inputs");
    }
    auto& io = attention_io_;
    io.hidden_states = address(hidden, at::kBFloat16, 6144, device_);
    io.residual = address(residual, at::kBFloat16, 6144, device_);
    io.key_cache = address(key_cache, at::kBFloat16, cache_slots_ * 192, device_);
    io.value_cache = address(value_cache, at::kBFloat16, cache_slots_ * 128, device_);
    io.position = address(position, at::kInt, 1, device_);
    io.slot_mapping = address(slot_mapping, at::kInt, 1, device_);
    io.window_pages = address(window_pages, at::kInt, 2, device_);
    io.window_groups = address(window_groups, at::kInt, 2, device_);
    io.rope_cos = address(rope_cos, at::kBFloat16, 64, device_);
    io.rope_sin = address(rope_sin, at::kBFloat16, 64, device_);
    if (io.key_cache != key_cache_address_ || io.value_cache != value_cache_address_) {
      throw std::invalid_argument("BridgeLayer: KV storage changed");
    }
    const uint64_t expected[] = {io.position, io.slot_mapping, io.window_pages,
                                 io.window_groups, io.rope_cos, io.rope_sin};
    for (size_t i = 0; i < 6; ++i) {
      if (expected[i] != metadata_addresses_[i]) {
        throw std::invalid_argument("BridgeLayer: static metadata storage changed");
      }
    }
    input_owners_.push_back(hidden);
    input_owners_.push_back(residual);
    full_attention_graph_->execute_prepare(io, runtime_->compute_stream());
    return query_owner_;
  }

  std::tuple<at::Tensor, at::Tensor> finish_full(const at::Tensor& context) {
    if (!full_attention_) {
      throw std::invalid_argument("BridgeLayer: finish_full requires a full layer");
    }
    if (context.dim() != 3 || context.size(0) != 1 || context.size(1) != 1 ||
        context.size(2) != 2048) {
      throw std::invalid_argument("BridgeLayer: expected BF16 [1,1,2048] context");
    }
    auto& io = attention_io_;
    io.context = address(context, at::kBFloat16, 2048, device_);
    full_attention_graph_->execute_finish(io, runtime_->compute_stream());
    if (hcclAllGather(reinterpret_cast<const void*>(io.output_local),
                      reinterpret_cast<void*>(attention_gathered_), 6144,
                      hcclBfloat16, runtime_->comm(), runtime_->compute_stream()) !=
        hcclSuccess) {
      throw std::runtime_error("BridgeLayer: attention AllGather failed");
    }
    attention_reduce_->execute(attention_gathered_, attention_output_,
                               runtime_->compute_stream());
    moe_->launch(runtime_->compute_stream(), moe_inputs_, moe_outputs_,
                 moe_workspace_);
    if (hcclAllGather(reinterpret_cast<const void*>(moe_outputs_.moe_local_fp32),
                      reinterpret_cast<void*>(moe_gathered_), 6144, hcclFloat32,
                      runtime_->comm(), runtime_->compute_stream()) != hcclSuccess) {
      throw std::runtime_error("BridgeLayer: MoE AllGather failed");
    }
    moe_reduce_->execute(moe_gathered_, layer_output_, runtime_->compute_stream());
    return {layer_output_owner_, residual_output_owner_};
  }

  std::tuple<at::Tensor, at::Tensor> forward(
      const at::Tensor& hidden, const at::Tensor& residual,
      const at::Tensor& key_cache, const at::Tensor& value_cache,
      const at::Tensor& position, const at::Tensor& slot_mapping,
      const at::Tensor& window_pages, const at::Tensor& window_groups,
      const at::Tensor& rope_cos, const at::Tensor& rope_sin) {
    if (full_attention_) {
      throw std::invalid_argument("BridgeLayer: full layers require prepare/finish");
    }
    if (hidden.dim() != 2 || hidden.size(0) != 1 || hidden.size(1) != 6144 ||
        residual.dim() != 2 || residual.size(0) != 1 || residual.size(1) != 6144) {
      throw std::invalid_argument("BridgeLayer: expected BF16 [1,6144] inputs");
    }
    auto& io = attention_io_;
    io.hidden_states = address(hidden, at::kBFloat16, 6144, device_);
    io.residual = address(residual, at::kBFloat16, 6144, device_);
    io.key_cache = address(key_cache, at::kBFloat16, cache_slots_ * 192, device_);
    io.value_cache = address(value_cache, at::kBFloat16, cache_slots_ * 128, device_);
    io.position = address(position, at::kInt, 1, device_);
    io.slot_mapping = address(slot_mapping, at::kInt, 1, device_);
    io.window_pages = address(window_pages, at::kInt, 2, device_);
    io.window_groups = address(window_groups, at::kInt, 2, device_);
    io.rope_cos = address(rope_cos, at::kBFloat16, 64, device_);
    io.rope_sin = address(rope_sin, at::kBFloat16, 64, device_);
    if (io.key_cache != key_cache_address_ || io.value_cache != value_cache_address_) {
      throw std::invalid_argument("BridgeLayer: KV storage changed");
    }
    const uint64_t expected[] = {io.position, io.slot_mapping, io.window_pages,
                                 io.window_groups, io.rope_cos, io.rope_sin};
    for (size_t i = 0; i < 6; ++i) {
      if (expected[i] != metadata_addresses_[i]) {
        throw std::invalid_argument("BridgeLayer: static metadata storage changed");
      }
    }
    if (input_owners_.size() >= 256) {
      throw std::invalid_argument("BridgeLayer: vertical replay capacity exceeded");
    }
    input_owners_.push_back(hidden);
    input_owners_.push_back(residual);

    attention_->execute(io, runtime_->compute_stream());
    if (hcclAllGather(reinterpret_cast<const void*>(io.output_local),
                      reinterpret_cast<void*>(attention_gathered_), 6144,
                      hcclBfloat16, runtime_->comm(), runtime_->compute_stream()) !=
        hcclSuccess) {
      throw std::runtime_error("BridgeLayer: attention AllGather failed");
    }
    attention_reduce_->execute(attention_gathered_, attention_output_,
                               runtime_->compute_stream());
    moe_->launch(runtime_->compute_stream(), moe_inputs_, moe_outputs_,
                 moe_workspace_);
    if (hcclAllGather(reinterpret_cast<const void*>(moe_outputs_.moe_local_fp32),
                      reinterpret_cast<void*>(moe_gathered_), 6144, hcclFloat32,
                      runtime_->comm(), runtime_->compute_stream()) != hcclSuccess) {
      throw std::runtime_error("BridgeLayer: MoE AllGather failed");
    }
    moe_reduce_->execute(moe_gathered_, layer_output_, runtime_->compute_stream());

    return {layer_output_owner_, residual_output_owner_};
  }

  void rebind(std::vector<at::Tensor> weights, const at::Tensor& key_cache,
              const at::Tensor& value_cache) {
    const size_t expected = full_attention_ ? 14 : 15;
    if (weights.size() != expected || keep_alive_.size() != expected + 2 ||
        key_cache.sizes() != keep_alive_[expected].sizes() ||
        value_cache.sizes() != keep_alive_[expected + 1].sizes()) {
      throw std::invalid_argument("BridgeLayer: incompatible layer rebind");
    }
    runtime_->synchronize_compute();
    for (size_t i = 0; i < weights.size(); ++i) {
      keep_alive_[i] = std::move(weights[i]);
    }
    keep_alive_[expected] = key_cache;
    keep_alive_[expected + 1] = value_cache;
    key_cache_address_ =
        address(keep_alive_[expected], at::kBFloat16, cache_slots_ * 192, device_);
    value_cache_address_ =
        address(keep_alive_[expected + 1], at::kBFloat16, cache_slots_ * 128, device_);
    auto rebound_weights = make_attention_weights(keep_alive_, device_, !full_attention_);
    if (full_attention_) {
      full_attention_graph_->rebind_weights(*rebound_weights);
    } else {
      attention_->rebind_weights(*rebound_weights);
    }
    weights_ = std::move(rebound_weights);
    moe_->rebind_weights(make_moe_weights(keep_alive_, device_, !full_attention_));
  }

  void synchronize() { runtime_->synchronize_compute(); }

  std::pair<uint64_t, uint64_t> workspace_info() const {
    return {arena_->workspace_addr(), arena_->workspace_bytes()};
  }

  std::pair<uint64_t, uint64_t> recipe_info() const {
    if (full_attention_) {
      return full_attention_graph_->recipe_identity();
    }
    return {attention_->recipe_identity(), moe_->recipe_identity()};
  }

 private:
  uint32_t rank_;
  synDeviceId device_;
  uint32_t cache_slots_ = 0;
  uint64_t key_cache_address_ = 0;
  uint64_t value_cache_address_ = 0;
  uint64_t metadata_addresses_[6] = {};
  std::vector<at::Tensor> keep_alive_;
  std::vector<at::Tensor> input_owners_;
  std::unique_ptr<DeviceRuntime> runtime_;
  at::Tensor arena_owner_;
  at::Tensor layer_output_owner_;
  at::Tensor residual_output_owner_;
  at::Tensor query_owner_;
  std::unique_ptr<HbmArena> arena_;
  std::unique_ptr<WeightManifest> weights_;
  bool full_attention_ = false;
  std::unique_ptr<MimoAttention> attention_;
  std::unique_ptr<MimoFullAttention> full_attention_graph_;
  std::unique_ptr<TpReduce> attention_reduce_;
  std::unique_ptr<TpReduce> moe_reduce_;
  std::unique_ptr<MimoMoeLayer1> moe_;
  MimoAttentionIo attention_io_{};
  MimoMoeLayer1::Inputs moe_inputs_{};
  MimoMoeLayer1::Outputs moe_outputs_{};
  uint64_t attention_output_ = 0;
  uint64_t attention_gathered_ = 0;
  uint64_t moe_gathered_ = 0;
  uint64_t layer_output_ = 0;
  uint64_t moe_workspace_ = 0;
};

std::unique_ptr<LayerProgram> active;

void initialize(uint32_t rank, synDeviceId device, uint64_t stream,
                std::vector<at::Tensor> weights, const at::Tensor& key_cache,
                const at::Tensor& value_cache,
                std::vector<at::Tensor> static_inputs,
                const at::Tensor& arena_slab,
                const std::string& unique_id_path, bool full_attention = false) {
  if (active) {
    throw std::runtime_error("BridgeLayer: already initialized");
  }
  active = std::make_unique<LayerProgram>(rank, device, stream, std::move(weights),
                                          key_cache, value_cache,
                                          std::move(static_inputs), arena_slab,
                                          unique_id_path, full_attention);
}

std::tuple<at::Tensor, at::Tensor> forward(
    const at::Tensor& hidden, const at::Tensor& residual,
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& position, const at::Tensor& slot_mapping,
    const at::Tensor& window_pages, const at::Tensor& window_groups,
    const at::Tensor& rope_cos, const at::Tensor& rope_sin) {
  if (!active) {
    throw std::runtime_error("BridgeLayer: not initialized");
  }
  return active->forward(hidden, residual, key_cache, value_cache, position,
                         slot_mapping, window_pages, window_groups, rope_cos, rope_sin);
}

at::Tensor prepare_full(
    const at::Tensor& hidden, const at::Tensor& residual,
    const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& position, const at::Tensor& slot_mapping,
    const at::Tensor& window_pages, const at::Tensor& window_groups,
    const at::Tensor& rope_cos, const at::Tensor& rope_sin) {
  if (!active) {
    throw std::runtime_error("BridgeLayer: not initialized");
  }
  return active->prepare_full(hidden, residual, key_cache, value_cache, position,
                              slot_mapping, window_pages, window_groups,
                              rope_cos, rope_sin);
}

std::tuple<at::Tensor, at::Tensor> finish_full(const at::Tensor& context) {
  if (!active) {
    throw std::runtime_error("BridgeLayer: not initialized");
  }
  return active->finish_full(context);
}

std::tuple<at::Tensor, at::Tensor> meta(
    const at::Tensor& hidden, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&) {
  return {at::empty({1, 6144}, hidden.options()),
          at::empty({1, 6144}, hidden.options())};
}

void synchronize() {
  if (!active) {
    throw std::runtime_error("BridgeLayer: not initialized");
  }
  active->synchronize();
}

std::pair<uint64_t, uint64_t> workspace_info() {
  if (!active) {
    throw std::runtime_error("BridgeLayer: not initialized");
  }
  return active->workspace_info();
}

void rebind(std::vector<at::Tensor> weights, const at::Tensor& key_cache,
            const at::Tensor& value_cache) {
  if (!active) {
    throw std::runtime_error("BridgeLayer: not initialized");
  }
  active->rebind(std::move(weights), key_cache, value_cache);
}

std::pair<uint64_t, uint64_t> recipe_info() {
  if (!active) {
    throw std::runtime_error("BridgeLayer: not initialized");
  }
  return active->recipe_info();
}

void finalize() { active.reset(); }

}  // namespace
}  // namespace gaudirt::bridge

TORCH_LIBRARY(gaudirt, m) {
  m.def("layer1(Tensor hidden, Tensor residual, Tensor(a!) key_cache, Tensor(a!) value_cache, Tensor position, Tensor slot_mapping, Tensor window_pages, Tensor window_groups, Tensor rope_cos, Tensor rope_sin) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(gaudirt, HPU, m) {
  m.impl("layer1", &gaudirt::bridge::forward);
}

TORCH_LIBRARY_IMPL(gaudirt, Meta, m) {
  m.impl("layer1", &gaudirt::bridge::meta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("workspace_info", &gaudirt::bridge::workspace_info);
  m.def("recipe_info", &gaudirt::bridge::recipe_info);
  m.def("rebind", &gaudirt::bridge::rebind);
  m.def("prepare_full", &gaudirt::bridge::prepare_full);
  m.def("finish_full", &gaudirt::bridge::finish_full);
  m.def("initialize", &gaudirt::bridge::initialize);
  m.def("synchronize", &gaudirt::bridge::synchronize);
  m.def("finalize", &gaudirt::bridge::finalize);
}
