// SPDX-License-Identifier: Apache-2.0
#include "layer0.hpp"
#include "layer_b.hpp"
#include "layer_c.hpp"
#include "model_argmax.hpp"
#include "model_embedding.hpp"
#include "model_head.hpp"
#include "step_metadata.hpp"
#include "prefill_metadata.hpp"
#include "prefill_bindings.hpp"
#include "prefill_chunks.hpp"

#include "gaudirt/bridge/rank_lifecycle.hpp"
#include "gaudirt/common/sha256.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/EmptyTensor.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace gaudirt::mimo {
namespace {

std::filesystem::path recipe_cache_directory(const std::filesystem::path& root,
                                             uint32_t rank) {
  if (root.empty()) return {};
  Dl_info builder{}, compiler{};
  if (!dladdr(reinterpret_cast<void*>(&recipe_cache_directory), &builder) ||
      !dladdr(reinterpret_cast<void*>(&synGraphCompile), &compiler)) {
    throw std::runtime_error("PrivateUse1: cannot identify recipe compiler inputs");
  }
  const char* kernels = std::getenv("GC_KERNEL_PATH");
  if (!kernels || !*kernels)
    throw std::invalid_argument("PrivateUse1: recipe cache requires GC_KERNEL_PATH");
  std::string inputs = "gaudirt-recipe-cache-v1\n";
  const auto add = [&](const std::filesystem::path& file) {
    inputs += common::sha256_file(file);
    inputs += '\n';
  };
  add(builder.dli_fname);
  add(compiler.dli_fname);
  add(std::filesystem::path(compiler.dli_fname).parent_path() / "libSynapseMme.so");
  const std::string paths(kernels);
  for (size_t start = 0; start < paths.size();) {
    const size_t end = paths.find(':', start);
    const std::string file = paths.substr(start, end - start);
    if (file.empty()) throw std::invalid_argument("PrivateUse1: empty TPC library path");
    add(file);
    if (end == std::string::npos) break;
    start = end + 1;
    if (start == paths.size()) throw std::invalid_argument("PrivateUse1: empty TPC library path");
  }
  const auto digest = common::sha256_bytes(
      reinterpret_cast<const uint8_t*>(inputs.data()), inputs.size());
  return root / digest / ("rank" + std::to_string(rank));
}

enum class Kind : int { A = 0, B = 1, C = 2 };

Kind checked_kind(int64_t value) {
  if (value < 0 || value > 2) {
    throw std::invalid_argument("PrivateUse1: layer kind must be 0(Type A), 1(Type B), or 2(Type C)");
  }
  return static_cast<Kind>(value);
}

struct LayerEntry {
  Kind kind;
  std::shared_ptr<runtime::WeightManifest> attention_weights;
  std::shared_ptr<runtime::WeightManifest> dense_weights;
  uint64_t residual_output = 0;
  uint64_t layer_output = 0;
  uint64_t prefill_residual_output = 0;
  uint64_t large_prefill_residual_output = 0;
};

struct PendingWeights {
  std::vector<runtime::WeightTensor> attention;
  std::vector<runtime::WeightTensor> dense;
};

bool is_full_layer(uint32_t layer) {
  constexpr std::array<uint32_t, 10> full{0, 7, 15, 23, 31, 39, 47, 55, 62, 69};
  return std::find(full.begin(), full.end(), layer) != full.end();
}

void check_syn(synStatus status, const char* operation) {
  if (status != synSuccess) {
    throw std::runtime_error(std::string("PrivateUse1: ") + operation +
                             " failed with Synapse status " +
                             std::to_string(static_cast<int>(status)));
  }
}

struct Instance {
  Kind kind;
  uint32_t cache_slots = 0;
  uint32_t page_count = 0;
  std::unique_ptr<Layer0> a;
  std::unique_ptr<LayerB> b;
  std::unique_ptr<LayerC> c;
  Layer0::Io a_io{};
  LayerB::Io b_io{};
  LayerC::Io c_io{};

  std::vector<uint64_t> recipe_identity() const {
    if (a) return a->recipe_identity();
    if (b) return b->recipe_identity();
    if (c) return c->recipe_identity();
    throw std::runtime_error("PrivateUse1: empty layer instance");
  }
};

class PrivateProgram final : public std::enable_shared_from_this<PrivateProgram> {
 public:
  PrivateProgram(uint32_t rank, uint32_t world,
                 const std::filesystem::path& unique_id,
                 const std::filesystem::path& recipe_cache,
                 uint64_t hbm_bytes)
      : rank_(rank), runtime_([&] {
          runtime::DeviceRuntime::Config config;
          config.module_id = rank;
          config.rank = rank;
          config.world_size = world;
          // The service launcher owns CPU placement and scheduling policy.
          config.pin_cpu = -1;
          config.uclamp_min = common::kInheritHostPolicy;
          return config;
        }()),
        arena_(runtime_.device_id(), hbm_bytes),
        recipe_cache_(recipe_cache_directory(recipe_cache, rank)) {
    if (world != 8 || rank >= world) {
      throw std::invalid_argument("PrivateUse1: only TP8 is supported");
    }
    runtime_.init_comm(bridge::exchange_unique_id(rank, unique_id));
    shutdown_buffer_ = arena_.allocate(sizeof(float));
    const float ready = 1.0f;
    arena_.copy_h2d_sync(runtime_.h2d_stream(), &ready, shutdown_buffer_, sizeof(ready));
  }

  ~PrivateProgram() {
    if (metadata_host_ || token_host_ || metadata_ready_ || compute_done_) {
      try { runtime_.synchronize_all(); } catch (...) {}
    }
    if (metadata_ready_) synEventDestroy(metadata_ready_);
    if (compute_done_) synEventDestroy(compute_done_);
    if (token_ready_) synEventDestroy(token_ready_);
    if (metadata_host_) synHostFree(runtime_.device_id(), metadata_host_, 0);
    if (token_host_) synHostFree(runtime_.device_id(), token_host_, 0);
  }

  void enable_model_fusion() {
    if (fused_model_ || !entries_.empty() || !pending_.empty() || model_finalized_)
      throw std::invalid_argument("PrivateUse1: enable model fusion before registering weights");
    fused_model_ = true;
  }

  void enable_prefill() {
    if (!fused_model_ || prefill_enabled_ || !entries_.empty() || !pending_.empty())
      throw std::invalid_argument("PrivateUse1: enable prefill after fusion and before weights");
    prefill_enabled_ = true;
  }

  void register_layer(uint32_t layer_id, int64_t kind_value,
                      const std::string& attention_weights,
                      const std::string& dense_weights,
                      uint32_t cache_slots, uint32_t page_count) {
    auto attention = std::make_shared<runtime::WeightManifest>(
        std::filesystem::path(attention_weights) / ("rank" + std::to_string(rank_)),
        runtime_, arena_);
    std::shared_ptr<runtime::WeightManifest> dense;
    if (!dense_weights.empty()) {
      dense = std::make_shared<runtime::WeightManifest>(
          std::filesystem::path(dense_weights) / ("rank" + std::to_string(rank_)),
          runtime_, arena_);
    }
    register_layer_manifest(layer_id, checked_kind(kind_value), std::move(attention),
                            std::move(dense), cache_slots, page_count);
  }

  void upload_weight(uint32_t layer_id, bool dense, const std::string& name,
                     const at::Tensor& tensor) {
    if (layer_id >= 70 || entries_.count(layer_id)) {
      throw std::invalid_argument("PrivateUse1: invalid or finalized weight layer_id");
    }
    if (name.empty() || tensor.device().type() != c10::kCPU ||
        !tensor.is_contiguous() || tensor.dim() <= 0 ||
        tensor.dim() > HABANA_DIM_MAX || tensor.numel() <= 0) {
      throw std::invalid_argument("PrivateUse1: weights must be named contiguous CPU tensors");
    }
    runtime::WeightTensor item;
    item.name = name;
    switch (tensor.scalar_type()) {
      case at::kBFloat16: item.dtype = syn_type_bf16; break;
      case at::kFloat: item.dtype = syn_type_float; break;
      case at::kFloat8_e4m3fn: item.dtype = syn_type_fp8_143; break;
      case at::kByte: item.dtype = syn_type_uint8; break;
      default:
        throw std::invalid_argument("PrivateUse1: unsupported weight dtype");
    }
    if (prefill_enabled_ && !dense && (name == "moe_gs" || name == "moe_ds")) {
      if (tensor.scalar_type() != at::kByte)
        throw std::invalid_argument("PrivateUse1: invalid MoE scale dtype");
      const auto* scales = tensor.const_data_ptr<uint8_t>();
      if (std::any_of(scales, scales + tensor.numel(), [](uint8_t x) { return x < 2 || x > 252; }))
        throw std::invalid_argument("PrivateUse1: MoE scale outside grouped decoder contract");
    }
    item.elements = static_cast<uint64_t>(tensor.numel());
    item.bytes = tensor.nbytes();
    if (!dense && (name == "qkv_scales" || name == "qkv_bias")) {
      if (tensor.scalar_type() != at::kFloat)
        throw std::invalid_argument("PrivateUse1: QKV scales and bias must be FP32");
      const auto* values = tensor.const_data_ptr<float>();
      for (uint64_t i = 0; i < item.elements; ++i) {
        if ((name == "qkv_scales" && (!std::isfinite(values[i]) || values[i] < 0x1p-20f)) ||
            (name == "qkv_bias" && values[i] != 0.0f))
          throw std::invalid_argument("PrivateUse1: QKV scale or bias outside model reducer contract");
      }
    }
    for (const int64_t size : tensor.sizes()) {
      item.shape.push_back(static_cast<uint64_t>(size));
    }
    std::reverse(item.shape.begin(), item.shape.end());
    auto& pending = pending_[layer_id];
    auto& tensors = dense ? pending.dense : pending.attention;
    if (std::any_of(tensors.begin(), tensors.end(), [&](const auto& existing) {
          return existing.name == name;
        })) {
      throw std::invalid_argument("PrivateUse1: duplicate streamed weight " + name);
    }
    item.device_address = arena_.allocate(item.bytes);
    arena_.copy_h2d_sync(runtime_.h2d_stream(), tensor.data_ptr(),
                         item.device_address, item.bytes);
    tensors.push_back(std::move(item));
  }

  void register_streamed_layer(uint32_t layer_id, int64_t kind_value,
                               uint32_t cache_slots, uint32_t page_count) {
    const auto it = pending_.find(layer_id);
    if (it == pending_.end() || it->second.attention.empty()) {
      throw std::invalid_argument("PrivateUse1: streamed attention weights are incomplete");
    }
    auto attention = std::make_shared<runtime::WeightManifest>(
        std::move(it->second.attention));
    std::shared_ptr<runtime::WeightManifest> dense;
    if (!it->second.dense.empty()) {
      dense = std::make_shared<runtime::WeightManifest>(std::move(it->second.dense));
    }
    register_layer_manifest(layer_id, checked_kind(kind_value), std::move(attention),
                            std::move(dense), cache_slots, page_count);
    pending_.erase(it);
  }

  void register_layer_manifest(
      uint32_t layer_id, Kind kind,
      std::shared_ptr<runtime::WeightManifest> attention,
      std::shared_ptr<runtime::WeightManifest> dense,
      uint32_t cache_slots, uint32_t page_count) {
    if (layer_id >= 70 || entries_.count(layer_id)) {
      throw std::invalid_argument("PrivateUse1: invalid or duplicate layer_id");
    }
    if (cache_slots == 0 || cache_slots % 128 != 0 || page_count == 0 ||
        (prefill_enabled_ && cache_slots > UINT32_MAX - 128) ||
        (kind == Kind::C && page_count != 2) ||
        (kind != Kind::C && page_count > 64)) {
      throw std::invalid_argument("PrivateUse1: invalid cache or page geometry");
    }
    const uint32_t graph_cache_slots = cache_slots + (prefill_enabled_ ? 128 : 0);
    if ((kind == Kind::A) != static_cast<bool>(dense)) {
      throw std::invalid_argument("PrivateUse1: dense weights are only valid for Type A");
    }
    // A layer consumes the preceding residual before its own output is
    // written on the same stream. Two buffers therefore serve all layers and
    // all prefill buckets; model weights and KV remain separately resident.
    if (prefill_enabled_ && !prefill_residual_slots_[0]) {
      prefill_residual_slots_[0] = arena_.allocate(12288 * 2048);
      prefill_residual_slots_[1] = arena_.allocate(12288 * 2048);
    }
    const uint64_t prefill_residual = prefill_enabled_ ? prefill_residual_slots_[layer_id % 2] : 0;
    LayerEntry entry{kind, std::move(attention), std::move(dense),
                     arena_.allocate(12288), arena_.allocate(12288),
                     prefill_residual, prefill_residual};
    auto [entry_it, inserted] = entries_.emplace(layer_id, std::move(entry));
    if (!inserted) {
      throw std::invalid_argument("PrivateUse1: duplicate layer_id");
    }
    if (!instances_.count(kind)) {
      instances_.emplace(kind, create_instance(kind, graph_cache_slots, page_count,
                                               *entry_it->second.attention_weights,
                                               entry_it->second.dense_weights.get()));
      if (prefill_enabled_)
        prefill_instances_.emplace(kind, create_instance(kind, graph_cache_slots, kind == Kind::C ? page_count : 32,
            *entry_it->second.attention_weights, entry_it->second.dense_weights.get(), 32));
      if (prefill_enabled_)
        large_prefill_instances_.emplace(kind, create_instance(kind, graph_cache_slots,
            kind == Kind::C ? page_count : 32, *entry_it->second.attention_weights,
            entry_it->second.dense_weights.get(), 512));
      if (prefill_enabled_)
        xlarge_prefill_instances_.emplace(kind, create_instance(kind, graph_cache_slots,
            kind == Kind::C ? page_count : 32, *entry_it->second.attention_weights,
            entry_it->second.dense_weights.get(), 2048));
    } else {
      const auto& instance = instances_.at(kind);
      if (instance->cache_slots != graph_cache_slots || instance->page_count != page_count) {
        throw std::invalid_argument("PrivateUse1: isomorphic layers must share cache geometry");
      }
    }
  }

  void upload_model_weight(const std::string& name, const at::Tensor& tensor) {
    if (model_finalized_ || model_weights_.count(name) ||
        tensor.device().type() != c10::kCPU || !tensor.is_contiguous() ||
        tensor.scalar_type() != at::kBFloat16) {
      throw std::invalid_argument("PrivateUse1: invalid or duplicate model weight");
    }
    if (name == "embedding" || name == "lm_head") {
      if (tensor.dim() != 2 || tensor.size(0) != 152576 || tensor.size(1) != 6144)
        throw std::invalid_argument("PrivateUse1: invalid model matrix geometry");
    } else if (name == "final_norm") {
      if (tensor.dim() != 1 || tensor.size(0) != 6144)
        throw std::invalid_argument("PrivateUse1: invalid final norm geometry");
    } else {
      throw std::invalid_argument("PrivateUse1: unknown model weight");
    }
    uint64_t bytes = tensor.nbytes();
    const auto* data = static_cast<const uint8_t*>(tensor.data_ptr());
    if (name == "lm_head") {
      bytes = static_cast<uint64_t>(ModelHead::kLocalVocabSize) * 6144 * sizeof(uint16_t);
      data += rank_ * bytes;
    }
    const uint64_t address = arena_.allocate(bytes);
    arena_.copy_h2d_sync(runtime_.h2d_stream(), data, address, bytes);
    model_weights_.emplace(name, address);
  }

  void register_model_kv(uint32_t layer, const at::Tensor& key, const at::Tensor& value) {
    if (prefill_enabled_ || model_finalized_ || layer >= 70 || model_keys_[layer] || !entries_.count(layer))
      throw std::invalid_argument("PrivateUse1: invalid or duplicate model KV layer");
    const auto& instance = *instances_.at(entries_.at(layer).kind);
    if (key.scalar_type() != at::kBFloat16 || value.scalar_type() != at::kBFloat16 ||
        key.numel() != static_cast<int64_t>(instance.cache_slots) * 192 ||
        value.numel() != static_cast<int64_t>(instance.cache_slots) * 128)
      throw std::invalid_argument("PrivateUse1: invalid model KV geometry");
    const uint64_t key_address = address_of(key);
    const uint64_t value_address = address_of(value);
    const uint64_t key_bytes = key.nbytes();
    const uint64_t value_bytes = value.nbytes();
    const uint64_t limit = arena_.base_addr() + arena_.used_bytes();
    if (key_bytes > limit - key_address || value_bytes > limit - value_address)
      throw std::invalid_argument("PrivateUse1: model KV extends outside the arena");
    const auto overlaps = [](uint64_t a, uint64_t as, uint64_t b, uint64_t bs) {
      return a < b + bs && b < a + as;
    };
    if (overlaps(key_address, key_bytes, value_address, value_bytes))
      throw std::invalid_argument("PrivateUse1: model key and value buffers overlap");
    for (uint32_t other = 0; other < 70; ++other) {
      if (!model_keys_[other]) continue;
      const auto& other_instance = *instances_.at(entries_.at(other).kind);
      const uint64_t other_key_bytes = static_cast<uint64_t>(other_instance.cache_slots) * 192 * 2;
      const uint64_t other_value_bytes = static_cast<uint64_t>(other_instance.cache_slots) * 128 * 2;
      if (overlaps(key_address, key_bytes, model_keys_[other], other_key_bytes) ||
          overlaps(key_address, key_bytes, model_values_[other], other_value_bytes) ||
          overlaps(value_address, value_bytes, model_keys_[other], other_key_bytes) ||
          overlaps(value_address, value_bytes, model_values_[other], other_value_bytes))
        throw std::invalid_argument("PrivateUse1: model KV layers overlap");
    }
    model_keys_[layer] = key_address;
    model_values_[layer] = value_address;
  }

  void register_empty_model_kv(uint32_t layer) {
    if (!prefill_enabled_ || model_finalized_ || layer >= 70 || model_keys_[layer] || !entries_.count(layer))
      throw std::invalid_argument("PrivateUse1: invalid empty KV registration");
    const auto slots = instances_.at(entries_.at(layer).kind)->cache_slots;
    const size_t key_bytes = static_cast<size_t>(slots) * 192 * 2;
    const size_t value_bytes = static_cast<size_t>(slots) * 128 * 2;
    std::vector<uint8_t> zeros(key_bytes, 0);
    model_keys_[layer] = arena_.allocate(key_bytes);
    model_values_[layer] = arena_.allocate(value_bytes);
    arena_.copy_h2d_sync(runtime_.h2d_stream(), zeros.data(), model_keys_[layer], key_bytes);
    arena_.copy_h2d_sync(runtime_.h2d_stream(), zeros.data(), model_values_[layer], value_bytes);
  }

  at::Tensor finalize_model(const at::Tensor& full_rope, const at::Tensor& swa_rope,
                            const at::Tensor& pages, int32_t initial_token) {
    if (model_finalized_ || entries_.size() != 70 || model_weights_.size() != 3 ||
        !model_weights_.count("embedding") || !model_weights_.count("lm_head") ||
        !model_weights_.count("final_norm") || initial_token < 0 || initial_token >= 152576)
      throw std::invalid_argument("PrivateUse1: incomplete model or invalid initial token");
    for (uint32_t layer = 0; layer < 70; ++layer) {
      if (!entries_.count(layer) || !model_keys_[layer] || !model_values_[layer])
        throw std::invalid_argument("PrivateUse1: missing model layer or KV");
      const Kind expected = layer == 0 ? Kind::A : (is_full_layer(layer) ? Kind::B : Kind::C);
      if (entries_.at(layer).kind != expected)
        throw std::invalid_argument("PrivateUse1: unexpected model layer kind");
    }
    const auto cpu = [](const at::Tensor& tensor, at::ScalarType dtype) {
      return tensor.device().type() == c10::kCPU && tensor.is_contiguous() &&
             tensor.scalar_type() == dtype;
    };
    if (!cpu(full_rope, at::kBFloat16) || !cpu(swa_rope, at::kBFloat16) ||
        !cpu(pages, at::kInt) || full_rope.dim() != 2 || swa_rope.dim() != 2 ||
        full_rope.size(1) != 64 || full_rope.sizes() != swa_rope.sizes() ||
        full_rope.size(0) < 128 || full_rope.size(0) > (prefill_enabled_ ? 4224 : 4096) ||
        (prefill_enabled_ && full_rope.size(0) != 4224) || pages.numel() != 64)
      throw std::invalid_argument("PrivateUse1: invalid shared decode metadata");
    const auto slots = instances_.at(Kind::A)->cache_slots;
    if (instances_.at(Kind::B)->cache_slots != slots ||
        instances_.at(Kind::C)->cache_slots != slots ||
        instances_.at(Kind::A)->page_count != 64 || instances_.at(Kind::B)->page_count != 64)
      throw std::invalid_argument("PrivateUse1: model layers must share KV page geometry");
    cache_slots_ = slots;
    physical_cache_slots_ = slots - (prefill_enabled_ ? 128 : 0);
    rope_rows_ = static_cast<uint32_t>(full_rope.size(0));
    rope_first_position_ = prefill_enabled_ ? 0 : 4096;
    full_rope_.resize(static_cast<size_t>(full_rope.numel()));
    swa_rope_.resize(static_cast<size_t>(swa_rope.numel()));
    std::memcpy(full_rope_.data(), full_rope.data_ptr(), full_rope.nbytes());
    std::memcpy(swa_rope_.data(), swa_rope.data_ptr(), swa_rope.nbytes());
    std::memcpy(model_pages_.data(), pages.data_ptr(), pages.nbytes());
    validate_page_update(model_pages_, model_pages_, 0, physical_cache_slots_);
    validate_step_metadata(rope_first_position_ + rope_rows_ - 1, rope_first_position_, rope_rows_,
                           full_rope_.data(), swa_rope_.data(), model_pages_,
                           cache_slots_);
    check_syn(synHostMalloc(runtime_.device_id(), sizeof(StepMetadata), 0, &metadata_host_),
              "metadata pinned allocation");
    if (reinterpret_cast<uintptr_t>(metadata_host_) % alignof(StepMetadata) != 0)
      throw std::runtime_error("PrivateUse1: pinned metadata is not aligned");
    new (metadata_host_) StepMetadata{};
    check_syn(synHostMalloc(runtime_.device_id(), sizeof(int32_t), 0, &token_host_),
              "token pinned allocation");
    check_syn(synEventCreate(&metadata_ready_, runtime_.device_id(), 0), "metadata event creation");
    check_syn(synEventCreate(&compute_done_, runtime_.device_id(), 0), "compute event creation");
    check_syn(synEventCreate(&token_ready_, runtime_.device_id(), 0), "token event creation");
    metadata_device_ = arena_.allocate(sizeof(StepMetadata));
    trigger_device_ = arena_.allocate(sizeof(int32_t));
    token_device_ = arena_.allocate(sizeof(int32_t));
    embedding_device_ = arena_.allocate(6144 * sizeof(uint16_t));
    embedding_status_ = arena_.allocate(sizeof(int32_t));
    normalized_device_ = arena_.allocate(6144 * sizeof(uint16_t));
    local_logits_device_ = arena_.allocate(ModelHead::kLocalVocabSize * sizeof(uint16_t));
    logits_device_ = arena_.allocate(ModelHead::kVocabSize * sizeof(uint16_t));
    const int32_t trigger_value = 0;
    arena_.copy_h2d_sync(runtime_.h2d_stream(), &trigger_value, trigger_device_, sizeof(trigger_value));
    arena_.copy_h2d_sync(runtime_.h2d_stream(), &initial_token, token_device_, sizeof(initial_token));
    embedding_ = std::make_unique<ModelEmbedding>(arena_, model_weights_.at("embedding"), &recipe_cache_);
    head_ = std::make_unique<ModelHead>(arena_, model_weights_.at("final_norm"),
                                        model_weights_.at("lm_head"), &recipe_cache_, fused_model_,
                                        ModelHead::kLocalVocabSize);
    argmax_ = std::make_unique<ModelArgmax>(arena_, &recipe_cache_);
    if (prefill_enabled_) {
      prefill_tokens_device_ = arena_.allocate(4096 * sizeof(int32_t));
      prefill_metadata_device_ = arena_.allocate(128 * sizeof(PrefillMetadata));
      large_prefill_metadata_device_ = arena_.allocate(8 * sizeof(LargePrefillMetadata));
      xlarge_prefill_metadata_device_ = arena_.allocate(4 * sizeof(XLargePrefillMetadata));
      prefill_embedding_device_ = arena_.allocate(2048 * 6144 * sizeof(uint16_t));
      prefill_status_device_ = arena_.allocate(2048 * sizeof(int32_t));
      prefill_reduced_device_ = arena_.allocate(6144 * sizeof(uint16_t));
      prefill_embedding_ = std::make_unique<ModelEmbedding>(arena_, model_weights_.at("embedding"), &recipe_cache_, 32);
      large_prefill_embedding_ = std::make_unique<ModelEmbedding>(arena_, model_weights_.at("embedding"), &recipe_cache_, 512);
      xlarge_prefill_embedding_ = std::make_unique<ModelEmbedding>(arena_, model_weights_.at("embedding"), &recipe_cache_, 2048);
      prefill_head_ = std::make_unique<ModelHead>(arena_, model_weights_.at("final_norm"), model_weights_.at("lm_head"), &recipe_cache_, false, ModelHead::kLocalVocabSize);
      prefill_reduce_ = std::make_unique<runtime::TpReduce>(arena_, 6144, syn_type_float, &recipe_cache_);
      prefill_allreduce_cast_ = std::make_unique<runtime::TpReduce>(arena_, 6144, syn_type_float, &recipe_cache_, 1);
    }
    const uint64_t workspace = arena_.workspace_addr();
    model_embedding_launch_ = embedding_->prepare(token_device_, embedding_device_,
                                                   embedding_status_);
    model_layers_.reserve(70);
    uint64_t hidden = embedding_device_;
    uint64_t residual = embedding_device_;
    for (uint32_t layer = 0; layer < 70; ++layer) {
      const bool full = is_full_layer(layer);
      const uint64_t base = metadata_device_;
      std::array<uint64_t, 11> addresses{
          hidden, residual,
          base + (full ? offsetof(StepMetadata, full_cos) : offsetof(StepMetadata, swa_cos)),
          base + (full ? offsetof(StepMetadata, full_sin) : offsetof(StepMetadata, swa_sin)),
          model_keys_[layer], model_values_[layer], base + offsetof(StepMetadata, slot),
          base + (full ? offsetof(StepMetadata, full_pages) : offsetof(StepMetadata, window_pages)),
          base + offsetof(StepMetadata, full_usage),
          base + (full ? offsetof(StepMetadata, full_groups) : offsetof(StepMetadata, window_groups)),
          base + offsetof(StepMetadata, position)};
      auto prepared = prepare_layer(layer, addresses);
      hidden = fused_model_ ? prepared.feed_forward_gathered : entries_.at(layer).layer_output;
      model_layers_.push_back(std::move(prepared));
      residual = entries_.at(layer).residual_output;
    }
    model_head_launch_ = head_->prepare(hidden, residual, normalized_device_, local_logits_device_);
    model_argmax_launch_ = argmax_->prepare(logits_device_, embedding_status_, token_device_);
    if (prefill_enabled_) {
      auto bind = [&]<typename Metadata>(ModelEmbedding& embedding, runtime::TpReduce& reduce,
                                       uint64_t base, std::vector<PreparedLayer>& layers,
                                       std::vector<std::pair<uint64_t*, uint64_t>>& metadata_bindings,
                                       std::vector<PreparedLayer>& short_layers,
                                       std::vector<std::pair<uint64_t*, uint64_t>>& short_bindings,
                                       runtime::PreparedLaunch& embedding_launch, uint64_t*& token_binding,
                                       runtime::PreparedLaunch& reduce_launch, runtime::PreparedLaunch& head_launch) {
        embedding_launch = embedding.prepare(prefill_tokens_device_, prefill_embedding_device_, prefill_status_device_);
        for (auto& tensor : embedding_launch.tensors) if (tensor.pTensorAddress == prefill_tokens_device_) {
          if (token_binding) throw std::runtime_error("PrivateUse1: ambiguous prefill token binding");
          token_binding = &tensor.pTensorAddress;
        }
        if (!token_binding) throw std::runtime_error("PrivateUse1: missing prefill token binding");
        layers.reserve(70);
        short_layers.reserve(70);
        uint64_t hidden = prefill_embedding_device_, residual = prefill_embedding_device_;
        for (uint32_t layer = 0; layer < 70; ++layer) {
          const bool full = is_full_layer(layer);
          std::array<uint64_t, 11> addresses{
              hidden, residual,
              base + (full ? offsetof(Metadata, full_cos) : offsetof(Metadata, swa_cos)),
              base + (full ? offsetof(Metadata, full_sin) : offsetof(Metadata, swa_sin)),
              model_keys_[layer], model_values_[layer], base + offsetof(Metadata, slots),
              base + (full ? offsetof(Metadata, full_pages) : offsetof(Metadata, window_pages)),
              base + offsetof(Metadata, full_usage),
              base + (full ? offsetof(Metadata, full_groups) : offsetof(Metadata, window_groups)),
              base + offsetof(Metadata, positions)};
          layers.push_back(prepare_layer(layer, addresses, Metadata::kRows));
          short_layers.push_back(prepare_layer(layer, addresses, Metadata::kRows, full));
          hidden = layers.back().feed_forward_gathered;
          residual = Metadata::kRows == 32 ? entries_.at(layer).prefill_residual_output
                                           : entries_.at(layer).large_prefill_residual_output;
        }
        metadata_bindings = prefill_metadata_bindings(layers, base, sizeof(Metadata));
        short_bindings = prefill_metadata_bindings(short_layers, base, sizeof(Metadata));
        const uint64_t last = (Metadata::kRows - 1) * 6144 * sizeof(uint16_t);
        const uint64_t rank_stride = Metadata::kRows * 6144 * sizeof(float);
        if constexpr (Metadata::kRows >= 512)
          reduce_launch = prefill_allreduce_cast_->prepare(hidden + (Metadata::kRows - 1) * 6144 * sizeof(float),
                                                            prefill_reduced_device_);
        else
          reduce_launch = reduce.prepare_strided(hidden + (Metadata::kRows - 1) * 6144 * sizeof(float),
                                                 prefill_reduced_device_, rank_stride);
        head_launch = prefill_head_->prepare(prefill_reduced_device_, residual + last,
                                            normalized_device_, local_logits_device_);
      };
      bind.template operator()<PrefillMetadata>(*prefill_embedding_, *prefill_reduce_, prefill_metadata_device_,
          prefill_layers_, prefill_metadata_bindings_, short_prefill_layers_, short_prefill_metadata_bindings_, prefill_embedding_launch_, prefill_token_binding_,
          prefill_reduce_launch_, prefill_head_launch_);
      bind.template operator()<LargePrefillMetadata>(*large_prefill_embedding_, *prefill_reduce_, large_prefill_metadata_device_,
          large_prefill_layers_, large_prefill_metadata_bindings_, short_large_prefill_layers_, short_large_prefill_metadata_bindings_, large_prefill_embedding_launch_, large_prefill_token_binding_,
          large_prefill_reduce_launch_, large_prefill_head_launch_);
      bind.template operator()<XLargePrefillMetadata>(*xlarge_prefill_embedding_, *prefill_reduce_, xlarge_prefill_metadata_device_,
          xlarge_prefill_layers_, xlarge_prefill_metadata_bindings_, short_xlarge_prefill_layers_, short_xlarge_prefill_metadata_bindings_, xlarge_prefill_embedding_launch_, xlarge_prefill_token_binding_,
          xlarge_prefill_reduce_launch_, xlarge_prefill_head_launch_);
      prefill_argmax_launch_ = argmax_->prepare(logits_device_, prefill_status_device_, token_device_);
    }
    if (arena_.workspace_addr() != workspace)
      throw std::runtime_error("PrivateUse1: workspace changed during model binding");
    model_position_ = prefill_enabled_ ? 0 : 4096;
    model_finalized_ = true;
    return wrap(trigger_device_, {1}, at::kInt);
  }

  void update_model_pages(const at::Tensor& pages) {
    if (!model_finalized_ || model_pending_ || pages.device().type() != c10::kCPU ||
        !pages.is_contiguous() || pages.scalar_type() != at::kInt || pages.numel() != 64)
      throw std::invalid_argument("PrivateUse1: invalid scheduled page table");
    std::array<int32_t, 64> next{};
    std::memcpy(next.data(), pages.data_ptr(), pages.nbytes());
    const uint32_t written = prefill_enabled_ && !prefill_complete_ ? prefill_rows_ : model_position_;
    validate_page_update(model_pages_, next, written, physical_cache_slots_);
    model_pages_ = next;
  }

  void reset_prefill() {
    if (!prefill_enabled_ || !model_finalized_ || model_pending_ || !prefill_complete_)
      throw std::invalid_argument("PrivateUse1: completed request required before reset");
    auto pages = model_pages_;
    std::sort(pages.begin(), pages.end());
    const auto stream = runtime_.compute_stream();
    for (size_t first = 0; first < pages.size();) {
      size_t end = first + 1;
      while (end < pages.size() && pages[end] == pages[end - 1] + 1) ++end;
      const size_t slot = static_cast<size_t>(pages[first]) * 128;
      const size_t slots = (end - first) * 128;
      for (size_t layer = 0; layer < model_keys_.size(); ++layer) {
        check_syn(synMemsetD8Async(model_keys_[layer] + slot * 192 * 2, 0,
                                   slots * 192 * 2, stream), "reset model key pages");
        check_syn(synMemsetD8Async(model_values_[layer] + slot * 128 * 2, 0,
                                   slots * 128 * 2, stream), "reset model value pages");
      }
      first = end;
    }
    model_position_ = 0;
    prefill_rows_ = 0;
    prompt_length_ = 0;
    prefill_complete_ = false;
  }

  int32_t prefill(const at::Tensor& tokens, const at::Tensor& full_rope,
                  const at::Tensor& swa_rope) {
    if (prefill_rows_ != 0)
      throw std::invalid_argument("PrivateUse1: full prefill requires a fresh request");
    return prefill_chunk(tokens, full_rope, swa_rope, static_cast<uint32_t>(tokens.numel()));
  }

  int32_t prefill_chunk(const at::Tensor& tokens, const at::Tensor& full_rope,
                        const at::Tensor& swa_rope, uint32_t prompt_length) {
    if (!prefill_enabled_ || !model_finalized_ || model_pending_ || prefill_complete_ ||
        model_position_ != 0)
      throw std::invalid_argument("PrivateUse1: prefill is not ready");
    const auto cpu = [](const at::Tensor& t, at::ScalarType type) {
      return t.device().type() == c10::kCPU && t.is_contiguous() && t.scalar_type() == type;
    };
    if (prompt_length == 0 || prompt_length > 4096 ||
        (prompt_length_ && prompt_length_ != prompt_length) || prefill_rows_ >= prompt_length ||
        !cpu(tokens, at::kInt) || tokens.numel() <= 0 ||
        tokens.numel() > prompt_length - prefill_rows_ ||
        !cpu(full_rope, at::kBFloat16) || !cpu(swa_rope, at::kBFloat16) ||
        full_rope.dim() != 2 || full_rope.size(0) != 4096 || full_rope.size(1) != 64 ||
        full_rope.sizes() != swa_rope.sizes())
      throw std::invalid_argument("PrivateUse1: invalid prompt or prefill RoPE");
    prompt_length_ = prompt_length;
    const auto* token_data = tokens.data_ptr<int32_t>();
    for (int64_t i = 0; i < tokens.numel(); ++i)
      if (token_data[i] < 0 || token_data[i] >= 152576)
        throw std::invalid_argument("PrivateUse1: prompt token out of range");
    const auto* full = reinterpret_cast<const uint16_t*>(full_rope.data_ptr());
    const auto* swa = reinterpret_cast<const uint16_t*>(swa_rope.data_ptr());
    const auto chunks = prefill_chunks(static_cast<uint32_t>(tokens.numel()));
    std::vector<PrefillMetadata> metadata;
    std::vector<LargePrefillMetadata> large_metadata;
    std::vector<XLargePrefillMetadata> xlarge_metadata;
    for (const auto& chunk : chunks) {
      if (chunk.rows == 2048) {
        xlarge_metadata.emplace_back();
        fill_prefill_metadata(xlarge_metadata.back(), prefill_rows_ + chunk.offset, 4096, full, swa,
                              model_pages_, cache_slots_, chunk.valid_rows, physical_cache_slots_);
      } else if (chunk.rows == 512) {
        large_metadata.emplace_back();
        fill_prefill_metadata(large_metadata.back(), prefill_rows_ + chunk.offset, 4096, full, swa,
                              model_pages_, cache_slots_, chunk.valid_rows, physical_cache_slots_);
      } else {
        metadata.emplace_back();
        fill_prefill_metadata(metadata.back(), prefill_rows_ + chunk.offset, 4096, full, swa,
                              model_pages_, cache_slots_, chunk.valid_rows, physical_cache_slots_);
      }
    }
    std::vector<int32_t> padded_tokens(token_data, token_data + tokens.numel());
    padded_tokens.resize(padded_tokens.size() + chunks.back().rows - chunks.back().valid_rows, 0);
    arena_.copy_h2d_sync(runtime_.h2d_stream(), padded_tokens.data(), prefill_tokens_device_,
                         padded_tokens.size() * sizeof(int32_t));
    if (!metadata.empty())
      arena_.copy_h2d_sync(runtime_.h2d_stream(), metadata.data(), prefill_metadata_device_,
                           metadata.size() * sizeof(PrefillMetadata));
    if (!large_metadata.empty())
      arena_.copy_h2d_sync(runtime_.h2d_stream(), large_metadata.data(), large_prefill_metadata_device_,
                           large_metadata.size() * sizeof(LargePrefillMetadata));
    if (!xlarge_metadata.empty())
      arena_.copy_h2d_sync(runtime_.h2d_stream(), xlarge_metadata.data(), xlarge_prefill_metadata_device_,
                           xlarge_metadata.size() * sizeof(XLargePrefillMetadata));
    const auto stream = runtime_.compute_stream();
    for (const auto& chunk : chunks) {
      const bool large = chunk.rows == 512, xlarge = chunk.rows == 2048;
      auto& token_binding = xlarge ? xlarge_prefill_token_binding_ : large ? large_prefill_token_binding_ : prefill_token_binding_;
      auto& embedding = xlarge ? xlarge_prefill_embedding_launch_ : large ? large_prefill_embedding_launch_ : prefill_embedding_launch_;
      const bool short_prefix = prefill_rows_ + chunk.offset + chunk.valid_rows <= 2048;
      auto& bindings = short_prefix
          ? (xlarge ? short_xlarge_prefill_metadata_bindings_ : large ? short_large_prefill_metadata_bindings_ : short_prefill_metadata_bindings_)
          : (xlarge ? xlarge_prefill_metadata_bindings_ : large ? large_prefill_metadata_bindings_ : prefill_metadata_bindings_);
      auto& layers = short_prefix
          ? (xlarge ? short_xlarge_prefill_layers_ : large ? short_large_prefill_layers_ : short_prefill_layers_)
          : (xlarge ? xlarge_prefill_layers_ : large ? large_prefill_layers_ : prefill_layers_);
      *token_binding = prefill_tokens_device_ + chunk.offset * sizeof(int32_t);
      embedding.execute(stream);
      const uint64_t base = xlarge ? xlarge_prefill_metadata_device_ + chunk.index * sizeof(XLargePrefillMetadata)
                            : large ? large_prefill_metadata_device_ + chunk.index * sizeof(LargePrefillMetadata)
                                    : prefill_metadata_device_ + chunk.index * sizeof(PrefillMetadata);
      for (auto [address, offset] : bindings) *address = base + offset;
      for (const auto& layer : layers) layer.execute_fused(stream);
    }
    if (prefill_rows_ + tokens.numel() != prompt_length_) {
      runtime_.synchronize_compute();
      prefill_rows_ += static_cast<uint32_t>(tokens.numel());
      return -1;
    }
    const bool last_large = chunks.back().rows == 512, last_xlarge = chunks.back().rows == 2048;
    auto& last_layers = last_xlarge ? xlarge_prefill_layers_ : last_large ? large_prefill_layers_ : prefill_layers_;
    auto& last_reduce = last_xlarge ? xlarge_prefill_reduce_launch_ : last_large ? large_prefill_reduce_launch_ : prefill_reduce_launch_;
    auto& last_head = last_xlarge ? xlarge_prefill_head_launch_ : last_large ? large_prefill_head_launch_ : prefill_head_launch_;
    const auto& last_chunk = chunks.back();
    const uint64_t final_hidden = last_layers.back().feed_forward_gathered +
        static_cast<uint64_t>(last_chunk.valid_rows - 1) * 6144 * sizeof(float);
    const uint64_t rank_stride = static_cast<uint64_t>(last_chunk.rows) * 6144 * sizeof(float);
    for (unsigned rank = 0; rank < (last_chunk.rows >= 512 ? 1U : 8U); ++rank)
      last_reduce.tensors[rank].pTensorAddress = final_hidden + rank * rank_stride;
    const uint64_t residual_base = last_chunk.rows == 32 ? entries_.at(69).prefill_residual_output
                                                          : entries_.at(69).large_prefill_residual_output;
    last_head.tensors[1].pTensorAddress = residual_base +
        static_cast<uint64_t>(last_chunk.valid_rows - 1) * 6144 * sizeof(uint16_t);
    last_reduce.execute(stream);
    last_head.execute(stream);
    if (hcclAllGather(reinterpret_cast<const void*>(local_logits_device_),
                      reinterpret_cast<void*>(logits_device_), ModelHead::kLocalVocabSize,
                      hcclBfloat16, runtime_.comm(), stream) != hcclSuccess)
      throw std::runtime_error("PrivateUse1: prefill logits AllGather failed");
    prefill_argmax_launch_.execute(stream);
    runtime_.synchronize_compute();
    check_syn(synMemCopyAsync(runtime_.d2h_stream(), token_device_, sizeof(int32_t),
                              reinterpret_cast<uint64_t>(token_host_), DRAM_TO_HOST),
              "prefill token D2H");
    check_syn(synStreamSynchronize(runtime_.d2h_stream()), "prefill token completion");
    const int32_t token = *static_cast<int32_t*>(token_host_);
    if (token < 0 || token >= 152576) throw std::runtime_error("PrivateUse1: invalid prefill token or logits");
    prefill_rows_ = prompt_length_;
    model_position_ = prompt_length_;
    prefill_complete_ = true;
    return token;
  }

  at::Tensor forward_model(const at::Tensor& trigger) {
    if (!model_finalized_ || model_pending_ || (prefill_enabled_ && !prefill_complete_) ||
        trigger.scalar_type() != at::kInt ||
        trigger.numel() != 1 || address_of(trigger) != trigger_device_)
      throw std::invalid_argument("PrivateUse1: invalid model step or unfinished previous step");
    if (model_position_ < rope_first_position_ || model_position_ - rope_first_position_ >= rope_rows_)
      throw std::invalid_argument("PrivateUse1: model position exceeds registered RoPE rows");
    auto& metadata = *static_cast<StepMetadata*>(metadata_host_);
    fill_step_metadata_prevalidated(metadata, model_position_, rope_first_position_, full_rope_.data(),
                                    swa_rope_.data(), model_pages_);
    model_pending_ = true;
    check_syn(synMemCopyAsync(runtime_.h2d_stream(), reinterpret_cast<uint64_t>(metadata_host_),
                              sizeof(StepMetadata), metadata_device_, HOST_TO_DRAM),
              "step metadata H2D");
    check_syn(synEventRecord(metadata_ready_, runtime_.h2d_stream()), "metadata event record");
    check_syn(synStreamWaitEvent(runtime_.compute_stream(), metadata_ready_, 0), "metadata wait");
    const auto stream = runtime_.compute_stream();
    model_embedding_launch_.execute(stream);
    if (fused_model_) {
      for (const auto& layer : model_layers_) layer.execute_fused(stream);
    } else {
      for (const auto& layer : model_layers_) layer.execute(stream);
    }
    model_head_launch_.execute(stream);
    if (hcclAllGather(reinterpret_cast<const void*>(local_logits_device_),
                      reinterpret_cast<void*>(logits_device_), ModelHead::kLocalVocabSize,
                      hcclBfloat16, runtime_.comm(), stream) != hcclSuccess)
      throw std::runtime_error("PrivateUse1: model logits AllGather failed");
    model_argmax_launch_.execute(stream);
    check_syn(synEventRecord(compute_done_, runtime_.compute_stream()), "compute event record");
    check_syn(synStreamWaitEvent(runtime_.d2h_stream(), compute_done_, 0), "output wait");
    check_syn(synMemCopyAsync(runtime_.d2h_stream(), token_device_, sizeof(int32_t),
                              reinterpret_cast<uint64_t>(token_host_), DRAM_TO_HOST),
              "token D2H");
    check_syn(synEventRecord(token_ready_, runtime_.d2h_stream()), "token completion event");
    return wrap(token_device_, {1}, at::kInt);
  }

  int32_t finish_model_step() {
    if (!model_pending_) throw std::invalid_argument("PrivateUse1: no pending model step");
    check_syn(synEventSynchronize(token_ready_), "token D2H completion");
    model_pending_ = false;
    const int32_t token = *static_cast<int32_t*>(token_host_);
    if (token < 0 || token >= 152576) {
      model_finalized_ = false;
      throw std::runtime_error("PrivateUse1: invalid token or nonfinite model logits");
    }
    ++model_position_;
    return token;
  }

  at::Tensor model_logits() {
    if (!model_finalized_ || model_pending_ || (prefill_enabled_ && !prefill_complete_))
      throw std::invalid_argument("PrivateUse1: model logits are not ready");
    return wrap(logits_device_, {1, 152576}, at::kBFloat16);
  }

  std::vector<at::Tensor> model_kv(uint32_t layer) {
    if (!model_finalized_ || model_pending_ || (prefill_enabled_ && !prefill_complete_) || layer >= 70)
      throw std::invalid_argument("PrivateUse1: model KV is not ready");
    return {wrap(model_keys_[layer], {physical_cache_slots_, 192}, at::kBFloat16),
            wrap(model_values_[layer], {physical_cache_slots_, 128}, at::kBFloat16)};
  }

  std::vector<at::Tensor> model_kv_slots(uint32_t layer, uint32_t first,
                                         uint32_t count) {
    if (!model_finalized_ || model_pending_ || (prefill_enabled_ && !prefill_complete_) ||
        layer >= 70 || count == 0 || first >= physical_cache_slots_ || count > physical_cache_slots_ - first)
      throw std::invalid_argument("PrivateUse1: invalid model KV slot range");
    return {wrap(model_keys_[layer] + static_cast<uint64_t>(first) * 192 * sizeof(uint16_t),
                 {count, 192}, at::kBFloat16),
            wrap(model_values_[layer] + static_cast<uint64_t>(first) * 128 * sizeof(uint16_t),
                 {count, 128}, at::kBFloat16)};
  }

  std::vector<at::Tensor> forward(uint32_t layer_id,
                                  const std::vector<at::Tensor>& inputs) {
    if (fused_model_)
      throw std::invalid_argument("PrivateUse1: fused model does not expose standalone layers");
    auto& entry = entries_.at(layer_id);
    auto& instance = *instances_.at(entry.kind);
    const auto addresses = validate_inputs(instance, inputs);
    execute_layer(layer_id, addresses);
    return {wrap(entry.residual_output, {1, 6144}, at::kBFloat16),
            wrap(entry.layer_output, {1, 6144}, at::kBFloat16)};
  }

  PreparedLayer prepare_layer(uint32_t layer_id,
                              const std::array<uint64_t, 11>& addresses,
                              uint32_t prefill = 0, bool short_prefix = false) {
    auto& entry = entries_.at(layer_id);
    auto& map = prefill == 2048 ? xlarge_prefill_instances_ : prefill == 512 ? large_prefill_instances_ : prefill == 32 ? prefill_instances_ : instances_;
    auto& instance = *map.at(entry.kind);
    const uint64_t residual_output = prefill >= 512 ? entry.large_prefill_residual_output
                                     : prefill == 32 ? entry.prefill_residual_output : entry.residual_output;
    if (entry.kind == Kind::A) {
      instance.a->rebind_weights(*entry.attention_weights, *entry.dense_weights);
      auto io = instance.a_io;
      set_common_attention(io.attention, addresses);
      io.attention.residual = addresses[0];
      io.dense_outputs.residual_out_bf16 = residual_output;
      io.layer_output_bf16 = entry.layer_output;
      return instance.a->prepare(io, short_prefix);
    }
    if (entry.kind == Kind::B) {
      instance.b->rebind_weights(*entry.attention_weights);
      auto io = instance.b_io;
      set_common_attention(io.attention, addresses);
      io.moe_outputs.residual_out_bf16 = residual_output;
      io.layer_output_bf16 = entry.layer_output;
      return instance.b->prepare(io, short_prefix);
    }
    instance.c->rebind_weights(*entry.attention_weights);
    auto io = instance.c_io;
    set_common_attention(io.attention, addresses);
    io.attention.window_pages = addresses[7];
    io.attention.window_groups = addresses[9];
    io.moe_outputs.residual_out_bf16 = residual_output;
    io.layer_output_bf16 = entry.layer_output;
    return instance.c->prepare(io);
  }

  void execute_layer(uint32_t layer_id, const std::array<uint64_t, 11>& addresses) {
    auto& entry = entries_.at(layer_id);
    auto& instance = *instances_.at(entry.kind);
    if (entry.kind == Kind::A) {
      instance.a->rebind_weights(*entry.attention_weights, *entry.dense_weights);
      auto& io = instance.a_io;
      set_common_attention(io.attention, addresses);
      io.attention.residual = addresses[0];
      io.dense_outputs.residual_out_bf16 = entry.residual_output;
      io.layer_output_bf16 = entry.layer_output;
      instance.a->execute(io, runtime_.compute_stream());
    } else if (entry.kind == Kind::B) {
      instance.b->rebind_weights(*entry.attention_weights);
      auto& io = instance.b_io;
      set_common_attention(io.attention, addresses);
      io.moe_outputs.residual_out_bf16 = entry.residual_output;
      io.layer_output_bf16 = entry.layer_output;
      instance.b->execute(io, runtime_.compute_stream());
    } else {
      instance.c->rebind_weights(*entry.attention_weights);
      auto& io = instance.c_io;
      set_common_attention(io.attention, addresses);
      io.attention.window_pages = addresses[7];
      io.attention.window_groups = addresses[9];
      io.moe_outputs.residual_out_bf16 = entry.residual_output;
      io.layer_output_bf16 = entry.layer_output;
      instance.c->execute(io, runtime_.compute_stream());
    }
  }

  std::vector<uint64_t> recipe_identity(int64_t kind_value) const {
    return instances_.at(checked_kind(kind_value))->recipe_identity();
  }

  std::vector<uint64_t> recipe_cache_stats() const {
    const auto [loaded, saved] = recipe_cache_.stats();
    return {loaded, saved};
  }

  at::Tensor wrap(uint64_t address, const std::vector<int64_t>& shape,
                  at::ScalarType dtype) {
    if (!owns_address(address)) {
      throw std::invalid_argument("PrivateUse1: address is outside the owned arena");
    }
    auto owner = shared_from_this();
    return at::from_blob(reinterpret_cast<void*>(address), shape,
                         [owner](void*) {},
                         at::TensorOptions().device(c10::Device(c10::kPrivateUse1, rank_))
                             .dtype(dtype).requires_grad(false));
  }

  uint64_t address_of(const at::Tensor& tensor) const {
    if (tensor.device().type() != c10::kPrivateUse1 || !tensor.is_contiguous()) {
      throw std::invalid_argument("PrivateUse1: expected a contiguous GaudiRT tensor");
    }
    const uint64_t address = reinterpret_cast<uint64_t>(tensor.data_ptr());
    if (!owns_address(address)) {
      throw std::invalid_argument("PrivateUse1: tensor does not belong to this rank");
    }
    return address;
  }

  bool owns_address(const void* pointer) const {
    return owns_address(reinterpret_cast<uint64_t>(pointer));
  }

  bool owns_address(uint64_t address) const {
    return address != 0 && address >= arena_.base_addr() &&
           address < arena_.base_addr() + arena_.used_bytes();
  }

  uint32_t rank() const noexcept { return rank_; }
  uint64_t allocate(size_t bytes) { return arena_.allocate(bytes); }

  void copy_h2d(uint64_t destination, const void* source, size_t bytes) {
    arena_.copy_h2d_sync(runtime_.h2d_stream(), source, destination, bytes);
  }

  void copy_d2h(void* destination, uint64_t source, size_t bytes) {
    runtime_.synchronize_compute();
    arena_.copy_d2h_sync(runtime_.d2h_stream(), source, destination, bytes);
  }

  void synchronize_compute() const { runtime_.synchronize_compute(); }
  void synchronize_all() const { runtime_.synchronize_all(); }

  void synchronize_shutdown() {
    if (model_pending_)
      throw std::runtime_error("PrivateUse1: finish the pending model step before shutdown");
    bridge::synchronize_shutdown(runtime_, shutdown_buffer_);
  }

 private:
  std::unique_ptr<Instance> create_instance(Kind kind, uint32_t cache_slots,
                                            uint32_t page_count,
                                            runtime::WeightManifest& attention_weights,
                                            runtime::WeightManifest* dense_weights,
                                            uint32_t rows = 1) {
    auto instance = std::make_unique<Instance>();
    instance->kind = kind;
    instance->cache_slots = cache_slots;
    instance->page_count = page_count;
    if (kind == Kind::A) {
      instance->a = std::make_unique<Layer0>(arena_, runtime_, attention_weights,
                                             *dense_weights, cache_slots, page_count,
                                             &recipe_cache_, fused_model_, rows);
      auto& io = instance->a_io;
      io.attention.output_local_fp32 = arena_.allocate(rows * 6144 * 4);
      io.dense_outputs.activation_fp32 = arena_.allocate(rows * 2048 * 4);
      io.dense_outputs.output_local_fp32 = arena_.allocate(rows * 6144 * 4);
      io.attention_output_bf16 = arena_.allocate(rows * 12288);
      io.attention_gathered_fp32 = rows >= 512 ? io.attention.output_local_fp32 : arena_.allocate(rows * 8 * 6144 * 4);
      io.down_gathered_fp32 = rows >= 512 ? io.dense_outputs.output_local_fp32 : arena_.allocate(rows * 8 * 6144 * 4);
    } else if (kind == Kind::B) {
      const auto* decode = rows == 1 ? nullptr : instances_.at(kind)->b.get();
      instance->b = std::make_unique<LayerB>(arena_, runtime_, attention_weights,
                                             cache_slots, page_count, &recipe_cache_,
                                             fused_model_, rows,
                                             decode ? decode->moe_lut() : 0,
                                             decode ? decode->moe_directions() : 0);
      auto& io = instance->b_io;
      io.attention.residual_out = arena_.allocate(rows * 12288);
      io.attention.output_local_fp32 = arena_.allocate(rows * 6144 * 4);
      io.moe_outputs.moe_local_fp32 = arena_.allocate(rows * 6144 * 4);
      io.attention_output_bf16 = arena_.allocate(rows * 12288);
      io.attention_gathered_fp32 = rows >= 512 ? io.attention.output_local_fp32 : arena_.allocate(rows * 8 * 6144 * 4);
      io.moe_gathered_fp32 = rows >= 512 ? io.moe_outputs.moe_local_fp32 : arena_.allocate(rows * 8 * 6144 * 4);
    } else {
      const auto* decode = rows == 1 ? nullptr : instances_.at(kind)->c.get();
      instance->c = std::make_unique<LayerC>(arena_, runtime_, attention_weights,
                                             cache_slots, &recipe_cache_, fused_model_, rows,
                                             decode ? decode->moe_lut() : 0,
                                             decode ? decode->moe_directions() : 0);
      auto& io = instance->c_io;
      io.attention.residual_out = arena_.allocate(rows * 12288);
      io.attention.output_local = arena_.allocate(rows * 6144 * (rows >= 512 ? 4 : 2));
      io.moe_outputs.moe_local_fp32 = arena_.allocate(rows * 6144 * 4);
      io.attention_output_bf16 = arena_.allocate(rows * 12288);
      io.attention_gathered_bf16 = rows >= 512 ? io.attention.output_local : arena_.allocate(rows * 8 * 6144 * 2);
      io.moe_gathered_fp32 = rows >= 512 ? io.moe_outputs.moe_local_fp32 : arena_.allocate(rows * 8 * 6144 * 4);
    }
    return instance;
  }

  std::array<uint64_t, 11> validate_inputs(const Instance& instance,
                                           const std::vector<at::Tensor>& inputs) const {
    if (inputs.size() != 11) {
      throw std::invalid_argument("PrivateUse1: expected eleven explicit dynamic inputs");
    }
    const std::array<at::ScalarType, 11> dtypes = {
        at::kBFloat16, at::kBFloat16, at::kBFloat16, at::kBFloat16,
        at::kBFloat16, at::kBFloat16, at::kInt, at::kInt, at::kInt,
        at::kInt, at::kInt};
    const std::array<int64_t, 11> elements = {
        6144, 6144, 64, 64, static_cast<int64_t>(instance.cache_slots * 192),
        static_cast<int64_t>(instance.cache_slots * 128), 1,
        static_cast<int64_t>(instance.page_count),
        static_cast<int64_t>(instance.page_count),
        static_cast<int64_t>(instance.page_count), 1};
    std::array<uint64_t, 11> addresses = {};
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].scalar_type() != dtypes[i] || inputs[i].numel() != elements[i]) {
        throw std::invalid_argument("PrivateUse1: invalid input dtype or geometry at index " +
                                    std::to_string(i));
      }
      addresses[i] = address_of(inputs[i]);
    }
    return addresses;
  }

  static void set_common_attention(MimoAttentionIo& io,
                                   const std::array<uint64_t, 11>& addresses) {
    io.hidden_states = addresses[0];
    io.residual = addresses[1];
    io.rope_cos = addresses[2];
    io.rope_sin = addresses[3];
    io.key_cache = addresses[4];
    io.value_cache = addresses[5];
    io.slot_mapping = addresses[6];
    io.full_block_list = addresses[7];
    io.full_block_usage = addresses[8];
    io.full_block_groups = addresses[9];
    io.position = addresses[10];
  }

  uint32_t rank_;
  runtime::DeviceRuntime runtime_;
  runtime::HbmArena arena_;
  uint64_t shutdown_buffer_ = 0;
  runtime::RecipeCache recipe_cache_;
  std::unordered_map<Kind, std::unique_ptr<Instance>> instances_;
  std::unordered_map<Kind, std::unique_ptr<Instance>> prefill_instances_;
  std::unordered_map<Kind, std::unique_ptr<Instance>> large_prefill_instances_;
  std::unordered_map<Kind, std::unique_ptr<Instance>> xlarge_prefill_instances_;
  std::unordered_map<uint32_t, LayerEntry> entries_;
  std::unordered_map<uint32_t, PendingWeights> pending_;
  std::unordered_map<std::string, uint64_t> model_weights_;
  std::vector<PreparedLayer> model_layers_;
  std::vector<PreparedLayer> prefill_layers_;
  std::vector<PreparedLayer> large_prefill_layers_;
  std::vector<PreparedLayer> xlarge_prefill_layers_;
  std::vector<PreparedLayer> short_prefill_layers_;
  std::vector<PreparedLayer> short_large_prefill_layers_;
  std::vector<PreparedLayer> short_xlarge_prefill_layers_;
  std::vector<std::pair<uint64_t*, uint64_t>> prefill_metadata_bindings_;
  std::vector<std::pair<uint64_t*, uint64_t>> large_prefill_metadata_bindings_;
  std::vector<std::pair<uint64_t*, uint64_t>> xlarge_prefill_metadata_bindings_;
  std::vector<std::pair<uint64_t*, uint64_t>> short_prefill_metadata_bindings_;
  std::vector<std::pair<uint64_t*, uint64_t>> short_large_prefill_metadata_bindings_;
  std::vector<std::pair<uint64_t*, uint64_t>> short_xlarge_prefill_metadata_bindings_;
  uint64_t* prefill_token_binding_ = nullptr;
  uint64_t* large_prefill_token_binding_ = nullptr;
  uint64_t* xlarge_prefill_token_binding_ = nullptr;
  runtime::PreparedLaunch prefill_embedding_launch_;
  runtime::PreparedLaunch large_prefill_embedding_launch_;
  runtime::PreparedLaunch xlarge_prefill_embedding_launch_;
  runtime::PreparedLaunch prefill_reduce_launch_;
  runtime::PreparedLaunch large_prefill_reduce_launch_;
  runtime::PreparedLaunch xlarge_prefill_reduce_launch_;
  runtime::PreparedLaunch prefill_head_launch_;
  runtime::PreparedLaunch large_prefill_head_launch_;
  runtime::PreparedLaunch xlarge_prefill_head_launch_;
  runtime::PreparedLaunch prefill_argmax_launch_;
  runtime::PreparedLaunch model_embedding_launch_;
  runtime::PreparedLaunch model_head_launch_;
  runtime::PreparedLaunch model_argmax_launch_;
  std::array<uint64_t, 2> prefill_residual_slots_{};
  std::array<uint64_t, 70> model_keys_{};
  std::array<uint64_t, 70> model_values_{};
  std::array<int32_t, 64> model_pages_{};
  std::vector<uint16_t> full_rope_;
  std::vector<uint16_t> swa_rope_;
  uint32_t cache_slots_ = 0;
  uint32_t physical_cache_slots_ = 0;
  uint32_t rope_rows_ = 0;
  uint32_t rope_first_position_ = 0;
  uint32_t model_position_ = 0;
  bool model_finalized_ = false;
  bool model_pending_ = false;
  bool fused_model_ = false;
  bool prefill_enabled_ = false;
  bool prefill_complete_ = false;
  uint32_t prefill_rows_ = 0;
  uint32_t prompt_length_ = 0;
  void* metadata_host_ = nullptr;
  void* token_host_ = nullptr;
  synEventHandle metadata_ready_ = nullptr;
  synEventHandle compute_done_ = nullptr;
  synEventHandle token_ready_ = nullptr;
  uint64_t metadata_device_ = 0;
  uint64_t trigger_device_ = 0;
  uint64_t token_device_ = 0;
  uint64_t embedding_device_ = 0;
  uint64_t embedding_status_ = 0;
  uint64_t normalized_device_ = 0;
  uint64_t local_logits_device_ = 0;
  uint64_t logits_device_ = 0;
  uint64_t prefill_tokens_device_ = 0;
  uint64_t prefill_metadata_device_ = 0;
  uint64_t large_prefill_metadata_device_ = 0;
  uint64_t xlarge_prefill_metadata_device_ = 0;
  uint64_t prefill_embedding_device_ = 0;
  uint64_t prefill_status_device_ = 0;
  uint64_t prefill_reduced_device_ = 0;
  std::unique_ptr<ModelEmbedding> prefill_embedding_;
  std::unique_ptr<ModelEmbedding> large_prefill_embedding_;
  std::unique_ptr<ModelEmbedding> xlarge_prefill_embedding_;
  std::unique_ptr<ModelHead> prefill_head_;
  std::unique_ptr<runtime::TpReduce> prefill_reduce_;
  std::unique_ptr<runtime::TpReduce> prefill_allreduce_cast_;
  std::unique_ptr<ModelEmbedding> embedding_;
  std::unique_ptr<ModelHead> head_;
  std::unique_ptr<ModelArgmax> argmax_;
};

std::mutex registry_mutex;
std::unordered_map<uint32_t, std::shared_ptr<PrivateProgram>> registry;

std::shared_ptr<PrivateProgram> get(uint32_t rank) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  const auto it = registry.find(rank);
  if (it == registry.end()) {
    throw std::runtime_error("PrivateUse1: rank is not initialized");
  }
  return it->second;
}

std::shared_ptr<PrivateProgram> get_sole_instance() {
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (registry.size() != 1) {
    throw std::runtime_error("PrivateUse1: expected exactly one initialized rank in this process");
  }
  return registry.begin()->second;
}

class Hooks final : public at::PrivateUse1HooksInterface {
 public:
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return true; }

  at::Device getDeviceFromPtr(void* data) const override {
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (const auto& [rank, instance] : registry) {
      if (instance->owns_address(data)) {
        return at::Device(c10::kPrivateUse1, rank);
      }
    }
    throw std::invalid_argument("PrivateUse1: pointer has no owning rank");
  }

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    std::lock_guard<std::mutex> lock(registry_mutex);
    return device_index >= 0 && registry.count(device_index) != 0;
  }
};

class Allocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t bytes) override {
    auto instance = get_sole_instance();
    if (bytes == 0) {
      return {nullptr, c10::Device(c10::kPrivateUse1, instance->rank())};
    }
    void* pointer = reinterpret_cast<void*>(instance->allocate(bytes));
    return c10::InefficientStdFunctionContext::makeDataPtr(
        pointer, [instance](void*) {}, c10::Device(c10::kPrivateUse1, instance->rank()));
  }

  void copy_data(void*, const void*, std::size_t) const override {
    TORCH_CHECK(false, "PrivateUse1: device-to-device allocator copies are not supported");
  }
};

class Guard final : public c10::impl::DeviceGuardImplInterface {
 public:
  c10::DeviceType type() const override { return c10::kPrivateUse1; }

  c10::Device exchangeDevice(c10::Device device) const override {
    auto instance = checked(device);
    return c10::Device(c10::kPrivateUse1, instance->rank());
  }

  c10::Device getDevice() const override {
    return c10::Device(c10::kPrivateUse1, get_sole_instance()->rank());
  }

  void setDevice(c10::Device device) const override { checked(device); }
  void uncheckedSetDevice(c10::Device) const noexcept override {}

  c10::Stream getStream(c10::Device device) const override {
    auto instance = checked(device);
    return c10::Stream(c10::Stream::DEFAULT,
                       c10::Device(c10::kPrivateUse1, instance->rank()));
  }

  c10::Stream exchangeStream(c10::Stream stream) const override {
    auto instance = checked(stream.device());
    return c10::Stream(c10::Stream::DEFAULT,
                       c10::Device(c10::kPrivateUse1, instance->rank()));
  }

  c10::DeviceIndex deviceCount() const noexcept override { return 8; }

  bool queryStream(const c10::Stream& stream) const override {
    checked(stream.device())->synchronize_compute();
    return true;
  }

  void synchronizeStream(const c10::Stream& stream) const override {
    checked(stream.device())->synchronize_compute();
  }

  void synchronizeDevice(c10::DeviceIndex device_index) const override {
    checked(c10::Device(c10::kPrivateUse1, device_index))->synchronize_all();
  }

 private:
  static std::shared_ptr<PrivateProgram> checked(c10::Device device) {
    auto instance = get_sole_instance();
    if (device.type() != c10::kPrivateUse1 ||
        (device.index() >= 0 &&
         static_cast<uint32_t>(device.index()) != instance->rank())) {
      throw std::invalid_argument("PrivateUse1: device does not belong to this process");
    }
    return instance;
  }
};

void initialize_backend() {
  static std::once_flag flag;
  static Hooks hooks;
  static Allocator allocator;
  static Guard guard;
  std::call_once(flag, [] {
    at::RegisterPrivateUse1HooksInterface(&hooks);
    c10::SetAllocator(c10::kPrivateUse1, &allocator, 1);
    c10::impl::DeviceGuardImplRegistrar registrar(c10::kPrivateUse1, &guard);
  });
}

at::Tensor& copy_tensor(at::Tensor& destination, const at::Tensor& source,
                        bool non_blocking) {
  TORCH_CHECK(!non_blocking, "PrivateUse1: non-blocking copies are not supported");
  TORCH_CHECK(destination.is_contiguous() && source.is_contiguous(),
              "PrivateUse1: copies require contiguous tensors");
  TORCH_CHECK(destination.scalar_type() == source.scalar_type() &&
                  destination.sizes() == source.sizes(),
              "PrivateUse1: copies require identical dtype and shape");
  if (destination.nbytes() == 0) return destination;
  auto instance = get_sole_instance();
  if (destination.device().type() == c10::kPrivateUse1 &&
      source.device().type() == c10::kCPU) {
    instance->copy_h2d(instance->address_of(destination), source.data_ptr(),
                       source.nbytes());
    return destination;
  }
  if (destination.device().type() == c10::kCPU &&
      source.device().type() == c10::kPrivateUse1) {
    instance->copy_d2h(destination.data_ptr(), instance->address_of(source),
                       source.nbytes());
    return destination;
  }
  TORCH_CHECK(false, "PrivateUse1: only CPU-to-device and device-to-CPU copies are supported");
}

void check_factory_options(const std::optional<at::Layout>& layout,
                           const std::optional<at::Device>& device,
                           const std::optional<bool>& pin_memory) {
  TORCH_CHECK(!layout || *layout == at::kStrided,
              "PrivateUse1: only strided tensors are supported");
  TORCH_CHECK(!pin_memory.value_or(false),
              "PrivateUse1: pinned device allocation is not supported");
  if (device) {
    auto instance = get_sole_instance();
    TORCH_CHECK(device->type() == c10::kPrivateUse1 &&
                    (device->index() < 0 ||
                     static_cast<uint32_t>(device->index()) == instance->rank()),
                "PrivateUse1: invalid allocation device");
  }
}

at::Tensor empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                 std::optional<at::Layout> layout, std::optional<at::Device> device,
                 std::optional<bool> pin_memory,
                 std::optional<at::MemoryFormat> memory_format) {
  check_factory_options(layout, device, pin_memory);
  return at::detail::empty_generic(size, c10::GetAllocator(c10::kPrivateUse1),
                                   c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                   dtype.value_or(at::kFloat), memory_format);
}

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         std::optional<at::ScalarType> dtype,
                         std::optional<at::Layout> layout,
                         std::optional<at::Device> device,
                         std::optional<bool> pin_memory) {
  check_factory_options(layout, device, pin_memory);
  return at::detail::empty_strided_generic(
      size, stride, c10::GetAllocator(c10::kPrivateUse1),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), dtype.value_or(at::kFloat));
}

void initialize(uint32_t rank, uint32_t world, const std::string& unique_id,
                const std::string& recipe_cache, uint64_t hbm_bytes) {
  if (hbm_bytes == 0) {
    throw std::invalid_argument("PrivateUse1: HBM budget must be greater than zero");
  }
  initialize_backend();
  auto instance = std::make_shared<PrivateProgram>(rank, world, unique_id,
                                                   recipe_cache, hbm_bytes);
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (!registry.empty()) {
    throw std::runtime_error("PrivateUse1: only one rank may be initialized per process");
  }
  registry.emplace(rank, std::move(instance));
}

void upload_weight(uint32_t rank, uint32_t layer_id, bool dense,
                   const std::string& name, const at::Tensor& tensor) {
  get(rank)->upload_weight(layer_id, dense, name, tensor);
}

void enable_model_fusion(uint32_t rank) { get(rank)->enable_model_fusion(); }
void enable_prefill(uint32_t rank) { get(rank)->enable_prefill(); }

void register_streamed_layer(uint32_t rank, uint32_t layer_id, int64_t kind,
                             uint32_t cache_slots, uint32_t page_count) {
  get(rank)->register_streamed_layer(layer_id, kind, cache_slots, page_count);
}

void upload_model_weight(uint32_t rank, const std::string& name, const at::Tensor& tensor) {
  get(rank)->upload_model_weight(name, tensor);
}

void register_model_kv(uint32_t rank, uint32_t layer_id,
                       const at::Tensor& key, const at::Tensor& value) {
  get(rank)->register_model_kv(layer_id, key, value);
}
void register_empty_model_kv(uint32_t rank, uint32_t layer_id) {
  get(rank)->register_empty_model_kv(layer_id);
}
int32_t prefill(uint32_t rank, const at::Tensor& tokens, const at::Tensor& full_rope,
                const at::Tensor& swa_rope) {
  return get(rank)->prefill(tokens, full_rope, swa_rope);
}
int32_t prefill_chunk(uint32_t rank, const at::Tensor& tokens, const at::Tensor& full_rope,
                      const at::Tensor& swa_rope, uint32_t prompt_length) {
  return get(rank)->prefill_chunk(tokens, full_rope, swa_rope, prompt_length);
}
void update_model_pages(uint32_t rank, const at::Tensor& pages) {
  get(rank)->update_model_pages(pages);
}
void reset_prefill(uint32_t rank) { get(rank)->reset_prefill(); }

at::Tensor finalize_model(uint32_t rank, const at::Tensor& full_rope,
                          const at::Tensor& swa_rope, const at::Tensor& pages,
                          int32_t initial_token) {
  return get(rank)->finalize_model(full_rope, swa_rope, pages, initial_token);
}

int32_t finish_model_step(uint32_t rank) { return get(rank)->finish_model_step(); }

at::Tensor model_logits(uint32_t rank) { return get(rank)->model_logits(); }

std::vector<at::Tensor> model_kv(uint32_t rank, uint32_t layer) {
  return get(rank)->model_kv(layer);
}

std::vector<at::Tensor> model_kv_slots(uint32_t rank, uint32_t layer,
                                       uint32_t first, uint32_t count) {
  return get(rank)->model_kv_slots(layer, first, count);
}

void register_layer(uint32_t rank, uint32_t layer_id, int64_t kind,
                    const std::string& attention_weights,
                    const std::string& dense_weights,
                    uint32_t cache_slots, uint32_t page_count) {
  try {
    get(rank)->register_layer(layer_id, kind, attention_weights, dense_weights,
                              cache_slots, page_count);
  } catch (const std::exception& error) {
    throw std::runtime_error("PrivateUse1: register layer " +
                             std::to_string(layer_id) + " with weights " +
                             attention_weights + ": " + error.what());
  }
}

std::vector<uint64_t> recipe_identity(uint32_t rank, int64_t kind) {
  return get(rank)->recipe_identity(kind);
}

std::vector<uint64_t> recipe_cache_stats(uint32_t rank) {
  return get(rank)->recipe_cache_stats();
}

void shutdown(uint32_t rank) {
  auto instance = get(rank);
  instance->synchronize_shutdown();
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (registry.erase(rank) != 1) {
    throw std::runtime_error("PrivateUse1: rank is not initialized");
  }
}

std::vector<at::Tensor> forward(int64_t layer_id, const std::vector<at::Tensor>& inputs) {
  if (layer_id < 0 || layer_id >= 70) {
    throw std::invalid_argument("PrivateUse1: invalid layer_id");
  }
  auto instance = get_sole_instance();
  return instance->forward(static_cast<uint32_t>(layer_id), inputs);
}

std::vector<at::Tensor> forward_meta(int64_t layer_id,
                                     const std::vector<at::Tensor>& inputs) {
  if (layer_id < 0 || layer_id >= 70 || inputs.size() != 11) {
    throw std::invalid_argument("PrivateUse1: invalid meta inputs");
  }
  const auto options = at::TensorOptions().device(at::kMeta).dtype(at::kBFloat16);
  return {at::empty({1, 6144}, options), at::empty({1, 6144}, options)};
}

at::Tensor model_step(const at::Tensor& trigger) {
  return get_sole_instance()->forward_model(trigger);
}

at::Tensor model_step_meta(const at::Tensor& trigger) {
  if (trigger.scalar_type() != at::kInt || trigger.numel() != 1)
    throw std::invalid_argument("PrivateUse1: invalid meta model trigger");
  return at::empty({1}, at::TensorOptions().device(at::kMeta).dtype(at::kInt));
}

}  // namespace
}  // namespace gaudirt::mimo

TORCH_LIBRARY(gaudirt, m) {
  m.def("layer(int layer_id, Tensor[] inputs) -> Tensor[]");
  m.def("model_step(Tensor trigger) -> Tensor");
}

TORCH_LIBRARY_IMPL(gaudirt, PrivateUse1, m) {
  m.impl("layer", &gaudirt::mimo::forward);
  m.impl("model_step", &gaudirt::mimo::model_step);
}

TORCH_LIBRARY_IMPL(gaudirt, Meta, m) {
  m.impl("layer", &gaudirt::mimo::forward_meta);
  m.impl("model_step", &gaudirt::mimo::model_step_meta);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("copy_", TORCH_FN(gaudirt::mimo::copy_tensor));
  m.impl("empty.memory_format", TORCH_FN(gaudirt::mimo::empty));
  m.impl("empty_strided", TORCH_FN(gaudirt::mimo::empty_strided));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  gaudirt::mimo::initialize_backend();
  m.def("initialize", &gaudirt::mimo::initialize);
  m.def("enable_model_fusion", &gaudirt::mimo::enable_model_fusion);
  m.def("enable_prefill", &gaudirt::mimo::enable_prefill);
  m.def("register_layer", &gaudirt::mimo::register_layer);
  m.def("upload_weight", &gaudirt::mimo::upload_weight);
  m.def("register_streamed_layer", &gaudirt::mimo::register_streamed_layer);
  m.def("upload_model_weight", &gaudirt::mimo::upload_model_weight);
  m.def("register_model_kv", &gaudirt::mimo::register_model_kv);
  m.def("register_empty_model_kv", &gaudirt::mimo::register_empty_model_kv);
  m.def("prefill", &gaudirt::mimo::prefill);
  m.def("prefill_chunk", &gaudirt::mimo::prefill_chunk);
  m.def("update_model_pages", &gaudirt::mimo::update_model_pages);
  m.def("reset_prefill", &gaudirt::mimo::reset_prefill);
  m.def("finalize_model", &gaudirt::mimo::finalize_model);
  m.def("finish_model_step", &gaudirt::mimo::finish_model_step);
  m.def("model_logits", &gaudirt::mimo::model_logits);
  m.def("model_kv", &gaudirt::mimo::model_kv);
  m.def("model_kv_slots", &gaudirt::mimo::model_kv_slots);
  m.def("recipe_identity", &gaudirt::mimo::recipe_identity);
  m.def("recipe_cache_stats", &gaudirt::mimo::recipe_cache_stats);
  m.def("shutdown", &gaudirt::mimo::shutdown);
}
