// SPDX-License-Identifier: Apache-2.0
#include "layer0.hpp"
#include "layer_b.hpp"
#include "layer_c.hpp"
#include "model_argmax.hpp"
#include "model_embedding.hpp"
#include "model_head.hpp"
#include "step_metadata.hpp"
#include "batch_step_metadata.hpp"
#include "prefill_metadata.hpp"
#include "prefill_bindings.hpp"
#include "prefill_chunks.hpp"

#include "gaudirt/bridge/rank_lifecycle.hpp"
#include "gaudirt/bridge/private_use1.hpp"
#include "gaudirt/common/sha256.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"

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

struct BucketExecutionPlan {
  std::unordered_map<Kind, std::unique_ptr<Instance>> instances;
  std::unique_ptr<ModelEmbedding> embedding;
  uint64_t metadata_device = 0;
  uint64_t metadata_stride = 0;
  std::vector<PreparedLayer> layers;
  std::vector<PreparedLayer> short_layers;
  std::vector<std::pair<uint64_t*, uint64_t>> metadata_bindings;
  std::vector<std::pair<uint64_t*, uint64_t>> short_metadata_bindings;
  uint64_t* token_binding = nullptr;
  runtime::PreparedLaunch embedding_launch;
  runtime::PreparedLaunch reduce_launch;
  runtime::PreparedLaunch head_launch;
};

struct BatchDecodePlan {
  BucketExecutionPlan execution;
  std::unique_ptr<ModelHead> head;
  std::unique_ptr<ModelArgmax> argmax;
  runtime::PreparedLaunch head_launch;
  runtime::PreparedLaunch argmax_launch;
  void* input_host = nullptr;
  void* output_host = nullptr;
  uint64_t input_device = 0;
  uint64_t output_device = 0;
  uint64_t embedding_device = 0;
  uint64_t status_device = 0;
  uint64_t normalized_device = 0;
  uint64_t local_logits = 0;
  uint64_t logits = 0;
  uint32_t rows = 0;
};

struct PrefillRequestState {
  uint32_t rows = 0;
  uint32_t prompt_length = 0;
  uint32_t position = 0;
  bool complete = false;
  std::vector<int32_t> pages;
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
    for (auto& plan : batch_decode_) {
      if (plan.input_host) synHostFree(runtime_.device_id(), plan.input_host, 0);
      if (plan.output_host) synHostFree(runtime_.device_id(), plan.output_host, 0);
    }
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

  void enable_batch_decode() {
    if (!prefill_enabled_ || batch_decode_enabled_ || !entries_.empty() || !pending_.empty())
      throw std::invalid_argument("PrivateUse1: enable batch decode before registering weights");
    batch_decode_enabled_ = true;
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
      if (batch_decode_enabled_) {
        for (size_t i = 0; i < batch_decode_.size(); ++i) {
          const uint32_t rows = i == 0 ? 4 : 8;
          batch_decode_[i].rows = rows;
          batch_decode_[i].execution.instances.emplace(kind, create_instance(
              kind, graph_cache_slots, page_count, *entry_it->second.attention_weights,
              entry_it->second.dense_weights.get(), rows));
        }
      }
      if (prefill_enabled_)
        prefill_buckets_[0].instances.emplace(kind, create_instance(kind, graph_cache_slots, kind == Kind::C ? page_count : 32,
            *entry_it->second.attention_weights, entry_it->second.dense_weights.get(), 32));
      if (prefill_enabled_)
        prefill_buckets_[1].instances.emplace(kind, create_instance(kind, graph_cache_slots,
            kind == Kind::C ? page_count : 32, *entry_it->second.attention_weights,
            entry_it->second.dense_weights.get(), 512));
      if (prefill_enabled_)
        prefill_buckets_[2].instances.emplace(kind, create_instance(kind, graph_cache_slots,
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
      prefill_buckets_[0].metadata_device = arena_.allocate(128 * sizeof(PrefillMetadata));
      prefill_buckets_[0].metadata_stride = sizeof(PrefillMetadata);
      prefill_buckets_[1].metadata_device = arena_.allocate(8 * sizeof(LargePrefillMetadata));
      prefill_buckets_[1].metadata_stride = sizeof(LargePrefillMetadata);
      prefill_buckets_[2].metadata_device = arena_.allocate(4 * sizeof(XLargePrefillMetadata));
      prefill_buckets_[2].metadata_stride = sizeof(XLargePrefillMetadata);
      prefill_embedding_device_ = arena_.allocate(2048 * 6144 * sizeof(uint16_t));
      prefill_status_device_ = arena_.allocate(2048 * sizeof(int32_t));
      prefill_reduced_device_ = arena_.allocate(6144 * sizeof(uint16_t));
      prefill_buckets_[0].embedding = std::make_unique<ModelEmbedding>(arena_, model_weights_.at("embedding"), &recipe_cache_, 32);
      prefill_buckets_[1].embedding = std::make_unique<ModelEmbedding>(arena_, model_weights_.at("embedding"), &recipe_cache_, 512);
      prefill_buckets_[2].embedding = std::make_unique<ModelEmbedding>(arena_, model_weights_.at("embedding"), &recipe_cache_, 2048);
      prefill_head_ = std::make_unique<ModelHead>(arena_, model_weights_.at("final_norm"), model_weights_.at("lm_head"), &recipe_cache_, false, ModelHead::kLocalVocabSize);
      prefill_reduce_ = std::make_unique<runtime::TpReduce>(arena_, 6144, syn_type_float, &recipe_cache_);
      prefill_allreduce_cast_ = std::make_unique<runtime::TpReduce>(arena_, 6144, syn_type_float, &recipe_cache_, 1);
    }
    if (batch_decode_enabled_) {
      batch_residual_slots_[0] = arena_.allocate(8 * 6144 * sizeof(uint16_t));
      batch_residual_slots_[1] = arena_.allocate(8 * 6144 * sizeof(uint16_t));
      for (auto& plan : batch_decode_) {
        const auto rows = plan.rows;
        const size_t bytes = rows == 4 ? sizeof(BatchDecodeInput<4>) : sizeof(BatchDecodeInput<8>);
        check_syn(synHostMalloc(runtime_.device_id(), bytes, 0, &plan.input_host), "batch metadata pinned allocation");
        check_syn(synHostMalloc(runtime_.device_id(), rows * sizeof(int32_t), 0, &plan.output_host), "batch output pinned allocation");
        plan.input_device = arena_.allocate(bytes);
        plan.output_device = arena_.allocate(rows * sizeof(int32_t));
        plan.embedding_device = arena_.allocate(rows * 6144 * sizeof(uint16_t));
        plan.status_device = arena_.allocate(rows * sizeof(int32_t));
        plan.normalized_device = arena_.allocate(rows * 6144 * sizeof(uint16_t));
        plan.local_logits = arena_.allocate(rows * ModelHead::kLocalVocabSize * sizeof(uint16_t));
        plan.logits = arena_.allocate(rows * ModelHead::kVocabSize * sizeof(uint16_t));
        plan.execution.embedding = std::make_unique<ModelEmbedding>(arena_, model_weights_.at("embedding"), &recipe_cache_, rows);
        plan.head = std::make_unique<ModelHead>(arena_, model_weights_.at("final_norm"), model_weights_.at("lm_head"),
                                                &recipe_cache_, true, ModelHead::kLocalVocabSize, rows);
        plan.argmax = std::make_unique<ModelArgmax>(arena_, &recipe_cache_, rows);
      }
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
      auto bind = [&]<typename Metadata>(BucketExecutionPlan& plan) {
        auto& embedding = *plan.embedding;
        auto& reduce = *prefill_reduce_;
        const uint64_t base = plan.metadata_device;
        auto& layers = plan.layers;
        auto& short_layers = plan.short_layers;
        auto& metadata_bindings = plan.metadata_bindings;
        auto& short_bindings = plan.short_metadata_bindings;
        auto& embedding_launch = plan.embedding_launch;
        auto& token_binding = plan.token_binding;
        auto& reduce_launch = plan.reduce_launch;
        auto& head_launch = plan.head_launch;
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
      bind.template operator()<PrefillMetadata>(prefill_buckets_[0]);
      bind.template operator()<LargePrefillMetadata>(prefill_buckets_[1]);
      bind.template operator()<XLargePrefillMetadata>(prefill_buckets_[2]);
      prefill_argmax_launch_ = argmax_->prepare(logits_device_, prefill_status_device_, token_device_);
    }
    if (batch_decode_enabled_) {
      auto bind = [&]<uint32_t Rows>(BatchDecodePlan& plan) {
        using Input = BatchDecodeInput<Rows>;
        using Metadata = BatchStepMetadata<Rows>;
        if (reinterpret_cast<uintptr_t>(plan.input_host) % alignof(Input))
          throw std::runtime_error("PrivateUse1: batch metadata is not aligned");
        new (plan.input_host) Input{};
        const uint64_t base = plan.input_device + offsetof(Input, metadata);
        plan.execution.embedding_launch = plan.execution.embedding->prepare(
            plan.input_device + offsetof(Input, tokens), plan.embedding_device, plan.status_device);
        plan.execution.layers.reserve(70);
        uint64_t hidden = plan.embedding_device, residual = plan.embedding_device;
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
          auto prepared = prepare_layer(layer, addresses, Rows, false,
                                        base + offsetof(Metadata, full_row_groups));
          hidden = prepared.feed_forward_gathered;
          residual = batch_residual_slots_[layer % 2];
          plan.execution.layers.push_back(std::move(prepared));
        }
        plan.head_launch = plan.head->prepare(hidden, residual, plan.normalized_device, plan.local_logits);
        plan.argmax_launch = plan.argmax->prepare(plan.logits, plan.status_device, plan.output_device);
      };
      bind.template operator()<4>(batch_decode_[0]);
      bind.template operator()<8>(batch_decode_[1]);
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

  int32_t prefill_request(const std::string& request_id, const at::Tensor& tokens,
                          const at::Tensor& full_rope, const at::Tensor& swa_rope,
                          uint32_t prompt_length, const std::vector<int32_t>& pages) {
    if (!batch_decode_enabled_ || !model_finalized_ || model_pending_ || request_id.empty() ||
        pages.empty() || pages.size() > 64)
      throw std::invalid_argument("PrivateUse1: invalid request prefill input");
    auto previous = request_prefills_.find(request_id);
    if (previous == request_prefills_.end() && request_prefills_.size() >= 8)
      throw std::invalid_argument("PrivateUse1: too many live requests");
    if (previous != request_prefills_.end()) {
      const auto& state = previous->second;
      if (state.complete || state.prompt_length != prompt_length || pages.size() < state.pages.size() ||
          !std::equal(state.pages.begin(), state.pages.end(), pages.begin()))
        throw std::invalid_argument("PrivateUse1: invalid request prefill continuation");
    }
    std::vector<bool> used(physical_cache_slots_ / 128, false);
    for (const auto page : pages) {
      if (page < 0 || static_cast<size_t>(page) >= used.size() || used[page])
        throw std::invalid_argument("PrivateUse1: invalid request KV pages");
      used[page] = true;
    }
    for (const auto& [other_id, state] : request_prefills_) {
      if (other_id == request_id) continue;
      for (const auto page : state.pages)
        if (used[page]) throw std::invalid_argument("PrivateUse1: request KV pages overlap");
    }
    const uint32_t prior_rows = previous == request_prefills_.end() ? 0 : previous->second.rows;
    if (tokens.numel() <= 0 || tokens.numel() > UINT32_MAX - prior_rows ||
        pages.size() < (static_cast<uint64_t>(prior_rows) + tokens.numel() + 127) / 128)
      throw std::invalid_argument("PrivateUse1: request lacks allocated prefill pages");
    std::copy(pages.begin(), pages.end(), model_pages_.begin());
    size_t count = pages.size();
    for (size_t page = 0; count < model_pages_.size() && page < used.size(); ++page)
      if (!used[page]) model_pages_[count++] = static_cast<int32_t>(page);
    if (count != model_pages_.size()) throw std::invalid_argument("PrivateUse1: insufficient page geometry");
    model_position_ = previous == request_prefills_.end() ? 0 : previous->second.position;
    prefill_rows_ = prior_rows;
    prompt_length_ = previous == request_prefills_.end() ? 0 : previous->second.prompt_length;
    prefill_complete_ = false;
    const int32_t next = prefill_chunk(tokens, full_rope, swa_rope, prompt_length);
    request_prefills_[request_id] = {prefill_rows_, prompt_length_, model_position_, prefill_complete_, pages};
    return next;
  }

  void release_request(const std::string& request_id) {
    if (!batch_decode_enabled_ || model_pending_ || request_prefills_.erase(request_id) != 1)
      throw std::invalid_argument("PrivateUse1: invalid request release");
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
      arena_.copy_h2d_sync(runtime_.h2d_stream(), metadata.data(), prefill_buckets_[0].metadata_device,
                           metadata.size() * sizeof(PrefillMetadata));
    if (!large_metadata.empty())
      arena_.copy_h2d_sync(runtime_.h2d_stream(), large_metadata.data(), prefill_buckets_[1].metadata_device,
                           large_metadata.size() * sizeof(LargePrefillMetadata));
    if (!xlarge_metadata.empty())
      arena_.copy_h2d_sync(runtime_.h2d_stream(), xlarge_metadata.data(), prefill_buckets_[2].metadata_device,
                           xlarge_metadata.size() * sizeof(XLargePrefillMetadata));
    const auto stream = runtime_.compute_stream();
    for (const auto& chunk : chunks) {
      auto& plan = bucket_for_rows(chunk.rows);
      const bool short_prefix = prefill_rows_ + chunk.offset + chunk.valid_rows <= 2048;
      auto& bindings = short_prefix ? plan.short_metadata_bindings : plan.metadata_bindings;
      auto& layers = short_prefix ? plan.short_layers : plan.layers;
      *plan.token_binding = prefill_tokens_device_ + chunk.offset * sizeof(int32_t);
      plan.embedding_launch.execute(stream);
      const uint64_t base = plan.metadata_device + chunk.index * plan.metadata_stride;
      for (auto [address, offset] : bindings) *address = base + offset;
      for (const auto& layer : layers) layer.execute_fused(stream);
    }
    if (prefill_rows_ + tokens.numel() != prompt_length_) {
      runtime_.synchronize_compute();
      prefill_rows_ += static_cast<uint32_t>(tokens.numel());
      return -1;
    }
    auto& last_plan = bucket_for_rows(chunks.back().rows);
    auto& last_layers = last_plan.layers;
    auto& last_reduce = last_plan.reduce_launch;
    auto& last_head = last_plan.head_launch;
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

  std::vector<int32_t> decode_batch(const std::vector<int32_t>& tokens,
                                    const std::vector<uint32_t>& positions,
                                    const std::vector<std::vector<int32_t>>& page_lists) {
    const size_t active = tokens.size();
    if (!batch_decode_enabled_ || !model_finalized_ || model_pending_ ||
        !active || active > 8 || positions.size() != active || page_lists.size() != active)
      throw std::invalid_argument("PrivateUse1: invalid batch decode inputs");
    std::array<BatchRequestStep, 8> requests{};
    for (size_t row = 0; row < active; ++row) {
      if (tokens[row] < 0 || tokens[row] >= static_cast<int32_t>(ModelHead::kVocabSize) ||
          page_lists[row].size() > requests[row].pages.size())
        throw std::invalid_argument("PrivateUse1: invalid batch token or page table");
      requests[row].position = positions[row];
      requests[row].page_count = static_cast<uint32_t>(page_lists[row].size());
      std::copy(page_lists[row].begin(), page_lists[row].end(), requests[row].pages.begin());
    }
    auto& plan = batch_decode_[active <= 4 ? 0 : 1];
    auto stage = [&]<uint32_t Rows>() {
      auto& input = *static_cast<BatchDecodeInput<Rows>*>(plan.input_host);
      validate_batch_step<Rows>(requests.data(), active, rope_rows_, full_rope_.data(), swa_rope_.data(),
                                physical_cache_slots_ / 128, physical_cache_slots_);
      fill_batch_step_metadata_prevalidated(input.metadata, requests.data(), active,
                                             full_rope_.data(), swa_rope_.data(), physical_cache_slots_);
      input.tokens.fill(0);
      std::copy(tokens.begin(), tokens.end(), input.tokens.begin());
      check_syn(synMemCopyAsync(runtime_.h2d_stream(), reinterpret_cast<uint64_t>(plan.input_host),
                                sizeof(input), plan.input_device, HOST_TO_DRAM), "batch metadata H2D");
    };
    if (plan.rows == 4) stage.template operator()<4>(); else stage.template operator()<8>();
    check_syn(synEventRecord(metadata_ready_, runtime_.h2d_stream()), "batch metadata event");
    check_syn(synStreamWaitEvent(runtime_.compute_stream(), metadata_ready_, 0), "batch metadata wait");
    const auto stream = runtime_.compute_stream();
    plan.execution.embedding_launch.execute(stream);
    for (const auto& layer : plan.execution.layers) layer.execute_fused(stream);
    plan.head_launch.execute(stream);
    if (hcclAllGather(reinterpret_cast<const void*>(plan.local_logits), reinterpret_cast<void*>(plan.logits),
                      static_cast<uint64_t>(ModelHead::kLocalVocabSize) * plan.rows,
                      hcclBfloat16, runtime_.comm(), stream) != hcclSuccess)
      throw std::runtime_error("PrivateUse1: batch logits AllGather failed");
    plan.argmax_launch.execute(stream);
    check_syn(synEventRecord(compute_done_, stream), "batch compute event");
    check_syn(synStreamWaitEvent(runtime_.d2h_stream(), compute_done_, 0), "batch output wait");
    check_syn(synMemCopyAsync(runtime_.d2h_stream(), plan.output_device, plan.rows * sizeof(int32_t),
                              reinterpret_cast<uint64_t>(plan.output_host), DRAM_TO_HOST), "batch token D2H");
    check_syn(synStreamSynchronize(runtime_.d2h_stream()), "batch token completion");
    const auto* output = static_cast<const int32_t*>(plan.output_host);
    std::vector<int32_t> result(output, output + active);
    if (std::any_of(result.begin(), result.end(), [](int32_t token) {
          return token < 0 || token >= static_cast<int32_t>(ModelHead::kVocabSize);
        }))
      throw std::runtime_error("PrivateUse1: invalid batch token or nonfinite logits");
    return result;
  }

  std::vector<int32_t> decode_requests(const std::vector<std::string>& request_ids,
                                       const std::vector<int32_t>& tokens,
                                       const std::vector<uint32_t>& positions,
                                       const std::vector<std::vector<int32_t>>& pages) {
    if (request_ids.empty() || request_ids.size() > 8 || tokens.size() != request_ids.size() ||
        positions.size() != request_ids.size() || pages.size() != request_ids.size())
      throw std::invalid_argument("PrivateUse1: invalid batch request geometry");
    std::unordered_map<std::string, size_t> scheduled;
    for (size_t row = 0; row < request_ids.size(); ++row) {
      const auto state = request_prefills_.find(request_ids[row]);
      if (state == request_prefills_.end() || !state->second.complete ||
          state->second.position != positions[row] || !scheduled.emplace(request_ids[row], row).second ||
          pages[row].size() < state->second.pages.size() ||
          !std::equal(state->second.pages.begin(), state->second.pages.end(), pages[row].begin()))
        throw std::invalid_argument("PrivateUse1: invalid decode request state");
    }
    std::vector<bool> owned(physical_cache_slots_ / 128, false);
    for (const auto& [id, state] : request_prefills_) {
      const auto row = scheduled.find(id);
      const auto& current = row == scheduled.end() ? state.pages : pages[row->second];
      if (current.size() > 64) throw std::invalid_argument("PrivateUse1: too many request pages");
      for (auto page : current) {
        if (page < 0 || static_cast<size_t>(page) >= owned.size() || owned[page])
          throw std::invalid_argument("PrivateUse1: overlapping request KV pages");
        owned[page] = true;
      }
    }
    auto result = decode_batch(tokens, positions, pages);
    for (size_t row = 0; row < request_ids.size(); ++row) {
      auto& state = request_prefills_.at(request_ids[row]);
      state.pages = pages[row];
      ++state.position;
    }
    return result;
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
                              uint32_t prefill = 0, bool short_prefix = false,
                              uint64_t full_row_groups = 0) {
    auto& entry = entries_.at(layer_id);
    auto& map = prefill ? bucket_for_rows(prefill).instances : instances_;
    auto& instance = *map.at(entry.kind);
    const uint64_t residual_output = (prefill == 4 || prefill == 8) ? batch_residual_slots_[layer_id % 2]
                                     : prefill >= 512 ? entry.large_prefill_residual_output
                                     : prefill == 32 ? entry.prefill_residual_output : entry.residual_output;
    if (entry.kind == Kind::A) {
      instance.a->rebind_weights(*entry.attention_weights, *entry.dense_weights);
      auto io = instance.a_io;
      set_common_attention(io.attention, addresses);
      io.attention.full_row_groups = full_row_groups;
      io.attention.residual = addresses[0];
      io.dense_outputs.residual_out_bf16 = residual_output;
      io.layer_output_bf16 = entry.layer_output;
      return instance.a->prepare(io, short_prefix);
    }
    if (entry.kind == Kind::B) {
      instance.b->rebind_weights(*entry.attention_weights);
      auto io = instance.b_io;
      set_common_attention(io.attention, addresses);
      io.attention.full_row_groups = full_row_groups;
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

  runtime::DeviceRuntime& device_runtime() noexcept { return runtime_; }
  runtime::HbmArena& hbm_arena() noexcept { return arena_; }

  void synchronize_shutdown() {
    if (model_pending_)
      throw std::runtime_error("PrivateUse1: finish the pending model step before shutdown");
    bridge::synchronize_shutdown(runtime_, shutdown_buffer_);
  }

 private:
  BucketExecutionPlan& bucket_for_rows(uint32_t rows) {
    switch (rows) {
      case 4: return batch_decode_[0].execution;
      case 8: return batch_decode_[1].execution;
      case 32: return prefill_buckets_[0];
      case 512: return prefill_buckets_[1];
      case 2048: return prefill_buckets_[2];
      default: throw std::invalid_argument("PrivateUse1: unsupported prefill bucket");
    }
  }

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
  std::array<BucketExecutionPlan, 3> prefill_buckets_;
  std::array<BatchDecodePlan, 2> batch_decode_;
  std::unordered_map<uint32_t, LayerEntry> entries_;
  std::unordered_map<uint32_t, PendingWeights> pending_;
  std::unordered_map<std::string, uint64_t> model_weights_;
  std::unordered_map<std::string, PrefillRequestState> request_prefills_;
  std::vector<PreparedLayer> model_layers_;
  runtime::PreparedLaunch prefill_argmax_launch_;
  runtime::PreparedLaunch model_embedding_launch_;
  runtime::PreparedLaunch model_head_launch_;
  runtime::PreparedLaunch model_argmax_launch_;
  std::array<uint64_t, 2> prefill_residual_slots_{};
  std::array<uint64_t, 2> batch_residual_slots_{};
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
  bool batch_decode_enabled_ = false;
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
  uint64_t prefill_embedding_device_ = 0;
  uint64_t prefill_status_device_ = 0;
  uint64_t prefill_reduced_device_ = 0;
  std::unique_ptr<ModelHead> prefill_head_;
  std::unique_ptr<runtime::TpReduce> prefill_reduce_;
  std::unique_ptr<runtime::TpReduce> prefill_allreduce_cast_;
  std::unique_ptr<ModelEmbedding> embedding_;
  std::unique_ptr<ModelHead> head_;
  std::unique_ptr<ModelArgmax> argmax_;
};

std::shared_ptr<PrivateProgram> get(uint32_t rank) {
  return std::static_pointer_cast<PrivateProgram>(bridge::get_rank(rank)->owner);
}

std::shared_ptr<PrivateProgram> get_sole_model() {
  return std::static_pointer_cast<PrivateProgram>(bridge::get_sole_instance()->owner);
}

void initialize(uint32_t rank, uint32_t world, const std::string& unique_id,
                const std::string& recipe_cache, uint64_t hbm_bytes) {
  if (hbm_bytes == 0) {
    throw std::invalid_argument("PrivateUse1: HBM budget must be greater than zero");
  }
  bridge::initialize_backend();
  auto instance = std::make_shared<PrivateProgram>(rank, world, unique_id,
                                                   recipe_cache, hbm_bytes);
  auto context = std::make_shared<bridge::RankContext>(
      bridge::RankContext{rank, &instance->device_runtime(), &instance->hbm_arena(), instance});
  bridge::register_rank(std::move(context));
}

void upload_weight(uint32_t rank, uint32_t layer_id, bool dense,
                   const std::string& name, const at::Tensor& tensor) {
  get(rank)->upload_weight(layer_id, dense, name, tensor);
}

void enable_model_fusion(uint32_t rank) { get(rank)->enable_model_fusion(); }
void enable_prefill(uint32_t rank) { get(rank)->enable_prefill(); }
void enable_batch_decode(uint32_t rank) { get(rank)->enable_batch_decode(); }

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
int32_t prefill_request(uint32_t rank, const std::string& request_id, const at::Tensor& tokens,
                        const at::Tensor& full_rope, const at::Tensor& swa_rope,
                        uint32_t prompt_length, const std::vector<int32_t>& pages) {
  return get(rank)->prefill_request(request_id, tokens, full_rope, swa_rope, prompt_length, pages);
}
void release_request(uint32_t rank, const std::string& request_id) { get(rank)->release_request(request_id); }

at::Tensor finalize_model(uint32_t rank, const at::Tensor& full_rope,
                          const at::Tensor& swa_rope, const at::Tensor& pages,
                          int32_t initial_token) {
  return get(rank)->finalize_model(full_rope, swa_rope, pages, initial_token);
}

int32_t finish_model_step(uint32_t rank) { return get(rank)->finish_model_step(); }
std::vector<int32_t> decode_batch(uint32_t rank, const std::vector<int32_t>& tokens,
                                  const std::vector<uint32_t>& positions,
                                  const std::vector<std::vector<int32_t>>& pages) {
  return get(rank)->decode_batch(tokens, positions, pages);
}
std::vector<int32_t> decode_requests(uint32_t rank, const std::vector<std::string>& ids,
                                     const std::vector<int32_t>& tokens,
                                     const std::vector<uint32_t>& positions,
                                     const std::vector<std::vector<int32_t>>& pages) {
  return get(rank)->decode_requests(ids, tokens, positions, pages);
}

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
  bridge::unregister_rank(rank);
}

std::vector<at::Tensor> forward(int64_t layer_id, const std::vector<at::Tensor>& inputs) {
  if (layer_id < 0 || layer_id >= 70) {
    throw std::invalid_argument("PrivateUse1: invalid layer_id");
  }
  auto instance = get_sole_model();
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
  return get_sole_model()->forward_model(trigger);
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
  m.impl("copy_", TORCH_FN(gaudirt::bridge::copy_tensor));
  m.impl("empty.memory_format", TORCH_FN(gaudirt::bridge::empty));
  m.impl("empty_strided", TORCH_FN(gaudirt::bridge::empty_strided));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  gaudirt::bridge::initialize_backend();
  m.def("initialize", &gaudirt::mimo::initialize);
  m.def("enable_model_fusion", &gaudirt::mimo::enable_model_fusion);
  m.def("enable_prefill", &gaudirt::mimo::enable_prefill);
  m.def("enable_batch_decode", &gaudirt::mimo::enable_batch_decode);
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
  m.def("prefill_request", &gaudirt::mimo::prefill_request);
  m.def("release_request", &gaudirt::mimo::release_request);
  m.def("finalize_model", &gaudirt::mimo::finalize_model);
  m.def("finish_model_step", &gaudirt::mimo::finish_model_step);
  m.def("decode_batch", &gaudirt::mimo::decode_batch);
  m.def("decode_requests", &gaudirt::mimo::decode_requests);
  m.def("model_logits", &gaudirt::mimo::model_logits);
  m.def("model_kv", &gaudirt::mimo::model_kv);
  m.def("model_kv_slots", &gaudirt::mimo::model_kv_slots);
  m.def("recipe_identity", &gaudirt::mimo::recipe_identity);
  m.def("recipe_cache_stats", &gaudirt::mimo::recipe_cache_stats);
  m.def("shutdown", &gaudirt::mimo::shutdown);
}
