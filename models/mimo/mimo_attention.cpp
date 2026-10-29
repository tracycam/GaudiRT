// SPDX-License-Identifier: Apache-2.0
#include "mimo_attention.hpp"

#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"

#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <climits>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gaudirt::runtime {
namespace {

constexpr float kRmsNormEpsilon = 1e-5F;
constexpr float kValueScale = 0.612F;
constexpr float kAttentionScale = 0.07216878364870322F;

void check(synStatus status, const std::string& operation) {
  if (status != synSuccess) {
    throw std::runtime_error("MimoAttention: " + operation + " failed, status " +
                             std::to_string(static_cast<int>(status)));
  }
}

enum class Binding {
  kHiddenStates,
  kResidual,
  kResidualOut,
  kNormalized,
  kRopeCos,
  kRopeSin,
  kKeyCache,
  kValueCache,
  kSlotMapping,
  kWindowPages,
  kWindowGroups,
  kPosition,
  kQkv,
  kQuery,
  kNewKey,
  kNewValue,
  kContext,
  kOutputLocal,
  kInputLayernorm,
  kQkvWeight,
  kQkvScales,
  kQkvBias,
  kAttentionSink,
  kOprojWeight,
};

struct ExternalTensor {
  const char* name;
  synTensor tensor = nullptr;
  synSectionHandle section = nullptr;
  Binding binding;
  std::vector<uint64_t> shape;
};

const WeightTensor& expect_weight(const WeightManifest& weights,
                                  const char* name, synDataType dtype,
                                  std::initializer_list<uint64_t> shape) {
  const auto& tensor = weights.get(name);
  if (tensor.dtype != dtype ||
      tensor.shape != std::vector<uint64_t>(shape) ||
      tensor.device_address == 0) {
    throw std::runtime_error(std::string("MimoAttention: invalid weight ") +
                             name);
  }
  return tensor;
}

}  // namespace

struct MimoAttention::Impl {
  HbmArena& arena;
  const WeightManifest* weights;
  uint32_t cache_slots;
  synGraphHandle graph = nullptr;
  synRecipeHandle recipe = nullptr;
  uint64_t workspace_size = 0;
  std::vector<ExternalTensor> externals;
  std::vector<synLaunchTensorInfo> launch;

  synTensor hidden_states = nullptr;
  synTensor residual = nullptr;
  synTensor residual_out = nullptr;
  synTensor normalized = nullptr;
  synTensor qkv = nullptr;
  synTensor query = nullptr;
  synTensor new_key = nullptr;
  synTensor new_value = nullptr;
  synTensor key_cache_after = nullptr;
  synTensor value_cache_after = nullptr;
  synTensor context = nullptr;
  synTensor output_local = nullptr;

  Impl(HbmArena& arena, const WeightManifest& weights, uint32_t cache_slots)
      : arena(arena), weights(&weights), cache_slots(cache_slots) {}

  ~Impl() { close(); }

  void close() noexcept {
    if (recipe) {
      synRecipeDestroy(recipe);
      recipe = nullptr;
    }
    if (graph) {
      synGraphDestroy(graph);
      graph = nullptr;
    }
    for (auto& external : externals) {
      if (external.section) {
        synSectionDestroy(external.section);
        external.section = nullptr;
      }
    }
  }

  synTensor external(const char* name, Binding binding, synDataType dtype,
                     std::initializer_list<uint64_t> sizes,
                     synSectionHandle shared_section = nullptr) {
    synTensorDescriptor descriptor = {};
    descriptor.m_name = name;
    descriptor.m_dataType = dtype;
    descriptor.m_dims = sizes.size();
    unsigned dim = 0;
    for (const uint64_t size : sizes) {
      if (size == 0 || size > UINT_MAX) {
        throw std::runtime_error("MimoAttention: invalid external tensor shape");
      }
      descriptor.m_sizes[dim] = size;
      descriptor.m_minSizes[dim] = size;
      ++dim;
    }
    synSectionHandle section = shared_section;
    const bool owns_section = section == nullptr;
    if (owns_section) {
      check(synSectionCreate(&section, 0, graph),
            std::string("create section for ") + name);
    }
    externals.push_back(
        {name, nullptr, owns_section ? section : nullptr, binding, sizes});
    try {
      if (owns_section) {
        check(synSectionSetPersistent(section, true),
              std::string("persist section for ") + name);
      }
      synTensor tensor = nullptr;
      check(synTensorCreate(&tensor, &descriptor, section, 0),
            std::string("create tensor ") + name);
      if (dtype == syn_type_fp8_143) {
        synFpQuantParam param{1.0, 7};
        synFpQuantMetadata metadata{dtype, &param, 1};
        check(synTensorSetQuantizationData(tensor, SYN_FP_QUANT_METADATA,
                                           &metadata, sizeof(metadata)),
              std::string("set FP8 metadata for ") + name);
      }
      externals.back().tensor = tensor;
      return tensor;
    } catch (...) {
      externals.pop_back();
      if (owns_section) synSectionDestroy(section);
      throw;
    }
  }

  synTensor intermediate(const char* name, synDataType dtype,
                         std::initializer_list<uint64_t> sizes) {
    const std::string full_name = "swa_" + std::string(name);
    synTensorDescriptor descriptor = {};
    descriptor.m_name = full_name.c_str();
    descriptor.m_dataType = dtype;
    descriptor.m_dims = sizes.size();
    unsigned dim = 0;
    for (const uint64_t size : sizes) {
      descriptor.m_sizes[dim] = size;
      descriptor.m_minSizes[dim] = size;
      ++dim;
    }
    synTensor tensor = nullptr;
    check(synTensorCreate(&tensor, &descriptor, nullptr, 0),
          std::string("create intermediate ") + name);
    return tensor;
  }

  void node(std::initializer_list<synTensor> inputs,
            std::initializer_list<synTensor> outputs, const void* params,
            unsigned params_size, const char* guid, const char* name) {
    check(synNodeCreate(graph, inputs.begin(), outputs.begin(), inputs.size(),
                        outputs.size(), params, params_size, guid, name, nullptr,
                        nullptr),
          std::string("create node ") + name);
  }

  void build() {
    if (cache_slots < 2 * MimoAttention::kKvBlockSize ||
        cache_slots % MimoAttention::kKvBlockSize != 0 ||
        cache_slots > static_cast<uint32_t>(INT_MAX) /
                          MimoAttention::kHeadDim) {
      throw std::runtime_error("MimoAttention: cache_slots must be a multiple "
                               "of 128 in the supported SWA range");
    }

    const auto& norm_weight =
        expect_weight(*weights, "input_layernorm", syn_type_bf16, {6144});
    const auto& qkv_weight = expect_weight(
        *weights, "qkv_weight", syn_type_fp8_143, {128, 3392, 48});
    const auto& qkv_scales =
        expect_weight(*weights, "qkv_scales", syn_type_float, {48, 27});
    const auto& qkv_bias =
        expect_weight(*weights, "qkv_bias", syn_type_float, {3392});
    const auto& sinks =
        expect_weight(*weights, "attention_sink", syn_type_bf16, {16});
    const auto& oproj_weight = expect_weight(*weights, "oproj_weight",
                                             syn_type_bf16, {2048, 6144});
    (void)norm_weight;
    (void)qkv_weight;
    (void)qkv_scales;
    (void)qkv_bias;
    (void)sinks;
    (void)oproj_weight;

    check(synGraphCreate(&graph, synDeviceGaudi2), "create graph");

    hidden_states = external("hidden_states", Binding::kHiddenStates,
                             syn_type_bf16, {6144, 1});
    residual = external("residual", Binding::kResidual, syn_type_bf16,
                        {6144, 1});
    residual_out = external("residual_out", Binding::kResidualOut,
                            syn_type_bf16, {6144, 1});
    normalized = external("normalized", Binding::kNormalized, syn_type_bf16,
                          {6144, 1});
    synTensor norm_gamma = external("input_layernorm", Binding::kInputLayernorm,
                                    syn_type_bf16, {6144});
    node({hidden_states, residual, norm_gamma}, {residual_out, normalized},
         &kRmsNormEpsilon, sizeof(kRmsNormEpsilon),
         "gk_residual_rmsnorm_bf16_v1", "input_residual_rmsnorm");

    synTensor qkv_packed = external("qkv_weight", Binding::kQkvWeight,
                                    syn_type_fp8_143, {128, 3392, 48});
    synTensor qkv_scale = external("qkv_scales", Binding::kQkvScales,
                                   syn_type_float, {48, 27});
    synTensor qkv_zero_bias = external("qkv_bias", Binding::kQkvBias,
                                       syn_type_float, {3392});
    synTensor activation_q = intermediate("qkv_activation_fp8", syn_type_fp8_143,
                                          {128, 1, 48});
    synTensor activation_scale = intermediate("qkv_activation_scale_fp32",
                                              syn_type_float, {1, 1, 48});
    node({normalized}, {activation_q, activation_scale}, nullptr, 0,
         "gk_block128_quant_v1", "qkv_block128_activation_quant");

    synTensor qkv_accumulator = intermediate("qkv_accumulator_fp32",
                                             syn_type_float, {3392, 1, 48});
    synGEMMParams qkv_gemm_params{false, true};
    qkv = external("qkv", Binding::kQkv, syn_type_bf16, {3392, 1});
    node({activation_q, qkv_packed}, {qkv_accumulator}, &qkv_gemm_params,
         sizeof(qkv_gemm_params), "batch_gemm", "qkv_block128_fp8_batch_gemm");
    node({qkv_accumulator, activation_scale, qkv_scale, qkv_zero_bias}, {qkv},
         nullptr, 0, "gk_block128_reduce_bias_v1", "qkv_block128_scale_finish");

    synTensor rope_cos = external("rope_cos", Binding::kRopeCos, syn_type_bf16,
                                  {64, 1, 1});
    synTensor rope_sin = external("rope_sin", Binding::kRopeSin, syn_type_bf16,
                                  {64, 1, 1});
    query = external("query", Binding::kQuery, syn_type_bf16, {192, 16, 1});
    new_key = external("new_key", Binding::kNewKey, syn_type_bf16,
                       {192, 1, 1});
    new_value = external("new_value", Binding::kNewValue, syn_type_bf16,
                         {128, 1, 1});
    node({qkv, rope_cos, rope_sin}, {query, new_key, new_value}, &kValueScale,
         sizeof(kValueScale), "gk_qkv_post_bf16_bf16_v1",
         "qkv_split_rope_vscale");

    synTensor key_cache = external("key_cache", Binding::kKeyCache,
                                   syn_type_bf16, {192, 1, cache_slots});
    const synSectionHandle key_cache_section = externals.back().section;
    synTensor value_cache = external("value_cache", Binding::kValueCache,
                                     syn_type_bf16, {128, 1, cache_slots});
    const synSectionHandle value_cache_section = externals.back().section;
    synTensor slot_mapping = external("slot_mapping", Binding::kSlotMapping,
                                      syn_type_int32, {1});
    key_cache_after = external("key_cache_after", Binding::kKeyCache,
                               syn_type_bf16, {192, 1, cache_slots},
                               key_cache_section);
    value_cache_after = external("value_cache_after", Binding::kValueCache,
                                 syn_type_bf16, {128, 1, cache_slots},
                                 value_cache_section);
    ns_IndexCopy::Params index_copy_params = {};
    index_copy_params.axis = 0;
    node({key_cache, slot_mapping, new_key}, {key_cache_after},
         &index_copy_params, sizeof(index_copy_params), "index_copy_fwd_bf16",
         "write_new_key");
    node({value_cache, slot_mapping, new_value}, {value_cache_after},
         &index_copy_params, sizeof(index_copy_params), "index_copy_fwd_bf16",
         "write_new_value");

    synTensor window_pages = external("window_pages", Binding::kWindowPages,
                                      syn_type_int32, {2});
    synTensor window_groups = external("window_groups", Binding::kWindowGroups,
                                       syn_type_int32, {2});
    synTensor position = external("position", Binding::kPosition,
                                  syn_type_int32, {1});
    synTensor sink = external("attention_sink", Binding::kAttentionSink,
                              syn_type_bf16, {16});
    context = external("context", Binding::kContext, syn_type_bf16,
                       {128, 16, 1});
    node({query, key_cache_after, value_cache_after, window_pages,
          window_groups, position, sink},
         {context}, &kAttentionScale, sizeof(kAttentionScale),
         "gk_swa128_window_head_fp32_v0", "swa128_fp32_attention");

    synTensor context_flat = intermediate("context_flat", syn_type_bf16,
                                          {2048, 1});
    node({context}, {context_flat}, nullptr, 0, "reshape", "flatten_context");
    synTensor oproj = external("oproj_weight", Binding::kOprojWeight,
                               syn_type_bf16, {2048, 6144});
    output_local = external("output_local", Binding::kOutputLocal,
                            syn_type_bf16, {6144, 1});
    synGEMMParams oproj_gemm_params{false, true};
    node({context_flat, oproj}, {output_local}, &oproj_gemm_params,
         sizeof(oproj_gemm_params), "gemm", "oproj_bf16_mme_local");

    check(synGraphCompile(&recipe, graph, "mimo_layer1_attention_b1_m1_tp8",
                          nullptr),
          "compile graph");
    check(synWorkspaceGetSize(&workspace_size, recipe), "get workspace size");
    arena.ensure_workspace(workspace_size);
    if (graph) {
      synGraphDestroy(graph);
      graph = nullptr;
    }
    for (auto& external : externals) {
      if (external.section) {
        synSectionDestroy(external.section);
        external.section = nullptr;
      }
    }
    launch.resize(externals.size());
    for (size_t i = 0; i < externals.size(); ++i) {
      launch[i] = {};
      launch[i].tensorName = externals[i].name;
      launch[i].tensorType = DATA_TENSOR;
      for (size_t dim = 0; dim < externals[i].shape.size(); ++dim) {
        launch[i].tensorSize[dim] = externals[i].shape[dim];
      }
    }
    initialize_launch_tensor_ids(recipe, launch.data(), launch.size());
  }

  uint64_t resolve(Binding binding, const MimoAttentionIo& io) const {
    switch (binding) {
      case Binding::kHiddenStates: return io.hidden_states;
      case Binding::kResidual: return io.residual;
      case Binding::kResidualOut: return io.residual_out;
      case Binding::kNormalized: return io.normalized;
      case Binding::kRopeCos: return io.rope_cos;
      case Binding::kRopeSin: return io.rope_sin;
      case Binding::kKeyCache: return io.key_cache;
      case Binding::kValueCache: return io.value_cache;
      case Binding::kSlotMapping: return io.slot_mapping;
      case Binding::kWindowPages: return io.window_pages;
      case Binding::kWindowGroups: return io.window_groups;
      case Binding::kPosition: return io.position;
      case Binding::kQkv: return io.qkv;
      case Binding::kQuery: return io.query;
      case Binding::kNewKey: return io.new_key;
      case Binding::kNewValue: return io.new_value;
      case Binding::kContext: return io.context;
      case Binding::kOutputLocal: return io.output_local;
      case Binding::kInputLayernorm:
        return weights->get("input_layernorm").device_address;
      case Binding::kQkvWeight: return weights->get("qkv_weight").device_address;
      case Binding::kQkvScales: return weights->get("qkv_scales").device_address;
      case Binding::kQkvBias: return weights->get("qkv_bias").device_address;
      case Binding::kAttentionSink:
        return weights->get("attention_sink").device_address;
      case Binding::kOprojWeight:
        return weights->get("oproj_weight").device_address;
    }
    throw std::runtime_error("MimoAttention: invalid tensor binding");
  }

  void execute(const MimoAttentionIo& io, synStreamHandle stream) {
    if (!stream || !recipe) {
      throw std::runtime_error("MimoAttention: invalid stream or recipe");
    }
    for (size_t i = 0; i < externals.size(); ++i) {
      const uint64_t address = resolve(externals[i].binding, io);
      if (address == 0) {
        throw std::runtime_error(std::string("MimoAttention: null address for ") +
                                 externals[i].name);
      }
      launch[i].pTensorAddress = address;
    }
    const uint64_t workspace = arena.ensure_workspace(workspace_size);
    check(synLaunch(stream, launch.data(), launch.size(), workspace, recipe, 0),
          "launch graph");
  }
};

MimoAttention::MimoAttention(HbmArena& arena, const WeightManifest& weights,
                             uint32_t cache_slots)
    : impl_(std::make_unique<Impl>(arena, weights, cache_slots)) {
  impl_->build();
}

MimoAttention::~MimoAttention() = default;

void MimoAttention::execute(const MimoAttentionIo& io, synStreamHandle stream) {
  impl_->execute(io, stream);
}

void MimoAttention::rebind_weights(const WeightManifest& weights) {
  expect_weight(weights, "input_layernorm", syn_type_bf16, {6144});
  expect_weight(weights, "qkv_weight", syn_type_fp8_143, {128, 3392, 48});
  expect_weight(weights, "qkv_scales", syn_type_float, {48, 27});
  expect_weight(weights, "qkv_bias", syn_type_float, {3392});
  expect_weight(weights, "attention_sink", syn_type_bf16, {16});
  expect_weight(weights, "oproj_weight", syn_type_bf16, {2048, 6144});
  impl_->weights = &weights;
}

uint32_t MimoAttention::cache_slots() const noexcept {
  return impl_->cache_slots;
}

uint64_t MimoAttention::workspace_bytes() const noexcept {
  return impl_->workspace_size;
}

uint64_t MimoAttention::recipe_identity() const noexcept {
  return reinterpret_cast<uint64_t>(impl_->recipe);
}

}  // namespace gaudirt::runtime
