// SPDX-License-Identifier: Apache-2.0
#include "mimo_full_attention.hpp"
#include "tp_reduce_graph.hpp"

#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"

#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <algorithm>
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

struct FullMmaSoftmaxParams {
  float scale;
  int32_t physical_pages;
};

void check(synStatus status, const std::string& operation) {
  if (status != synSuccess) {
    throw std::runtime_error("MimoFullAttention: " + operation +
                             " failed, status " +
                             std::to_string(static_cast<int>(status)));
  }
}

enum class Binding {
  kHiddenStates,
  kResidual,
  kResidualOut,
  kRopeCos,
  kRopeSin,
  kKeyCache,
  kValueCache,
  kSlotMapping,
  kFullBlockList,
  kFullBlockUsage,
  kFullBlockGroups,
  kPosition,
  kOutputLocal,
  kInputLayernorm,
  kQkvWeight,
  kQkvScales,
  kQkvBias,
  kOprojWeight,
};

struct ExternalTensor {
  const char* name;
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
    throw std::runtime_error(std::string("MimoFullAttention: invalid weight ") +
                             name);
  }
  return tensor;
}

}  // namespace

struct MimoFullAttention::Impl {
  HbmArena& arena;
  const WeightManifest* weights;
  uint32_t cache_slots;
  uint32_t rows;
  bool pure_rmsnorm;
  bool gathered_input;
  uint32_t page_count;
  synGraphHandle prepare_graph = nullptr;
  synSectionHandle reduction_section = nullptr;
  synRecipeHandle prepare_recipe = nullptr;
  RecipeCache* cache = nullptr;
  uint64_t prepare_workspace_size = 0;
  std::vector<ExternalTensor> prepare_externals;
  std::vector<synLaunchTensorInfo> prepare_launch;

  Impl(HbmArena& arena, const WeightManifest& weights, uint32_t cache_slots,
       bool pure_rmsnorm, uint32_t page_count, bool gathered_input, uint32_t rows)
      : arena(arena), weights(&weights), cache_slots(cache_slots),
        rows(rows), pure_rmsnorm(pure_rmsnorm), gathered_input(gathered_input),
        page_count(page_count) {
    if (pure_rmsnorm && gathered_input)
      throw std::invalid_argument("MimoFullAttention: pure input cannot be gathered");
  }

  ~Impl() { close(); }

  void close() noexcept {
    if (prepare_recipe) synRecipeDestroy(prepare_recipe);
    destroy_graphs();
  }

  void destroy_graphs() noexcept {
    if (prepare_graph) synGraphDestroy(prepare_graph);
    prepare_graph = nullptr;
    if (reduction_section) {
      synSectionDestroy(reduction_section);
      reduction_section = nullptr;
    }
    for (auto& external : prepare_externals) {
      if (external.section) {
        synSectionDestroy(external.section);
        external.section = nullptr;
      }
    }
  }

  synTensor external(synGraphHandle graph, std::vector<ExternalTensor>& externals,
                     const char* name, Binding binding, synDataType dtype,
                     std::initializer_list<uint64_t> sizes,
                     synSectionHandle shared_section = nullptr) {
    synTensorDescriptor descriptor = {};
    descriptor.m_name = name;
    descriptor.m_dataType = dtype;
    descriptor.m_dims = sizes.size();
    unsigned dim = 0;
    for (const uint64_t size : sizes) {
      if (size == 0 || size > UINT_MAX) {
        throw std::runtime_error("MimoFullAttention: invalid external shape");
      }
      descriptor.m_sizes[dim] = descriptor.m_minSizes[dim] = size;
      ++dim;
    }
    synSectionHandle section = shared_section;
    const bool owns_section = section == nullptr;
    if (owns_section) {
      check(synSectionCreate(&section, 0, graph),
            std::string("create section for ") + name);
    }
    externals.push_back({name, owns_section ? section : nullptr, binding, sizes});
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
      return tensor;
    } catch (...) {
      externals.pop_back();
      if (owns_section) synSectionDestroy(section);
      throw;
    }
  }

  synTensor intermediate(synGraphHandle, const char* name,
                         synDataType dtype,
                         std::initializer_list<uint64_t> sizes) {
    const std::string full_name = (pure_rmsnorm ? "pure_" : "residual_") +
                                  std::string(rows == 1 ? "" : "m32_") +
                                  std::string(name);
    synTensorDescriptor descriptor = {};
    descriptor.m_name = full_name.c_str();
    descriptor.m_dataType = dtype;
    descriptor.m_dims = sizes.size();
    unsigned dim = 0;
    for (const uint64_t size : sizes) {
      descriptor.m_sizes[dim] = descriptor.m_minSizes[dim] = size;
      ++dim;
    }
    synTensor tensor = nullptr;
    check(synTensorCreate(&tensor, &descriptor, nullptr, 0),
          std::string("create intermediate ") + name);
    return tensor;
  }

  void node(synGraphHandle graph, std::initializer_list<synTensor> inputs,
            std::initializer_list<synTensor> outputs, const void* params,
            unsigned params_size, const char* guid, const char* name) {
    check(synNodeCreate(graph, inputs.begin(), outputs.begin(), inputs.size(),
                        outputs.size(), params, params_size, guid, name, nullptr,
                        nullptr),
          std::string("create node ") + name);
  }

  void build() {
    if (rows != 1 && rows != 32)
      throw std::invalid_argument("MimoFullAttention: rows must be 1 or 32");
    if (cache_slots < 2 * MimoAttention::kKvBlockSize ||
        cache_slots % MimoAttention::kKvBlockSize != 0 ||
        cache_slots > static_cast<uint32_t>(INT_MAX) /
                          MimoAttention::kHeadDim) {
      throw std::runtime_error("MimoFullAttention: invalid cache geometry");
    }
    if (page_count == 0 || page_count > 64) {
      throw std::runtime_error("MimoFullAttention: invalid page count");
    }
    expect_weight(*weights, "input_layernorm", syn_type_bf16, {6144});
    expect_weight(*weights, "qkv_weight", syn_type_fp8_143, {128, 3392, 48});
    expect_weight(*weights, "qkv_scales", syn_type_float, {48, 27});
    expect_weight(*weights, "qkv_bias", syn_type_float, {3392});
    expect_weight(*weights, "oproj_weight", syn_type_bf16, {2048, 6144});
    build_prepare();
    destroy_graphs();
  }

  void build_prepare() {
    check(synGraphCreate(&prepare_graph, synDeviceGaudi2), "create prepare graph");
    synTensor hidden_states = nullptr;
    if (gathered_input) {
      synTensor gathered = external(prepare_graph, prepare_externals, "hidden_gathered",
                                   Binding::kHiddenStates, syn_type_float, {6144, 8 * rows});
      hidden_states = gaudirt::mimo::append_tp8_reduce(prepare_graph, gathered, syn_type_float,
                                                       reduction_section, rows);
    } else {
      hidden_states = external(prepare_graph, prepare_externals, "hidden_states",
                               Binding::kHiddenStates, syn_type_bf16, {6144, rows});
    }
    synTensor normalized = intermediate(prepare_graph, "normalized", syn_type_bf16, {6144, rows});
    synTensor norm_gamma = external(
        prepare_graph, prepare_externals, "input_layernorm",
        Binding::kInputLayernorm, syn_type_bf16, {6144});
    if (pure_rmsnorm) {
      node(prepare_graph, {hidden_states, norm_gamma}, {normalized},
           &kRmsNormEpsilon, sizeof(kRmsNormEpsilon),
           "gk_pure_rmsnorm_bf16_v1", "input_pure_rmsnorm");
    } else {
      synTensor residual = external(prepare_graph, prepare_externals, "residual",
                                    Binding::kResidual, syn_type_bf16, {6144, rows});
      synTensor residual_out = external(
          prepare_graph, prepare_externals, "residual_out",
          Binding::kResidualOut, syn_type_bf16, {6144, rows});
      node(prepare_graph, {hidden_states, residual, norm_gamma},
           {residual_out, normalized}, &kRmsNormEpsilon, sizeof(kRmsNormEpsilon),
           "gk_residual_rmsnorm_bf16_v1", "input_residual_rmsnorm");
    }

    synTensor qkv_packed = external(
        prepare_graph, prepare_externals, "qkv_weight", Binding::kQkvWeight,
        syn_type_fp8_143, {128, 3392, 48});
    synTensor qkv_scale = external(prepare_graph, prepare_externals,
                                   "qkv_scales", Binding::kQkvScales,
                                   syn_type_float, {48, 27});
    synTensor qkv_zero_bias = external(prepare_graph, prepare_externals,
                                       "qkv_bias", Binding::kQkvBias,
                                       syn_type_float, {3392});
    synTensor activation_q = intermediate(prepare_graph, "qkv_activation_fp8",
                                          syn_type_fp8_143, {128, rows, 48});
    synTensor activation_scale = intermediate(prepare_graph,
                                              "qkv_activation_scale_fp32",
                                              syn_type_float, {1, rows, 48});
    node(prepare_graph, {normalized}, {activation_q, activation_scale}, nullptr,
         0, "gk_block128_quant_v1", "qkv_block128_activation_quant");

    synTensor qkv_accumulator = intermediate(prepare_graph,
                                             "qkv_accumulator_fp32",
                                             syn_type_float, {3392, rows, 48});
    synGEMMParams qkv_gemm_params{false, true};
    synTensor qkv = intermediate(prepare_graph, "qkv", syn_type_bf16, {3392, rows});
    node(prepare_graph, {activation_q, qkv_packed}, {qkv_accumulator},
         &qkv_gemm_params, sizeof(qkv_gemm_params), "batch_gemm",
         "qkv_block128_fp8_batch_gemm");
    node(prepare_graph,
         {qkv_accumulator, activation_scale, qkv_scale, qkv_zero_bias}, {qkv},
         nullptr, 0, "gk_block128_neumaier_bf16_compensated_v1", "qkv_block128_scale_finish");

    synTensor rope_cos = external(prepare_graph, prepare_externals, "rope_cos",
                                  Binding::kRopeCos, syn_type_bf16, {64, 1, rows});
    synTensor rope_sin = external(prepare_graph, prepare_externals, "rope_sin",
                                  Binding::kRopeSin, syn_type_bf16, {64, 1, rows});
    synTensor query = intermediate(prepare_graph, "query", syn_type_bf16, {192, 16, rows});
    synTensor new_key = intermediate(prepare_graph, "new_key", syn_type_bf16, {192, 1, rows});
    synTensor new_value = intermediate(prepare_graph, "new_value", syn_type_bf16, {128, 1, rows});
    node(prepare_graph, {qkv, rope_cos, rope_sin},
         {query, new_key, new_value}, &kValueScale, sizeof(kValueScale),
         "gk_qkv_post_f32_f32_v1", "qkv_split_rope_vscale");

    synTensor key_cache = external(prepare_graph, prepare_externals, "key_cache",
                                   Binding::kKeyCache, syn_type_bf16,
                                   {192, 1, cache_slots});
    const synSectionHandle key_cache_section = prepare_externals.back().section;
    synTensor value_cache = external(prepare_graph, prepare_externals,
                                     "value_cache", Binding::kValueCache,
                                     syn_type_bf16, {128, 1, cache_slots});
    const synSectionHandle value_cache_section = prepare_externals.back().section;
    synTensor slot_mapping = external(prepare_graph, prepare_externals,
                                      "slot_mapping", Binding::kSlotMapping,
                                      syn_type_int32, {rows});
    synTensor key_cache_after = external(
        prepare_graph, prepare_externals, "key_cache_after", Binding::kKeyCache,
        syn_type_bf16, {192, 1, cache_slots}, key_cache_section);
    synTensor value_cache_after = external(
        prepare_graph, prepare_externals, "value_cache_after",
        Binding::kValueCache, syn_type_bf16, {128, 1, cache_slots},
        value_cache_section);
    ns_IndexCopy::Params index_copy_params = {};
    index_copy_params.axis = 0;
    node(prepare_graph, {key_cache, slot_mapping, new_key}, {key_cache_after},
         &index_copy_params, sizeof(index_copy_params), "index_copy_fwd_bf16",
         "write_new_key");
    node(prepare_graph, {value_cache, slot_mapping, new_value},
         {value_cache_after}, &index_copy_params, sizeof(index_copy_params),
         "index_copy_fwd_bf16", "write_new_value");

    synTensor block_list = external(prepare_graph, prepare_externals, "block_list",
                                    Binding::kFullBlockList, syn_type_int32,
                                    {page_count});
    synTensor block_usage = external(prepare_graph, prepare_externals, "block_usage",
                                     Binding::kFullBlockUsage, syn_type_int32,
                                     {page_count});
    synTensor block_groups = external(prepare_graph, prepare_externals, "block_groups",
                                      Binding::kFullBlockGroups, syn_type_int32,
                                      {page_count});
    const uint64_t tokens = static_cast<uint64_t>(page_count) * 128;
    synTensor gathered_key = intermediate(prepare_graph, "full_gathered_key",
                                          syn_type_bf16, {192, tokens});
    synTensor gathered_value = intermediate(prepare_graph, "full_gathered_value",
                                            syn_type_bf16, {128, tokens});
    node(prepare_graph,
         {key_cache_after, value_cache_after, block_list, block_usage, block_groups},
         {gathered_key, gathered_value}, nullptr, 0,
         "gk_full_mme_gather_v1", "full_gather_shared_kv");
    synTensor query_matrix = intermediate(prepare_graph, "full_query_matrix",
                                           syn_type_bf16, {192, 16 * rows});
    node(prepare_graph, {query}, {query_matrix}, nullptr, 0, "reshape", "full_query_reshape");
    synTensor scores = intermediate(prepare_graph, "full_attention_scores",
                                     syn_type_float, {tokens, 16 * rows});
    const synGEMMParams scores_params{false, true};
    node(prepare_graph, {query_matrix, gathered_key}, {scores}, &scores_params,
         sizeof(scores_params), "gemm", "full_qk_mme");
    synTensor probabilities = intermediate(prepare_graph, "full_attention_probabilities",
                                            syn_type_bf16, {tokens, 16 * rows});
    synTensor corrections = intermediate(prepare_graph, "full_attention_probabilities_correction",
                                        syn_type_bf16, {tokens, 16 * rows});
    const FullMmaSoftmaxParams softmax_params{kAttentionScale,
                                               static_cast<int32_t>(cache_slots / 128)};
    if (rows == 1) {
      node(prepare_graph, {scores, block_list, block_usage, block_groups}, {probabilities, corrections},
           &softmax_params, sizeof(softmax_params), "gk_full_mme_softmax_v2",
           "full_masked_softmax");
    } else {
      synTensor position = external(prepare_graph, prepare_externals, "position",
                                    Binding::kPosition, syn_type_int32, {rows});
      node(prepare_graph, {scores, block_list, block_usage, block_groups, position},
           {probabilities, corrections}, &softmax_params, sizeof(softmax_params),
           "gk_full_mme_prefill_softmax_v1", "full_causal_prefill_softmax");
    }
    synTensor context_main = intermediate(prepare_graph, "full_context_main",
                                          syn_type_float, {128, 16 * rows});
    synTensor context_correction = intermediate(prepare_graph, "full_context_correction",
                                                syn_type_float, {128, 16 * rows});
    synTensor context_float = intermediate(prepare_graph, "full_context_float",
                                           syn_type_float, {128, 16 * rows});
    synTensor context_matrix = intermediate(prepare_graph, "full_context_matrix",
                                            syn_type_bf16, {128, 16 * rows});
    const synGEMMParams context_params{false, false};
    node(prepare_graph, {probabilities, gathered_value}, {context_main}, &context_params,
         sizeof(context_params), "gemm", "full_pv_mme");
    node(prepare_graph, {corrections, gathered_value}, {context_correction}, &context_params,
         sizeof(context_params), "gemm", "full_pv_correction_mme");
    node(prepare_graph, {context_main, context_correction}, {context_float}, nullptr, 0,
         "add_fwd_f32", "full_pv_combine");
    ns_CastKernel::ParamsV3 context_cast{};
    context_cast.round_mode = CAST_ROUND_HALF_NE;
    node(prepare_graph, {context_float}, {context_matrix}, &context_cast, sizeof(context_cast),
         "cast_f32_to_bf16", "full_context_round");
    synTensor context = intermediate(prepare_graph, "context", syn_type_bf16, {128, 16, rows});
    node(prepare_graph, {context_matrix}, {context}, nullptr, 0, "reshape", "full_context_reshape");

    synTensor context_flat = intermediate(prepare_graph, "context_flat",
                                          syn_type_bf16, {2048, rows});
    node(prepare_graph, {context}, {context_flat}, nullptr, 0, "reshape",
         "flatten_context");
    synTensor oproj = external(prepare_graph, prepare_externals, "oproj_weight",
                               Binding::kOprojWeight, syn_type_bf16, {2048, 6144});
    synTensor output_local = external(prepare_graph, prepare_externals,
                                      "output_local", Binding::kOutputLocal,
                                      syn_type_float, {6144, rows});
    synGEMMParams oproj_gemm_params{false, true};
    node(prepare_graph, {context_flat, oproj}, {output_local},
         &oproj_gemm_params, sizeof(oproj_gemm_params), "gemm",
         "oproj_bf16_fp32_mme_local");
    const std::string recipe_name = pure_rmsnorm
        ? (rows == 1 ? "mimo_full_attention_b1_m1_tp8_pure" : "mimo_full_prefill_b1_m32_tp8_pure")
        : (rows == 1 ? "mimo_full_attention_b1_m1_tp8_residual" : "mimo_full_prefill_b1_m32_tp8_residual");
    prepare_recipe = compile_cached(
        prepare_graph, recipe_name,
        std::string("full_mme_neumaier_v4_") + (pure_rmsnorm ? "pure" : "residual") +
            "_c" + std::to_string(cache_slots) + "_p" + std::to_string(page_count) +
            (gathered_input ? "_gathered_tp1" : "_direct") + (rows == 1 ? "" : "_m32"),
        cache);
    check(synWorkspaceGetSize(&prepare_workspace_size, prepare_recipe),
          "get prepare workspace size");
    arena.ensure_workspace(prepare_workspace_size);
    prepare_launch.resize(prepare_externals.size());
    for (size_t i = 0; i < prepare_externals.size(); ++i) {
      prepare_launch[i] = {};
      prepare_launch[i].tensorName = prepare_externals[i].name;
      prepare_launch[i].tensorType = DATA_TENSOR;
      for (size_t dim = 0; dim < prepare_externals[i].shape.size(); ++dim) {
        prepare_launch[i].tensorSize[dim] = prepare_externals[i].shape[dim];
      }
    }
    initialize_launch_tensor_ids(prepare_recipe, prepare_launch.data(),
                                 prepare_launch.size());
  }

  uint64_t resolve(Binding binding, const MimoAttentionIo& io) const {
    switch (binding) {
      case Binding::kHiddenStates: return io.hidden_states;
      case Binding::kResidual: return io.residual;
      case Binding::kResidualOut: return io.residual_out;
      case Binding::kRopeCos: return io.rope_cos;
      case Binding::kRopeSin: return io.rope_sin;
      case Binding::kKeyCache: return io.key_cache;
      case Binding::kValueCache: return io.value_cache;
      case Binding::kSlotMapping: return io.slot_mapping;
      case Binding::kFullBlockList: return io.full_block_list;
      case Binding::kFullBlockUsage: return io.full_block_usage;
      case Binding::kFullBlockGroups: return io.full_block_groups;
      case Binding::kPosition: return io.position;
      case Binding::kOutputLocal: return io.output_local_fp32;
      case Binding::kInputLayernorm:
        return weights->get("input_layernorm").device_address;
      case Binding::kQkvWeight: return weights->get("qkv_weight").device_address;
      case Binding::kQkvScales: return weights->get("qkv_scales").device_address;
      case Binding::kQkvBias: return weights->get("qkv_bias").device_address;
      case Binding::kOprojWeight:
        return weights->get("oproj_weight").device_address;
    }
    throw std::runtime_error("MimoFullAttention: invalid tensor binding");
  }

  PreparedLaunch bind_graph(synRecipeHandle recipe,
                            const std::vector<ExternalTensor>& externals,
                            const std::vector<synLaunchTensorInfo>& prototype,
                            uint64_t workspace_size, const MimoAttentionIo& io) {
    auto tensors = prototype;
    for (size_t i = 0; i < externals.size(); ++i) {
      tensors[i].pTensorAddress = resolve(externals[i].binding, io);
      if (!tensors[i].pTensorAddress)
        throw std::runtime_error(std::string("MimoFullAttention: null address for ") +
                                 externals[i].name);
    }
    PreparedLaunch result{recipe, arena.ensure_workspace(workspace_size),
                          std::move(tensors), 0};
    result.validate();
    return result;
  }

  PreparedLaunch prepare(const MimoAttentionIo& io) {
    return bind_graph(prepare_recipe, prepare_externals, prepare_launch,
                      prepare_workspace_size, io);
  }

  void launch(synRecipeHandle recipe, std::vector<ExternalTensor>& externals,
              std::vector<synLaunchTensorInfo>& launch, uint64_t workspace_size,
              const MimoAttentionIo& io, synStreamHandle stream,
              const char* label) {
    if (!stream || !recipe) {
      throw std::runtime_error(std::string("MimoFullAttention: invalid ") + label);
    }
    for (size_t i = 0; i < externals.size(); ++i) {
      const uint64_t address = resolve(externals[i].binding, io);
      if (address == 0) {
        throw std::runtime_error(std::string("MimoFullAttention: null address for ") +
                                 externals[i].name);
      }
      launch[i].pTensorAddress = address;
    }
    const uint64_t workspace = arena.ensure_workspace(workspace_size);
    check(synLaunch(stream, launch.data(), launch.size(), workspace, recipe, 0),
          std::string("launch ") + label);
  }

  void execute(const MimoAttentionIo& io, synStreamHandle stream) {
    launch(prepare_recipe, prepare_externals, prepare_launch,
           prepare_workspace_size, io, stream, "attention graph");
  }
};

MimoFullAttention::MimoFullAttention(HbmArena& arena,
                                     const WeightManifest& weights,
                                     uint32_t cache_slots, bool pure_rmsnorm,
                                     uint32_t page_count, RecipeCache* cache,
                                     bool gathered_input, uint32_t rows)
    : impl_(std::make_unique<Impl>(arena, weights, cache_slots, pure_rmsnorm,
                                  page_count, gathered_input, rows)) {
  impl_->cache = cache;
  impl_->build();
}

MimoFullAttention::~MimoFullAttention() = default;

void MimoFullAttention::execute(const MimoAttentionIo& io,
                                synStreamHandle stream) {
  impl_->execute(io, stream);
}

PreparedLaunch MimoFullAttention::prepare(
    const MimoAttentionIo& io) {
  return impl_->prepare(io);
}

void MimoFullAttention::rebind_weights(const WeightManifest& weights) {
  expect_weight(weights, "input_layernorm", syn_type_bf16, {6144});
  expect_weight(weights, "qkv_weight", syn_type_fp8_143, {128, 3392, 48});
  expect_weight(weights, "qkv_scales", syn_type_float, {48, 27});
  expect_weight(weights, "qkv_bias", syn_type_float, {3392});
  expect_weight(weights, "oproj_weight", syn_type_bf16, {2048, 6144});
  impl_->weights = &weights;
}

uint64_t MimoFullAttention::workspace_bytes() const noexcept {
  return impl_->prepare_workspace_size;
}

uint64_t MimoFullAttention::recipe_identity() const noexcept {
  return reinterpret_cast<uint64_t>(impl_->prepare_recipe);
}

}  // namespace gaudirt::runtime
