// SPDX-License-Identifier: Apache-2.0
#include "mimo_dense_mlp.hpp"

#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"

#include <perf_lib_layer_params.h>
#include <synapse_common_types.hpp>

#include <algorithm>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace gaudirt::runtime {
namespace {

constexpr float kRmsNormEpsilon = 1e-5F;

void check(synStatus status, const std::string& operation) {
  if (status != synSuccess) {
    throw std::runtime_error("MimoDenseMlp: " + operation +
                             " failed, status " +
                             std::to_string(static_cast<int>(status)));
  }
}

enum class Binding {
  kAttention,
  kResidualIn,
  kGamma,
  kResidualOut,
  kGateUpWeight,
  kGateUpScales,
  kGateUpBias,
  kActivation,
  kActivationFp32,
  kDownWeight,
  kDownScales,
  kDownBias,
  kOutputLocal,
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
    throw std::runtime_error(std::string("MimoDenseMlp: invalid weight ") + name);
  }
  return tensor;
}

const WeightTensor& expect_dynamic_weight(const WeightManifest& weights,
                                          const char* name, synDataType dtype,
                                          size_t rank) {
  const auto& tensor = weights.get(name);
  if (tensor.dtype != dtype || tensor.shape.size() != rank ||
      tensor.device_address == 0) {
    throw std::runtime_error(std::string("MimoDenseMlp: invalid weight ") + name);
  }
  return tensor;
}

}  // namespace

struct MimoDenseMlp::Impl {
  HbmArena& arena;
  const WeightManifest* weights;
  synGraphHandle gate_up_graph = nullptr;
  synGraphHandle down_graph = nullptr;
  synRecipeHandle gate_up_recipe = nullptr;
  synRecipeHandle down_recipe = nullptr;
  uint64_t gate_up_workspace_size = 0;
  uint64_t down_workspace_size = 0;
  uint64_t gate_up_n = 0;
  uint64_t down_k = 0;
  std::vector<ExternalTensor> gate_up_externals;
  std::vector<ExternalTensor> down_externals;
  std::vector<synLaunchTensorInfo> gate_up_launch;
  std::vector<synLaunchTensorInfo> down_launch;

  Impl(HbmArena& arena, const WeightManifest& weights)
      : arena(arena), weights(&weights) {}

  ~Impl() {
    if (gate_up_recipe) synRecipeDestroy(gate_up_recipe);
    if (down_recipe) synRecipeDestroy(down_recipe);
    destroy_graphs();
  }

  void destroy_graphs() noexcept {
    if (gate_up_graph) synGraphDestroy(gate_up_graph);
    if (down_graph) synGraphDestroy(down_graph);
    gate_up_graph = down_graph = nullptr;
    for (auto& external : gate_up_externals) {
      if (external.section) {
        synSectionDestroy(external.section);
        external.section = nullptr;
      }
    }
    for (auto& external : down_externals) {
      if (external.section) {
        synSectionDestroy(external.section);
        external.section = nullptr;
      }
    }
  }

  synTensor external(synGraphHandle graph, std::vector<ExternalTensor>& externals,
                     const char* name, Binding binding, synDataType dtype,
                     std::initializer_list<uint64_t> sizes) {
    synTensorDescriptor descriptor = {};
    descriptor.m_name = name;
    descriptor.m_dataType = dtype;
    descriptor.m_dims = sizes.size();
    unsigned dim = 0;
    for (const uint64_t size : sizes) {
      if (size == 0 || size > UINT_MAX) {
        throw std::runtime_error("MimoDenseMlp: invalid external shape");
      }
      descriptor.m_sizes[dim] = descriptor.m_minSizes[dim] = size;
      ++dim;
    }
    synSectionHandle section = nullptr;
    check(synSectionCreate(&section, 0, graph),
          std::string("create section for ") + name);
    externals.push_back({name, section, binding, sizes});
    try {
      check(synSectionSetPersistent(section, true),
            std::string("persist section for ") + name);
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
      synSectionDestroy(section);
      throw;
    }
  }

  synTensor intermediate(synGraphHandle, const char* name, synDataType dtype,
                         std::initializer_list<uint64_t> sizes) {
    synTensorDescriptor descriptor = {};
    descriptor.m_name = name;
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
    const auto& gate_up_weight =
        expect_dynamic_weight(*weights, "gate_up_weight", syn_type_fp8_143, 3);
    gate_up_n = gate_up_weight.shape[1];
    down_k = gate_up_n / 2;
    if (gate_up_weight.shape.size() != 3 || gate_up_weight.shape[0] != 128 ||
        gate_up_n == 0 || gate_up_n % 2 != 0 || down_k == 0) {
      throw std::runtime_error("MimoDenseMlp: invalid gate_up geometry");
    }
    const uint64_t groups = gate_up_weight.shape[2];
    expect_weight(*weights, "gate_up_scales", syn_type_float,
                  {groups, (gate_up_n + 127) / 128});
    expect_weight(*weights, "gate_up_bias", syn_type_float, {gate_up_n});
    const auto& down_weight =
        expect_dynamic_weight(*weights, "down_weight", syn_type_fp8_143, 3);
    if (down_weight.shape.size() != 3 || down_weight.shape[0] != 128 ||
        down_weight.shape[1] != 6144) {
      throw std::runtime_error("MimoDenseMlp: invalid down geometry");
    }
    const uint64_t down_groups = down_weight.shape[2];
    expect_weight(*weights, "down_scales", syn_type_float, {down_groups, 48});
    expect_weight(*weights, "down_bias", syn_type_float, {6144});
    build_gate_up(groups);
    build_down(down_groups);
    destroy_graphs();
  }

  void build_gate_up(uint64_t groups) {
    check(synGraphCreate(&gate_up_graph, synDeviceGaudi2),
          "create gate_up graph");
    synTensor attention = external(gate_up_graph, gate_up_externals, "attention",
                                   Binding::kAttention, syn_type_bf16, {6144, 1});
    synTensor residual_in = external(gate_up_graph, gate_up_externals,
                                     "residual_in", Binding::kResidualIn,
                                     syn_type_bf16, {6144, 1});
    synTensor gamma = external(gate_up_graph, gate_up_externals,
                               "post_attention_gamma", Binding::kGamma,
                               syn_type_bf16, {6144});
    synTensor residual_out = external(gate_up_graph, gate_up_externals,
                                      "residual_out", Binding::kResidualOut,
                                      syn_type_bf16, {6144, 1});
    synTensor normalized = intermediate(gate_up_graph, "normalized_fp32",
                                        syn_type_float, {6144, 1});
    node(gate_up_graph, {attention, residual_in, gamma},
         {residual_out, normalized}, &kRmsNormEpsilon, sizeof(kRmsNormEpsilon),
         "gk_dense_residual_rmsnorm_fp32_v1", "post_residual_rmsnorm_fp32");

    synTensor weight = external(gate_up_graph, gate_up_externals,
                                "gate_up_weight", Binding::kGateUpWeight,
                                syn_type_fp8_143, {128, gate_up_n, groups});
    synTensor weight_scale = external(gate_up_graph, gate_up_externals,
                                      "gate_up_scales", Binding::kGateUpScales,
                                      syn_type_float,
                                      {groups, (gate_up_n + 127) / 128});
    synTensor bias = external(gate_up_graph, gate_up_externals, "gate_up_bias",
                              Binding::kGateUpBias, syn_type_float, {gate_up_n});
    synTensor gate_up = intermediate(gate_up_graph, "gate_up_fp32",
                                     syn_type_float, {gate_up_n, 1});
    node(gate_up_graph, {normalized, weight, weight_scale, bias}, {gate_up}, nullptr, 0,
         "gk_dense_linear_fp32_v1", "gate_up_fp32_linear");
    synTensor activation_wide = external(gate_up_graph, gate_up_externals,
                                         "activation_fp32", Binding::kActivationFp32,
                                         syn_type_float, {down_k, 1});
    node(gate_up_graph, {gate_up}, {activation_wide}, nullptr, 0,
         "gk_dense_swiglu_fp32_v1", "swiglu_fp32");
    check(synGraphCompile(&gate_up_recipe, gate_up_graph,
                          "mimo_dense_gate_up_b1_m1_tp8", nullptr),
          "compile gate_up graph");
    check(synWorkspaceGetSize(&gate_up_workspace_size, gate_up_recipe),
          "get gate_up workspace size");
    arena.ensure_workspace(gate_up_workspace_size);
    gate_up_launch.resize(gate_up_externals.size());
    for (size_t i = 0; i < gate_up_externals.size(); ++i) {
      gate_up_launch[i] = {};
      gate_up_launch[i].tensorName = gate_up_externals[i].name;
      gate_up_launch[i].tensorType = DATA_TENSOR;
      for (size_t dim = 0; dim < gate_up_externals[i].shape.size(); ++dim) {
        gate_up_launch[i].tensorSize[dim] = gate_up_externals[i].shape[dim];
      }
    }
    initialize_launch_tensor_ids(gate_up_recipe, gate_up_launch.data(),
                                 gate_up_launch.size());
  }

  void build_down(uint64_t groups) {
    check(synGraphCreate(&down_graph, synDeviceGaudi2), "create down graph");
    synTensor activation = external(down_graph, down_externals, "activation",
                                    Binding::kActivation, syn_type_float,
                                    {down_k, 1});
    synTensor weight = external(down_graph, down_externals, "down_weight",
                                Binding::kDownWeight, syn_type_fp8_143,
                                {128, 6144, groups});
    synTensor weight_scale = external(down_graph, down_externals, "down_scales",
                                      Binding::kDownScales, syn_type_float,
                                      {groups, 48});
    synTensor bias = external(down_graph, down_externals, "down_bias",
                              Binding::kDownBias, syn_type_float, {6144});
    synTensor output_local = external(down_graph, down_externals, "output_local",
                                      Binding::kOutputLocal, syn_type_float,
                                      {6144, 1});
    node(down_graph, {activation, weight, weight_scale, bias}, {output_local},
         nullptr, 0, "gk_dense_linear_fp32_v1", "down_fp32_linear");
    check(synGraphCompile(&down_recipe, down_graph,
                          "mimo_dense_down_b1_m1_tp8", nullptr),
          "compile down graph");
    check(synWorkspaceGetSize(&down_workspace_size, down_recipe),
          "get down workspace size");
    arena.ensure_workspace(down_workspace_size);
    down_launch.resize(down_externals.size());
    for (size_t i = 0; i < down_externals.size(); ++i) {
      down_launch[i] = {};
      down_launch[i].tensorName = down_externals[i].name;
      down_launch[i].tensorType = DATA_TENSOR;
      for (size_t dim = 0; dim < down_externals[i].shape.size(); ++dim) {
        down_launch[i].tensorSize[dim] = down_externals[i].shape[dim];
      }
    }
    initialize_launch_tensor_ids(down_recipe, down_launch.data(),
                                 down_launch.size());
  }

  uint64_t resolve(Binding binding, const Inputs& inputs,
                   const Outputs& outputs) const {
    switch (binding) {
      case Binding::kAttention: return inputs.attention_bf16;
      case Binding::kResidualIn: return inputs.residual_in_bf16;
      case Binding::kGamma: return inputs.post_attention_gamma_bf16;
      case Binding::kResidualOut: return outputs.residual_out_bf16;
      case Binding::kGateUpWeight:
        return weights->get("gate_up_weight").device_address;
      case Binding::kGateUpScales:
        return weights->get("gate_up_scales").device_address;
      case Binding::kGateUpBias: return weights->get("gate_up_bias").device_address;
      case Binding::kActivation: return outputs.activation_fp32;
      case Binding::kActivationFp32: return outputs.activation_fp32;
      case Binding::kDownWeight: return weights->get("down_weight").device_address;
      case Binding::kDownScales: return weights->get("down_scales").device_address;
      case Binding::kDownBias: return weights->get("down_bias").device_address;
      case Binding::kOutputLocal: return outputs.output_local_fp32;
    }
    throw std::runtime_error("MimoDenseMlp: invalid tensor binding");
  }

  void launch(synRecipeHandle recipe, std::vector<ExternalTensor>& externals,
              std::vector<synLaunchTensorInfo>& launch, uint64_t workspace_size,
              const Inputs& inputs, const Outputs& outputs, synStreamHandle stream,
              const char* label) {
    if (!stream || !recipe) {
      throw std::runtime_error(std::string("MimoDenseMlp: invalid ") + label);
    }
    for (size_t i = 0; i < externals.size(); ++i) {
      const uint64_t address = resolve(externals[i].binding, inputs, outputs);
      if (address == 0) {
        throw std::runtime_error(std::string("MimoDenseMlp: null address for ") +
                                 externals[i].name);
      }
      launch[i].pTensorAddress = address;
    }
    const uint64_t workspace = arena.ensure_workspace(workspace_size);
    check(synLaunch(stream, launch.data(), launch.size(), workspace, recipe, 0),
          std::string("launch ") + label);
  }

  void execute(const Inputs& inputs, const Outputs& outputs,
               synStreamHandle stream) {
    launch(gate_up_recipe, gate_up_externals, gate_up_launch,
           gate_up_workspace_size, inputs, outputs, stream, "gate_up graph");
    launch(down_recipe, down_externals, down_launch, down_workspace_size, inputs,
           outputs, stream, "down graph");
  }
};

MimoDenseMlp::MimoDenseMlp(HbmArena& arena, const WeightManifest& weights)
    : impl_(std::make_unique<Impl>(arena, weights)) {
  impl_->build();
}

MimoDenseMlp::~MimoDenseMlp() = default;

void MimoDenseMlp::execute(const Inputs& inputs, const Outputs& outputs,
                           synStreamHandle stream) {
  impl_->execute(inputs, outputs, stream);
}

void MimoDenseMlp::rebind_weights(const WeightManifest& weights) {
  const auto& gate_up = expect_dynamic_weight(weights, "gate_up_weight",
                                               syn_type_fp8_143, 3);
  const auto& down = expect_dynamic_weight(weights, "down_weight",
                                            syn_type_fp8_143, 3);
  expect_dynamic_weight(weights, "gate_up_scales", syn_type_float, 2);
  expect_dynamic_weight(weights, "gate_up_bias", syn_type_float, 1);
  expect_dynamic_weight(weights, "down_scales", syn_type_float, 2);
  expect_dynamic_weight(weights, "down_bias", syn_type_float, 1);
  const auto& old_gate_up = impl_->weights->get("gate_up_weight");
  const auto& old_down = impl_->weights->get("down_weight");
  if (gate_up.shape != old_gate_up.shape || down.shape != old_down.shape ||
      weights.get("gate_up_scales").shape !=
          impl_->weights->get("gate_up_scales").shape ||
      weights.get("down_scales").shape !=
          impl_->weights->get("down_scales").shape) {
    throw std::invalid_argument("MimoDenseMlp: incompatible rebind geometry");
  }
  impl_->weights = &weights;
}

uint64_t MimoDenseMlp::workspace_bytes() const noexcept {
  return std::max(impl_->gate_up_workspace_size, impl_->down_workspace_size);
}

std::pair<uint64_t, uint64_t> MimoDenseMlp::recipe_identity() const noexcept {
  return {reinterpret_cast<uint64_t>(impl_->gate_up_recipe),
          reinterpret_cast<uint64_t>(impl_->down_recipe)};
}

}  // namespace gaudirt::runtime
