// SPDX-License-Identifier: Apache-2.0
#include "mimo_moe.hpp"
#include "tp_reduce_graph.hpp"

#include "gaudirt/runtime/recipe_registry.hpp"

#include <perf_lib_layer_params.h>

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace gaudirt::runtime {
namespace {

using Sizes = std::initializer_list<uint64_t>;

void check(synStatus status, const char* where) {
  if (status != synSuccess) {
    throw std::runtime_error(std::string("MimoMoeLayer1: ") + where +
                             " failed with Synapse status " +
                             std::to_string(static_cast<int>(status)));
  }
}

template <size_t N>
void validate_addresses(const std::array<uint64_t, N>& addresses,
                        const char* where) {
  for (size_t i = 0; i < addresses.size(); ++i) {
    if (addresses[i] == 0) {
      throw std::invalid_argument(std::string("MimoMoeLayer1: ") + where +
                                  " contains a null HBM address");
    }
    for (size_t j = i + 1; j < addresses.size(); ++j) {
      if (addresses[i] == addresses[j]) {
        throw std::invalid_argument(std::string("MimoMoeLayer1: ") + where +
                                    " contains aliased HBM addresses");
      }
    }
  }
}

struct RouterPostParams {
  float factor;
  int32_t renormalize;
  int32_t apply_scale;
};
static_assert(sizeof(RouterPostParams) == 12);

}  // namespace

MimoMoeLayer1::MimoMoeLayer1(uint32_t tp_rank, const Weights& weights,
                             const char* graph_variant, RecipeCache* cache,
                             bool gathered_attention, synDataType gathered_type)
    : tp_rank_(tp_rank), gathered_attention_(gathered_attention),
      gathered_type_(gathered_type) {
  if (tp_rank_ >= kTpSize) {
    throw std::invalid_argument("MimoMoeLayer1: TP rank must be in [0,8)");
  }
  if (graph_variant == nullptr || *graph_variant == '\0') {
    throw std::invalid_argument("MimoMoeLayer1: graph variant must not be empty");
  }
  if (gathered_attention_ && gathered_type_ != syn_type_bf16 &&
      gathered_type_ != syn_type_float)
    throw std::invalid_argument("MimoMoeLayer1: invalid gathered dtype");
  validate_addresses(
      std::array<uint64_t, 9>{weights.post_attention_gamma_bf16,
                              weights.router_weight_bf16,
                              weights.router_correction_bias_fp32,
                              weights.gate_up_packed_u8,
                              weights.gate_up_scales_u8,
                              weights.down_packed_u8,
                              weights.down_scales_u8,
                              weights.mxfp4_lut_bf16,
                              weights.down_combine_directions_u8},
      "weights");

  synGraphHandle graph = nullptr;
  std::vector<synSectionHandle> sections;
  sections.reserve(kGatheredSlotCount);
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "synGraphCreate");

    auto external = [&](Slot slot, const char* name, synDataType dtype,
                        Sizes sizes, uint64_t address) -> synTensor {
      synSectionHandle section = nullptr;
      check(synSectionCreate(&section, 0, graph), name);
      sections.push_back(section);
      check(synSectionSetPersistent(section, true), name);
      synTensorDescriptor descriptor{};
      descriptor.m_name = name;
      descriptor.m_dataType = dtype;
      descriptor.m_dims = static_cast<unsigned>(sizes.size());
      unsigned dim = 0;
      for (uint64_t size : sizes) {
        descriptor.m_sizes[dim] = descriptor.m_minSizes[dim] = size;
        ++dim;
      }
      synTensor tensor = nullptr;
      check(synTensorCreate(&tensor, &descriptor, section, 0), name);

      synLaunchTensorInfo info{};
      info.tensorName = name;
      info.pTensorAddress = address;
      info.tensorType = DATA_TENSOR;
      dim = 0;
      for (uint64_t size : sizes) {
        info.tensorSize[dim++] = static_cast<uint32_t>(size);
      }
      launch_tensors_[slot] = info;
      return tensor;
    };

    auto temporary = [&](const char* name, synDataType dtype,
                         Sizes sizes) -> synTensor {
      const std::string full_name =
          std::string("mimo_") + graph_variant + "_" + name;
      synTensorDescriptor descriptor{};
      descriptor.m_name = full_name.c_str();
      descriptor.m_dataType = dtype;
      descriptor.m_dims = static_cast<unsigned>(sizes.size());
      unsigned dim = 0;
      for (uint64_t size : sizes) {
        descriptor.m_sizes[dim] = descriptor.m_minSizes[dim] = size;
        ++dim;
      }
      synTensor tensor = nullptr;
      check(synTensorCreate(&tensor, &descriptor, nullptr, 0), name);
      return tensor;
    };

    synTensor attention = nullptr;
    if (gathered_attention_) {
      constexpr const char* names[] = {
          "mimo1_attention_rank0", "mimo1_attention_rank1",
          "mimo1_attention_rank2", "mimo1_attention_rank3",
          "mimo1_attention_rank4", "mimo1_attention_rank5",
          "mimo1_attention_rank6", "mimo1_attention_rank7"};
      std::array<synTensor, 8> ranks{};
      for (unsigned rank = 0; rank < ranks.size(); ++rank) {
        const auto slot = static_cast<Slot>(rank == 0 ? kAttention : kSlotCount + rank - 1);
        ranks[rank] = external(slot, names[rank], gathered_type_, {kHiddenSize, 1}, 0);
      }
      sections.push_back(nullptr);
      attention = gaudirt::mimo::append_tp8_reduce(graph, ranks, gathered_type_, sections.back());
    } else {
      attention = external(kAttention, "mimo1_attention", syn_type_bf16,
                           {kHiddenSize, 1}, 0);
    }
    synTensor residual_in =
        external(kResidualIn, "mimo1_residual_in", syn_type_bf16,
                 {kHiddenSize, 1}, 0);
    synTensor gamma = external(kGamma, "mimo1_post_attention_gamma",
                               syn_type_bf16, {kHiddenSize},
                               weights.post_attention_gamma_bf16);
    synTensor router_weight =
        external(kRouterWeight, "mimo1_router_weight", syn_type_bf16,
                 {kHiddenSize, kExpertCount}, weights.router_weight_bf16);
    synTensor router_bias =
        external(kRouterBias, "mimo1_router_correction_bias", syn_type_single,
                 {kExpertCount, 1}, weights.router_correction_bias_fp32);
    synTensor gate_up_weight = external(
        kGateUpWeight, "mimo1_gate_up_packed", syn_type_uint8,
        {256, static_cast<uint64_t>(kExpertCount) * kHiddenSize},
        weights.gate_up_packed_u8);
    synTensor gate_up_scale = external(
        kGateUpScale, "mimo1_gate_up_scales", syn_type_uint8,
        {512, static_cast<uint64_t>(kExpertCount) * kHiddenSize / 32},
        weights.gate_up_scales_u8);
    synTensor down_weight = external(
        kDownWeight, "mimo1_down_packed", syn_type_uint8,
        {256, static_cast<uint64_t>(kExpertCount) * 12 * kLocalDownK},
        weights.down_packed_u8);
    synTensor down_scale = external(
        kDownScale, "mimo1_down_scales", syn_type_uint8,
        {512, static_cast<uint64_t>(kExpertCount) * 12 * kLocalDownK / 32},
        weights.down_scales_u8);
    synTensor lut = external(kLut, "mimo1_mxfp4_lut", syn_type_bf16, {512, 1},
                             weights.mxfp4_lut_bf16);
    synTensor directions =
        external(kDirections, "mimo1_down_combine_directions", syn_type_uint8,
                 {256, 2}, weights.down_combine_directions_u8);
    synTensor residual_out =
        external(kResidualOut, "mimo1_residual_out", syn_type_bf16,
                 {kHiddenSize, 1}, 0);
    synTensor normalized = temporary("normalized", syn_type_bf16, {kHiddenSize, 1});
    synTensor route_ids = temporary("topk_ids", syn_type_int32, {kTopK, 1});
    synTensor route_weights = temporary("topk_weights", syn_type_single, {kTopK, 1});
    synTensor moe_out = external(kMoeOut, "mimo1_moe_local_fp32",
                                 syn_type_single, {kHiddenSize, 1}, 0);

    synTensor router_logits =
        temporary("mimo1_router_logits", syn_type_single, {kExpertCount, 1});
    synTensor scores =
        temporary("mimo1_router_scores", syn_type_single, {kExpertCount, 1});
    synTensor biased_scores = temporary("mimo1_router_biased_scores",
                                        syn_type_single, {kExpertCount, 1});
    synTensor topk_values =
        temporary("mimo1_topk_values", syn_type_single, {kTopK, 1});
    synTensor selected_ids =
        temporary("mimo1_selected_ids", syn_type_int32, {kTopK, 1});
    synTensor route_weight_sum =
        temporary("mimo1_route_weight_sum", syn_type_single, {1, 1});
    synTensor gate_up_partial = temporary(
        "mimo1_gate_up_partial", syn_type_single, {kTopK * 1536, 1});
    synTensor gate_activation = temporary("mimo1_gate_activation",
                                          syn_type_bf16, {kLocalDownK, kTopK});
    synTensor down_partial = temporary(
        "mimo1_down_partial", syn_type_single, {kTopK * kHiddenSize, 1});

    auto node = [&](const char* guid, const char* name,
                    std::initializer_list<synTensor> inputs,
                    std::initializer_list<synTensor> outputs,
                    const void* params, unsigned params_size) {
      check(synNodeCreate(graph, inputs.begin(), outputs.begin(),
                          static_cast<uint32_t>(inputs.size()),
                          static_cast<uint32_t>(outputs.size()), params,
                          params_size, guid, name, nullptr, nullptr),
            name);
    };

    const float epsilon = 1e-5F;
    node("gk_residual_rmsnorm_bf16_v1", "mimo1_post_rmsnorm",
         {attention, residual_in, gamma}, {residual_out, normalized}, &epsilon,
         sizeof(epsilon));

    const synGEMMParams gemm_params{false, true};
    node("gemm", "mimo1_router_gemm", {normalized, router_weight},
         {router_logits}, &gemm_params, sizeof(gemm_params));

    const ns_SigmoidKernel::Params sigmoid_params{NO_SATURATION_SIGMOID};
    node("sigmoid_fwd_f32", "mimo1_router_sigmoid", {router_logits}, {scores},
         &sigmoid_params, sizeof(sigmoid_params));
    node("add_fwd_f32", "mimo1_router_correction", {scores, router_bias},
         {biased_scores}, nullptr, 0);

    ns_TopkNodeV2::ParamsV4 topk_params{};
    topk_params.bsw = kTopK;
    topk_params.axis = 0;
    topk_params.bottomK = false;
    topk_params.isVcData = false;
    node("topk", "mimo1_router_top8", {biased_scores},
         {topk_values, selected_ids}, &topk_params, sizeof(topk_params));

    const RouterPostParams router_post_params{1.0F, 1, 0};
    node("gk_router_post8_vector_experiment", "mimo1_router_post8",
         {scores, selected_ids}, {route_weights, route_ids, route_weight_sum},
         &router_post_params, sizeof(router_post_params));

    node("gk_moe_gp_scale_tail_v1", "mimo1_gate_up_mxfp4",
         {gate_up_weight, gate_up_scale, normalized, lut, route_ids},
         {gate_up_partial}, nullptr, 0);

    const int32_t gate_splits = 3;
    node("ub_compact_gate", "mimo1_gate_silu", {gate_up_partial, route_ids},
         {gate_activation}, &gate_splits, sizeof(gate_splits));

    node("downact_direct_down_broadcast", "mimo1_down_mxfp4",
         {down_weight, down_scale, gate_activation, lut, route_ids},
         {down_partial}, nullptr, 0);

    const int32_t combine_topk = kTopK;
    node("pf_combine_f32", "mimo1_fp32_combine",
         {down_partial, route_weights, directions}, {moe_out}, &combine_topk,
         sizeof(combine_topk));

    const std::string recipe_name =
        std::string("mimo_") + graph_variant + "_moe_b1_m1_tp8_rank" +
        std::to_string(tp_rank_);
    recipe_ = compile_cached(
        graph, recipe_name,
        std::string("moe_internal_v2_") + graph_variant + "_rank" + std::to_string(tp_rank_) +
            (gathered_attention_ ? "_gathered" : "_direct"),
        cache);
    check(synWorkspaceGetSize(&workspace_bytes_, recipe_),
          "synWorkspaceGetSize");
    initialize_launch_tensor_ids(recipe_, launch_tensors_.data(),
                                 gathered_attention_ ? kGatheredSlotCount : kSlotCount);

    synGraphDestroy(graph);
    graph = nullptr;
    for (synSectionHandle& section : sections) {
      synSectionDestroy(section);
      section = nullptr;
    }
  } catch (...) {
    if (recipe_ != nullptr) {
      synRecipeDestroy(recipe_);
      recipe_ = nullptr;
    }
    if (graph != nullptr) {
      synGraphDestroy(graph);
    }
    for (synSectionHandle& section : sections) {
      if (section != nullptr) {
        synSectionDestroy(section);
      }
    }
    throw;
  }
}

MimoMoeLayer1::~MimoMoeLayer1() {
  if (recipe_ != nullptr) {
    synRecipeDestroy(recipe_);
  }
}

void MimoMoeLayer1::rebind_weights(const Weights& weights) {
  validate_addresses(
      std::array<uint64_t, 9>{weights.post_attention_gamma_bf16,
                              weights.router_weight_bf16,
                              weights.router_correction_bias_fp32,
                              weights.gate_up_packed_u8,
                              weights.gate_up_scales_u8,
                              weights.down_packed_u8,
                              weights.down_scales_u8,
                              weights.mxfp4_lut_bf16,
                              weights.down_combine_directions_u8},
      "weights");
  launch_tensors_[kGamma].pTensorAddress = weights.post_attention_gamma_bf16;
  launch_tensors_[kRouterWeight].pTensorAddress = weights.router_weight_bf16;
  launch_tensors_[kRouterBias].pTensorAddress = weights.router_correction_bias_fp32;
  launch_tensors_[kGateUpWeight].pTensorAddress = weights.gate_up_packed_u8;
  launch_tensors_[kGateUpScale].pTensorAddress = weights.gate_up_scales_u8;
  launch_tensors_[kDownWeight].pTensorAddress = weights.down_packed_u8;
  launch_tensors_[kDownScale].pTensorAddress = weights.down_scales_u8;
  launch_tensors_[kLut].pTensorAddress = weights.mxfp4_lut_bf16;
  launch_tensors_[kDirections].pTensorAddress = weights.down_combine_directions_u8;
}

PreparedLaunch MimoMoeLayer1::prepare(const Inputs& inputs,
                                      const Outputs& outputs,
                                      uint64_t workspace_addr) {
  if ((workspace_bytes_ == 0) != (workspace_addr == 0))
    throw std::invalid_argument("MimoMoeLayer1: invalid prepared workspace");
  auto tensors = launch_tensors_;
  tensors[kAttention].pTensorAddress = inputs.attention_bf16;
  const size_t count = gathered_attention_ ? kGatheredSlotCount : kSlotCount;
  if (gathered_attention_) {
    const uint64_t rank_bytes = kHiddenSize * (gathered_type_ == syn_type_bf16 ? 2 : 4);
    for (size_t rank = 1; rank < kTpSize; ++rank)
      tensors[kSlotCount + rank - 1].pTensorAddress = inputs.attention_bf16 + rank * rank_bytes;
  }
  tensors[kResidualIn].pTensorAddress = inputs.residual_in_bf16;
  tensors[kResidualOut].pTensorAddress = outputs.residual_out_bf16;
  tensors[kMoeOut].pTensorAddress = outputs.moe_local_fp32;
  std::vector<uint64_t> addresses(count);
  for (size_t i = 0; i < addresses.size(); ++i)
    addresses[i] = tensors[i].pTensorAddress;
  for (size_t i = 0; i < addresses.size(); ++i) {
    for (size_t j = i + 1; j < addresses.size(); ++j)
      if (addresses[i] == addresses[j])
        throw std::invalid_argument("MimoMoeLayer1: aliased prepared bindings");
    if (workspace_addr && addresses[i] == workspace_addr)
      throw std::invalid_argument("MimoMoeLayer1: workspace aliases a binding");
  }
  PreparedLaunch result{recipe_, workspace_addr,
                        {tensors.begin(), tensors.begin() + count}, 0};
  result.validate();
  return result;
}

void MimoMoeLayer1::launch(synStreamHandle stream, const Inputs& inputs,
                           const Outputs& outputs, uint64_t workspace_addr) {
  if (gathered_attention_)
    throw std::invalid_argument("MimoMoeLayer1: gathered model requires prepared launch");
  if (stream == nullptr) {
    throw std::invalid_argument("MimoMoeLayer1: null launch stream");
  }
  if ((workspace_bytes_ == 0) != (workspace_addr == 0)) {
    throw std::invalid_argument(
        "MimoMoeLayer1: workspace address does not match compiled workspace");
  }

  launch_tensors_[kAttention].pTensorAddress = inputs.attention_bf16;
  launch_tensors_[kResidualIn].pTensorAddress = inputs.residual_in_bf16;
  launch_tensors_[kResidualOut].pTensorAddress = outputs.residual_out_bf16;
  launch_tensors_[kMoeOut].pTensorAddress = outputs.moe_local_fp32;

  std::array<uint64_t, kSlotCount> addresses{};
  for (size_t i = 0; i < addresses.size(); ++i) {
    addresses[i] = launch_tensors_[i].pTensorAddress;
  }
  validate_addresses(addresses, "launch bindings");
  for (uint64_t address : addresses) {
    if (address == workspace_addr && workspace_addr != 0) {
      throw std::invalid_argument(
          "MimoMoeLayer1: workspace aliases a tensor binding");
    }
  }

  check(synLaunch(stream, launch_tensors_.data(),
                  static_cast<uint32_t>(kSlotCount), workspace_addr,
                  recipe_, 0),
        "synLaunch");
}

std::array<uint16_t, 512> MimoMoeLayer1::make_mxfp4_lut_bf16() {
  constexpr std::array<uint16_t, 16> values = {
      0x0000, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0,
      0x8000, 0xbf00, 0xbf80, 0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0};
  std::array<uint16_t, 512> table{};
  for (size_t byte = 0; byte < 256; ++byte) {
    table[2 * byte] = values[byte % 16];
    table[2 * byte + 1] = values[byte / 16];
  }
  return table;
}

std::array<uint8_t, 512> MimoMoeLayer1::make_down_combine_directions() {
  std::array<uint8_t, 512> directions{};
  for (size_t parity = 0; parity < 2; ++parity) {
    for (size_t byte = 0; byte < 256; ++byte) {
      const size_t lane = byte / 4;
      directions[parity * 256 + byte] = static_cast<uint8_t>(
          lane % 16 / 2 + lane / 16 % 2 * 32 +
          (lane % 2 == parity ? 128 : 0));
    }
  }
  return directions;
}

}  // namespace gaudirt::runtime
