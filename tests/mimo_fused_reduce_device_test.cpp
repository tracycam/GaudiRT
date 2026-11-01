// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include "gaudirt/runtime/tp_reduce.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "mimo_attention.hpp"
#include "mimo_dense_mlp.hpp"
#include "mimo_full_attention.hpp"
#include "mimo_moe.hpp"
#include "model_head.hpp"
#include "tp_reduce_graph.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
constexpr size_t kHidden = 6144;

void check(synStatus status) {
  if (status != synSuccess) {
    std::fprintf(stderr, "Synapse status %d\n", static_cast<int>(status));
    throw std::runtime_error("fused reduction device test failed");
  }
}

uint16_t bf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

class RmsGraph {
 public:
  RmsGraph(gaudirt::runtime::HbmArena& arena, bool fused, synDataType input_type)
      : arena_(arena), fused_(fused), input_type_(input_type) {
    synGraphHandle graph = nullptr;
    std::vector<synSectionHandle> sections;
    check(synGraphCreate(&graph, synDeviceGaudi2));
    try {
      auto external = [&](const char* name, synDataType dtype, bool vector = false,
                          uint64_t rows = 1) {
        synSectionHandle section = nullptr;
        check(synSectionCreate(&section, 0, graph));
        sections.push_back(section);
        check(synSectionSetPersistent(section, true));
        synTensorDescriptor descriptor{};
        descriptor.m_name = name;
        descriptor.m_dataType = dtype;
        descriptor.m_dims = vector ? 1 : 2;
        descriptor.m_sizes[0] = descriptor.m_minSizes[0] = kHidden;
        if (!vector) descriptor.m_sizes[1] = descriptor.m_minSizes[1] = rows;
        synTensor tensor = nullptr;
        check(synTensorCreate(&tensor, &descriptor, section, 0));
        synLaunchTensorInfo info{};
        info.tensorName = name;
        info.tensorType = DATA_TENSOR;
        info.tensorSize[0] = kHidden;
        if (!vector) info.tensorSize[1] = rows;
        launch_.push_back(info);
        return tensor;
      };
      synTensor attention = nullptr;
      if (fused_) {
        synTensor gathered = external("gathered", input_type_, false, 8);
        synSectionHandle reduced_section = nullptr;
        attention = gaudirt::mimo::append_tp8_reduce(graph, gathered, input_type_, reduced_section);
        sections.push_back(reduced_section);
      } else {
        attention = external("attention", syn_type_bf16);
      }
      synTensor residual = external("residual", syn_type_bf16);
      synTensor gamma = external("gamma", syn_type_bf16, true);
      synTensor residual_out = external("residual_out", syn_type_bf16);
      synTensor normalized = external("normalized", syn_type_bf16);
      const float epsilon = 1e-5f;
      synTensor inputs[] = {attention, residual, gamma};
      synTensor outputs[] = {residual_out, normalized};
      check(synNodeCreate(graph, inputs, outputs, 3, 2, &epsilon, sizeof(epsilon),
                          "gk_residual_rmsnorm_bf16_v1", "rmsnorm", nullptr, nullptr));
      std::fprintf(stderr, "compiling %s RMS graph, input type %d\n",
                   fused_ ? "fused" : "direct", static_cast<int>(input_type_));
      check(synGraphCompile(&recipe_, graph, fused_ ? "fused_reduce_rms" : "direct_rms", nullptr));
      check(synWorkspaceGetSize(&workspace_bytes_, recipe_));
      arena_.ensure_workspace(workspace_bytes_);
      gaudirt::runtime::initialize_launch_tensor_ids(recipe_, launch_.data(), launch_.size());
      synGraphDestroy(graph);
      graph = nullptr;
      for (auto& section : sections) { synSectionDestroy(section); section = nullptr; }
    } catch (...) {
      if (recipe_) synRecipeDestroy(recipe_);
      if (graph) synGraphDestroy(graph);
      for (auto section : sections) if (section) synSectionDestroy(section);
      throw;
    }
  }

  ~RmsGraph() { if (recipe_) synRecipeDestroy(recipe_); }

  void execute(uint64_t attention, uint64_t residual, uint64_t gamma,
               uint64_t residual_out, uint64_t normalized, synStreamHandle stream) {
    auto tensors = launch_;
    tensors[0].pTensorAddress = attention;
    tensors[1].pTensorAddress = residual;
    tensors[2].pTensorAddress = gamma;
    tensors[3].pTensorAddress = residual_out;
    tensors[4].pTensorAddress = normalized;
    gaudirt::runtime::PreparedLaunch prepared{recipe_, arena_.ensure_workspace(workspace_bytes_),
                                              std::move(tensors), 0};
    prepared.validate();
    prepared.execute(stream);
  }

 private:
  gaudirt::runtime::HbmArena& arena_;
  bool fused_;
  synDataType input_type_;
  synRecipeHandle recipe_ = nullptr;
  uint64_t workspace_bytes_ = 0;
  std::vector<synLaunchTensorInfo> launch_;
};
}  // namespace

int main() {
  using gaudirt::runtime::DeviceRuntime;
  using gaudirt::runtime::HbmArena;
  DeviceRuntime::Config config{};
  config.module_id = 0;
  DeviceRuntime runtime(config);
  HbmArena arena(runtime.device_id(), 64ULL * 1024 * 1024);
  const uint64_t gathered = arena.allocate(8 * kHidden * sizeof(float));
  const uint64_t reduced = arena.allocate(kHidden * sizeof(uint16_t));
  const uint64_t residual = arena.allocate(kHidden * sizeof(uint16_t));
  const uint64_t gamma = arena.allocate(kHidden * sizeof(uint16_t));
  const uint64_t old_residual = arena.allocate(kHidden * sizeof(uint16_t));
  const uint64_t old_normal = arena.allocate(kHidden * sizeof(uint16_t));
  const uint64_t new_residual = arena.allocate(kHidden * sizeof(uint16_t));
  const uint64_t new_normal = arena.allocate(kHidden * sizeof(uint16_t));
  RmsGraph direct(arena, false, syn_type_bf16);
  std::vector<uint16_t> host_residual(kHidden), host_gamma(kHidden);
  std::vector<uint16_t> actual(kHidden), expected(kHidden);
  for (const auto dtype : {syn_type_bf16, syn_type_float}) {
    gaudirt::runtime::TpReduce reduce(arena, kHidden, dtype);
    RmsGraph fused(arena, true, dtype);
    for (int sample = 0; sample < 4; ++sample) {
      std::vector<float> wide(8 * kHidden);
      std::vector<uint16_t> narrow(8 * kHidden);
      for (size_t i = 0; i < kHidden; ++i) {
        host_residual[i] = bf16(static_cast<float>(static_cast<int>(i % 97) - 48) / 71);
        host_gamma[i] = bf16(0.5f + static_cast<float>(i % 17) / 32);
        for (size_t rank = 0; rank < 8; ++rank) {
          const int v = static_cast<int>((i * 13 + rank * 19 + sample * 7) % 113) - 56;
          wide[rank * kHidden + i] = static_cast<float>(v) / (31 + rank * 7);
          narrow[rank * kHidden + i] = bf16(wide[rank * kHidden + i]);
        }
      }
      arena.copy_h2d_sync(runtime.h2d_stream(), host_residual.data(), residual, kHidden * 2);
      arena.copy_h2d_sync(runtime.h2d_stream(), host_gamma.data(), gamma, kHidden * 2);
      if (dtype == syn_type_bf16)
        arena.copy_h2d_sync(runtime.h2d_stream(), narrow.data(), gathered, narrow.size() * 2);
      else
        arena.copy_h2d_sync(runtime.h2d_stream(), wide.data(), gathered, wide.size() * 4);
      reduce.execute(gathered, reduced, runtime.compute_stream());
      direct.execute(reduced, residual, gamma, old_residual, old_normal, runtime.compute_stream());
      fused.execute(gathered, residual, gamma, new_residual, new_normal, runtime.compute_stream());
      runtime.synchronize_compute();
      for (const auto& pair : {std::array<uint64_t, 2>{old_residual, new_residual},
                               std::array<uint64_t, 2>{old_normal, new_normal}}) {
        arena.copy_d2h_sync(runtime.d2h_stream(), pair[0], expected.data(), kHidden * 2);
        arena.copy_d2h_sync(runtime.d2h_stream(), pair[1], actual.data(), kHidden * 2);
        if (expected != actual)
          throw std::runtime_error("fused reduction and RMSNorm differ bitwise");
      }
    }
  }
  std::printf("FP32 and BF16 TP8 reduction fused with RMSNorm match bitwise.\n");

  // Compile the actual model consumer graphs. These addresses are placeholders;
  // no model graph is launched in this test.
  using gaudirt::runtime::WeightTensor;
  auto weight = [&](const char* name, synDataType dtype, std::vector<uint64_t> shape) {
    WeightTensor tensor;
    tensor.name = name;
    tensor.dtype = dtype;
    tensor.shape = std::move(shape);
    tensor.elements = 1;
    for (auto size : tensor.shape) tensor.elements *= size;
    const uint64_t element_bytes = dtype == syn_type_bf16 ? 2 : dtype == syn_type_float ? 4 : 1;
    tensor.bytes = tensor.elements * element_bytes;
    tensor.device_address = arena.allocate(256);
    return tensor;
  };
  std::vector<WeightTensor> attention_weights;
  attention_weights.push_back(weight("input_layernorm", syn_type_bf16, {6144}));
  attention_weights.push_back(weight("qkv_weight", syn_type_fp8_143, {128, 3392, 48}));
  attention_weights.push_back(weight("qkv_scales", syn_type_float, {48, 27}));
  attention_weights.push_back(weight("qkv_bias", syn_type_float, {3392}));
  attention_weights.push_back(weight("attention_sink", syn_type_bf16, {16}));
  attention_weights.push_back(weight("oproj_weight", syn_type_bf16, {2048, 6144}));
  gaudirt::runtime::WeightManifest attention_manifest(std::move(attention_weights));
  std::vector<WeightTensor> dense_weights;
  dense_weights.push_back(weight("gate_up_weight", syn_type_fp8_143, {128, 4096, 48}));
  dense_weights.push_back(weight("gate_up_scales", syn_type_float, {48, 32}));
  dense_weights.push_back(weight("gate_up_bias", syn_type_float, {4096}));
  dense_weights.push_back(weight("down_weight", syn_type_fp8_143, {128, 6144, 16}));
  dense_weights.push_back(weight("down_scales", syn_type_float, {16, 48}));
  dense_weights.push_back(weight("down_bias", syn_type_float, {6144}));
  gaudirt::runtime::WeightManifest dense_manifest(std::move(dense_weights));
  constexpr uint32_t cache_slots = 205184;
  std::fprintf(stderr, "compiling fused SWA attention\n");
  gaudirt::runtime::MimoAttention swa(arena, attention_manifest, cache_slots, nullptr, true);
  std::fprintf(stderr, "compiling fused full attention\n");
  gaudirt::runtime::MimoFullAttention full(arena, attention_manifest, cache_slots,
                                          false, 64, nullptr, true);
  std::fprintf(stderr, "compiling fused dense\n");
  gaudirt::runtime::MimoDenseMlp dense(arena, dense_manifest, nullptr, true);
  gaudirt::runtime::MimoMoeLayer1::Weights moe{};
  moe.post_attention_gamma_bf16 = arena.allocate(256);
  moe.router_weight_bf16 = arena.allocate(256);
  moe.router_correction_bias_fp32 = arena.allocate(256);
  moe.gate_up_packed_u8 = arena.allocate(256);
  moe.gate_up_scales_u8 = arena.allocate(256);
  moe.down_packed_u8 = arena.allocate(256);
  moe.down_scales_u8 = arena.allocate(256);
  moe.mxfp4_lut_bf16 = arena.allocate(256);
  moe.down_combine_directions_u8 = arena.allocate(256);
  std::fprintf(stderr, "compiling fused SWA MoE\n");
  gaudirt::runtime::MimoMoeLayer1 swa_moe(0, moe, "swa", nullptr, true, syn_type_bf16);
  std::fprintf(stderr, "compiling fused full MoE\n");
  gaudirt::runtime::MimoMoeLayer1 full_moe(0, moe, "full_attention", nullptr, true, syn_type_float);
  std::fprintf(stderr, "compiling fused model head\n");
  gaudirt::mimo::ModelHead head(arena, arena.allocate(256), arena.allocate(256), nullptr, true);
  std::printf("All fused model consumer graphs compiled.\n");
}
