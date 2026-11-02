// SPDX-License-Identifier: Apache-2.0
// Execute the grouped graph with one analytically checkable shared weight set.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "mimo_moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main() {
  using namespace gaudirt::runtime;
    using namespace gaudirt::mimo;
  constexpr uint32_t rows = 512, hidden = 6144;
  DeviceRuntime::Config config{};
  DeviceRuntime runtime(config);
  HbmArena arena(runtime.device_id(), 4ULL * 1024 * 1024 * 1024);
  auto upload = [&](const void* data, size_t bytes) {
    auto address = arena.allocate(bytes);
    arena.copy_h2d_sync(runtime.h2d_stream(), data, address, bytes);
    return address;
  };
  auto filled = [&](size_t bytes, uint8_t value) {
    std::vector<uint8_t> data(bytes, value);
    return upload(data.data(), data.size());
  };
  MimoMoeLayer1::Weights w{};
  std::vector<uint16_t> gamma(hidden, 0x3f80);
  std::fill(gamma.begin() + hidden / 2, gamma.end(), 0x3f00);
  w.post_attention_gamma_bf16 = upload(gamma.data(), gamma.size() * 2);
  w.router_weight_bf16 = filled(MimoMoeLayer1::kRouterWeightBf16Bytes, 0);
  w.router_correction_bias_fp32 = filled(MimoMoeLayer1::kRouterBiasFp32Bytes, 0);
  // Each nibble decodes to +0.5 and E8M0=120 scales it to 1/256.
  w.gate_up_packed_u8 = filled(MimoMoeLayer1::kGateUpPackedWeightBytes, 0x11);
  w.gate_up_scales_u8 = filled(MimoMoeLayer1::kGateUpScaleBytes, 120);
  w.down_packed_u8 = filled(MimoMoeLayer1::kDownPackedWeightBytes, 0x11);
  w.down_scales_u8 = filled(MimoMoeLayer1::kDownScaleBytes, 120);
  auto lut = MimoMoeLayer1::make_mxfp4_lut_bf16();
  auto directions = MimoMoeLayer1::make_down_combine_directions();
  w.mxfp4_lut_bf16 = upload(lut.data(), sizeof(lut));
  w.down_combine_directions_u8 = upload(directions.data(), sizeof(directions));
  std::vector<uint16_t> attention(rows * hidden, 0), residual(rows * hidden, 0x3f80);
  MimoMoeLayer1::Inputs in{upload(attention.data(), attention.size() * 2),
                           upload(residual.data(), residual.size() * 2)};
  MimoMoeLayer1::Outputs out{arena.allocate(residual.size() * 2),
                             arena.allocate(residual.size() * sizeof(float))};
  MimoMoeLayer1 model(0, w, "grouped_math_test", nullptr, false, syn_type_float, rows);
  model.launch(runtime.compute_stream(), in, out, arena.ensure_workspace(model.workspace_bytes()));
  runtime.synchronize_compute();
  std::vector<float> actual(rows * hidden);
  std::vector<uint16_t> actual_residual(rows * hidden);
  arena.copy_d2h_sync(runtime.d2h_stream(), out.moe_local_fp32, actual.data(), actual.size() * 4);
  arena.copy_d2h_sync(runtime.d2h_stream(), out.residual_out_bf16, actual_residual.data(), actual_residual.size() * 2);
  // Normalized gamma values are exactly representable in BF16. Each expert
  // therefore has gate=up=(3072*1 + 3072*0.5)/256 = 18. All eight experts
  // have equal weights, so routing normalization leaves the output unchanged.
  const double gate = 18.0;
  const double reference = gate * gate / (1.0 + std::exp(-gate));
  double max_abs = 0.0;
  for (size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]) || actual_residual[i] != residual[i])
      throw std::runtime_error("non-finite output or residual mismatch");
    max_abs = std::max(max_abs, std::abs(actual[i] - reference));
  }
  std::printf("M512 grouped MoE vs FP64: reference=%.9g max_abs=%.9g relative=%.9g\n",
              reference, max_abs, max_abs / reference);
  if (max_abs / reference > 0.003) throw std::runtime_error("grouped MoE exceeds FP64 error gate");
}
