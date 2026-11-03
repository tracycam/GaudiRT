// SPDX-License-Identifier: Apache-2.0
// Compile the shared-weight grouped graph without allocating model weights.
#include "gaudirt/runtime/device_runtime.hpp"
#include "mimo_moe.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) throw std::invalid_argument("expected static rows and optional gathered-bf16");
  const bool gathered_bf16 = argc == 3 && std::string(argv[2]) == "gathered-bf16";
  if (argc == 3 && !gathered_bf16) throw std::invalid_argument("unknown attention mode");
  const auto rows = static_cast<uint32_t>(std::strtoul(argv[1], nullptr, 10));
  gaudirt::runtime::DeviceRuntime::Config config{};
  gaudirt::runtime::DeviceRuntime runtime(config);
  gaudirt::mimo::MimoMoeLayer1::Weights w{};
  w.post_attention_gamma_bf16 = 0x1000;
  w.router_weight_bf16 = 0x2000;
  w.router_correction_bias_fp32 = 0x3000;
  w.gate_up_packed_u8 = 0x4000;
  w.gate_up_scales_u8 = 0x5000;
  w.down_packed_u8 = 0x6000;
  w.down_scales_u8 = 0x7000;
  w.mxfp4_lut_bf16 = 0x8000;
  w.down_combine_directions_u8 = 0x9000;
  gaudirt::mimo::MimoMoeLayer1 model(0, w, "grouped_test", nullptr, gathered_bf16,
                                    gathered_bf16 ? syn_type_bf16 : syn_type_float, rows);
  std::printf("grouped M=%u graph compiled, workspace=%llu bytes\n", rows,
              static_cast<unsigned long long>(model.workspace_bytes()));
}
