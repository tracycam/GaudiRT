// SPDX-License-Identifier: Apache-2.0
// Compare the graph's local MoE output against a frozen real-weight fixture.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "mimo_moe.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
std::vector<uint8_t> read(const std::filesystem::path& path) {
  std::vector<uint8_t> bytes(std::filesystem::file_size(path));
  std::ifstream in(path, std::ios::binary);
  if (!in.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
    throw std::runtime_error("cannot read fixture");
  return bytes;
}
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: mimo_moe_output_device_test WEIGHT_MANIFEST_DIR FIXTURE_DIR\n";
    return 2;
  }
  try {
    using namespace gaudirt::runtime;
    DeviceRuntime::Config config{};
    config.module_id = 0;
    DeviceRuntime runtime(config);
    HbmArena arena(runtime.device_id(), 2ULL * 1024 * 1024 * 1024);
    WeightManifest weights(argv[1], runtime, arena);
    MimoMoeLayer1::Weights w{};
    w.post_attention_gamma_bf16 = weights.get("post_attention_layernorm").device_address;
    w.router_weight_bf16 = weights.get("router_weight").device_address;
    w.router_correction_bias_fp32 = weights.get("router_bias").device_address;
    w.gate_up_packed_u8 = weights.get("moe_gp").device_address;
    w.gate_up_scales_u8 = weights.get("moe_gs").device_address;
    w.down_packed_u8 = weights.get("moe_dp").device_address;
    w.down_scales_u8 = weights.get("moe_ds").device_address;
    const auto lut = MimoMoeLayer1::make_mxfp4_lut_bf16();
    w.mxfp4_lut_bf16 = arena.allocate(sizeof(lut));
    arena.copy_h2d_sync(runtime.h2d_stream(), lut.data(), w.mxfp4_lut_bf16, sizeof(lut));
    const auto directions = MimoMoeLayer1::make_down_combine_directions();
    w.down_combine_directions_u8 = arena.allocate(sizeof(directions));
    arena.copy_h2d_sync(runtime.h2d_stream(), directions.data(),
                        w.down_combine_directions_u8, sizeof(directions));
    const auto fixture = std::filesystem::path(argv[2]);
    auto upload = [&](const char* name) {
      const auto data = read(fixture / name);
      const auto address = arena.allocate(data.size());
      arena.copy_h2d_sync(runtime.h2d_stream(), data.data(), address, data.size());
      return address;
    };
    MimoMoeLayer1::Inputs inputs{upload("attention.bin"), upload("residual_in.bin")};
    MimoMoeLayer1::Outputs outputs{arena.allocate(6144 * 2), arena.allocate(6144 * 4)};
    MimoMoeLayer1 moe(0, w, "swa");
    moe.launch(runtime.compute_stream(), inputs, outputs, arena.ensure_workspace(moe.workspace_bytes()));
    runtime.synchronize_compute();
    const auto expected = read(fixture / "moe_local.bin");
    std::vector<uint8_t> actual(expected.size());
    arena.copy_d2h_sync(runtime.d2h_stream(), outputs.moe_local_fp32, actual.data(), actual.size());
    if (actual != expected) throw std::runtime_error("real-weight MoE local output differs");
    std::cout << "real-weight MoE local output matches bitwise\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
