// SPDX-License-Identifier: Apache-2.0
// Compare four real dense prefill layers using separate residual buffers and
// two alternating buffers. Each layer reads the previous layer's residual.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "mimo_dense_mlp.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  try {
    using namespace gaudirt::runtime;
    using namespace gaudirt::mimo;
    DeviceRuntime runtime(DeviceRuntime::Config{});
    HbmArena arena(runtime.device_id(), 2ULL * 1024 * 1024 * 1024);
    WeightManifest weights(argv[1], runtime, arena);
    constexpr size_t rows = 32, hidden = 6144, bytes = rows * hidden * 2;
    auto read = [&](const char* name) {
      const auto path = std::filesystem::path(argv[2]) / "layer0" / name;
      std::vector<uint8_t> row(std::filesystem::file_size(path)), all(bytes);
      std::ifstream in(path, std::ios::binary);
      if (row.size() != hidden * 2 || !in.read(reinterpret_cast<char*>(row.data()), row.size()))
        throw std::runtime_error("invalid fixture");
      for (size_t r = 0; r < rows; ++r) std::copy(row.begin(), row.end(), all.begin() + r * row.size());
      return all;
    };
    auto attention = read("layer_output.bin"), residual = read("residual_out.bin");
    auto upload = [&](const void* data, size_t size) {
      const auto addr = arena.allocate(size); arena.copy_h2d_sync(runtime.h2d_stream(), data, addr, size); return addr;
    };
    MimoDenseMlp::Inputs input{};
    input.attention_bf16 = upload(attention.data(), bytes);
    input.residual_in_bf16 = upload(residual.data(), bytes);
    input.post_attention_gamma_bf16 = weights.get("post_attention_gamma").device_address;
    MimoDenseMlp graph(arena, weights, nullptr, false, rows);
    std::array<uint64_t, 4> separate{}; for (auto& addr : separate) addr = arena.allocate(bytes);
    std::array<uint64_t, 2> shared{}; for (auto& addr : shared) addr = arena.allocate(bytes);
    MimoDenseMlp::Outputs output{};
    output.activation_fp32 = arena.allocate(rows * 2048 * 4);
    const auto reference_output = arena.allocate(rows * hidden * 4);
    const auto reused_output = arena.allocate(rows * hidden * 4);
    const auto initial = input.residual_in_bf16;
    for (int path = 0; path < 2; ++path) {
      input.residual_in_bf16 = initial;
      output.output_local_fp32 = path ? reused_output : reference_output;
      for (size_t layer = 0; layer < 4; ++layer) {
        output.residual_out_bf16 = path ? shared[layer % 2] : separate[layer];
        graph.execute(input, output, runtime.compute_stream());
        input.residual_in_bf16 = output.residual_out_bf16;
      }
    }
    runtime.synchronize_compute();
    std::vector<uint8_t> a(bytes), b(bytes);
    for (size_t layer : {size_t(2), size_t(3)}) {
      arena.copy_d2h_sync(runtime.d2h_stream(), separate[layer], a.data(), bytes);
      arena.copy_d2h_sync(runtime.d2h_stream(), shared[layer % 2], b.data(), bytes);
      if (a != b) throw std::runtime_error("reused residual differs");
    }
    a.resize(rows * hidden * 4); b.resize(a.size());
    arena.copy_d2h_sync(runtime.d2h_stream(), reference_output, a.data(), a.size());
    arena.copy_d2h_sync(runtime.d2h_stream(), reused_output, b.data(), b.size());
    if (a != b) throw std::runtime_error("output differs after residual reuse");
    std::cout << "Four dense prefill layers: alternating residuals and output match bitwise\n";
    return 0;
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
