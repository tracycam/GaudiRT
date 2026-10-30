// SPDX-License-Identifier: Apache-2.0
// Run one real-weight SWA rank against a frozen full-model input.
#include "gaudirt/common/sha256.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "mimo_attention.hpp"

#include <array>
#include <cstring>
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
  if (argc != 4) {
    std::cerr << "usage: mimo_swa_output_device_test WEIGHTS RANK_FIXTURE OUTPUT\n";
    return 2;
  }
  try {
    using namespace gaudirt::runtime;
    const auto fixture = std::filesystem::path(argv[2]);
    const auto layer = fixture / "layer1";
    const auto slots = std::filesystem::file_size(layer / "key_cache.bin") / (192 * 2);
    DeviceRuntime::Config config{};
    config.module_id = 0;
    DeviceRuntime runtime(config);
    HbmArena arena(runtime.device_id(), 2ULL * 1024 * 1024 * 1024);
    WeightManifest weights(argv[1], runtime, arena);
    auto upload_bytes = [&](const void* data, size_t size) {
      const auto address = arena.allocate(size);
      arena.copy_h2d_sync(runtime.h2d_stream(), data, address, size);
      return address;
    };
    auto upload = [&](const std::filesystem::path& name) {
      const auto bytes = read(name);
      return upload_bytes(bytes.data(), bytes.size());
    };
    MimoAttentionIo io{};
    io.hidden_states = upload(fixture / "layer0/layer_output.bin");
    io.residual = upload(fixture / "layer0/residual_out.bin");
    io.residual_out = arena.allocate(6144 * 2);
    const auto rope = read(layer / "rope.bin");
    if (rope.size() < 128) throw std::runtime_error("invalid RoPE fixture");
    std::array<uint16_t, 64> row{};
    std::memcpy(row.data(), rope.data(), sizeof(row));
    std::array<uint16_t, 64> cos{}, sin{};
    for (size_t i = 0; i < cos.size(); ++i) {
      cos[i] = row[i % 32];
      sin[i] = row[32 + i % 32];
    }
    io.rope_cos = upload_bytes(cos.data(), sizeof(cos));
    io.rope_sin = upload_bytes(sin.data(), sizeof(sin));
    io.key_cache = upload(layer / "key_cache.bin");
    io.value_cache = upload(layer / "value_cache.bin");
    io.slot_mapping = upload(layer / "slot_mapping.bin");
    const auto page_bytes = read(layer / "window_pages.bin");
    if (page_bytes.size() != 8) throw std::runtime_error("invalid window pages");
    std::array<int32_t, 2> pages{}, groups{};
    std::memcpy(pages.data(), page_bytes.data(), page_bytes.size());
    for (size_t i = 0; i < pages.size(); ++i) groups[i] = pages[i] < 0 ? -1 : 0;
    io.window_pages = upload_bytes(pages.data(), sizeof(pages));
    io.window_groups = upload_bytes(groups.data(), sizeof(groups));
    const int32_t position = 4096;
    io.position = upload_bytes(&position, sizeof(position));
    io.output_local = arena.allocate(6144 * 2);
#ifdef ATTENTION_EXTERNAL_TEST_INTERMEDIATES
    io.normalized = arena.allocate(6144 * 2);
    io.qkv = arena.allocate(3392 * 2);
    io.query = arena.allocate(16 * 192 * 2);
    io.new_key = arena.allocate(192 * 2);
    io.new_value = arena.allocate(128 * 2);
    io.context = arena.allocate(16 * 128 * 2);
#endif
    MimoAttention attention(arena, weights, slots);
    attention.execute(io, runtime.compute_stream());
    runtime.synchronize_compute();
    std::ofstream out(argv[3]);
    auto digest = [&](const char* name, uint64_t address, size_t size) {
      std::vector<uint8_t> bytes(size);
      arena.copy_d2h_sync(runtime.d2h_stream(), address, bytes.data(), bytes.size());
      out << name << ' ' << gaudirt::common::sha256_bytes(bytes.data(), bytes.size()) << '\n';
    };
    digest("attention_local", io.output_local, 6144 * 2);
    digest("residual_out", io.residual_out, 6144 * 2);
    digest("key_cache", io.key_cache, slots * 192 * 2);
    digest("value_cache", io.value_cache, slots * 128 * 2);
    if (!out) throw std::runtime_error("cannot write results");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
