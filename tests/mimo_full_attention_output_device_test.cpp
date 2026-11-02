// SPDX-License-Identifier: Apache-2.0
// Run one real-weight Full Attention rank and save its output and KV hashes.
#include "gaudirt/common/sha256.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "mimo_full_attention.hpp"

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
    std::cerr << "usage: mimo_full_attention_output_device_test WEIGHTS FIXTURE OUTPUT\n";
    return 2;
  }
  try {
    using namespace gaudirt::runtime;
    using namespace gaudirt::mimo;
    const auto fixture = std::filesystem::path(argv[2]);
    const auto slots = std::filesystem::file_size(fixture / "key_cache_before.bin") / (192 * 2);
    const auto pages = std::filesystem::file_size(fixture / "block_list.bin") / 4;
    const auto slot_bytes = read(fixture / "slot_mapping.bin");
    if (slot_bytes.size() != sizeof(int32_t)) throw std::runtime_error("invalid slot fixture");
    int32_t slot = 0;
    std::memcpy(&slot, slot_bytes.data(), sizeof(slot));
    if (slot < 0 || static_cast<size_t>(slot) >= slots)
      throw std::runtime_error("slot outside KV cache");
    DeviceRuntime::Config config{};
    config.module_id = 0;
    DeviceRuntime runtime(config);
    HbmArena arena(runtime.device_id(), 512ULL * 1024 * 1024);
    WeightManifest weights(argv[1], runtime, arena);
    auto upload = [&](const char* name) {
      const auto bytes = read(fixture / name);
      const auto address = arena.allocate(bytes.size());
      arena.copy_h2d_sync(runtime.h2d_stream(), bytes.data(), address, bytes.size());
      return address;
    };
    MimoAttentionIo io{};
    io.hidden_states = upload("residual_in.bin");
    io.residual = io.hidden_states;
    io.residual_out = arena.allocate(6144 * 2);
    io.rope_cos = upload("rope_cos.bin");
    io.rope_sin = upload("rope_sin.bin");
    io.key_cache = upload("key_cache_before.bin");
    io.value_cache = upload("value_cache_before.bin");
    io.slot_mapping = upload("slot_mapping.bin");
    io.full_block_list = upload("block_list.bin");
    io.full_block_usage = upload("block_usage.bin");
    io.full_block_groups = upload("block_groups.bin");
    io.output_local_fp32 = arena.allocate(6144 * 4);
#ifdef ATTENTION_EXTERNAL_TEST_INTERMEDIATES
    // Only the reference test binary built against the preceding graph needs these.
    io.normalized = arena.allocate(6144 * 2);
    io.qkv = arena.allocate(3392 * 2);
    io.query = arena.allocate(16 * 192 * 2);
    io.new_key = arena.allocate(192 * 2);
    io.new_value = arena.allocate(128 * 2);
    io.context = arena.allocate(16 * 128 * 2);
#endif
    MimoFullAttention attention(arena, weights, slots, true, pages);
    attention.execute(io, runtime.compute_stream());
    runtime.synchronize_compute();
    std::ofstream out(argv[3]);
    auto digest = [&](const char* name, uint64_t address, size_t size) {
      std::vector<uint8_t> bytes(size);
      arena.copy_d2h_sync(runtime.d2h_stream(), address, bytes.data(), bytes.size());
      if (std::string(name) == "attention_local") {
        std::ofstream values(std::string(argv[3]) + ".attention.bin", std::ios::binary);
        values.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        if (!values) throw std::runtime_error("cannot write attention output");
      }
      out << name << ' ' << gaudirt::common::sha256_bytes(bytes.data(), bytes.size()) << '\n';
    };
    auto check_untouched = [&](const char* label, const char* filename,
                               uint64_t address, size_t bytes_per_slot) {
      const auto before = read(fixture / filename);
      std::vector<uint8_t> after(before.size());
      arena.copy_d2h_sync(runtime.d2h_stream(), address, after.data(), after.size());
      const size_t offset = static_cast<size_t>(slot) * bytes_per_slot;
      if (offset + bytes_per_slot > before.size() ||
          std::memcmp(before.data(), after.data(), offset) != 0 ||
          std::memcmp(before.data() + offset + bytes_per_slot,
                      after.data() + offset + bytes_per_slot,
                      before.size() - offset - bytes_per_slot) != 0)
        throw std::runtime_error(std::string(label) + " touched another KV slot");
      std::ofstream values(std::string(argv[3]) + "." + label + ".bin", std::ios::binary);
      values.write(reinterpret_cast<const char*>(after.data() + offset), bytes_per_slot);
      if (!values) throw std::runtime_error("cannot write updated KV slot");
      out << label << "_untouched identical\n";
    };
    digest("attention_local", io.output_local_fp32, 6144 * 4);
    digest("key_cache", io.key_cache, slots * 192 * 2);
    digest("value_cache", io.value_cache, slots * 128 * 2);
    check_untouched("key", "key_cache_before.bin", io.key_cache, 192 * 2);
    check_untouched("value", "value_cache_before.bin", io.value_cache, 128 * 2);
    if (!out) throw std::runtime_error("cannot write results");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
