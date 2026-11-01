// SPDX-License-Identifier: Apache-2.0
// Compare one real-weight M=512 SWA attention graph against 512 sequential
// M=1 graphs with identical per-row inputs and initial paged KV state.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "mimo_attention.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned kRows = 512;
std::vector<uint8_t> read(const std::filesystem::path& path) {
  std::vector<uint8_t> data(std::filesystem::file_size(path));
  std::ifstream in(path, std::ios::binary);
  if (!in.read(reinterpret_cast<char*>(data.data()), data.size())) throw std::runtime_error("cannot read fixture");
  return data;
}
float widen(uint16_t value) {
  const uint32_t bits = uint32_t{value} << 16;
  float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}
}

int main(int argc, char** argv) {
  if (argc != 4) { std::cerr << "usage: TEST WEIGHTS RANK_FIXTURE MODULE\n"; return 2; }
  try {
    using namespace gaudirt::runtime;
    const std::filesystem::path fixture(argv[2]), layer = fixture / "layer2";
    const auto initial_key = read(layer / "key_cache.bin"), initial_value = read(layer / "value_cache.bin");
    if (initial_key.size() % (192*2) || initial_value.size() % (128*2) ||
        initial_key.size() / (192*2) != initial_value.size() / (128*2))
      throw std::runtime_error("invalid cache geometry");
    const unsigned cache_slots = initial_key.size() / (192*2);
    DeviceRuntime::Config config{};
    const std::string module = argv[3];
    if (module.size() != 1 || module[0] < '0' || module[0] > '7')
      throw std::invalid_argument("module must be 0..7");
    config.module_id = module[0] - '0';
    DeviceRuntime runtime(config);
    HbmArena arena(runtime.device_id(), 2ULL * 1024 * 1024 * 1024);
    WeightManifest weights(argv[1], runtime, arena);
    const auto upload = [&](const void* data, size_t bytes) {
      const auto address = arena.allocate(bytes);
      arena.copy_h2d_sync(runtime.h2d_stream(), data, address, bytes);
      return address;
    };
    const auto repeat = [&](const std::vector<uint8_t>& row) {
      std::vector<uint8_t> data(row.size()*kRows);
      for (unsigned i=0; i<kRows; ++i) std::copy(row.begin(),row.end(),data.begin()+i*row.size());
      return data;
    };
    const auto hidden = repeat(read(fixture / "layer1/layer_output.bin"));
    const auto residual = repeat(read(fixture / "layer1/residual_out.bin"));
    if (hidden.size() != 6144*2*kRows || residual.size() != hidden.size())
      throw std::runtime_error("invalid hidden or residual fixture");
    const auto rope = read(fixture / "layer1/rope.bin");
    if (rope.size() < 128*64*2) throw std::runtime_error("invalid rope fixture");
    std::vector<uint16_t> cos(kRows*64), sin(kRows*64);
    for (unsigned r=0; r<kRows; ++r) for (unsigned i=0; i<64; ++i) {
      std::memcpy(&cos[r*64+i], rope.data()+((r%128)*64+i%32)*2,2);
      std::memcpy(&sin[r*64+i], rope.data()+((r%128)*64+32+i%32)*2,2);
    }
    const auto slot_bytes = read(layer / "slot_mapping.bin");
    const auto page_bytes = read(layer / "window_pages.bin");
    if (slot_bytes.size()!=4 || page_bytes.size()!=8) throw std::runtime_error("invalid page fixture");
    int32_t first; std::array<int32_t,2> pages1{},groups1{0,0};
    std::memcpy(&first,slot_bytes.data(),4); std::memcpy(pages1.data(),page_bytes.data(),8);
    if (first<0 || unsigned(first)+kRows>cache_slots || first%128 || pages1[1]*128!=first)
      throw std::runtime_error("invalid starting decode page");
    std::array<int32_t,kRows> slots{},positions{};
    std::array<int32_t,kRows*2> pages{},groups{};
    for (unsigned r=0;r<kRows;++r) {
      slots[r]=first+r; positions[r]=4096+r;
      const int pos=4096+r, first_page=(pos-127)/128, last_page=pos/128;
      pages[2*r]=34+first_page; groups[2*r]=r;
      pages[2*r+1]=first_page==last_page?-1:34+last_page; groups[2*r+1]=first_page==last_page?-1:r;
    }
    MimoAttentionIo batched{};
    batched.hidden_states=upload(hidden.data(),hidden.size()); batched.residual=upload(residual.data(),residual.size());
    batched.residual_out=arena.allocate(hidden.size()); batched.rope_cos=upload(cos.data(),cos.size()*2);
    batched.rope_sin=upload(sin.data(),sin.size()*2);
    batched.key_cache=upload(initial_key.data(),initial_key.size()); batched.value_cache=upload(initial_value.data(),initial_value.size());
    batched.slot_mapping=upload(slots.data(),sizeof(slots)); batched.position=upload(positions.data(),sizeof(positions));
    batched.window_pages=upload(pages.data(),sizeof(pages)); batched.window_groups=upload(groups.data(),sizeof(groups));
    batched.output_local=arena.allocate(hidden.size());
    MimoAttentionIo serial=batched;
    serial.key_cache=upload(initial_key.data(),initial_key.size()); serial.value_cache=upload(initial_value.data(),initial_value.size());
    serial.window_pages=upload(pages1.data(),sizeof(pages1)); serial.window_groups=upload(groups1.data(),sizeof(groups1));
    const uint64_t serial_output=arena.allocate(hidden.size()), serial_residual=arena.allocate(hidden.size());
    std::array<uint64_t,kRows> serial_cos{},serial_sin{},serial_slots{},serial_positions{},serial_pages{},serial_groups{};
    for (unsigned row=0;row<kRows;++row) {
      serial_cos[row]=upload(cos.data()+row*64,64*2);
      serial_sin[row]=upload(sin.data()+row*64,64*2);
      serial_slots[row]=upload(&slots[row],sizeof(slots[row]));
      serial_positions[row]=upload(&positions[row],sizeof(positions[row]));
      std::array<int32_t,2> row_pages{pages[row*2],pages[row*2+1]},row_groups{0,pages[row*2+1]<0?-1:0};
      serial_pages[row]=upload(row_pages.data(),sizeof(row_pages)); serial_groups[row]=upload(row_groups.data(),sizeof(row_groups));
    }
    MimoAttention one(arena,weights,cache_slots);
    MimoAttention many(arena,weights,cache_slots,nullptr,false,kRows);
    for (unsigned row=0;row<kRows;++row) {
      serial.hidden_states=batched.hidden_states+row*6144*2; serial.residual=batched.residual+row*6144*2;
      serial.residual_out=serial_residual+row*6144*2; serial.output_local=serial_output+row*6144*2;
      serial.rope_cos=serial_cos[row]; serial.rope_sin=serial_sin[row];
      serial.slot_mapping=serial_slots[row]; serial.position=serial_positions[row];
      serial.window_pages=serial_pages[row]; serial.window_groups=serial_groups[row];
      one.execute(serial,runtime.compute_stream());
    }
    many.execute(batched,runtime.compute_stream()); runtime.synchronize_compute();
    std::vector<uint16_t> expected(6144*kRows),actual(expected.size());
    arena.copy_d2h_sync(runtime.d2h_stream(),serial_output,expected.data(),expected.size()*2);
    arena.copy_d2h_sync(runtime.d2h_stream(),batched.output_local,actual.data(),actual.size()*2);
    double max_abs=0,max_row_relative_l2=0;
    for (unsigned row=0;row<kRows;++row) {
      double diff=0,norm=0;
      for (unsigned i=0;i<6144;++i) {
        const double a=widen(actual[row*6144+i]), b=widen(expected[row*6144+i]);
        if (!std::isfinite(a) || !std::isfinite(b)) throw std::runtime_error("non-finite attention output");
        diff+=(a-b)*(a-b); norm+=b*b; max_abs=std::max(max_abs,std::abs(a-b));
      }
      max_row_relative_l2=std::max(max_row_relative_l2,std::sqrt(diff/std::max(norm,1e-30)));
    }
    const auto verify_kv = [&](uint64_t a, uint64_t b, const std::vector<uint8_t>& original, size_t width) {
      std::vector<uint8_t> x(original.size()),y(original.size());
      arena.copy_d2h_sync(runtime.d2h_stream(),a,x.data(),x.size());
      arena.copy_d2h_sync(runtime.d2h_stream(),b,y.data(),y.size());
      const size_t begin=size_t(first)*width, end=(size_t(first)+kRows)*width;
      if (!std::equal(x.begin(),x.begin()+begin,original.begin()) ||
          !std::equal(x.begin()+end,x.end(),original.begin()+end) ||
          !std::equal(y.begin(),y.begin()+begin,original.begin()) ||
          !std::equal(y.begin()+end,y.end(),original.begin()+end))
        throw std::runtime_error("untouched cache changed");
      if (!std::equal(x.begin()+begin,x.begin()+end,y.begin()+begin))
        throw std::runtime_error("batched and sequential written KV differ");
    };
    verify_kv(batched.key_cache,serial.key_cache,initial_key,192*2);
    verify_kv(batched.value_cache,serial.value_cache,initial_value,128*2);
    std::cout << "M512 real-weight SWA vs sequential M1: max_abs=" << max_abs
              << " max_row_relative_l2=" << max_row_relative_l2 << " KV=bitwise\n";
    if (max_row_relative_l2>0.003) throw std::runtime_error("attention row exceeds quick-screen tolerance");
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
