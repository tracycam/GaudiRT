// SPDX-License-Identifier: Apache-2.0
// Compare one real-weight M=32 SWA attention graph against 32 sequential
// M=1 graphs with identical per-row inputs and initial paged KV state.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "layer_c.hpp"
#include <chrono>
#include <thread>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned kRows = 32;
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
  if (argc != 5 && argc != 6) { std::cerr << "usage: TEST WEIGHTS RANK_FIXTURE MODULE HCCL_ID_FILE [--empty]\n"; return 2; }
  try {
    using namespace gaudirt::runtime;
    using namespace gaudirt::mimo;
    const std::filesystem::path fixture(argv[2]), layer = fixture / "layer2";
    auto initial_key = read(layer / "key_cache.bin"), initial_value = read(layer / "value_cache.bin");
    const bool empty = argc == 6;
    if (empty && std::string(argv[5]) != "--empty") throw std::invalid_argument("invalid mode");
    if (empty) { std::fill(initial_key.begin(),initial_key.end(),0); std::fill(initial_value.begin(),initial_value.end(),0); }
    if (initial_key.size() % (192*2) || initial_value.size() % (128*2) ||
        initial_key.size() / (192*2) != initial_value.size() / (128*2))
      throw std::runtime_error("invalid cache geometry");
    const unsigned cache_slots = initial_key.size() / (192*2);
    DeviceRuntime::Config config{};
    const std::string module = argv[3];
    if (module.size() != 1 || module[0] < '0' || module[0] > '7')
      throw std::invalid_argument("module must be 0..7");
    config.module_id = module[0] - '0'; config.rank=config.module_id; config.world_size=8;
    DeviceRuntime runtime(config);
    hcclUniqueId id{}; const std::filesystem::path id_path(argv[4]);
    if(config.rank==0) {
      if(hcclGetUniqueId(&id)!=hcclSuccess) throw std::runtime_error("cannot make HCCL id");
      auto temp=id_path; temp += ".tmp"; std::ofstream out(temp,std::ios::binary);
      out.write(reinterpret_cast<const char*>(&id),sizeof(id)); out.close(); std::filesystem::rename(temp,id_path);
    } else {
      for(unsigned i=0;!std::filesystem::exists(id_path) && i<2400;++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
      std::ifstream in(id_path,std::ios::binary); if(!in.read(reinterpret_cast<char*>(&id),sizeof(id))) throw std::runtime_error("cannot read HCCL id");
    }
    runtime.init_comm(id);
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
    const auto* teacher = std::getenv("GAUDIRT_TEACHER_ROOT");
    auto hidden = repeat(read(teacher ? std::filesystem::path(teacher) / "layer01/output.bf16" : fixture / "layer1/layer_output.bin"));
    auto residual = repeat(read(teacher ? std::filesystem::path(teacher) / "layer01/residual.bf16" : fixture / "layer1/residual_out.bin"));
    const auto* prompt_rows = std::getenv("GAUDIRT_PROMPT_ROWS");
    if(prompt_rows) { if(!empty) throw std::invalid_argument("prompt rows require empty KV"); hidden=read(std::filesystem::path(prompt_rows)/"embedding.bf16"); residual=hidden; }
    if(const auto* layer_rows = std::getenv("GAUDIRT_LAYER_ROWS")) {
      if(!prompt_rows) throw std::invalid_argument("layer rows require prompt RoPE");
      hidden=read(std::filesystem::path(layer_rows)/"hidden-rows.bf16");
      residual=read(std::filesystem::path(layer_rows)/"residual-rows.bf16");
    }
    if (hidden.size() != 6144*2*kRows || residual.size() != hidden.size())
      throw std::runtime_error("invalid hidden or residual fixture");
    const auto rope = read(prompt_rows ? std::filesystem::path(prompt_rows)/"swa-rope.bf16" : fixture / "layer1/rope.bin");
    if (rope.size() < kRows*64*2) throw std::runtime_error("invalid rope fixture");
    std::vector<uint16_t> cos(kRows*64), sin(kRows*64);
    for (unsigned r=0; r<kRows; ++r) for (unsigned i=0; i<64; ++i) {
      std::memcpy(&cos[r*64+i], rope.data()+(r*64+i%32)*2,2);
      std::memcpy(&sin[r*64+i], rope.data()+(r*64+32+i%32)*2,2);
    }
    if(empty && !prompt_rows) { std::fill(cos.begin(),cos.end(),0x3f80); std::fill(sin.begin(),sin.end(),0); }
    const auto slot_bytes = read(layer / "slot_mapping.bin");
    const auto page_bytes = read(layer / "window_pages.bin");
    if (slot_bytes.size()!=4 || page_bytes.size()!=8) throw std::runtime_error("invalid page fixture");
    int32_t first; std::array<int32_t,2> pages1{},groups1{0,0};
    std::memcpy(&first,slot_bytes.data(),4); std::memcpy(pages1.data(),page_bytes.data(),8);
    if (empty) { first=0; pages1={0,-1}; groups1={0,-1}; }
    if (first<0 || unsigned(first)+kRows>cache_slots || first%128 || (!empty && pages1[1]*128!=first))
      throw std::runtime_error("invalid starting decode page");
    std::array<int32_t,kRows> slots{},positions{};
    std::array<int32_t,kRows*2> pages{},groups{};
    for (unsigned r=0;r<kRows;++r) {
      slots[r]=first+r; positions[r]=(empty?0:4096)+r;
      pages[2*r]=pages1[0]; pages[2*r+1]=pages1[1];
      groups[2*r]=r; groups[2*r+1]=empty ? -1 : static_cast<int32_t>(r);
    }
    std::vector<float> gathered_hidden(8*kRows*6144);
    for(unsigned rank=0;rank<8;++rank) for(unsigned r=0;r<kRows;++r) for(unsigned c=0;c<6144;++c) {
      uint16_t bits; std::memcpy(&bits,hidden.data()+(r*6144+c)*2,2);
      gathered_hidden[(rank*kRows+r)*6144+c]=widen(bits)/8.0f;
    }
    std::array<uint64_t,kRows> gathered_single{};
    for(unsigned r=0;r<kRows;++r) {
      std::vector<float> row(8*6144);
      for(unsigned rank=0;rank<8;++rank) std::copy_n(gathered_hidden.data()+(rank*kRows+r)*6144,6144,row.data()+rank*6144);
      gathered_single[r]=upload(row.data(),row.size()*4);
    }
    MimoAttentionIo batched{};
    batched.hidden_states=upload(gathered_hidden.data(),gathered_hidden.size()*4); batched.residual=upload(residual.data(),residual.size());
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
    std::array<uint64_t,kRows> serial_cos{},serial_sin{},serial_slots{},serial_positions{};
    for (unsigned row=0;row<kRows;++row) {
      serial_cos[row]=upload(cos.data()+row*64,64*2);
      serial_sin[row]=upload(sin.data()+row*64,64*2);
      serial_slots[row]=upload(&slots[row],sizeof(slots[row]));
      serial_positions[row]=upload(&positions[row],sizeof(positions[row]));
    }
    gaudirt::mimo::LayerC one(arena,runtime,weights,cache_slots,nullptr,true);
    gaudirt::mimo::LayerC many(arena,runtime,weights,cache_slots,nullptr,true,kRows);
    const auto make_io=[&](MimoAttentionIo io,unsigned rows) {
      gaudirt::mimo::LayerC::Io layer_io{}; layer_io.attention=io;
      layer_io.attention_gathered_bf16=arena.allocate(8ULL*rows*6144*2);
      layer_io.moe_outputs.residual_out_bf16=arena.allocate(rows*6144*2);
      layer_io.moe_outputs.moe_local_fp32=arena.allocate(rows*6144*4);
      layer_io.moe_gathered_fp32=arena.allocate(8ULL*rows*6144*4);
      return layer_io;
    };
    auto batch_io=make_io(batched,kRows);
    auto serial_io=make_io(serial,1);
    std::vector<float> expected(8*kRows*6144),actual(expected.size());
    std::vector<float> one_result(8*6144);
    for (unsigned row=0;row<kRows;++row) {
      serial_io.attention.hidden_states=gathered_single[row]; serial_io.attention.residual=batched.residual+row*6144*2;
      serial_io.attention.residual_out=serial_residual+row*6144*2;
      serial_io.attention.rope_cos=serial_cos[row]; serial_io.attention.rope_sin=serial_sin[row];
      serial_io.attention.slot_mapping=serial_slots[row]; serial_io.attention.position=serial_positions[row];
      auto launch=one.prepare(serial_io); launch.execute_fused(runtime.compute_stream()); runtime.synchronize_compute();
      arena.copy_d2h_sync(runtime.d2h_stream(),serial_io.moe_gathered_fp32,one_result.data(),one_result.size()*4);
      for(unsigned rank=0;rank<8;++rank) std::copy_n(one_result.data()+rank*6144,6144,expected.data()+(rank*kRows+row)*6144);
    }
    auto launch=many.prepare(batch_io); launch.execute_fused(runtime.compute_stream()); runtime.synchronize_compute();
    arena.copy_d2h_sync(runtime.d2h_stream(),batch_io.moe_gathered_fp32,actual.data(),actual.size()*4);
    double max_abs=0,max_row_relative_l2=0;
    for (unsigned row=0;row<8*kRows;++row) {
      double diff=0,norm=0;
      for (unsigned i=0;i<6144;++i) {
        const double a=actual[row*6144+i], b=expected[row*6144+i];
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
    std::cout << "M32 TP8 real-weight SWA layer vs sequential M1: max_abs=" << max_abs
              << " max_row_relative_l2=" << max_row_relative_l2 << " KV=bitwise empty=" << empty << "\n";
    if (max_row_relative_l2>0.003) throw std::runtime_error("attention row exceeds quick-screen tolerance");
    if(const char* output_dir=std::getenv("GAUDIRT_TEST_OUTPUT_DIR")) {
      const auto prefix=std::filesystem::path(output_dir)/(std::string("rank")+module+"-C");
      std::ofstream partial(prefix.string()+"-partial.f32",std::ios::binary);
      for(unsigned rank=0;rank<8;++rank) partial.write(reinterpret_cast<const char*>(actual.data()+rank*kRows*6144),6144*4);
      std::vector<uint16_t> residual_row(6144);
      arena.copy_d2h_sync(runtime.d2h_stream(),batch_io.moe_outputs.residual_out_bf16,residual_row.data(),residual_row.size()*2);
      std::ofstream residual_file(prefix.string()+"-residual.bf16",std::ios::binary);
      residual_file.write(reinterpret_cast<const char*>(residual_row.data()),residual_row.size()*2);
      if(!partial || !residual_file) throw std::runtime_error("cannot write layer comparison output");
    }
    float marker=1.0f; const auto barrier=upload(&marker,sizeof(marker));
    if(hcclAllReduce(reinterpret_cast<const void*>(barrier),reinterpret_cast<void*>(barrier),1,hcclFloat32,hcclSum,runtime.comm(),runtime.compute_stream())!=hcclSuccess) throw std::runtime_error("barrier failed");
    runtime.synchronize_compute();
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
