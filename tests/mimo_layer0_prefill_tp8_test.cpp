// SPDX-License-Identifier: Apache-2.0
// Compare M=32 full attention with 32 sequential M=1 graphs on frozen inputs.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "layer0.hpp"
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
constexpr unsigned kRows=32,kPages=64;
std::vector<uint8_t> read(const std::filesystem::path& path) {
  std::vector<uint8_t> data(std::filesystem::file_size(path));
  std::ifstream in(path,std::ios::binary);
  if(!in.read(reinterpret_cast<char*>(data.data()),data.size())) throw std::runtime_error("cannot read fixture");
  return data;
}
}
int main(int argc,char** argv) {
  if(argc!=6 && argc!=7) { std::cerr << "usage: TEST WEIGHTS DENSE_WEIGHTS RANK_FIXTURE MODULE HCCL_ID_FILE [--empty]\n"; return 2; }
  try {
    using namespace gaudirt::runtime;
    const std::filesystem::path fixture(argv[3]),layer=fixture/"layer0";
    auto initial_key=read(layer/"key_cache.bin"),initial_value=read(layer/"value_cache.bin");
    const bool empty=argc==7; if(empty && std::string(argv[6])!="--empty") throw std::invalid_argument("invalid mode");
    if(empty) { std::fill(initial_key.begin(),initial_key.end(),0); std::fill(initial_value.begin(),initial_value.end(),0); }
    if(initial_key.size()%(192*2) || initial_value.size()%(128*2) ||
       initial_key.size()/(192*2)!=initial_value.size()/(128*2)) throw std::runtime_error("invalid cache geometry");
    const unsigned cache_slots=initial_key.size()/(192*2);
    const std::string module=argv[4]; if(module.size()!=1 || module[0]<'0' || module[0]>'7') throw std::invalid_argument("module must be 0..7");
    DeviceRuntime::Config cfg{}; cfg.module_id=module[0]-'0'; cfg.rank=cfg.module_id; cfg.world_size=8; DeviceRuntime runtime(cfg);
    hcclUniqueId id{}; const std::filesystem::path id_path(argv[5]);
    if(cfg.rank==0) { if(hcclGetUniqueId(&id)!=hcclSuccess) throw std::runtime_error("cannot make HCCL id");
      auto temp=id_path; temp += ".tmp"; std::ofstream out(temp,std::ios::binary); out.write(reinterpret_cast<const char*>(&id),sizeof(id)); out.close(); std::filesystem::rename(temp,id_path);
    } else { for(unsigned i=0;!std::filesystem::exists(id_path) && i<2400;++i) std::this_thread::sleep_for(std::chrono::milliseconds(50));
      std::ifstream in(id_path,std::ios::binary); if(!in.read(reinterpret_cast<char*>(&id),sizeof(id))) throw std::runtime_error("cannot read HCCL id"); }
    runtime.init_comm(id);
    HbmArena arena(runtime.device_id(),2ULL*1024*1024*1024); WeightManifest weights(argv[1],runtime,arena); WeightManifest dense_weights(argv[2],runtime,arena);
    const auto upload=[&](const void* data,size_t bytes) { auto addr=arena.allocate(bytes); arena.copy_h2d_sync(runtime.h2d_stream(),data,addr,bytes); return addr; };
    const auto hidden_row=read(fixture/"embedding_output.bin"); if(hidden_row.size()!=6144*2) throw std::runtime_error("invalid hidden fixture");
    std::vector<uint8_t> hidden(hidden_row.size()*kRows);
    for(unsigned r=0;r<kRows;++r) std::copy(hidden_row.begin(),hidden_row.end(),hidden.begin()+r*hidden_row.size());
    const auto* prompt_rows=std::getenv("GAUDIRT_PROMPT_ROWS");
    if(prompt_rows) { if(!empty) throw std::invalid_argument("prompt rows require empty KV"); hidden=read(std::filesystem::path(prompt_rows)/"embedding.bf16"); }
    if(hidden.size()!=kRows*6144*2) throw std::runtime_error("invalid prompt rows");
    const auto rope=read(prompt_rows ? std::filesystem::path(prompt_rows)/"full-rope.bf16" : layer/"rope.bin"); if(rope.size()<kRows*64*2) throw std::runtime_error("invalid rope fixture");
    std::vector<uint16_t> cos(kRows*64),sin(kRows*64);
    for(unsigned r=0;r<kRows;++r) for(unsigned i=0;i<64;++i) {
      std::memcpy(&cos[r*64+i],rope.data()+(r*64+i%32)*2,2);
      std::memcpy(&sin[r*64+i],rope.data()+(r*64+32+i%32)*2,2);
    }
    if(empty && !prompt_rows) { std::fill(cos.begin(),cos.end(),0x3f80); std::fill(sin.begin(),sin.end(),0); }
    const auto slot_bytes=read(layer/"slot_mapping.bin"),page_bytes=read(layer/"block_list.bin"),
               usage_bytes=read(layer/"block_usage.bin"),group_bytes=read(layer/"block_groups.bin");
    if(slot_bytes.size()!=4 || page_bytes.size()!=kPages*4 || usage_bytes.size()!=kPages*4 || group_bytes.size()!=kPages*4)
      throw std::runtime_error("invalid page fixture");
    int32_t first; std::array<int32_t,kPages> pages{},usage{},groups{};
    std::memcpy(&first,slot_bytes.data(),4); std::memcpy(pages.data(),page_bytes.data(),page_bytes.size());
    std::memcpy(usage.data(),usage_bytes.data(),usage_bytes.size()); std::memcpy(groups.data(),group_bytes.data(),group_bytes.size());
    if(empty) { first=0; for(unsigned p=0;p<kPages;++p) { pages[p]=p; usage[p]=1; groups[p]=p==0?0:-1; } }
    const unsigned active_page=empty?0:32;
    if(first<0 || unsigned(first)+kRows>cache_slots || first%128 || pages[active_page]*128!=first || usage[active_page]!=1 || groups[active_page]!=0)
      throw std::runtime_error("invalid starting full-attention page");
    std::array<int32_t,kRows> slots{},positions{};
    for(unsigned r=0;r<kRows;++r) { slots[r]=first+r; positions[r]=(empty?0:4096)+r; }
    MimoAttentionIo batch{};
    batch.hidden_states=upload(hidden.data(),hidden.size()); batch.rope_cos=upload(cos.data(),cos.size()*2); batch.rope_sin=upload(sin.data(),sin.size()*2);
    batch.key_cache=upload(initial_key.data(),initial_key.size()); batch.value_cache=upload(initial_value.data(),initial_value.size());
    batch.slot_mapping=upload(slots.data(),sizeof(slots)); batch.position=upload(positions.data(),sizeof(positions));
    batch.full_block_list=upload(pages.data(),sizeof(pages)); batch.full_block_groups=upload(groups.data(),sizeof(groups));
    usage[active_page]=kRows; batch.full_block_usage=upload(usage.data(),sizeof(usage)); batch.output_local_fp32=arena.allocate(6144*kRows*sizeof(float));
    MimoAttentionIo serial=batch;
    serial.key_cache=upload(initial_key.data(),initial_key.size()); serial.value_cache=upload(initial_value.data(),initial_value.size());
    const uint64_t serial_output=arena.allocate(6144*kRows*sizeof(float));
    std::array<uint64_t,kRows> serial_cos{},serial_sin{},serial_slots{},serial_usage{};
    for(unsigned r=0;r<kRows;++r) {
      serial_cos[r]=upload(cos.data()+r*64,64*2); serial_sin[r]=upload(sin.data()+r*64,64*2);
      serial_slots[r]=upload(&slots[r],sizeof(slots[r])); usage[active_page]=r+1; serial_usage[r]=upload(usage.data(),sizeof(usage));
    }
    gaudirt::mimo::Layer0 one(arena,runtime,weights,dense_weights,cache_slots,kPages,nullptr,true);
    gaudirt::mimo::Layer0 many(arena,runtime,weights,dense_weights,cache_slots,kPages,nullptr,true,kRows);
    const auto make_io=[&](MimoAttentionIo attention,unsigned rows) {
      gaudirt::mimo::Layer0::Io io{}; io.attention=attention;
      io.attention_gathered_fp32=arena.allocate(8ULL*rows*6144*4);
      io.dense_outputs.residual_out_bf16=arena.allocate(rows*6144*2);
      io.dense_outputs.activation_fp32=arena.allocate(rows*2048*4);
      io.dense_outputs.output_local_fp32=arena.allocate(rows*6144*4);
      io.down_gathered_fp32=arena.allocate(8ULL*rows*6144*4); return io;
    };
    auto batch_io=make_io(batch,kRows), serial_io=make_io(serial,1);
    std::vector<float> actual(8*kRows*6144),expected(actual.size()),one_result(8*6144);
    for(unsigned r=0;r<kRows;++r) {
      serial_io.attention.hidden_states=batch.hidden_states+r*6144*2;
      serial_io.attention.rope_cos=serial_cos[r]; serial_io.attention.rope_sin=serial_sin[r];
      serial_io.attention.slot_mapping=serial_slots[r]; serial_io.attention.full_block_usage=serial_usage[r];
      auto launch=one.prepare(serial_io); launch.execute_fused(runtime.compute_stream()); runtime.synchronize_compute();
      arena.copy_d2h_sync(runtime.d2h_stream(),serial_io.down_gathered_fp32,one_result.data(),one_result.size()*4);
      for(unsigned rank=0;rank<8;++rank) std::copy_n(one_result.data()+rank*6144,6144,expected.data()+(rank*kRows+r)*6144);
    }
    auto launch=many.prepare(batch_io); launch.execute_fused(runtime.compute_stream()); runtime.synchronize_compute();
    arena.copy_d2h_sync(runtime.d2h_stream(),batch_io.down_gathered_fp32,actual.data(),actual.size()*4);
    double max_abs=0,max_row_relative_l2=0;
    for(unsigned r=0;r<8*kRows;++r) {
      double diff=0,norm=0;
      for(unsigned i=0;i<6144;++i) { const double a=actual[r*6144+i],b=expected[r*6144+i];
        if(!std::isfinite(a)||!std::isfinite(b)) throw std::runtime_error("non-finite attention output");
        diff+=(a-b)*(a-b); norm+=b*b; max_abs=std::max(max_abs,std::abs(a-b)); }
      max_row_relative_l2=std::max(max_row_relative_l2,std::sqrt(diff/std::max(norm,1e-30)));
    }
    const auto verify_kv=[&](uint64_t a,uint64_t b,const std::vector<uint8_t>& original,size_t width) {
      std::vector<uint8_t> x(original.size()),y(original.size());
      arena.copy_d2h_sync(runtime.d2h_stream(),a,x.data(),x.size()); arena.copy_d2h_sync(runtime.d2h_stream(),b,y.data(),y.size());
      const size_t begin=size_t(first)*width,end=(size_t(first)+kRows)*width;
      if(!std::equal(x.begin(),x.begin()+begin,original.begin()) || !std::equal(x.begin()+end,x.end(),original.begin()+end) ||
         !std::equal(y.begin(),y.begin()+begin,original.begin()) || !std::equal(y.begin()+end,y.end(),original.begin()+end))
        throw std::runtime_error("untouched cache changed");
      if(!std::equal(x.begin()+begin,x.begin()+end,y.begin()+begin)) throw std::runtime_error("batched and sequential written KV differ");
    };
    verify_kv(batch.key_cache,serial.key_cache,initial_key,192*2); verify_kv(batch.value_cache,serial.value_cache,initial_value,128*2);
    std::cout << "M32 TP8 real-weight full+dense layer vs sequential M1: max_abs=" << max_abs << " max_row_relative_l2=" << max_row_relative_l2 << " KV=bitwise empty=" << empty << "\n";
    if(max_row_relative_l2>0.003) throw std::runtime_error("attention row exceeds quick-screen tolerance");
    if(const char* output_dir=std::getenv("GAUDIRT_TEST_OUTPUT_DIR")) {
      const auto prefix=std::filesystem::path(output_dir)/(std::string("rank")+module+"-A");
      std::ofstream partial(prefix.string()+"-partial.f32",std::ios::binary);
      for(unsigned rank=0;rank<8;++rank) partial.write(reinterpret_cast<const char*>(actual.data()+rank*kRows*6144),6144*4);
      std::vector<uint16_t> residual_row(6144);
      arena.copy_d2h_sync(runtime.d2h_stream(),batch_io.dense_outputs.residual_out_bf16,residual_row.data(),residual_row.size()*2);
      std::ofstream residual_file(prefix.string()+"-residual.bf16",std::ios::binary);
      residual_file.write(reinterpret_cast<const char*>(residual_row.data()),residual_row.size()*2);
      if(!partial || !residual_file) throw std::runtime_error("cannot write layer comparison output");
    }
    float marker=1.0f; const auto barrier=upload(&marker,sizeof(marker));
    if(hcclAllReduce(reinterpret_cast<const void*>(barrier),reinterpret_cast<void*>(barrier),1,hcclFloat32,hcclSum,runtime.comm(),runtime.compute_stream())!=hcclSuccess) throw std::runtime_error("barrier failed");
    runtime.synchronize_compute();
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
