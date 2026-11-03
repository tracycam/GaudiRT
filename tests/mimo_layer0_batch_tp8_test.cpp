// SPDX-License-Identifier: Apache-2.0
// Eight-rank synthetic Layer0 B4/B8 execution against independent M1 requests.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "layer0.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>
#include <vector>

static uint16_t bf16(float value) {
  uint32_t bits; std::memcpy(&bits, &value, 4); bits += 0x7fff + ((bits >> 16) & 1); return bits >> 16;
}
int main(int argc, char** argv) {
  using namespace gaudirt::runtime;
  using namespace gaudirt::mimo;
  if (argc != 4) throw std::invalid_argument("usage: test rank rows shared_id_file");
  unsigned rank = std::strtoul(argv[1], nullptr, 10), rows = std::strtoul(argv[2], nullptr, 10);
  if (rank >= 8 || (rows != 4 && rows != 8)) throw std::invalid_argument("invalid rank or rows");
  constexpr unsigned hidden = 6144, slots = 9 * 128, pages = 2;
  DeviceRuntime::Config cfg{}; cfg.module_id = rank; cfg.rank = rank; cfg.world_size = 8;
  DeviceRuntime device(cfg);
  hcclUniqueId id{}; std::filesystem::path path(argv[3]);
  if (rank == 0) {
    if (hcclGetUniqueId(&id) != hcclSuccess) throw std::runtime_error("HCCL ID failed");
    auto tmp = path; tmp += ".tmp"; std::ofstream out(tmp, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&id), sizeof(id)); out.close(); std::filesystem::rename(tmp, path);
  } else {
    for (unsigned i = 0; !std::filesystem::exists(path) && i < 2400; ++i)
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    std::ifstream in(path, std::ios::binary);
    if (!in.read(reinterpret_cast<char*>(&id), sizeof(id))) throw std::runtime_error("HCCL ID read failed");
  }
  device.init_comm(id);
  HbmArena arena(device.device_id(), 1024ULL * 1024 * 1024);
  auto upload = [&](const void* p, size_t bytes) { auto a = arena.allocate(bytes); arena.copy_h2d_sync(device.h2d_stream(), p, a, bytes); return a; };
  auto weight = [&](const char* name, synDataType type, std::vector<uint64_t> shape, const void* data, size_t bytes) {
    WeightTensor t; t.name=name; t.dtype=type; t.shape=std::move(shape); t.elements=1;
    for (auto n : t.shape) t.elements*=n;
    t.bytes=bytes; t.device_address=upload(data,bytes); return t;
  };
  std::vector<uint16_t> gamma(hidden,bf16(1)), oproj(2048*hidden,bf16(1.0f/1024));
  std::vector<uint8_t> qkv(128*3392*48,0), gate(128*4096*48,0x38), down(128*hidden*16,0x38);
  std::vector<float> scales(48*27,1), bias(3392,0), gs(48*32,1.0f/16384), ds(16*48,1.0f/16384), gb(4096,0), db(hidden,0);
  std::vector<WeightTensor> a, d;
  a.push_back(weight("input_layernorm",syn_type_bf16,{hidden},gamma.data(),gamma.size()*2));
  a.push_back(weight("qkv_weight",syn_type_fp8_143,{128,3392,48},qkv.data(),qkv.size()));
  a.push_back(weight("qkv_scales",syn_type_float,{48,27},scales.data(),scales.size()*4));
  a.push_back(weight("qkv_bias",syn_type_float,{3392},bias.data(),bias.size()*4));
  a.push_back(weight("oproj_weight",syn_type_bf16,{2048,hidden},oproj.data(),oproj.size()*2));
  d.push_back(weight("post_attention_gamma",syn_type_bf16,{hidden},gamma.data(),gamma.size()*2));
  d.push_back(weight("gate_up_weight",syn_type_fp8_143,{128,4096,48},gate.data(),gate.size()));
  d.push_back(weight("gate_up_scales",syn_type_float,{48,32},gs.data(),gs.size()*4));
  d.push_back(weight("gate_up_bias",syn_type_float,{4096},gb.data(),gb.size()*4));
  d.push_back(weight("down_weight",syn_type_fp8_143,{128,hidden,16},down.data(),down.size()));
  d.push_back(weight("down_scales",syn_type_float,{16,48},ds.data(),ds.size()*4));
  d.push_back(weight("down_bias",syn_type_float,{hidden},db.data(),db.size()*4));
  WeightManifest attn_weights(std::move(a)), dense_weights(std::move(d));
  Layer0 single(arena,device,attn_weights,dense_weights,slots,pages,nullptr,true,1);
  Layer0 batch(arena,device,attn_weights,dense_weights,slots,pages,nullptr,true,rows);
  std::vector<uint16_t> hidden_data(rows*hidden),cos(rows*64,bf16(1)),sin(rows*64,0);
  for (unsigned r=0;r<rows;++r) for(unsigned i=0;i<hidden;++i) hidden_data[r*hidden+i]=bf16(0.125f + r*0.03125f + (i%3)*0.0078125f);
  std::vector<uint16_t> keys(slots*192,bf16(-8)),values(slots*128,bf16(-8));
  for(unsigned r=0;r<rows;++r){std::fill_n(keys.begin()+r*128*192,192,bf16(0));std::fill_n(values.begin()+r*128*128,128,bf16(float(r+1)));}
  const auto original_keys=keys,original_values=values;
  std::vector<int32_t> slot(rows),table(rows*pages),usage(rows*pages,2),groups(rows*pages),row_groups(rows);
  for(unsigned r=0;r<rows;++r){slot[r]=r*128+1;row_groups[r]=100+r;table[r*pages]=r;groups[r*pages]=row_groups[r];table[r*pages+1]=(r+1)%rows;groups[r*pages+1]=999;}
  const auto h=upload(hidden_data.data(),hidden_data.size()*2), c=upload(cos.data(),cos.size()*2), s=upload(sin.data(),sin.size()*2);
  const auto bk=upload(keys.data(),keys.size()*2), bv=upload(values.data(),values.size()*2);
  const auto rk=upload(keys.data(),keys.size()*2), rv=upload(values.data(),values.size()*2);
  MimoAttentionIo ai{}; ai.hidden_states=h; ai.rope_cos=c; ai.rope_sin=s; ai.key_cache=bk; ai.value_cache=bv;
  ai.slot_mapping=upload(slot.data(),slot.size()*4); ai.full_block_list=upload(table.data(),table.size()*4);
  ai.full_block_usage=upload(usage.data(),usage.size()*4); ai.full_block_groups=upload(groups.data(),groups.size()*4);
  ai.full_row_groups=upload(row_groups.data(),row_groups.size()*4); ai.output_local_fp32=arena.allocate(rows*hidden*4);
  auto make_io=[&](MimoAttentionIo attention,unsigned n){Layer0::Io io{};io.attention=attention;
    io.attention_gathered_fp32=arena.allocate(8ULL*n*hidden*4);io.dense_outputs.residual_out_bf16=arena.allocate(n*hidden*2);
    io.dense_outputs.activation_fp32=arena.allocate(n*2048*4);io.dense_outputs.output_local_fp32=arena.allocate(n*hidden*4);
    io.down_gathered_fp32=arena.allocate(8ULL*n*hidden*4);return io;};
  auto batched=make_io(ai,rows);
  auto reference=make_io(ai,1); reference.attention.key_cache=rk;reference.attention.value_cache=rv;reference.attention.output_local_fp32=arena.allocate(hidden*4);
  std::vector<float> expected(8*rows*hidden), actual(expected.size()), one(8*hidden);
  for(unsigned r=0;r<rows;++r){
    std::array<int32_t,pages> p{int32_t(r),int32_t((r+1)%rows)},u{2,0},g{0,-1};
    reference.attention.hidden_states=h+r*hidden*2;reference.attention.rope_cos=c+r*64*2;reference.attention.rope_sin=s+r*64*2;
    reference.attention.slot_mapping=upload(&slot[r],4);reference.attention.full_block_list=upload(p.data(),sizeof(p));
    reference.attention.full_block_usage=upload(u.data(),sizeof(u));reference.attention.full_block_groups=upload(g.data(),sizeof(g));
    single.prepare(reference).execute_fused(device.compute_stream());device.synchronize_compute();
    arena.copy_d2h_sync(device.d2h_stream(),reference.down_gathered_fp32,one.data(),one.size()*4);
    for(unsigned t=0;t<8;++t)std::copy_n(one.data()+t*hidden,hidden,expected.data()+(t*rows+r)*hidden);
  }
  batch.prepare(batched).execute_fused(device.compute_stream());device.synchronize_compute();
  arena.copy_d2h_sync(device.d2h_stream(),batched.down_gathered_fp32,actual.data(),actual.size()*4);
  double max_abs=0,max_rel=0;
  for(unsigned r=0;r<8*rows;++r){double diff=0,norm=0;for(unsigned i=0;i<hidden;++i){
    const double x=actual[r*hidden+i],y=expected[r*hidden+i];if(!std::isfinite(x)||!std::isfinite(y))throw std::runtime_error("non-finite output");
    diff+=(x-y)*(x-y);norm+=y*y;max_abs=std::max(max_abs,std::abs(x-y));}
    max_rel=std::max(max_rel,std::sqrt(diff/std::max(norm,1e-30)));}
  if(max_rel>0.003)throw std::runtime_error("layer output mismatch");
  auto verify_kv=[&](uint64_t x,uint64_t y,const std::vector<uint16_t>& orig,unsigned width){
    std::vector<uint16_t> a(orig.size()),b(orig.size());arena.copy_d2h_sync(device.d2h_stream(),x,a.data(),a.size()*2);arena.copy_d2h_sync(device.d2h_stream(),y,b.data(),b.size()*2);
    if(a!=b)throw std::runtime_error("batch and single KV differ");
    for(unsigned pos=0;pos<slots;++pos){bool touched=false;for(auto v:slot)if(pos==unsigned(v))touched=true;
      if(!touched&&!std::equal(a.begin()+pos*width,a.begin()+(pos+1)*width,orig.begin()+pos*width))throw std::runtime_error("untouched KV changed");}
  };
  verify_kv(bk,rk,original_keys,192);verify_kv(bv,rv,original_values,128);
  std::printf("rank=%u B%u Layer0 max_abs=%.9g max_row_relative_l2=%.9g KV=bitwise\n",rank,rows,max_abs,max_rel);
  float marker=1;auto barrier=upload(&marker,sizeof(marker));
  if(hcclAllReduce((void*)barrier,(void*)barrier,1,hcclFloat32,hcclSum,device.comm(),device.compute_stream())!=hcclSuccess)throw std::runtime_error("exit barrier failed");
  device.synchronize_compute();
}
