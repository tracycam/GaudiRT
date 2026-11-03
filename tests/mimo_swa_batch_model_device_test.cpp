// SPDX-License-Identifier: Apache-2.0
// B4/B8 SWA Attention must isolate request pages and match independent M1 rows.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "mimo_attention.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
float widen(uint16_t bits) {
  uint32_t raw=static_cast<uint32_t>(bits)<<16;float value;std::memcpy(&value,&raw,4);return value;
}
uint16_t bf16(float value) {
  uint32_t bits;std::memcpy(&bits,&value,4);bits+=0x7fff+((bits>>16)&1);return bits>>16;
}
}
int main(int argc,char** argv) {
  using namespace gaudirt::runtime;using namespace gaudirt::mimo;
  const unsigned rows=argc>=2?std::strtoul(argv[1],nullptr,10):4;
  if(rows!=4 && rows!=8)throw std::invalid_argument("rows must be 4 or 8");
  if(argc>3 || (argc==3 && std::string(argv[2])!="padded"))throw std::invalid_argument("expected optional padded mode");
  const unsigned active=argc==3?3:rows;
  constexpr unsigned hidden=6144,slots=9*128,pages=2,scratch_page=8;
  DeviceRuntime device(DeviceRuntime::Config{});HbmArena arena(device.device_id(),512ULL*1024*1024);
  auto upload=[&](const void* data,size_t bytes){auto addr=arena.allocate(bytes);arena.copy_h2d_sync(device.h2d_stream(),data,addr,bytes);return addr;};
  auto weight=[&](const char* name,synDataType dtype,std::vector<uint64_t> shape,const void* data,size_t bytes){
    WeightTensor t;t.name=name;t.dtype=dtype;t.shape=std::move(shape);t.elements=1;for(auto n:t.shape)t.elements*=n;
    t.bytes=bytes;t.device_address=upload(data,bytes);return t;
  };
  std::vector<uint16_t> gamma(hidden,bf16(1)),oproj(2048*hidden,bf16(1.0f/1024));
  std::vector<uint8_t> qkv(128*3392*48,0);std::vector<float> scales(48*27,1),bias(3392,0);
  std::vector<WeightTensor> ws;
  ws.push_back(weight("input_layernorm",syn_type_bf16,{hidden},gamma.data(),gamma.size()*2));
  ws.push_back(weight("qkv_weight",syn_type_fp8_143,{128,3392,48},qkv.data(),qkv.size()));
  ws.push_back(weight("qkv_scales",syn_type_float,{48,27},scales.data(),scales.size()*4));
  ws.push_back(weight("qkv_bias",syn_type_float,{3392},bias.data(),bias.size()*4));
  ws.push_back(weight("oproj_weight",syn_type_bf16,{2048,hidden},oproj.data(),oproj.size()*2));
  std::vector<uint16_t> sink(16,bf16(0));
  ws.push_back(weight("attention_sink",syn_type_bf16,{16},sink.data(),sink.size()*2));
  WeightManifest weights(std::move(ws));
  MimoAttention single(arena,weights,slots,nullptr,false,1);
  MimoAttention batch4(arena,weights,slots,nullptr,false,4);
  MimoAttention batch8(arena,weights,slots,nullptr,false,8);
  auto& batch=rows==4?batch4:batch8;
  std::vector<uint16_t> h(rows*hidden,bf16(0.5f)),cos(rows*64,bf16(1)),sin(rows*64,0);
  std::vector<uint16_t> keys(slots*192,bf16(-8)),values(slots*128,bf16(-8));
  for(unsigned row=0;row<active;++row){
    std::fill_n(keys.begin()+row*128*192,192,bf16(0));
    std::fill_n(values.begin()+row*128*128,128,bf16(static_cast<float>(row+1)));
  }
  const auto original_keys=keys,original_values=values;
  std::vector<int32_t> slot(rows),tables(rows*pages),groups(rows*pages),row_groups(rows);
  for(unsigned row=0;row<rows;++row){
    if(row<active){
      slot[row]=row*128+1;row_groups[row]=row;
      tables[row*pages]=row;groups[row*pages]=row_groups[row];
      tables[row*pages+1]=(row+1)%active;groups[row*pages+1]=999;
    } else {
      slot[row]=scratch_page*128+row-active;row_groups[row]=-1;
      tables[row*pages]=tables[row*pages+1]=scratch_page;
      groups[row*pages]=groups[row*pages+1]=-1;
    }
  }
  const auto hidden_dev=upload(h.data(),h.size()*2),cos_dev=upload(cos.data(),cos.size()*2),sin_dev=upload(sin.data(),sin.size()*2);
  const auto residual_dev=upload(h.data(),h.size()*2);
  const auto batch_key=upload(keys.data(),keys.size()*2),batch_value=upload(values.data(),values.size()*2);
  const auto ref_key=upload(keys.data(),keys.size()*2),ref_value=upload(values.data(),values.size()*2);
  const auto slot_dev=upload(slot.data(),slot.size()*4),table_dev=upload(tables.data(),tables.size()*4);
  const auto groups_dev=upload(groups.data(),groups.size()*4);
  std::vector<int32_t> position(rows,1);
  const auto position_dev=upload(position.data(),position.size()*4);
  const auto out_dev=arena.allocate(rows*hidden*2),one_out=arena.allocate(hidden*2);
  const auto batch_residual_out=arena.allocate(rows*hidden*2),one_residual_out=arena.allocate(hidden*2);
  MimoAttentionIo io{};io.hidden_states=hidden_dev;io.rope_cos=cos_dev;io.rope_sin=sin_dev;
  io.key_cache=batch_key;io.value_cache=batch_value;io.slot_mapping=slot_dev;
  io.residual=residual_dev;io.residual_out=batch_residual_out;
  io.window_pages=table_dev;io.window_groups=groups_dev;io.position=position_dev;io.output_local=out_dev;
  batch.prepare(io).execute(device.compute_stream());device.synchronize_compute();
  std::vector<uint16_t> got(rows*hidden),expected(rows*hidden);
  arena.copy_d2h_sync(device.d2h_stream(),out_dev,got.data(),got.size()*2);
  for(unsigned row=0;row<active;++row){
    std::array<int32_t,pages> page{static_cast<int32_t>(row),static_cast<int32_t>((row+1)%active)};
    std::array<int32_t,pages> group{0,-1};
    MimoAttentionIo one{};one.hidden_states=hidden_dev+row*hidden*2;one.rope_cos=cos_dev+row*64*2;one.rope_sin=sin_dev+row*64*2;
    one.key_cache=ref_key;one.value_cache=ref_value;one.slot_mapping=upload(&slot[row],sizeof(slot[row]));
    one.residual=residual_dev+row*hidden*2;one.residual_out=one_residual_out;
    one.window_pages=upload(page.data(),sizeof(page));one.window_groups=upload(group.data(),sizeof(group));
    one.position=upload(&position[row],sizeof(position[row]));
    one.output_local=one_out;single.prepare(one).execute(device.compute_stream());device.synchronize_compute();
    arena.copy_d2h_sync(device.d2h_stream(),one_out,expected.data()+row*hidden,hidden*2);
  }
  float max_error=0,max_reference=0;
  for(size_t i=0;i<got.size();++i){float actual=widen(got[i]),reference=widen(expected[i]);
    if(!std::isfinite(actual))throw std::runtime_error("non-finite output");
    max_error=std::max(max_error,std::abs(actual-reference));max_reference=std::max(max_reference,std::abs(reference));}
  if(max_error>0.003f*max_reference)throw std::runtime_error("batch attention differs from M1");
  for(unsigned row=0;row<active;++row)if(!(widen(got[row*hidden])>0))throw std::runtime_error("attention lost nonzero row value");
  std::vector<uint16_t> batch_k(keys.size()),batch_v(values.size()),ref_k(keys.size()),ref_v(values.size());
  arena.copy_d2h_sync(device.d2h_stream(),batch_key,batch_k.data(),batch_k.size()*2);
  arena.copy_d2h_sync(device.d2h_stream(),batch_value,batch_v.data(),batch_v.size()*2);
  arena.copy_d2h_sync(device.d2h_stream(),ref_key,ref_k.data(),ref_k.size()*2);
  arena.copy_d2h_sync(device.d2h_stream(),ref_value,ref_v.data(),ref_v.size()*2);
  const auto real_key_end=scratch_page*128*192,real_value_end=scratch_page*128*128;
  if(!std::equal(batch_k.begin(),batch_k.begin()+real_key_end,ref_k.begin()) ||
     !std::equal(batch_v.begin(),batch_v.begin()+real_value_end,ref_v.begin()))
    throw std::runtime_error("batch real KV differs from M1");
  for(unsigned p=0;p<slots;++p){
    bool touched=false;for(auto pos:slot)if(p==static_cast<unsigned>(pos))touched=true;
    if(touched)continue;
    if(!std::equal(batch_k.begin()+p*192,batch_k.begin()+(p+1)*192,original_keys.begin()+p*192) ||
       !std::equal(batch_v.begin()+p*128,batch_v.begin()+(p+1)*128,original_values.begin()+p*128))
      throw std::runtime_error("untouched KV was modified");
  }
  std::printf("B%u active=%u SWA Attention vs M1 max_abs=%.9g, real KV bitwise and untouched slots passed\n",rows,active,max_error);
}
