// SPDX-License-Identifier: Apache-2.0
// Eight-rank synthetic MoE layer B4/B8 execution against independent M1 requests.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "batch_step_metadata.hpp"
#ifdef TEST_SWA
#include "layer_c.hpp"
#else
#include "layer_b.hpp"
#endif
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
#ifdef TEST_SWA
  using Layer = LayerC;
#else
  using Layer = LayerB;
#endif
  if (argc != 4 && argc != 5) throw std::invalid_argument("usage: test rank rows shared_id_file [active]");
  unsigned rank = std::strtoul(argv[1], nullptr, 10), rows = std::strtoul(argv[2], nullptr, 10);
  const unsigned active=argc==5?std::strtoul(argv[4],nullptr,10):rows;
  if (rank >= 8 || (rows != 4 && rows != 8) || !active || active > rows) throw std::invalid_argument("invalid rank, rows or active count");
  constexpr unsigned hidden = 6144, slots = 9 * 128, pages = 64;
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
  HbmArena arena(device.device_id(), 2ULL * 1024 * 1024 * 1024);
  auto upload = [&](const void* p, size_t bytes) { auto a = arena.allocate(bytes); arena.copy_h2d_sync(device.h2d_stream(), p, a, bytes); return a; };
  auto weight = [&](const char* name, synDataType type, std::vector<uint64_t> shape, const void* data, size_t bytes) {
    WeightTensor t; t.name=name; t.dtype=type; t.shape=std::move(shape); t.elements=1;
    for (auto n : t.shape) t.elements*=n;
    t.bytes=bytes; t.device_address=upload(data,bytes); return t;
  };
  std::vector<uint16_t> gamma(hidden,bf16(1)), oproj(2048*hidden,bf16(1.0f/1024));
  std::vector<uint8_t> qkv(128*3392*48,0);
  std::vector<float> scales(48*27,1), bias(3392,0), router_bias(384,-10);
  std::vector<uint16_t> router(hidden*384,0), sink(16,0);
  for(unsigned e=368;e<384;++e){router[e*hidden]=bf16(e%2?-1.0f:1.0f);router_bias[e]=(e-368)*0.0001f;}
  std::vector<WeightTensor> a;
  a.push_back(weight("input_layernorm",syn_type_bf16,{hidden},gamma.data(),gamma.size()*2));
  a.push_back(weight("qkv_weight",syn_type_fp8_143,{128,3392,48},qkv.data(),qkv.size()));
  a.push_back(weight("qkv_scales",syn_type_float,{48,27},scales.data(),scales.size()*4));
  a.push_back(weight("qkv_bias",syn_type_float,{3392},bias.data(),bias.size()*4));
  a.push_back(weight("oproj_weight",syn_type_bf16,{2048,hidden},oproj.data(),oproj.size()*2));
#ifdef TEST_SWA
  a.push_back(weight("attention_sink",syn_type_bf16,{16},sink.data(),sink.size()*2));
#endif
  a.push_back(weight("post_attention_layernorm",syn_type_bf16,{hidden},gamma.data(),gamma.size()*2));
  a.push_back(weight("router_weight",syn_type_bf16,{hidden,384},router.data(),router.size()*2));
  a.push_back(weight("router_bias",syn_type_float,{384},router_bias.data(),router_bias.size()*4));
  auto filled_weight = [&](const char* name, uint64_t bytes, uint8_t value){
    WeightTensor t; t.name=name; t.bytes=bytes; t.elements=bytes; t.shape={bytes}; t.dtype=syn_type_uint8;
    t.device_address=arena.allocate(bytes);
    if(synMemsetD8Async(t.device_address,value,bytes,device.compute_stream())!=synSuccess)throw std::runtime_error("weight fill failed");
    return t;
  };
  a.push_back(filled_weight("moe_gp",MimoMoeLayer1::kGateUpPackedWeightBytes,0x11));
  a.push_back(filled_weight("moe_gs",MimoMoeLayer1::kGateUpScaleBytes,110));
  a.push_back(filled_weight("moe_dp",MimoMoeLayer1::kDownPackedWeightBytes,0x11));
  a.push_back(filled_weight("moe_ds",MimoMoeLayer1::kDownScaleBytes,110));
  device.synchronize_compute();
  WeightManifest weights(std::move(a));
#ifdef TEST_SWA
  Layer single(arena,device,weights,slots,nullptr,true,1);
  Layer batch(arena,device,weights,slots,nullptr,true,rows);
#else
  Layer single(arena,device,weights,slots,pages,nullptr,true,1);
  Layer batch(arena,device,weights,slots,pages,nullptr,true,rows);
#endif
  std::vector<uint16_t> hidden_data(rows*hidden),cos(rows*64,bf16(1)),sin(rows*64,0);
  for (unsigned r=0;r<rows;++r) for(unsigned i=0;i<hidden;++i) hidden_data[r*hidden+i]=bf16(0.125f + r*0.03125f + (i%3)*0.0078125f);
  std::vector<uint16_t> keys(slots*192,bf16(-8)),values(slots*128,bf16(-8));
  for(unsigned r=0;r<rows;++r){std::fill_n(keys.begin()+r*128*192,192,bf16(0));std::fill_n(values.begin()+r*128*128,128,bf16(float(r+1)));}
  const auto original_keys=keys,original_values=values;
  std::array<BatchRequestStep,8> requests{};
  for(unsigned r=0;r<active;++r){requests[r].position=1;requests[r].pages[0]=r;requests[r].page_count=1;}
  std::vector<uint16_t> rope(2*64,0);
  for(unsigned i=0;i<32;++i)rope[64+i]=bf16(1);
  validate_batch_step<8>(requests.data(),active,2,rope.data(),rope.data(),8,8*128);
  BatchStepMetadata<8> metadata{};
  fill_batch_step_metadata_prevalidated(metadata,requests.data(),active,rope.data(),rope.data(),8*128);
  // A foreign page exists in the table but must not enter the request's attention.
  for(unsigned r=0;r<active;++r){metadata.full_pages[r*pages+1]=(r+1)%active;
    metadata.full_usage[r*pages+1]=2;metadata.full_groups[r*pages+1]=999;
    metadata.window_pages[r*2+1]=(r+1)%active;metadata.window_groups[r*2+1]=999;}
  const auto& slot=metadata.slots;
  std::vector<float> gathered_hidden(8*rows*hidden);
  for(unsigned t=0;t<8;++t)for(unsigned r=0;r<rows;++r)for(unsigned i=0;i<hidden;++i)
    gathered_hidden[(t*rows+r)*hidden+i]=0.015625f+r*0.00390625f+(i%3)*0.0009765625f;
  std::vector<uint64_t> single_hidden(rows);
  for(unsigned r=0;r<rows;++r){std::vector<float> row(8*hidden);
    for(unsigned t=0;t<8;++t)std::copy_n(gathered_hidden.data()+(t*rows+r)*hidden,hidden,row.data()+t*hidden);
    single_hidden[r]=upload(row.data(),row.size()*4);}
  const auto h=upload(gathered_hidden.data(),gathered_hidden.size()*4), c=upload(cos.data(),cos.size()*2), s=upload(sin.data(),sin.size()*2);
  const auto residual=upload(hidden_data.data(),hidden_data.size()*2);
  const auto bk=upload(keys.data(),keys.size()*2), bv=upload(values.data(),values.size()*2);
  const auto rk=upload(keys.data(),keys.size()*2), rv=upload(values.data(),values.size()*2);
  MimoAttentionIo ai{}; ai.hidden_states=h; ai.rope_cos=c; ai.rope_sin=s; ai.key_cache=bk; ai.value_cache=bv;
  ai.slot_mapping=upload(slot.data(),rows*4); ai.full_block_list=upload(metadata.full_pages.data(),rows*pages*4);
  ai.full_block_usage=upload(metadata.full_usage.data(),rows*pages*4); ai.full_block_groups=upload(metadata.full_groups.data(),rows*pages*4);
  ai.full_row_groups=upload(metadata.full_row_groups.data(),rows*4);
  ai.residual=residual;ai.residual_out=arena.allocate(rows*hidden*2);
#ifdef TEST_SWA
  ai.window_pages=upload(metadata.window_pages.data(),rows*2*4);
  ai.window_groups=upload(metadata.window_groups.data(),rows*2*4);
  ai.position=upload(metadata.positions.data(),rows*4);ai.output_local=arena.allocate(rows*hidden*2);
#else
  ai.output_local_fp32=arena.allocate(rows*hidden*4);
#endif
  auto make_io=[&](MimoAttentionIo attention,unsigned n){Layer::Io io{};io.attention=attention;
#ifdef TEST_SWA
    io.attention_gathered_bf16=arena.allocate(8ULL*n*hidden*2);
#else
    io.attention_gathered_fp32=arena.allocate(8ULL*n*hidden*4);
#endif
    io.moe_outputs.residual_out_bf16=arena.allocate(n*hidden*2);
    io.moe_outputs.moe_local_fp32=arena.allocate(n*hidden*4);
    io.moe_gathered_fp32=arena.allocate(8ULL*n*hidden*4);return io;};
  auto batched=make_io(ai,rows);
  auto reference=make_io(ai,1); reference.attention.key_cache=rk;reference.attention.value_cache=rv;
  reference.attention.residual_out=arena.allocate(hidden*2);
#ifdef TEST_SWA
  reference.attention.output_local=arena.allocate(hidden*2);
#else
  reference.attention.output_local_fp32=arena.allocate(hidden*4);
#endif
  std::vector<float> expected(8*rows*hidden), actual(expected.size()), one(8*hidden);
  for(unsigned r=0;r<active;++r){
    std::array<int32_t,pages> p{},u{},g{};p.fill(-1);g.fill(-1);
    p[0]=r;p[1]=(r+1)%active;u[0]=2;g[0]=0;
    reference.attention.hidden_states=single_hidden[r];reference.attention.residual=residual+r*hidden*2;reference.attention.rope_cos=c+r*64*2;reference.attention.rope_sin=s+r*64*2;
    reference.attention.slot_mapping=upload(&slot[r],4);reference.attention.full_block_list=upload(p.data(),sizeof(p));
    reference.attention.full_block_usage=upload(u.data(),sizeof(u));reference.attention.full_block_groups=upload(g.data(),sizeof(g));
#ifdef TEST_SWA
    int32_t pos=1;reference.attention.position=upload(&pos,4);
    std::array<int32_t,2> wp{int32_t(r),int32_t((r+1)%active)},wg{0,-1};
    reference.attention.window_pages=upload(wp.data(),sizeof(wp));reference.attention.window_groups=upload(wg.data(),sizeof(wg));
#endif
    single.prepare(reference).execute_fused(device.compute_stream());device.synchronize_compute();
    arena.copy_d2h_sync(device.d2h_stream(),reference.moe_gathered_fp32,one.data(),one.size()*4);
    for(unsigned t=0;t<8;++t)std::copy_n(one.data()+t*hidden,hidden,expected.data()+(t*rows+r)*hidden);
  }
  batch.prepare(batched).execute_fused(device.compute_stream());device.synchronize_compute();
  arena.copy_d2h_sync(device.d2h_stream(),batched.moe_gathered_fp32,actual.data(),actual.size()*4);
  double max_abs=0,max_rel=0;
  for(unsigned r=0;r<8*rows;++r){if(r%rows>=active)continue;double diff=0,norm=0;for(unsigned i=0;i<hidden;++i){
    const double x=actual[r*hidden+i],y=expected[r*hidden+i];if(!std::isfinite(x)||!std::isfinite(y))throw std::runtime_error("non-finite output");
    diff+=(x-y)*(x-y);norm+=y*y;max_abs=std::max(max_abs,std::abs(x-y));}
    max_rel=std::max(max_rel,std::sqrt(diff/std::max(norm,1e-30)));}
  if(max_rel>0.003)throw std::runtime_error("layer output mismatch");
  auto verify_kv=[&](uint64_t x,uint64_t y,const std::vector<uint16_t>& orig,unsigned width){
    std::vector<uint16_t> a(orig.size()),b(orig.size());arena.copy_d2h_sync(device.d2h_stream(),x,a.data(),a.size()*2);arena.copy_d2h_sync(device.d2h_stream(),y,b.data(),b.size()*2);
    if(!std::equal(a.begin(),a.begin()+8*128*width,b.begin()))throw std::runtime_error("batch and single real KV differ");
    for(unsigned pos=0;pos<slots;++pos){bool touched=false;for(unsigned r=0;r<rows;++r)if(pos==unsigned(slot[r]))touched=true;
      if(!touched&&!std::equal(a.begin()+pos*width,a.begin()+(pos+1)*width,orig.begin()+pos*width))throw std::runtime_error("untouched KV changed");}
  };
  verify_kv(bk,rk,original_keys,192);verify_kv(bv,rv,original_values,128);
  std::printf("rank=%u B%u active=%u MoE layer max_abs=%.9g max_row_relative_l2=%.9g real KV=bitwise\n",rank,rows,active,max_abs,max_rel);
  float marker=1;auto barrier=upload(&marker,sizeof(marker));
  if(hcclAllReduce((void*)barrier,(void*)barrier,1,hcclFloat32,hcclSum,device.comm(),device.compute_stream())!=hcclSuccess)throw std::runtime_error("exit barrier failed");
  device.synchronize_compute();
}
