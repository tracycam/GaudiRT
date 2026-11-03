// SPDX-License-Identifier: Apache-2.0
// Small-batch dense MLP row isolation against independent M1 graphs.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "mimo_dense_mlp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
uint16_t bf16(float value){uint32_t bits;std::memcpy(&bits,&value,4);bits+=0x7fff+((bits>>16)&1);return bits>>16;}
}
int main(int argc,char** argv){
  using namespace gaudirt::runtime;using gaudirt::mimo::MimoDenseMlp;
  const unsigned rows=argc==2?std::strtoul(argv[1],nullptr,10):4;
  if(rows!=4 && rows!=8 && rows!=32)throw std::invalid_argument("rows must be 4, 8 or 32");
  constexpr unsigned hidden=6144,intermediate=2048;
  DeviceRuntime device(DeviceRuntime::Config{});HbmArena arena(device.device_id(),1024ULL*1024*1024);
  auto upload=[&](const void* data,size_t bytes){auto address=arena.allocate(bytes);arena.copy_h2d_sync(device.h2d_stream(),data,address,bytes);return address;};
  auto make_weight=[&](const char* name,synDataType type,std::vector<uint64_t> shape,const void* data,size_t bytes){
    WeightTensor w;w.name=name;w.dtype=type;w.shape=std::move(shape);w.elements=1;for(auto n:w.shape)w.elements*=n;
    w.bytes=bytes;w.device_address=upload(data,bytes);return w;
  };
  std::vector<uint8_t> gate(128*4096*48,0x38),down(128*hidden*16,0x38);
  std::vector<float> gate_scales(48*32,1.0f/16384),down_scales(16*48,1.0f/16384),gate_bias(4096,0),down_bias(hidden,0);
  std::vector<WeightTensor> tensors;
  tensors.push_back(make_weight("gate_up_weight",syn_type_fp8_143,{128,4096,48},gate.data(),gate.size()));
  tensors.push_back(make_weight("gate_up_scales",syn_type_float,{48,32},gate_scales.data(),gate_scales.size()*4));
  tensors.push_back(make_weight("gate_up_bias",syn_type_float,{4096},gate_bias.data(),gate_bias.size()*4));
  tensors.push_back(make_weight("down_weight",syn_type_fp8_143,{128,hidden,16},down.data(),down.size()));
  tensors.push_back(make_weight("down_scales",syn_type_float,{16,48},down_scales.data(),down_scales.size()*4));
  tensors.push_back(make_weight("down_bias",syn_type_float,{hidden},down_bias.data(),down_bias.size()*4));
  WeightManifest weights(std::move(tensors));
  MimoDenseMlp one(arena,weights,nullptr,false,1),four(arena,weights,nullptr,false,4);
  MimoDenseMlp eight(arena,weights,nullptr,false,8),thirtytwo(arena,weights,nullptr,false,32);
  auto& batch=rows==4?four:rows==8?eight:thirtytwo;
  std::vector<uint16_t> attention(rows*hidden),residual(rows*hidden),gamma(hidden,bf16(1));
  for(unsigned row=0;row<rows;++row)for(unsigned i=0;i<hidden;++i){
    const float sign=row%2?-1.0f:1.0f;
    attention[row*hidden+i]=bf16(sign*((row+1)*0.03125f+(i%5)*0.015625f));
    residual[row*hidden+i]=bf16((i%3)*0.0078125f);
  }
  MimoDenseMlp::Inputs in{};in.attention_bf16=upload(attention.data(),attention.size()*2);
  in.residual_in_bf16=upload(residual.data(),residual.size()*2);in.post_attention_gamma_bf16=upload(gamma.data(),gamma.size()*2);
  MimoDenseMlp::Outputs out{};out.residual_out_bf16=arena.allocate(rows*hidden*2);
  out.activation_fp32=arena.allocate(rows*intermediate*4);out.output_local_fp32=arena.allocate(rows*hidden*4);
  const auto ref_residual=arena.allocate(rows*hidden*2),ref_activation=arena.allocate(rows*intermediate*4),ref_output=arena.allocate(rows*hidden*4);
  for(unsigned row=0;row<rows;++row){
    MimoDenseMlp::Inputs i{in.attention_bf16+row*hidden*2,in.residual_in_bf16+row*hidden*2,in.post_attention_gamma_bf16};
    MimoDenseMlp::Outputs o{ref_residual+row*hidden*2,ref_activation+row*intermediate*4,ref_output+row*hidden*4};
    one.execute(i,o,device.compute_stream());
  }
  batch.execute(in,out,device.compute_stream());device.synchronize_compute();
  std::vector<float> actual(rows*hidden),expected(rows*hidden);
  std::vector<uint16_t> actual_residual(rows*hidden),expected_residual(rows*hidden);
  arena.copy_d2h_sync(device.d2h_stream(),out.output_local_fp32,actual.data(),actual.size()*4);
  arena.copy_d2h_sync(device.d2h_stream(),ref_output,expected.data(),expected.size()*4);
  arena.copy_d2h_sync(device.d2h_stream(),out.residual_out_bf16,actual_residual.data(),actual_residual.size()*2);
  arena.copy_d2h_sync(device.d2h_stream(),ref_residual,expected_residual.data(),expected_residual.size()*2);
  if(actual_residual!=expected_residual)throw std::runtime_error("dense residual mismatch");
  if(expected[0]==expected[hidden])throw std::runtime_error("dense reference rows are not distinguishable");
  double max_relative=0,max_abs=0,max_value=0;
  for(unsigned row=0;row<rows;++row){double diff=0,norm=0;
    for(unsigned i=0;i<hidden;++i){const double a=actual[row*hidden+i],b=expected[row*hidden+i];
      if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite dense output");
      diff+=(a-b)*(a-b);norm+=b*b;max_abs=std::max(max_abs,std::abs(a-b));max_value=std::max(max_value,std::abs(b));}
    max_relative=std::max(max_relative,std::sqrt(diff/std::max(norm,1e-30)));
  }
  if(max_value==0 || max_relative>0.003)throw std::runtime_error("dense output mismatch");
  std::printf("B%u synthetic dense vs M1 max_abs=%.9g max_row_relative_l2=%.9g residual=bitwise\n",rows,max_abs,max_relative);
}
