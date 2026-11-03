// SPDX-License-Identifier: Apache-2.0
// B4/B8 MoE row isolation against independent M1 graphs with synthetic weights.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "mimo_moe.hpp"

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
uint16_t bf16(float value){uint32_t bits;std::memcpy(&bits,&value,4);bits+=0x7fff+((bits>>16)&1);return bits>>16;}
void check(synStatus status,const char* op){if(status!=synSuccess)throw std::runtime_error(std::string(op)+": "+std::to_string(status));}
}
int main(int argc,char** argv){
  using namespace gaudirt::runtime;using gaudirt::mimo::MimoMoeLayer1;
  const unsigned rows=argc==2?std::strtoul(argv[1],nullptr,10):4;
  if(rows!=4 && rows!=8 && rows!=32)throw std::invalid_argument("rows must be 4, 8 or 32");
  constexpr unsigned hidden=6144,experts=384;
  DeviceRuntime device(DeviceRuntime::Config{});HbmArena arena(device.device_id(),2ULL*1024*1024*1024);
  auto upload=[&](const void* data,size_t bytes){auto address=arena.allocate(bytes);arena.copy_h2d_sync(device.h2d_stream(),data,address,bytes);return address;};
  auto fill=[&](size_t bytes,uint8_t value){auto address=arena.allocate(bytes);check(synMemsetD8Async(address,value,bytes,device.compute_stream()),"memset");return address;};
  std::vector<uint16_t> gamma(hidden,bf16(1)),router(hidden*experts,0);
  std::vector<float> bias(experts,-10);
  for(unsigned e=368;e<experts;++e){
    router[e*hidden]=bf16(e%2?-1.0f:1.0f);
    bias[e]=(e-368)*0.0001f;
  }
  MimoMoeLayer1::Weights w{};
  w.post_attention_gamma_bf16=upload(gamma.data(),gamma.size()*2);
  w.router_weight_bf16=upload(router.data(),router.size()*2);
  w.router_correction_bias_fp32=upload(bias.data(),bias.size()*4);
  w.gate_up_packed_u8=fill(MimoMoeLayer1::kGateUpPackedWeightBytes,0x11);
  w.gate_up_scales_u8=fill(MimoMoeLayer1::kGateUpScaleBytes,110);
  w.down_packed_u8=fill(MimoMoeLayer1::kDownPackedWeightBytes,0x11);
  w.down_scales_u8=fill(MimoMoeLayer1::kDownScaleBytes,110);
  for(unsigned e=369;e<experts;e+=2){
    check(synMemsetD8Async(w.gate_up_packed_u8+static_cast<uint64_t>(e)*hidden*256,
                           0x22,hidden*256,device.compute_stream()),"vary gate weights");
    check(synMemsetD8Async(w.down_packed_u8+static_cast<uint64_t>(e)*12*256*256,
                           0x22,12*256*256,device.compute_stream()),"vary down weights");
  }
  device.synchronize_compute();
  const auto lut=MimoMoeLayer1::make_mxfp4_lut_bf16();
  const auto directions=MimoMoeLayer1::make_down_combine_directions();
  w.mxfp4_lut_bf16=upload(lut.data(),sizeof(lut));w.down_combine_directions_u8=upload(directions.data(),sizeof(directions));
  MimoMoeLayer1 one(0,w,"swa",nullptr,false,syn_type_float,1);
  MimoMoeLayer1 four(0,w,"swa",nullptr,false,syn_type_float,4);
  MimoMoeLayer1 eight(0,w,"swa",nullptr,false,syn_type_float,8);
  MimoMoeLayer1 thirtytwo(0,w,"swa",nullptr,false,syn_type_float,32);
  auto& batch=rows==4?four:rows==8?eight:thirtytwo;
  std::vector<uint16_t> attention(rows*hidden),residual(rows*hidden);
  for(unsigned r=0;r<rows;++r){
    for(unsigned i=0;i<hidden;++i){
      const float sign=r%2?-1.0f:1.0f;
      attention[r*hidden+i]=bf16(sign*((r+1)*0.03125f+(i%5)*0.015625f));
      residual[r*hidden+i]=bf16((i%3)*0.0078125f);
    }
  }
  MimoMoeLayer1::Inputs in{};in.attention_bf16=upload(attention.data(),attention.size()*2);in.residual_in_bf16=upload(residual.data(),residual.size()*2);
  MimoMoeLayer1::Outputs out{};out.residual_out_bf16=arena.allocate(rows*hidden*2);out.moe_local_fp32=arena.allocate(rows*hidden*4);
  const auto reference_residual=arena.allocate(rows*hidden*2),reference_output=arena.allocate(rows*hidden*4);
  const auto workspace=arena.ensure_workspace(std::max({one.workspace_bytes(),four.workspace_bytes(),eight.workspace_bytes(),thirtytwo.workspace_bytes()}));
  for(unsigned r=0;r<rows;++r){
    MimoMoeLayer1::Inputs in_one{in.attention_bf16+r*hidden*2,in.residual_in_bf16+r*hidden*2};
    MimoMoeLayer1::Outputs out_one{reference_residual+r*hidden*2,reference_output+r*hidden*4};
    one.launch(device.compute_stream(),in_one,out_one,workspace);
  }
  batch.launch(device.compute_stream(),in,out,workspace);device.synchronize_compute();
  std::vector<float> actual(rows*hidden),expected(rows*hidden);std::vector<uint16_t> actual_residual(rows*hidden),expected_residual(rows*hidden);
  arena.copy_d2h_sync(device.d2h_stream(),out.moe_local_fp32,actual.data(),actual.size()*4);
  arena.copy_d2h_sync(device.d2h_stream(),reference_output,expected.data(),expected.size()*4);
  arena.copy_d2h_sync(device.d2h_stream(),out.residual_out_bf16,actual_residual.data(),actual_residual.size()*2);
  arena.copy_d2h_sync(device.d2h_stream(),reference_residual,expected_residual.data(),expected_residual.size()*2);
  if(actual_residual!=expected_residual)throw std::runtime_error("MoE residual mismatch");
  if(expected[0]==expected[hidden])throw std::runtime_error("MoE reference rows are not distinguishable");
  double max_relative=0,max_abs=0,max_value=0;
  for(unsigned r=0;r<rows;++r){double diff=0,norm=0;
    for(unsigned i=0;i<hidden;++i){const double a=actual[r*hidden+i],b=expected[r*hidden+i];
      if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite MoE output");
      diff+=(a-b)*(a-b);norm+=b*b;max_abs=std::max(max_abs,std::abs(a-b));max_value=std::max(max_value,std::abs(b));}
    max_relative=std::max(max_relative,std::sqrt(diff/std::max(norm,1e-30)));
  }
  if(max_value==0 || max_relative>0.003)throw std::runtime_error("MoE output mismatch");
  std::printf("B%u synthetic MoE vs M1 max_abs=%.9g max_row_relative_l2=%.9g residual=bitwise\n",rows,max_abs,max_relative);
}
