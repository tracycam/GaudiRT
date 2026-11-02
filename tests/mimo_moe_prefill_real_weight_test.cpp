// SPDX-License-Identifier: Apache-2.0
// Compare M=32 MoE against 32 serial M=1 graphs with real weights.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "mimo_moe.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr unsigned kRows=32;
std::vector<uint8_t> read(const std::filesystem::path& path) {
  std::vector<uint8_t> data(std::filesystem::file_size(path)); std::ifstream in(path,std::ios::binary);
  if(!in.read(reinterpret_cast<char*>(data.data()),data.size())) throw std::runtime_error("cannot read fixture");
  return data;
}
}
int main(int argc,char** argv) {
  if(argc!=4) { std::cerr << "usage: TEST WEIGHTS RANK_FIXTURE MODULE\n"; return 2; }
  try {
    using namespace gaudirt::runtime;
    using namespace gaudirt::mimo;
    const std::string module=argv[3]; if(module.size()!=1 || module[0]<'0' || module[0]>'7') throw std::invalid_argument("module must be 0..7");
    DeviceRuntime::Config cfg{}; cfg.module_id=module[0]-'0'; DeviceRuntime runtime(cfg);
    HbmArena arena(runtime.device_id(),2ULL*1024*1024*1024); WeightManifest weights(argv[1],runtime,arena);
    const std::filesystem::path fixture(argv[2]);
    const auto repeat=[](const std::vector<uint8_t>& row) {
      if(row.size()!=6144*2) throw std::runtime_error("invalid input row");
      std::vector<uint8_t> data(row.size()*kRows); for(unsigned r=0;r<kRows;++r) std::copy(row.begin(),row.end(),data.begin()+r*row.size()); return data;
    };
    auto attention=repeat(read(fixture/"layer0/layer_output.bin"));
    auto residual=repeat(read(fixture/"layer0/residual_out.bin"));
    const auto other_attention=read(fixture/"layer1/layer_output.bin");
    const auto other_residual=read(fixture/"layer1/residual_out.bin");
    if(other_attention.size()!=6144*2 || other_residual.size()!=6144*2) throw std::runtime_error("invalid second row");
    for(unsigned r=1;r<kRows;r+=2) { std::copy(other_attention.begin(),other_attention.end(),attention.begin()+r*6144*2); std::copy(other_residual.begin(),other_residual.end(),residual.begin()+r*6144*2); }
    const auto upload=[&](const void* data,size_t bytes) { auto addr=arena.allocate(bytes); arena.copy_h2d_sync(runtime.h2d_stream(),data,addr,bytes); return addr; };
    MimoMoeLayer1::Weights w{};
    w.post_attention_gamma_bf16=weights.get("post_attention_layernorm").device_address;
    w.router_weight_bf16=weights.get("router_weight").device_address;
    w.router_correction_bias_fp32=weights.get("router_bias").device_address;
    w.gate_up_packed_u8=weights.get("moe_gp").device_address;
    w.gate_up_scales_u8=weights.get("moe_gs").device_address;
    w.down_packed_u8=weights.get("moe_dp").device_address;
    w.down_scales_u8=weights.get("moe_ds").device_address;
    const auto lut=MimoMoeLayer1::make_mxfp4_lut_bf16();
    const auto directions=MimoMoeLayer1::make_down_combine_directions();
    w.mxfp4_lut_bf16=upload(lut.data(),sizeof(lut));
    w.down_combine_directions_u8=upload(directions.data(),sizeof(directions));
    MimoMoeLayer1::Inputs batched{};
    batched.attention_bf16=upload(attention.data(),attention.size()); batched.residual_in_bf16=upload(residual.data(),residual.size());
    MimoMoeLayer1::Outputs bout{};
    bout.residual_out_bf16=arena.allocate(6144*kRows*2); bout.moe_local_fp32=arena.allocate(6144*kRows*4);
    MimoMoeLayer1::Inputs serial=batched; MimoMoeLayer1::Outputs sout{};
    sout.residual_out_bf16=arena.allocate(6144*kRows*2); sout.moe_local_fp32=arena.allocate(6144*kRows*4);
    const auto serial_residual=sout.residual_out_bf16, serial_output=sout.moe_local_fp32;
    MimoMoeLayer1 one(cfg.module_id,w,"swa"); MimoMoeLayer1 many(cfg.module_id,w,"swa",nullptr,false,syn_type_float,kRows);
    const auto workspace=arena.ensure_workspace(std::max(one.workspace_bytes(),many.workspace_bytes()));
    for(unsigned r=0;r<kRows;++r) {
      serial.attention_bf16=batched.attention_bf16+r*6144*2; serial.residual_in_bf16=batched.residual_in_bf16+r*6144*2;
      sout.residual_out_bf16=serial_residual+r*6144*2; sout.moe_local_fp32=serial_output+r*6144*4;
      one.launch(runtime.compute_stream(),serial,sout,workspace);
    }
    many.launch(runtime.compute_stream(),batched,bout,workspace); runtime.synchronize_compute();
    std::vector<float> actual(6144*kRows),expected(actual.size());
    arena.copy_d2h_sync(runtime.d2h_stream(),bout.moe_local_fp32,actual.data(),actual.size()*4);
    arena.copy_d2h_sync(runtime.d2h_stream(),serial_output,expected.data(),expected.size()*4);
    std::vector<uint16_t> actual_res(6144*kRows),expected_res(actual_res.size());
    arena.copy_d2h_sync(runtime.d2h_stream(),bout.residual_out_bf16,actual_res.data(),actual_res.size()*2);
    arena.copy_d2h_sync(runtime.d2h_stream(),serial_residual,expected_res.data(),expected_res.size()*2);
    if(actual_res!=expected_res) throw std::runtime_error("residual mismatch");
    double max_abs=0,max_row_relative_l2=0;
    for(unsigned r=0;r<kRows;++r) { double diff=0,norm=0;
      for(unsigned i=0;i<6144;++i) { const double a=actual[r*6144+i],b=expected[r*6144+i];
        if(!std::isfinite(a)||!std::isfinite(b)) throw std::runtime_error("non-finite MoE output");
        diff+=(a-b)*(a-b); norm+=b*b; max_abs=std::max(max_abs,std::abs(a-b)); }
      max_row_relative_l2=std::max(max_row_relative_l2,std::sqrt(diff/std::max(norm,1e-30)));
    }
    std::cout << "M32 real-weight MoE vs sequential M1: max_abs=" << max_abs << " max_row_relative_l2=" << max_row_relative_l2 << " residual=bitwise\n";
    if(max_row_relative_l2>0.003) throw std::runtime_error("MoE row exceeds quick-screen tolerance");
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
