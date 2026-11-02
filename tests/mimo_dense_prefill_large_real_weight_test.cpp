// SPDX-License-Identifier: Apache-2.0
// Compare M=512 dense MLP against an M=1 graph with real weights.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/weight_manifest.hpp"
#include "mimo_dense_mlp.hpp"
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
constexpr unsigned kRows=512;
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
    const auto attention=repeat(read(fixture/"layer0/layer_output.bin"));
    const auto residual=repeat(read(fixture/"layer0/residual_out.bin"));
    const auto upload=[&](const void* data,size_t bytes) { auto addr=arena.allocate(bytes); arena.copy_h2d_sync(runtime.h2d_stream(),data,addr,bytes); return addr; };
    MimoDenseMlp::Inputs batched{};
    batched.attention_bf16=upload(attention.data(),attention.size()); batched.residual_in_bf16=upload(residual.data(),residual.size());
    batched.post_attention_gamma_bf16=weights.get("post_attention_gamma").device_address;
    MimoDenseMlp::Outputs bout{};
    bout.residual_out_bf16=arena.allocate(6144*kRows*2); bout.activation_fp32=arena.allocate(2048*kRows*4); bout.output_local_fp32=arena.allocate(6144*kRows*4);
    MimoDenseMlp::Inputs serial=batched; MimoDenseMlp::Outputs sout{};
    sout.residual_out_bf16=arena.allocate(6144*kRows*2); sout.activation_fp32=arena.allocate(2048*kRows*4); sout.output_local_fp32=arena.allocate(6144*kRows*4);
    const auto serial_residual=sout.residual_out_bf16, serial_activation=sout.activation_fp32, serial_output=sout.output_local_fp32;
    MimoDenseMlp one(arena,weights); MimoDenseMlp many(arena,weights,nullptr,false,kRows);
    for(unsigned r=0;r<1;++r) {
      serial.attention_bf16=batched.attention_bf16+r*6144*2; serial.residual_in_bf16=batched.residual_in_bf16+r*6144*2;
      sout.residual_out_bf16=serial_residual+r*6144*2; sout.activation_fp32=serial_activation+r*2048*4; sout.output_local_fp32=serial_output+r*6144*4;
      one.execute(serial,sout,runtime.compute_stream());
    }
    many.execute(batched,bout,runtime.compute_stream()); runtime.synchronize_compute();
    std::vector<float> actual(6144*kRows),expected(actual.size());
    arena.copy_d2h_sync(runtime.d2h_stream(),bout.output_local_fp32,actual.data(),actual.size()*4);
    arena.copy_d2h_sync(runtime.d2h_stream(),serial_output,expected.data(),expected.size()*4);
    std::vector<uint16_t> actual_res(6144*kRows),expected_res(actual_res.size());
    arena.copy_d2h_sync(runtime.d2h_stream(),bout.residual_out_bf16,actual_res.data(),actual_res.size()*2);
    arena.copy_d2h_sync(runtime.d2h_stream(),serial_residual,expected_res.data(),expected_res.size()*2);
    for(unsigned r=0;r<kRows;++r) for(unsigned i=0;i<6144;++i)
      if(actual_res[r*6144+i]!=expected_res[i]) throw std::runtime_error("residual mismatch");
    double max_abs=0,max_row_relative_l2=0;
    for(unsigned r=0;r<kRows;++r) { double diff=0,norm=0;
      for(unsigned i=0;i<6144;++i) { const double a=actual[r*6144+i],b=expected[i];
        if(!std::isfinite(a)||!std::isfinite(b)) throw std::runtime_error("non-finite dense output");
        diff+=(a-b)*(a-b); norm+=b*b; max_abs=std::max(max_abs,std::abs(a-b)); }
      max_row_relative_l2=std::max(max_row_relative_l2,std::sqrt(diff/std::max(norm,1e-30)));
    }
    std::cout << "M512 real-weight dense vs M1: max_abs=" << max_abs << " max_row_relative_l2=" << max_row_relative_l2 << " residual=bitwise\n";
    if(max_row_relative_l2>0.003) throw std::runtime_error("dense row exceeds quick-screen tolerance");
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
