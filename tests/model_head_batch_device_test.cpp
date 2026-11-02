// SPDX-License-Identifier: Apache-2.0
// Batched head output must match independent M=1 heads on identical inputs.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "model_head.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

int main() {
  using namespace gaudirt::runtime;
  using gaudirt::mimo::ModelHead;
  DeviceRuntime runtime(DeviceRuntime::Config{});
  HbmArena arena(runtime.device_id(),1024ULL*1024*1024);
  constexpr size_t hidden=6144,vocab=ModelHead::kLocalVocabSize;
  auto upload=[&](const void* data,size_t bytes){auto address=arena.allocate(bytes);arena.copy_h2d_sync(runtime.h2d_stream(),data,address,bytes);return address;};
  std::vector<uint16_t> weight(hidden*vocab,0x3c00),gamma(hidden,0x3f80);
  const auto weight_dev=upload(weight.data(),weight.size()*2),gamma_dev=upload(gamma.data(),gamma.size()*2);
  ModelHead single(arena,gamma_dev,weight_dev,nullptr,false,vocab,1);
  ModelHead four(arena,gamma_dev,weight_dev,nullptr,true,vocab,4);
  ModelHead eight(arena,gamma_dev,weight_dev,nullptr,true,vocab,8);
  const auto one_hidden=arena.allocate(hidden*2),one_residual=arena.allocate(hidden*2);
  const auto one_norm=arena.allocate(hidden*2),one_logits=arena.allocate(vocab*2);
  for(unsigned rows:{4u,8u}) {
    auto& batch=rows==4?four:eight;
    std::vector<float> gathered(8*rows*hidden,0);
    std::vector<uint16_t> residual(rows*hidden),direct(rows*hidden);
    for(unsigned row=0;row<rows;++row) {
      const bool odd=row%2;
      for(unsigned rank=0;rank<8;++rank)
        std::fill_n(gathered.begin()+(rank*rows+row)*hidden,hidden,odd?0.0625f:0.03125f);
      std::fill_n(direct.begin()+row*hidden,hidden,odd?0x3f00:0x3e80);
      std::fill_n(residual.begin()+row*hidden,hidden,odd?0x3e80:0x3d80);
    }
    const auto gathered_dev=upload(gathered.data(),gathered.size()*4);
    const auto residual_dev=upload(residual.data(),residual.size()*2);
    const auto norm_dev=arena.allocate(rows*hidden*2),logits_dev=arena.allocate(rows*vocab*2);
    batch.prepare(gathered_dev,residual_dev,norm_dev,logits_dev).execute(runtime.compute_stream());
    runtime.synchronize_compute();
    std::vector<uint16_t> actual_norm(rows*hidden),actual_logits(rows*vocab),expected_norm(hidden),expected_logits(vocab);
    arena.copy_d2h_sync(runtime.d2h_stream(),norm_dev,actual_norm.data(),actual_norm.size()*2);
    arena.copy_d2h_sync(runtime.d2h_stream(),logits_dev,actual_logits.data(),actual_logits.size()*2);
    for(unsigned row=0;row<rows;++row){
      arena.copy_h2d_sync(runtime.h2d_stream(),direct.data()+row*hidden,one_hidden,hidden*2);
      arena.copy_h2d_sync(runtime.h2d_stream(),residual.data()+row*hidden,one_residual,hidden*2);
      single.prepare(one_hidden,one_residual,one_norm,one_logits).execute(runtime.compute_stream());
      runtime.synchronize_compute();
      arena.copy_d2h_sync(runtime.d2h_stream(),one_norm,expected_norm.data(),hidden*2);
      arena.copy_d2h_sync(runtime.d2h_stream(),one_logits,expected_logits.data(),vocab*2);
      if(!std::equal(expected_norm.begin(),expected_norm.end(),actual_norm.begin()+row*hidden) ||
         !std::equal(expected_logits.begin(),expected_logits.end(),actual_logits.begin()+row*vocab))
        throw std::runtime_error("batch head differs from the corresponding single-row head");
    }
    std::printf("B%u gathered head matches individual M1 heads bitwise\n",rows);
  }
}
