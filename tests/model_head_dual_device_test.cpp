// SPDX-License-Identifier: Apache-2.0
// Decode gathered head and prefill direct head must coexist and agree.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "model_head.hpp"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  try {
    using namespace gaudirt::runtime;
    using gaudirt::mimo::ModelHead;
    DeviceRuntime::Config cfg{}; cfg.module_id=0; DeviceRuntime runtime(cfg);
    HbmArena arena(runtime.device_id(),1024ULL*1024*1024);
    constexpr size_t hidden=6144, vocab=ModelHead::kLocalVocabSize;
    const auto upload=[&](const void* src,size_t bytes) { auto addr=arena.allocate(bytes); arena.copy_h2d_sync(runtime.h2d_stream(),src,addr,bytes); return addr; };
    std::vector<uint16_t> weights(hidden*vocab,0x3c00),gamma(hidden,0x3f80),input(hidden,0x3f00),residual(hidden,0x3e80);
    std::vector<float> gathered(8*hidden,0.0f); std::fill(gathered.begin(),gathered.begin()+hidden,0.5f);
    const auto weight=upload(weights.data(),weights.size()*2), norm_gamma=upload(gamma.data(),gamma.size()*2);
    const auto direct_input=upload(input.data(),input.size()*2), gather_input=upload(gathered.data(),gathered.size()*4);
    const auto residual_input=upload(residual.data(),residual.size()*2);
    const auto norm1=arena.allocate(hidden*2),norm2=arena.allocate(hidden*2),out1=arena.allocate(vocab*2),out2=arena.allocate(vocab*2);
    ModelHead decode(arena,norm_gamma,weight,nullptr,true,vocab);
    ModelHead prefill(arena,norm_gamma,weight,nullptr,false,vocab);
    decode.prepare(gather_input,residual_input,norm1,out1).execute(runtime.compute_stream());
    prefill.prepare(direct_input,residual_input,norm2,out2).execute(runtime.compute_stream());
    runtime.synchronize_compute();
    std::vector<uint16_t> first(vocab),second(vocab),n1(hidden),n2(hidden);
    arena.copy_d2h_sync(runtime.d2h_stream(),out1,first.data(),vocab*2);
    arena.copy_d2h_sync(runtime.d2h_stream(),out2,second.data(),vocab*2);
    arena.copy_d2h_sync(runtime.d2h_stream(),norm1,n1.data(),hidden*2);
    arena.copy_d2h_sync(runtime.d2h_stream(),norm2,n2.data(),hidden*2);
    if(first!=second || n1!=n2) throw std::runtime_error("dual model head outputs differ");
    std::cout << "gathered and direct shard heads coexist and match bitwise\n";
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
