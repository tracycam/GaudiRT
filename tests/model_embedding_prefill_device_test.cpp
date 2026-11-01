// SPDX-License-Identifier: Apache-2.0
// Check 32 device token lookups, including invalid IDs, against table rows.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "model_embedding.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  try {
    using namespace gaudirt::runtime;
    DeviceRuntime::Config config{}; config.module_id=0;
    DeviceRuntime runtime(config);
    HbmArena arena(runtime.device_id(), 3ULL*1024*1024*1024);
    constexpr size_t width=6144, rows=32;
    const auto table=arena.allocate(152576ULL*width*2);
    std::array<int32_t,rows> tokens{},status{};
    std::vector<uint16_t> expected(rows*width), actual(expected.size());
    for(size_t r=0;r<rows;++r) {
      tokens[r]=r==3 ? -1 : r==17 ? 152576 : static_cast<int32_t>(100+r);
      if(tokens[r]<0 || tokens[r]>=152576) continue;
      for(size_t c=0;c<width;++c) expected[r*width+c]=static_cast<uint16_t>(0x3f00+(r+c)%256);
      arena.copy_h2d_sync(runtime.h2d_stream(),expected.data()+r*width,table+tokens[r]*width*2,width*2);
    }
    const auto token=arena.allocate(sizeof(tokens)), out=arena.allocate(actual.size()*2), state=arena.allocate(sizeof(status));
    arena.copy_h2d_sync(runtime.h2d_stream(),tokens.data(),token,sizeof(tokens));
    gaudirt::mimo::ModelEmbedding embedding(arena,table,nullptr,rows);
    embedding.execute(token,out,state,runtime.compute_stream()); runtime.synchronize_compute();
    arena.copy_d2h_sync(runtime.d2h_stream(),out,actual.data(),actual.size()*2);
    arena.copy_d2h_sync(runtime.d2h_stream(),state,status.data(),sizeof(status));
    if(actual!=expected) throw std::runtime_error("embedding row mismatch");
    for(size_t r=0;r<rows;++r) if(status[r] != (r==3 || r==17)) throw std::runtime_error("token status mismatch");
    const auto token_one=arena.allocate(4), out_one=arena.allocate(width*2), state_one=arena.allocate(4);
    gaudirt::mimo::ModelEmbedding one(arena,table);
    std::vector<uint16_t> single(width);
    for(size_t r : {size_t(0),size_t(3),size_t(17),size_t(31)}) {
      arena.copy_h2d_sync(runtime.h2d_stream(),&tokens[r],token_one,4);
      one.execute(token_one,out_one,state_one,runtime.compute_stream()); runtime.synchronize_compute();
      int32_t single_status;
      arena.copy_d2h_sync(runtime.d2h_stream(),out_one,single.data(),width*2);
      arena.copy_d2h_sync(runtime.d2h_stream(),state_one,&single_status,4);
      if(single_status!=status[r] || !std::equal(single.begin(),single.end(),expected.begin()+r*width))
        throw std::runtime_error("M1 embedding regression");
    }
    std::cout << "M32 and M1 embedding table rows and invalid token status bitwise match\n";
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
