// SPDX-License-Identifier: Apache-2.0
// Coexisting M1/B4/B8/M32 lookup graphs and per-row invalid token status.
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
  using namespace gaudirt::runtime;
  using gaudirt::mimo::ModelEmbedding;
  DeviceRuntime runtime(DeviceRuntime::Config{});
  HbmArena arena(runtime.device_id(), 3ULL*1024*1024*1024);
  constexpr size_t width=6144, max_rows=32;
  const auto table=arena.allocate(152576ULL*width*2);
  std::array<int32_t,max_rows> tokens{}, expected_status{};
  std::vector<uint16_t> expected(max_rows*width);
  for(size_t r=0;r<max_rows;++r){
    tokens[r]=r==3?-1:r==7?152576:static_cast<int32_t>(100+r);
    expected_status[r]=(r==3 || r==7);
    if(expected_status[r])continue;
    for(size_t c=0;c<width;++c)expected[r*width+c]=static_cast<uint16_t>(0x3f00+(r+c)%256);
    arena.copy_h2d_sync(runtime.h2d_stream(),expected.data()+r*width,table+tokens[r]*width*2,width*2);
  }
  ModelEmbedding one(arena,table,nullptr,1),four(arena,table,nullptr,4),eight(arena,table,nullptr,8),thirtytwo(arena,table,nullptr,32);
  const auto token=arena.allocate(max_rows*4), out=arena.allocate(max_rows*width*2), status=arena.allocate(max_rows*4);
  arena.copy_h2d_sync(runtime.h2d_stream(),tokens.data(),token,tokens.size()*4);
  for(unsigned rows : {1u,4u,8u,32u}){
    auto& graph=rows==1?one:rows==4?four:rows==8?eight:thirtytwo;
    graph.execute(token,out,status,runtime.compute_stream());runtime.synchronize_compute();
    std::vector<uint16_t> actual(rows*width);std::vector<int32_t> states(rows);
    arena.copy_d2h_sync(runtime.d2h_stream(),out,actual.data(),actual.size()*2);
    arena.copy_d2h_sync(runtime.d2h_stream(),status,states.data(),states.size()*4);
    if(!std::equal(actual.begin(),actual.end(),expected.begin()) || !std::equal(states.begin(),states.end(),expected_status.begin()))
      throw std::runtime_error("batch embedding or status mismatch");
  }
  std::cout << "M1/B4/B8/M32 embedding and invalid token status bitwise match\n";
}
