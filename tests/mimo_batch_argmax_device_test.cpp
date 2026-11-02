// SPDX-License-Identifier: Apache-2.0
// Greedy selection directly from rank-major TP8 vocabulary shards.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include "model_argmax.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(synStatus status, const char* op) {
  if (status != synSuccess) throw std::runtime_error(std::string(op) + ": " + std::to_string(status));
}
}

int main(int argc, char** argv) {
  using namespace gaudirt::runtime;
  const unsigned rows=argc==2?std::strtoul(argv[1],nullptr,10):4;
  if(rows!=4 && rows!=8)throw std::invalid_argument("rows must be 4 or 8");
  constexpr unsigned shard=19072,ranks=8,vocab=shard*ranks;
  DeviceRuntime device(DeviceRuntime::Config{});
  HbmArena arena(device.device_id(),64ULL*1024*1024);
  synGraphHandle graph=nullptr;synRecipeHandle recipe=nullptr;
  std::array<synSectionHandle,3> sections{};
  std::array<synTensor,3> external{};std::array<synLaunchTensorInfo,3> binding{};
  const char* names[]={"logits","status","tokens"};
  const synDataType types[]={syn_type_bf16,syn_type_int32,syn_type_int32};
  const std::vector<std::vector<uint64_t>> shapes={{shard,rows,ranks},{rows},{rows}};
  check(synGraphCreate(&graph,synDeviceGaudi2),"create graph");
  for(size_t i=0;i<external.size();++i){
    check(synSectionCreate(&sections[i],0,graph),"create section");
    check(synSectionSetPersistent(sections[i],true),"persistent section");
    synTensorDescriptor d{};d.m_name=names[i];d.m_dataType=types[i];d.m_dims=shapes[i].size();
    binding[i].tensorName=names[i];binding[i].tensorType=DATA_TENSOR;
    for(size_t j=0;j<shapes[i].size();++j)binding[i].tensorSize[j]=d.m_sizes[j]=d.m_minSizes[j]=shapes[i][j];
    check(synTensorCreate(&external[i],&d,sections[i],0),"external tensor");
  }
  std::array<synTensor,3> partial{};
  const char* pnames[]={"partial_max","partial_id","partial_bad"};
  for(size_t i=0;i<partial.size();++i){
    synTensorDescriptor d{};d.m_name=pnames[i];d.m_dataType=i==0?syn_type_float:syn_type_int32;
    d.m_dims=2;d.m_sizes[0]=d.m_minSizes[0]=24;d.m_sizes[1]=d.m_minSizes[1]=rows;
    check(synTensorCreate(&partial[i],&d,nullptr,0),"partial tensor");
  }
  check(synNodeCreate(graph,&external[0],partial.data(),1,3,nullptr,0,
    "gk_mimo_argmax_batch_parts_bf16_v1","batch_argmax_parts",nullptr,nullptr),"parts node");
  std::array<synTensor,4> finish{partial[0],partial[1],partial[2],external[1]};
  check(synNodeCreate(graph,finish.data(),&external[2],4,1,nullptr,0,
    "gk_mimo_argmax_batch_finish_v1","batch_argmax_finish",nullptr,nullptr),"finish node");
  check(synGraphCompile(&recipe,graph,"mimo_batch_argmax_test",nullptr),"compile");
  uint64_t workspace=0;check(synWorkspaceGetSize(&workspace,recipe),"workspace");
  initialize_launch_tensor_ids(recipe,binding.data(),binding.size());
  check(synGraphDestroy(graph),"destroy graph");for(auto section:sections)check(synSectionDestroy(section),"destroy section");
  std::vector<uint16_t> logits(static_cast<size_t>(vocab)*rows,0xc000);
  std::vector<int32_t> status(rows),actual(rows,-99),expected(rows);
  auto offset=[&](unsigned row,unsigned token){return (static_cast<size_t>(token/shard)*rows+row)*shard+token%shard;};
  for(unsigned row=0;row<rows;++row){
    unsigned token=(row*23479+101)%vocab;expected[row]=token;logits[offset(row,token)]=0x4000;
  }
  // Equal maxima across distinct ranks retain the lowest global token id.
  logits[offset(0,expected[0])]=0xc000;logits[offset(0,15)]=logits[offset(0,shard+3)]=0x4000;expected[0]=15;
  logits[offset(1,expected[1])]=0xc000;logits[offset(1,vocab-1)]=0x4000;expected[1]=vocab-1;
  expected[2]=-1;
  status[3]=1;expected[3]=-1;
  const std::array<size_t,3> bytes{logits.size()*sizeof(uint16_t),status.size()*sizeof(int32_t),actual.size()*sizeof(int32_t)};
  for(size_t i=0;i<binding.size();++i)binding[i].pTensorAddress=arena.allocate(bytes[i]);
  arena.copy_h2d_sync(device.h2d_stream(),status.data(),binding[1].pTensorAddress,bytes[1]);
  PreparedLaunch launch{recipe,arena.ensure_workspace(workspace),{binding.begin(),binding.end()},0};
  launch.validate();
  for (uint16_t nonfinite : std::array<uint16_t,4>{0x7f80,0xff80,0x7fc0,0xffc0}) {
    logits[offset(2,vocab-5)]=nonfinite;
    arena.copy_h2d_sync(device.h2d_stream(),logits.data(),binding[0].pTensorAddress,bytes[0]);
    launch.execute(device.compute_stream());device.synchronize_compute();
    arena.copy_d2h_sync(device.d2h_stream(),binding[2].pTensorAddress,actual.data(),bytes[2]);
    for(unsigned row=0;row<rows;++row)if(actual[row]!=expected[row]){
      std::fprintf(stderr,"row %u expected %d got %d\n",row,expected[row],actual[row]);
      throw std::runtime_error("batch argmax mismatch");
    }
  }
  gaudirt::mimo::ModelArgmax model_argmax(arena, nullptr, rows);
  auto model_launch=model_argmax.prepare(binding[0].pTensorAddress,binding[1].pTensorAddress,binding[2].pTensorAddress);
  model_launch.execute(device.compute_stream());device.synchronize_compute();
  arena.copy_d2h_sync(device.d2h_stream(),binding[2].pTensorAddress,actual.data(),bytes[2]);
  if(actual!=expected)throw std::runtime_error("ModelArgmax batch output mismatch");
  check(synRecipeDestroy(recipe),"destroy recipe");
  std::printf("B%u rank-major argmax, ties, nonfinite values and status passed\n",rows);
}
