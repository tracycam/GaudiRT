// SPDX-License-Identifier: Apache-2.0
// Multi-request Full-Attention page and probability isolation.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

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
void check(synStatus status, const char* op) {
  if (status != synSuccess) throw std::runtime_error(std::string(op) + ": " + std::to_string(status));
}
float widen(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result; std::memcpy(&result, &bits, sizeof(result)); return result;
}
}

int main(int argc, char** argv) {
  using namespace gaudirt::runtime;
  const unsigned batch = argc >= 2 ? std::strtoul(argv[1], nullptr, 10) : 4;
  if (batch != 4 && batch != 8) throw std::invalid_argument("batch must be 4 or 8");
  if (argc > 3 || (argc == 3 && std::string(argv[2]) != "padded"))
    throw std::invalid_argument("expected optional padded mode");
  const unsigned active = argc == 3 ? 3 : batch;
  constexpr unsigned pages = 2, page_size = 128, slots = 4 * page_size, heads = 16;
  DeviceRuntime device(DeviceRuntime::Config{});
  HbmArena arena(device.device_id(), 64ULL * 1024 * 1024);
  synGraphHandle graph = nullptr; synRecipeHandle recipe = nullptr;
  std::array<synSectionHandle, 11> sections{};
  std::array<synTensor, 11> tensor{};
  std::array<synLaunchTensorInfo, 11> binding{};
  const char* names[] = {"key", "value", "pages", "usage", "groups", "row_groups",
                         "scores", "keys_out", "values_out", "prob_hi", "prob_lo"};
  const synDataType types[] = {syn_type_bf16, syn_type_bf16, syn_type_int32, syn_type_int32,
    syn_type_int32, syn_type_int32, syn_type_float, syn_type_bf16, syn_type_bf16,
    syn_type_bf16, syn_type_bf16};
  const std::vector<std::vector<uint64_t>> shapes = {{192,1,slots}, {128,1,slots},
    {pages,batch}, {pages,batch}, {pages,batch}, {batch}, {pages*page_size,heads,batch},
    {192,pages*page_size,batch}, {128,pages*page_size,batch},
    {pages*page_size,heads,batch}, {pages*page_size,heads,batch}};
  check(synGraphCreate(&graph, synDeviceGaudi2), "create graph");
  for (size_t i=0; i<tensor.size(); ++i) {
    check(synSectionCreate(&sections[i],0,graph), "create section");
    check(synSectionSetPersistent(sections[i],true), "persistent section");
    synTensorDescriptor desc{}; desc.m_name=names[i]; desc.m_dataType=types[i];
    desc.m_dims=shapes[i].size();
    binding[i].tensorName=names[i]; binding[i].tensorType=DATA_TENSOR;
    for(size_t d=0;d<shapes[i].size();++d)
      binding[i].tensorSize[d]=desc.m_sizes[d]=desc.m_minSizes[d]=shapes[i][d];
    check(synTensorCreate(&tensor[i],&desc,sections[i],0), "create tensor");
  }
  std::array<synTensor,6> gather_in{tensor[0],tensor[1],tensor[2],tensor[3],tensor[4],tensor[5]};
  std::array<synTensor,2> gather_out{tensor[7],tensor[8]};
  check(synNodeCreate(graph,gather_in.data(),gather_out.data(),6,2,nullptr,0,
    "gk_full_mme_batch_gather_v1","batch_gather",nullptr,nullptr), "gather node");
  std::array<synTensor,5> softmax_in{tensor[6],tensor[2],tensor[3],tensor[4],tensor[5]};
  std::array<synTensor,2> softmax_out{tensor[9],tensor[10]};
  struct Params { float scale; int32_t physical_pages; } params{1.0f,4};
  check(synNodeCreate(graph,softmax_in.data(),softmax_out.data(),5,2,&params,sizeof(params),
    "gk_full_mme_batch_softmax_v1","batch_softmax",nullptr,nullptr), "softmax node");
  check(synGraphCompile(&recipe,graph,"full_mme_batch_group_test",nullptr), "compile");
  uint64_t workspace=0;check(synWorkspaceGetSize(&workspace,recipe),"workspace");
  initialize_launch_tensor_ids(recipe,binding.data(),binding.size());
  check(synGraphDestroy(graph),"destroy graph");
  for(auto section:sections) check(synSectionDestroy(section),"destroy section");

  const uint16_t page_values[] = {0x3f80,0x4000,0x4040,0x4080};
  std::vector<uint16_t> key(192*slots),value(128*slots);
  for(unsigned p=0;p<4;++p) for(unsigned t=0;t<page_size;++t) {
    std::fill_n(key.begin()+(p*page_size+t)*192,192,page_values[p]);
    std::fill_n(value.begin()+(p*page_size+t)*128,128,page_values[p]);
  }
  std::vector<int32_t> table(pages*batch,3), usage(pages*batch,1), groups(pages*batch,-1), row_groups(batch,-1);
  row_groups[0]=10;row_groups[1]=20;row_groups[2]=30;
  table[0]=0;usage[0]=2;groups[0]=10;table[1]=1;usage[1]=3;groups[1]=20;
  table[2]=1;usage[2]=3;groups[2]=20;table[3]=0;usage[3]=2;groups[3]=10;
  table[4]=2;usage[4]=1;groups[4]=30;table[5]=3;usage[5]=1;groups[5]=30;
  for (unsigned row=3;row<active;++row) {
    row_groups[row]=40+row;
    table[row*pages]=row%4;usage[row*pages]=1+row%3;groups[row*pages]=row_groups[row];
    table[row*pages+1]=(row+1)%4;usage[row*pages+1]=2;groups[row*pages+1]=10;
  }
  std::vector<float> scores(pages*page_size*heads*batch,8.0f);
  for(unsigned row=0;row<batch;++row)for(unsigned page=0;page<pages;++page)
    for(int t=0;t<usage[row*pages+page];++t) if(row_groups[row]>=0 && groups[row*pages+page]==row_groups[row])
      for(unsigned h=0;h<heads;++h)
        scores[(row*heads+h)*pages*page_size+page*page_size+t]=-0.25f*(row+1)*(t+page);
  std::vector<uint16_t> kout(192*pages*page_size*batch),vout(128*pages*page_size*batch);
  std::vector<uint16_t> hi(scores.size()),lo(scores.size());
  std::array<void*,11> host{key.data(),value.data(),table.data(),usage.data(),groups.data(),row_groups.data(),scores.data(),kout.data(),vout.data(),hi.data(),lo.data()};
  std::array<size_t,11> bytes{key.size()*2,value.size()*2,table.size()*4,usage.size()*4,groups.size()*4,row_groups.size()*4,scores.size()*4,kout.size()*2,vout.size()*2,hi.size()*2,lo.size()*2};
  for(size_t i=0;i<tensor.size();++i) {
    binding[i].pTensorAddress=arena.allocate(bytes[i]);
    if(i<7) arena.copy_h2d_sync(device.h2d_stream(),host[i],binding[i].pTensorAddress,bytes[i]);
  }
  PreparedLaunch launch{recipe,arena.ensure_workspace(workspace),{binding.begin(),binding.end()},0};
  launch.validate();launch.execute(device.compute_stream());device.synchronize_compute();
  for(size_t i=7;i<tensor.size();++i)arena.copy_d2h_sync(device.d2h_stream(),binding[i].pTensorAddress,host[i],bytes[i]);
  double max_error=0;
  for(unsigned row=0;row<batch;++row) {
    double denominator=0;
    for(unsigned page=0;page<pages;++page)
      if(row_groups[row]>=0 && groups[row*pages+page]==row_groups[row])
        for(int t=0;t<usage[row*pages+page];++t)
          denominator+=std::exp(static_cast<double>(scores[(row*heads)*pages*page_size+page*page_size+t]));
    for(unsigned page=0;page<pages;++page)for(unsigned t=0;t<page_size;++t) {
      const auto index=row*pages+page;
      const bool valid=row_groups[row]>=0 && groups[index]==row_groups[row] && t<static_cast<unsigned>(usage[index]);
      const uint16_t expected=valid?page_values[table[index]]:0;
      const size_t token=(row*pages*page_size+page*page_size+t);
      for(unsigned c=0;c<192;++c) if(kout[token*192+c]!=expected) throw std::runtime_error("cross-request key leakage");
      for(unsigned c=0;c<128;++c) if(vout[token*128+c]!=expected) throw std::runtime_error("cross-request value leakage");
      for(unsigned h=0;h<heads;++h) {
        const size_t i=(row*heads+h)*pages*page_size+page*page_size+t;
        const double actual=widen(hi[i])+widen(lo[i]);
        if (!std::isfinite(actual)) throw std::runtime_error("non-finite probability");
        const double expected_probability=valid ? std::exp(static_cast<double>(scores[i]))/denominator : 0;
        max_error=std::max(max_error,std::abs(actual-expected_probability));
      }
    }
  }
  check(synRecipeDestroy(recipe),"destroy recipe");
  if(max_error>5e-5)throw std::runtime_error("batch softmax error");
  std::printf("B%u active=%u isolated gather and softmax max_abs=%.9g\n",batch,active,max_error);
}
