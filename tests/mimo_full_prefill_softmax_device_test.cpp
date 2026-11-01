// SPDX-License-Identifier: Apache-2.0
// Compare 32 causal full-attention rows against independent FP64 softmax.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(synStatus status) { if (status != synSuccess) throw std::runtime_error("full prefill softmax: Synapse failure"); }
float widen(uint16_t value) { const uint32_t bits=uint32_t{value}<<16; float out; std::memcpy(&out,&bits,4); return out; }
struct Params { float scale; int32_t physical_pages; };
}
int main(int argc, char** argv) {
  using namespace gaudirt::runtime;
  if (argc>2 || (argc==2 && std::string(argv[1])!="boundary")) throw std::invalid_argument("expected no argument or boundary");
  const bool boundary=argc==2;
  constexpr unsigned tokens=256, heads=512;
  DeviceRuntime runtime(DeviceRuntime::Config{});
  HbmArena arena(runtime.device_id(),64ULL*1024*1024);
  synGraphHandle graph=nullptr; synRecipeHandle recipe=nullptr;
  std::array<synSectionHandle,7> sections{}; std::array<synTensor,7> tensors{};
  std::array<synLaunchTensorInfo,7> bindings{};
  const char* names[]={"scores","pages","usage","groups","positions","probabilities","corrections"};
  const synDataType types[]={syn_type_float,syn_type_int32,syn_type_int32,syn_type_int32,syn_type_int32,syn_type_bf16,syn_type_bf16};
  const unsigned dims[]={2,1,1,1,1,2,2};
  const uint64_t sizes[][2]={{tokens,heads},{2,1},{2,1},{2,1},{32,1},{tokens,heads},{tokens,heads}};
  check(synGraphCreate(&graph,synDeviceGaudi2));
  for (unsigned i=0;i<tensors.size();++i) {
    check(synSectionCreate(&sections[i],0,graph)); check(synSectionSetPersistent(sections[i],true));
    synTensorDescriptor d{}; d.m_name=names[i]; d.m_dataType=types[i]; d.m_dims=dims[i];
    for (unsigned j=0;j<dims[i];++j) { d.m_sizes[j]=d.m_minSizes[j]=sizes[i][j]; bindings[i].tensorSize[j]=sizes[i][j]; }
    check(synTensorCreate(&tensors[i],&d,sections[i],0)); bindings[i].tensorName=names[i]; bindings[i].tensorType=DATA_TENSOR;
  }
  const Params p{1.f/std::sqrt(192.f),2};
  check(synNodeCreate(graph,tensors.data(),tensors.data()+5,5,2,&p,sizeof(p),"gk_full_mme_prefill_softmax_v1","full_causal_prefill_softmax",nullptr,nullptr));
  check(synGraphCompile(&recipe,graph,"mimo_full_prefill_softmax_test",nullptr));
  uint64_t workspace=0; check(synWorkspaceGetSize(&workspace,recipe));
  initialize_launch_tensor_ids(recipe,bindings.data(),bindings.size());
  check(synGraphDestroy(graph)); for (auto section:sections) check(synSectionDestroy(section));
  std::vector<float> scores(tokens*heads);
  for (unsigned h=0;h<heads;++h) for(unsigned t=0;t<tokens;++t)
    scores[h*tokens+t]=(int(t%17)-8)*0.13f+(int(h%7)-3)*0.011f;
  std::array<int32_t,2> pages{1,0},usage{128,32},groups{0,0};
  std::array<int32_t,32> positions{}; for(unsigned r=0;r<32;++r) positions[r]=(boundary?112:0)+r;
  std::vector<uint16_t> high(tokens*heads),low(tokens*heads);
  const void* inputs[]={scores.data(),pages.data(),usage.data(),groups.data(),positions.data()};
  const size_t bytes[]={scores.size()*4,sizeof(pages),sizeof(usage),sizeof(groups),sizeof(positions),high.size()*2,low.size()*2};
  for(unsigned i=0;i<7;++i) {
    bindings[i].pTensorAddress=arena.allocate(bytes[i]);
    if(i<5) arena.copy_h2d_sync(runtime.h2d_stream(),inputs[i],bindings[i].pTensorAddress,bytes[i]);
  }
  PreparedLaunch launch{recipe,arena.ensure_workspace(workspace),{bindings.begin(),bindings.end()},0};
  launch.validate(); launch.execute(runtime.compute_stream()); runtime.synchronize_compute();
  arena.copy_d2h_sync(runtime.d2h_stream(),bindings[5].pTensorAddress,high.data(),bytes[5]);
  arena.copy_d2h_sync(runtime.d2h_stream(),bindings[6].pTensorAddress,low.data(),bytes[6]);
  double max_abs=0;
  for(unsigned h=0;h<heads;++h) {
    const unsigned end=unsigned(positions[h/16]);
    double denominator=0; for(unsigned t=0;t<=end;++t) denominator+=std::exp(double(scores[h*tokens+t])*double(p.scale));
    for(unsigned t=0;t<tokens;++t) {
      const double expected=t<=end ? std::exp(double(scores[h*tokens+t])*double(p.scale))/denominator : 0;
      const double actual=double(widen(high[h*tokens+t]))+double(widen(low[h*tokens+t]));
      if (!std::isfinite(actual) || (t>end && actual!=0)) throw std::runtime_error("invalid masked probability");
      max_abs=std::max(max_abs,std::abs(actual-expected));
    }
  }
  std::printf("M=32 full causal softmax %s max_abs=%.9g masked=zero\n",boundary?"page-boundary":"empty-cache",max_abs);
  check(synRecipeDestroy(recipe));
  if(max_abs>2e-5) throw std::runtime_error("full prefill softmax differs from FP64");
}
