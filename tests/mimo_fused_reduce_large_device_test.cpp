// SPDX-License-Identifier: Apache-2.0
// Check the TP8 rank/row layout consumed by large prefill graphs.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/prepared_launch.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include "gaudirt/runtime/tp_reduce.hpp"
#include "tp_reduce_graph.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {
void check(synStatus s) { if (s != synSuccess) throw std::runtime_error("large TP reduction Synapse failure"); }
uint16_t bf16(float x) { uint32_t b; std::memcpy(&b, &x, 4); b += 0x7fff + ((b >> 16) & 1); return b >> 16; }
}
int main() {
  using namespace gaudirt::runtime;
#ifndef GAUDIRT_TEST_ROWS
#define GAUDIRT_TEST_ROWS 512
#endif
  constexpr unsigned hidden = 6144, rows = GAUDIRT_TEST_ROWS, ranks = 8;
  DeviceRuntime::Config config{}; DeviceRuntime runtime(config);
  HbmArena arena(runtime.device_id(), 512ULL * 1024 * 1024);
  for (auto dtype : {syn_type_float, syn_type_bf16}) {
    synGraphHandle graph = nullptr; synRecipeHandle recipe = nullptr;
    std::array<synSectionHandle, 2> sections{};
    std::array<synTensor, 2> tensor{};
    std::array<synLaunchTensorInfo, 2> binding{};
    check(synGraphCreate(&graph, synDeviceGaudi2));
    auto external = [&](int i, const char* name, synDataType type, unsigned count) {
      check(synSectionCreate(&sections[i], 0, graph)); check(synSectionSetPersistent(sections[i], true));
      synTensorDescriptor d{}; d.m_name=name; d.m_dataType=type; d.m_dims=2;
      d.m_sizes[0]=d.m_minSizes[0]=hidden; d.m_sizes[1]=d.m_minSizes[1]=count;
      check(synTensorCreate(&tensor[i], &d, sections[i], 0));
      binding[i].tensorName=name; binding[i].tensorType=DATA_TENSOR;
      binding[i].tensorSize[0]=hidden; binding[i].tensorSize[1]=count;
    };
    external(0,"input",dtype,rows*ranks); external(1,"output",syn_type_bf16,rows);
    synSectionHandle intermediate_section = nullptr;
    auto reduced=gaudirt::mimo::append_tp8_reduce(graph,tensor[0],dtype,intermediate_section,rows);
    check(synNodeCreate(graph,&reduced,&tensor[1],1,1,nullptr,0,"reshape","output_copy",nullptr,nullptr));
    check(synGraphCompile(&recipe,graph,"tp8_m512_test",nullptr));
    uint64_t ws=0; check(synWorkspaceGetSize(&ws,recipe));
    initialize_launch_tensor_ids(recipe,binding.data(),binding.size());
    check(synGraphDestroy(graph)); check(synSectionDestroy(intermediate_section));
    for(auto s:sections) check(synSectionDestroy(s));
    const size_t values=static_cast<size_t>(rows)*ranks*hidden;
    std::vector<float> wide(values); std::vector<uint16_t> narrow(values);
    for (unsigned rank=0;rank<ranks;++rank) for (unsigned row=0;row<rows;++row)
      for(unsigned c=0;c<hidden;++c) {
        const float x=static_cast<float>(static_cast<int>((row+c)%13)-6+static_cast<int>(rank))/16;
        const size_t i=(static_cast<size_t>(rank)*rows+row)*hidden+c;
        wide[i]=x; narrow[i]=bf16(x);
      }
    const auto bytes=values*(dtype==syn_type_float?4:2);
    binding[0].pTensorAddress=arena.allocate(bytes); binding[1].pTensorAddress=arena.allocate(rows*hidden*2);
    arena.copy_h2d_sync(runtime.h2d_stream(),dtype==syn_type_float?static_cast<void*>(wide.data()):static_cast<void*>(narrow.data()),binding[0].pTensorAddress,bytes);
    PreparedLaunch launch{recipe,arena.ensure_workspace(ws),{binding.begin(),binding.end()},0};
    launch.validate(); launch.execute(runtime.compute_stream()); runtime.synchronize_compute();
    std::vector<uint16_t> actual(static_cast<size_t>(rows)*hidden);
    arena.copy_d2h_sync(runtime.d2h_stream(),binding[1].pTensorAddress,actual.data(),actual.size()*2);
    for(unsigned row=0;row<rows;++row) for(unsigned c=0;c<hidden;++c) {
      float sum=0; for(unsigned rank=0;rank<ranks;++rank) sum+=wide[(static_cast<size_t>(rank)*rows+row)*hidden+c];
      if(actual[static_cast<size_t>(row)*hidden+c]!=bf16(sum)) throw std::runtime_error("TP8 row/rank reduction mismatch");
    }
    const uint64_t element_bytes = dtype == syn_type_float ? 4 : 2;
    const uint64_t rank_stride = static_cast<uint64_t>(rows) * hidden * element_bytes;
    const uint64_t last_input = binding[0].pTensorAddress + (rows - 1) * hidden * element_bytes;
    const uint64_t last_output = arena.allocate(hidden * 2);
    TpReduce last_reduce(arena, hidden, dtype);
    auto last = last_reduce.prepare_strided(last_input, last_output, rank_stride);
    last.execute(runtime.compute_stream()); runtime.synchronize_compute();
    std::vector<uint16_t> final_row(hidden);
    arena.copy_d2h_sync(runtime.d2h_stream(), last_output, final_row.data(), hidden * 2);
    for (unsigned c=0; c<hidden; ++c)
      if (final_row[c] != actual[(rows - 1) * hidden + c])
        throw std::runtime_error("strided last-row TP8 reduction mismatch");
    check(synRecipeDestroy(recipe));
  }
  std::printf("M%u BF16 and FP32 TP8 reductions match CPU bitwise\n", rows);
}
