// SPDX-License-Identifier: Apache-2.0
// A synHostMalloc allocation is already mapped; copy directly into it.
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>

int main() {
  try {
    using namespace gaudirt::runtime;
    DeviceRuntime::Config cfg{}; cfg.module_id=0; DeviceRuntime runtime(cfg);
    HbmArena arena(runtime.device_id(),1024*1024);
    int32_t value=1729; const auto device=arena.allocate(sizeof(value));
    arena.copy_h2d_sync(runtime.h2d_stream(),&value,device,sizeof(value));
    void* pinned=nullptr;
    if(synHostMalloc(runtime.device_id(),sizeof(value),0,&pinned)!=synSuccess) throw std::runtime_error("pinned allocation failed");
    for(int i=0;i<4;++i) {
      *static_cast<int32_t*>(pinned)=0;
      if(synMemCopyAsync(runtime.d2h_stream(),device,sizeof(value),reinterpret_cast<uint64_t>(pinned),DRAM_TO_HOST)!=synSuccess ||
         synStreamSynchronize(runtime.d2h_stream())!=synSuccess || *static_cast<int32_t*>(pinned)!=value)
        throw std::runtime_error("pinned D2H result mismatch");
    }
    if(synHostFree(runtime.device_id(),pinned,0)!=synSuccess) throw std::runtime_error("pinned free failed");
    std::cout << "synHostMalloc direct D2H and repeated host reads passed\n";
    return 0;
  } catch(const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
