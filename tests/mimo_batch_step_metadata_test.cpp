// SPDX-License-Identifier: Apache-2.0
#include "batch_step_metadata.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace {
void require(bool value) { if (!value) throw std::runtime_error("batch metadata contract failed"); }
template <unsigned Rows>
void run(unsigned active) {
  using namespace gaudirt::mimo;
  constexpr unsigned physical_pages=256, rope_rows=4224;
  std::vector<uint16_t> full(rope_rows*64), swa(rope_rows*64);
  for(size_t i=0;i<full.size();++i){full[i]=static_cast<uint16_t>(i);swa[i]=static_cast<uint16_t>(i+7);}
  std::array<BatchRequestStep,Rows> requests{};
  const std::array<unsigned,8> positions{0,127,128,4095,255,511,1024,4096};
  unsigned next_page=0;
  for(unsigned row=0;row<active;++row){
    requests[row].position=positions[row];requests[row].page_count=positions[row]/128+1;
    for(unsigned p=0;p<requests[row].page_count;++p)requests[row].pages[p]=next_page++;
  }
  BatchStepMetadata<Rows> batch{};
  validate_batch_step<Rows>(requests.data(),active,rope_rows,full.data(),swa.data(),physical_pages,physical_pages*128);
  fill_batch_step_metadata_prevalidated(batch,requests.data(),active,full.data(),swa.data(),physical_pages*128);
  const auto base=reinterpret_cast<uintptr_t>(&batch);
  for(auto address : {reinterpret_cast<uintptr_t>(batch.full_cos.data()),reinterpret_cast<uintptr_t>(batch.full_sin.data()),
                      reinterpret_cast<uintptr_t>(batch.swa_cos.data()),reinterpret_cast<uintptr_t>(batch.swa_sin.data()),
                      reinterpret_cast<uintptr_t>(batch.positions.data()),reinterpret_cast<uintptr_t>(batch.slots.data()),
                      reinterpret_cast<uintptr_t>(batch.full_row_groups.data()),reinterpret_cast<uintptr_t>(batch.full_pages.data()),
                      reinterpret_cast<uintptr_t>(batch.full_usage.data()),reinterpret_cast<uintptr_t>(batch.full_groups.data()),
                      reinterpret_cast<uintptr_t>(batch.window_pages.data()),reinterpret_cast<uintptr_t>(batch.window_groups.data())})
    require((address-base)%256==0);
  for(unsigned row=0;row<Rows;++row){
    if(row>=active){require(batch.slots[row]==static_cast<int32_t>(physical_pages*128+row-active));
      require(batch.full_row_groups[row]==-1 && batch.window_groups[row*2]==-1 && batch.window_groups[row*2+1]==-1);
      for(unsigned p=0;p<64;++p)require(batch.full_groups[row*64+p]==-1);
      continue;}
    StepMetadata one{};
    fill_step_metadata_prevalidated(one,requests[row].position,0,full.data(),swa.data(),requests[row].pages);
    require(batch.positions[row]==one.position && batch.slots[row]==one.slot && batch.full_row_groups[row]==static_cast<int32_t>(row));
    for(unsigned c=0;c<64;++c){require(batch.full_cos[row*64+c]==one.full_cos[c] && batch.full_sin[row*64+c]==one.full_sin[c]);
      require(batch.swa_cos[row*64+c]==one.swa_cos[c] && batch.swa_sin[row*64+c]==one.swa_sin[c]);}
    for(unsigned p=0;p<64;++p){if(one.full_groups[p]==0){require(batch.full_pages[row*64+p]==one.full_pages[p]);
      require(batch.full_usage[row*64+p]==one.full_usage[p] && batch.full_groups[row*64+p]==static_cast<int32_t>(row));}
      else require(batch.full_groups[row*64+p]==-1 && batch.full_pages[row*64+p]==-1);}
    for(unsigned p=0;p<2;++p){require(batch.window_pages[row*2+p]==one.window_pages[p]);
      require(batch.window_groups[row*2+p]==(one.window_groups[p]==0?static_cast<int32_t>(row):-1));}
  }
  if(active>1){requests[1].pages[0]=requests[0].pages[0];bool rejected=false;
    try{validate_batch_step<Rows>(requests.data(),active,rope_rows,full.data(),swa.data(),physical_pages,physical_pages*128);}
    catch(const std::invalid_argument&){rejected=true;}require(rejected);}
}
}
int main(){run<4>(3);run<4>(4);run<8>(3);run<8>(8);}
