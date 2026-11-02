// SPDX-License-Identifier: Apache-2.0
#include "prefill_metadata.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
  using namespace gaudirt::mimo;
  std::array<int32_t,64> pages{};
  for(int i=0;i<64;++i) pages[i]=63-i;
  std::vector<uint16_t> full(4096*64),swa(4096*64);
  for(size_t i=0;i<full.size();++i) { full[i]=i%65536; swa[i]=(i+13)%65536; }
  auto verify = [&]<uint32_t Rows>() {
  for(unsigned base=0;base+Rows<=4096;base+=32) {
    PrefillMetadataT<Rows> metadata{};
    fill_prefill_metadata(metadata,base,4096,full.data(),swa.data(),pages,8192);
    for(unsigned row=0;row<Rows;++row) {
      const unsigned pos=base+row, page=pos/128, first=(pos>=127?pos-127:0)/128;
      if(metadata.positions[row]!=int(pos) || metadata.slots[row]!=pages[page]*128+int(pos%128) ||
         metadata.window_pages[row*2]!=pages[first] || metadata.window_groups[row*2]!=int(row) ||
         metadata.window_pages[row*2+1]!=(first<page?pages[page]:-1) ||
         metadata.window_groups[row*2+1]!=(first<page?int(row):-1)) throw std::runtime_error("wrong position or pages");
      for(unsigned i=0;i<64;++i) {
        if(metadata.full_cos[row*64+i]!=full[pos*64+i%32] || metadata.full_sin[row*64+i]!=full[pos*64+32+i%32] ||
           metadata.swa_cos[row*64+i]!=swa[pos*64+i%32] || metadata.swa_sin[row*64+i]!=swa[pos*64+32+i%32])
          throw std::runtime_error("wrong rope");
      }
    }
    for(unsigned page=0;page<64;++page) {
      const int count=int(base+Rows)-int(page*128);
      if(metadata.full_pages[page]!=pages[page] || metadata.full_groups[page]!=(count>0?0:-1) ||
         metadata.full_usage[page]!=std::clamp(count,1,128)) throw std::runtime_error("wrong full page metadata");
    }
  }
  };
  verify.template operator()<32>();
  verify.template operator()<512>();
  verify.template operator()<1024>();
  verify.template operator()<2048>();
  for (unsigned valid : {1, 17, 31}) {
    PrefillMetadata metadata{};
    fill_prefill_metadata(metadata, 4096 - valid, 4096, full.data(), swa.data(),
                          pages, 8320, valid, 8192);
    for (unsigned row = 0; row < 32; ++row) {
      const unsigned pos = 4096 - valid + std::min(row, valid - 1);
      const int slot = row < valid ? pages[pos / 128] * 128 + int(pos % 128)
                                   : 8192 + int(row - valid);
      if (metadata.positions[row] != int(pos) || metadata.slots[row] != slot)
        throw std::runtime_error("incorrect valid or isolated padding row");
    }
    if (metadata.full_usage[31] != 128 || metadata.full_groups[32] != -1)
      throw std::runtime_error("padding extended the valid KV prefix");
  }
  bool rejected = false;
  try {
    PrefillMetadata metadata{};
    fill_prefill_metadata(metadata, 0, 4096, full.data(), swa.data(), pages, 8320, 1, 0);
  } catch (const std::invalid_argument&) { rejected = true; }
  if (!rejected) throw std::runtime_error("padding aliases a scheduled page");
  std::cout << "4096-token chunk metadata, permutations, windows, and RoPE verified\n";
}
