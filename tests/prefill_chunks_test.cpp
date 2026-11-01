// SPDX-License-Identifier: Apache-2.0
#include "prefill_chunks.hpp"
#include <stdexcept>

int main() {
  using gaudirt::mimo::prefill_chunks;
  for (unsigned count : {32, 480, 512, 544, 992, 1024, 1056, 1536, 2048, 2080, 4064, 4096}) {
    auto chunks = prefill_chunks(count);
    unsigned offset=0, small=0, large=0, xlarge=0;
    for (auto chunk : chunks) {
      if (chunk.offset!=offset || (chunk.rows!=32 && chunk.rows!=512 && chunk.rows!=2048) ||
          chunk.index!=(chunk.rows==2048?xlarge++:chunk.rows==512?large++:small++)) throw std::runtime_error("invalid prefill chunk plan");
      offset+=chunk.rows;
    }
    if (offset!=count || xlarge!=count/2048 || large!=(count%2048)/512 || small!=(count%512)/32)
      throw std::runtime_error("prefill chunk plan lost tokens");
  }
  for(unsigned bad : {0,31,33,4097}) {
    bool rejected=false; try { (void)prefill_chunks(bad); } catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected) throw std::runtime_error("invalid count accepted");
  }
}
