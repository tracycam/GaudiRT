#include "gaudirt/runtime/generation_epoch_tracker.hpp"
#include <vector>
#include <cstdio>
int main() {
  gaudirt::runtime::LegacyAddressEventMap events;
  events.reserve(2392);
  std::vector<uint64_t> addresses(2392);
  for (size_t i=0;i<addresses.size();++i) addresses[i]=0x100000000ULL+i*4096;
  events.add_producer(addresses, 1, 1);
  events.add_producer(addresses, 1, 2);
  auto result=events.get_wait_events(addresses);
  std::printf("returned events: %zu\n",result.size());
}
