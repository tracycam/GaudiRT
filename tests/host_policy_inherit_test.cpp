// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/common/host_policy.hpp"

#include <cstring>
#include <stdexcept>

int main() {
  cpu_set_t before{}, after{};
  gaudirt::common::SchedAttr sched_before{}, sched_after{};
  if (sched_getaffinity(0, sizeof(before), &before) != 0 ||
      syscall(SYS_sched_getattr, 0, &sched_before, sizeof(sched_before), 0) != 0)
    throw std::runtime_error("cannot inspect the current host policy");
  gaudirt::common::lock_host_policy(-1, gaudirt::common::kInheritHostPolicy);
  if (sched_getaffinity(0, sizeof(after), &after) != 0 ||
      syscall(SYS_sched_getattr, 0, &sched_after, sizeof(sched_after), 0) != 0 ||
      !CPU_EQUAL(&before, &after) ||
      sched_before.sched_policy != sched_after.sched_policy ||
      sched_before.sched_nice != sched_after.sched_nice ||
      sched_before.sched_util_min != sched_after.sched_util_min ||
      sched_before.sched_util_max != sched_after.sched_util_max)
    throw std::runtime_error("inherited host policy changed");
}
