#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

namespace gaudirt::common {

struct SchedAttr {
  uint32_t size;
  uint32_t sched_policy;
  uint64_t sched_flags;
  int32_t sched_nice;
  uint32_t sched_priority;
  uint64_t sched_runtime;
  uint64_t sched_deadline;
  uint64_t sched_period;
  uint32_t sched_util_min;
  uint32_t sched_util_max;
};

struct HostPolicyStatus {
  bool uclamp_applied = false;
  uint32_t uclamp_min = 0;
  uint32_t uclamp_max = 1024;
  int pinned_cpu = -1;
};

inline HostPolicyStatus lock_host_policy(int target_cpu = -1, uint32_t util_min = 1024) {
  HostPolicyStatus st{};
  st.uclamp_min = util_min;
  st.uclamp_max = 1024;

  if (target_cpu >= 0) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(target_cpu, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) == 0) {
      st.pinned_cpu = target_cpu;
    }
  } else {
    st.pinned_cpu = sched_getcpu();
  }

  SchedAttr attr{};
  attr.size = sizeof(SchedAttr);
  attr.sched_flags = 0x20 | 0x40; // SCHED_FLAG_UTIL_CLAMP_MIN | SCHED_FLAG_UTIL_CLAMP_MAX
  attr.sched_util_min = util_min;
  attr.sched_util_max = 1024;

  long rc = syscall(SYS_sched_setattr, 0, &attr, 0);
  st.uclamp_applied = (rc == 0);
  return st;
}

inline bool has_unwanted_instrumentation() {
  const char* preload = std::getenv("LD_PRELOAD");
  if (preload && preload[0] != '\0') {
    return true;
  }
  const char* prof = std::getenv("HABANA_PROFILE");
  if (prof && std::strcmp(prof, "0") != 0 && prof[0] != '\0') {
    return true;
  }
  return false;
}

} // namespace gaudirt::common
