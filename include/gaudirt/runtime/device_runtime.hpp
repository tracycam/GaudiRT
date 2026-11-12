// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/common/host_policy.hpp"

#include <synapse_api.h>

#include <cstdint>

namespace gaudirt::runtime {

// Device and primary stream lifecycle for a single Gaudi2 module.
// Collective communication is managed independently per communicator group
// in CommunicatorRegistry / ProcessGroupGaudiRT.
class DeviceRuntime {
 public:
  struct Config {
    uint32_t module_id = 0;
    uint32_t uclamp_min = 1024;
    int pin_cpu = -1;
  };

  explicit DeviceRuntime(const Config& cfg);
  ~DeviceRuntime();

  DeviceRuntime(const DeviceRuntime&) = delete;
  DeviceRuntime& operator=(const DeviceRuntime&) = delete;

  void synchronize_compute() const;
  void synchronize_all() const;

  synDeviceId device_id() const noexcept { return device_id_; }
  uint32_t module_id() const noexcept { return cfg_.module_id; }

  synStreamHandle compute_stream() const noexcept { return compute_stream_; }
  synStreamHandle h2d_stream() const noexcept { return h2d_stream_; }
  synStreamHandle d2h_stream() const noexcept { return d2h_stream_; }
  bool uclamp_applied() const noexcept { return host_policy_.uclamp_applied; }

 private:
  Config cfg_;
  common::HostPolicyStatus host_policy_;
  synDeviceId device_id_ = SYN_INVALID_DEVICE_ID;
  synStreamHandle compute_stream_ = nullptr;
  synStreamHandle h2d_stream_ = nullptr;
  synStreamHandle d2h_stream_ = nullptr;
};

}  // namespace gaudirt::runtime
