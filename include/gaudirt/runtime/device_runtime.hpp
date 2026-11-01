// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/common/host_policy.hpp"

#include <hccl.h>
#include <synapse_api.h>

#include <cstdint>
#include <memory>
#include <string>

namespace gaudirt::runtime {

// Device, stream, and collective runtime context for a single Gaudi2
// module/rank. Owns Synapse initialization and device acquisition by default;
// an existing device and compute stream can instead be borrowed from a caller.
class DeviceRuntime {
 public:
  struct Config {
    uint32_t module_id = 0;
    uint32_t rank = 0;
    uint32_t world_size = 1;
    uint32_t uclamp_min = 1024;
    uint32_t uclamp_max = 1024;
    int pin_cpu = -1;
    synDeviceId borrowed_device_id = SYN_INVALID_DEVICE_ID;
  };

  explicit DeviceRuntime(const Config& cfg);
  ~DeviceRuntime();

  DeviceRuntime(const DeviceRuntime&) = delete;
  DeviceRuntime& operator=(const DeviceRuntime&) = delete;

  // Initialize 8-card (or N-card) HCCL communicator using a shared unique ID.
  void init_comm(const hcclUniqueId& uid);
  // Replace the owned compute stream with a non-null caller-owned stream. The
  // borrowed stream is synchronized but never destroyed by this runtime.
  void use_compute_stream(synStreamHandle stream);

  // Synchronize the primary compute stream or all streams on this device.
  void synchronize_compute() const;
  void synchronize_all() const;

  synDeviceId device_id() const noexcept { return device_id_; }
  uint32_t module_id() const noexcept { return cfg_.module_id; }
  uint32_t rank() const noexcept { return cfg_.rank; }
  uint32_t world_size() const noexcept { return cfg_.world_size; }

  synStreamHandle compute_stream() const noexcept { return compute_stream_; }
  synStreamHandle h2d_stream() const noexcept { return h2d_stream_; }
  synStreamHandle d2h_stream() const noexcept { return d2h_stream_; }
  hcclComm_t comm() const noexcept { return comm_; }
  bool uclamp_applied() const noexcept { return host_policy_.uclamp_applied; }

 private:
  Config cfg_;
  common::HostPolicyStatus host_policy_;
  synDeviceId device_id_ = SYN_INVALID_DEVICE_ID;
  synStreamHandle compute_stream_ = nullptr;
  synStreamHandle h2d_stream_ = nullptr;
  synStreamHandle d2h_stream_ = nullptr;
  hcclComm_t comm_ = nullptr;
  bool owns_device_ = true;
  bool owns_compute_stream_ = false;
};

}  // namespace gaudirt::runtime
