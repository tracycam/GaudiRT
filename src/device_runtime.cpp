// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/common/synapse_raii.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace gaudirt::runtime {

DeviceRuntime::DeviceRuntime(const Config& cfg)
    : cfg_(cfg),
      host_policy_(common::lock_host_policy(cfg.pin_cpu, cfg.uclamp_min)) {
  setenv("GC_KERNEL_PATH", "/usr/lib/habanalabs/libtpc_kernels.so", 0);
  const std::string mod_str = std::to_string(cfg_.module_id);
  setenv("ID", mod_str.c_str(), 1);
  setenv("HLS_MODULE_ID", mod_str.c_str(), 1);

  if (synInitialize() != synSuccess) {
    throw std::runtime_error("DeviceRuntime: synInitialize failed");
  }
  if (!common::device_module_exists(cfg_.module_id)) {
    synDestroy();
    throw std::runtime_error("DeviceRuntime: unknown module " +
                             std::to_string(cfg_.module_id));
  }

  const synModuleId mod = static_cast<synModuleId>(cfg_.module_id);
  synStatus acq = synFail;
  for (int attempt = 0; attempt < 40; ++attempt) {
    acq = synDeviceAcquireByModuleId(&device_id_, mod);
    if (acq == synSuccess) {
      break;
    }
    usleep(25000);  // 25 ms backoff for kernel fd release
  }
  if (acq != synSuccess) {
    synDestroy();
    throw std::runtime_error("DeviceRuntime: failed to acquire module " +
                             std::to_string(cfg_.module_id) + ", status " +
                             std::to_string(static_cast<int>(acq)));
  }

  if (synStreamCreateGeneric(&compute_stream_, device_id_, 0) != synSuccess ||
      synStreamCreateGeneric(&h2d_stream_, device_id_, 0) != synSuccess ||
      synStreamCreateGeneric(&d2h_stream_, device_id_, 0) != synSuccess) {
    if (compute_stream_) synStreamDestroy(compute_stream_);
    if (h2d_stream_) synStreamDestroy(h2d_stream_);
    if (d2h_stream_) synStreamDestroy(d2h_stream_);
    synDeviceRelease(device_id_);
    synDestroy();
    throw std::runtime_error("DeviceRuntime: failed to create device streams");
  }
}

DeviceRuntime::~DeviceRuntime() {
  if (compute_stream_) synStreamSynchronize(compute_stream_);
  if (h2d_stream_) synStreamSynchronize(h2d_stream_);
  if (d2h_stream_) synStreamSynchronize(d2h_stream_);

  if (d2h_stream_) {
    synStreamDestroy(d2h_stream_);
    d2h_stream_ = nullptr;
  }
  if (h2d_stream_) {
    synStreamDestroy(h2d_stream_);
    h2d_stream_ = nullptr;
  }
  if (compute_stream_) {
    synStreamDestroy(compute_stream_);
    compute_stream_ = nullptr;
  }
  if (device_id_ != SYN_INVALID_DEVICE_ID) {
    synDeviceRelease(device_id_);
    device_id_ = SYN_INVALID_DEVICE_ID;
  }
  synDestroy();
}

void DeviceRuntime::synchronize_compute() const {
  if (synStreamSynchronize(compute_stream_) != synSuccess) {
    throw std::runtime_error("DeviceRuntime: compute stream sync failed");
  }
}

void DeviceRuntime::synchronize_all() const {
  if (synDeviceSynchronize(device_id_) != synSuccess) {
    throw std::runtime_error("DeviceRuntime: synDeviceSynchronize failed");
  }
}

}  // namespace gaudirt::runtime
