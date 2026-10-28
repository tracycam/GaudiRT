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
      host_policy_(common::lock_host_policy(cfg.pin_cpu, cfg.uclamp_min)),
      owns_device_(cfg.borrowed_device_id == SYN_INVALID_DEVICE_ID) {
  setenv("GC_KERNEL_PATH", "/usr/lib/habanalabs/libtpc_kernels.so", 0);
  if (cfg_.world_size > 1) {
    const std::string mod_str = std::to_string(cfg_.module_id);
    setenv("ID", mod_str.c_str(), 1);
    setenv("HLS_MODULE_ID", mod_str.c_str(), 1);
  }

  if (owns_device_ && synInitialize() != synSuccess) {
    throw std::runtime_error("DeviceRuntime: synInitialize failed");
  }
  if (!common::device_module_exists(cfg_.module_id)) {
    if (owns_device_) synDestroy();
    throw std::runtime_error("DeviceRuntime: unknown module " +
                             std::to_string(cfg_.module_id));
  }

  if (owns_device_) {
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
  } else {
    device_id_ = cfg_.borrowed_device_id;
  }

  if (synStreamCreateGeneric(&compute_stream_, device_id_, 0) != synSuccess ||
      synStreamCreateGeneric(&h2d_stream_, device_id_, 0) != synSuccess ||
      synStreamCreateGeneric(&d2h_stream_, device_id_, 0) != synSuccess) {
    if (compute_stream_) synStreamDestroy(compute_stream_);
    if (h2d_stream_) synStreamDestroy(h2d_stream_);
    if (d2h_stream_) synStreamDestroy(d2h_stream_);
    if (owns_device_) {
      synDeviceRelease(device_id_);
      synDestroy();
    }
    throw std::runtime_error("DeviceRuntime: failed to create device streams");
  }
  owns_compute_stream_ = true;
}

DeviceRuntime::~DeviceRuntime() {
  if (compute_stream_) {
    synStreamSynchronize(compute_stream_);
  }
  if (h2d_stream_) {
    synStreamSynchronize(h2d_stream_);
  }
  if (d2h_stream_) {
    synStreamSynchronize(d2h_stream_);
  }
  if (comm_) {
    hcclCommDestroy(comm_);
    comm_ = nullptr;
  }
  if (d2h_stream_) {
    synStreamDestroy(d2h_stream_);
    d2h_stream_ = nullptr;
  }
  if (h2d_stream_) {
    synStreamDestroy(h2d_stream_);
    h2d_stream_ = nullptr;
  }
  if (compute_stream_ && owns_compute_stream_) {
    synStreamDestroy(compute_stream_);
  }
  compute_stream_ = nullptr;
  if (owns_device_) {
    if (device_id_ != SYN_INVALID_DEVICE_ID) {
      synDeviceRelease(device_id_);
      device_id_ = SYN_INVALID_DEVICE_ID;
    }
    synDestroy();
  }
}

void DeviceRuntime::init_comm(const hcclUniqueId& uid) {
  if (comm_ != nullptr) {
    hcclCommDestroy(comm_);
    comm_ = nullptr;
  }
  hcclResult_t res = hcclCommInitRank(
      &comm_, static_cast<int>(cfg_.world_size), uid, static_cast<int>(cfg_.rank));
  if (res != hcclSuccess) {
    throw std::runtime_error("DeviceRuntime: hcclCommInitRank failed");
  }
}

void DeviceRuntime::use_compute_stream(synStreamHandle stream) {
  if (!stream) {
    throw std::invalid_argument("DeviceRuntime: cannot borrow a null compute stream");
  }
  if (owns_compute_stream_) {
    if (synStreamSynchronize(compute_stream_) != synSuccess ||
        synStreamDestroy(compute_stream_) != synSuccess) {
      throw std::runtime_error("DeviceRuntime: failed to release owned compute stream");
    }
  }
  compute_stream_ = stream;
  owns_compute_stream_ = false;
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
