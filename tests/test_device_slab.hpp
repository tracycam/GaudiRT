// SPDX-License-Identifier: Apache-2.0
#pragma once

// Minimal contiguous HBM test slab for standalone C++ kernel/descriptor tests
// that place adjacent guard regions inside a single physical allocation.
// Production runtime memory uses BlockCachingAllocator.

#include <synapse_api.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace gaudirt::testing {

class TestDeviceSlab {
 public:
  static constexpr uint64_t kAlignment = 256;

  TestDeviceSlab(synDeviceId device, uint64_t capacity_bytes)
      : device_(device), capacity_(align_up(capacity_bytes)) {
    if (synDeviceMalloc(device_, capacity_, 0, 0, &base_) != synSuccess) {
      throw std::runtime_error("TestDeviceSlab: synDeviceMalloc failed");
    }
  }

  ~TestDeviceSlab() {
    if (base_ != 0) synDeviceFree(device_, base_, 0);
  }

  TestDeviceSlab(const TestDeviceSlab&) = delete;
  TestDeviceSlab& operator=(const TestDeviceSlab&) = delete;

  uint64_t allocate(uint64_t bytes) {
    const uint64_t aligned = align_up(bytes == 0 ? kAlignment : bytes);
    if (aligned > capacity_ - offset_) {
      throw std::runtime_error("TestDeviceSlab: out of slab capacity");
    }
    const uint64_t addr = base_ + offset_;
    offset_ += aligned;
    return addr;
  }

  uint64_t ensure_workspace(uint64_t bytes) {
    if (bytes == 0) return 0;
    if (bytes <= workspace_bytes_) return workspace_addr_;
    workspace_addr_ = allocate(bytes);
    workspace_bytes_ = align_up(bytes);
    return workspace_addr_;
  }

  void copy_h2d_sync(synStreamHandle stream, const void* host_src,
                     uint64_t dev_dst, uint64_t bytes) {
    if (bytes == 0) return;
    const uint64_t host_va = reinterpret_cast<uint64_t>(host_src);
    if (synHostMap(device_, bytes, host_src) != synSuccess) {
      throw std::runtime_error("TestDeviceSlab: synHostMap H2D failed");
    }
    const synStatus copy = synMemCopyAsync(stream, host_va, bytes, dev_dst, HOST_TO_DRAM);
    const synStatus sync = copy == synSuccess ? synStreamSynchronize(stream) : copy;
    synHostUnmap(device_, host_src);
    if (sync != synSuccess) {
      throw std::runtime_error("TestDeviceSlab: H2D transfer failed");
    }
  }

  void copy_d2h_sync(synStreamHandle stream, uint64_t dev_src,
                     void* host_dst, uint64_t bytes) {
    if (bytes == 0) return;
    const uint64_t host_va = reinterpret_cast<uint64_t>(host_dst);
    if (synHostMap(device_, bytes, host_dst) != synSuccess) {
      throw std::runtime_error("TestDeviceSlab: synHostMap D2H failed");
    }
    const synStatus copy = synMemCopyAsync(stream, dev_src, bytes, host_va, DRAM_TO_HOST);
    const synStatus sync = copy == synSuccess ? synStreamSynchronize(stream) : copy;
    synHostUnmap(device_, host_dst);
    if (sync != synSuccess) {
      throw std::runtime_error("TestDeviceSlab: D2H transfer failed");
    }
  }

  uint64_t base_addr() const noexcept { return base_; }
  uint64_t capacity_bytes() const noexcept { return capacity_; }

 private:
  static constexpr uint64_t align_up(uint64_t n) noexcept {
    return (n + kAlignment - 1) & ~(kAlignment - 1);
  }

  synDeviceId device_;
  uint64_t base_ = 0;
  uint64_t capacity_ = 0;
  uint64_t offset_ = 0;
  uint64_t workspace_addr_ = 0;
  uint64_t workspace_bytes_ = 0;
};

}  // namespace gaudirt::testing
