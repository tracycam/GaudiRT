// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gaudirt::runtime {

// Pre-allocated HBM slab arena and shared workspace manager with O(1)
// address-generation tracking and pinned-host DMA helpers.
class HbmArena {
 public:
  static constexpr uint64_t kDefaultAlignment = 256;

  explicit HbmArena(synDeviceId device_id, uint64_t slab_bytes);
  struct BorrowedSlab {};
  HbmArena(synDeviceId device_id, uint64_t base_address, uint64_t slab_bytes,
           BorrowedSlab);
  ~HbmArena();

  HbmArena(const HbmArena&) = delete;
  HbmArena& operator=(const HbmArena&) = delete;

  // Sub-allocate aligned HBM memory from the pre-allocated slab.
  uint64_t allocate(uint64_t bytes, uint64_t alignment = kDefaultAlignment);

  // Reset the sub-allocation cursor to 0 and increment the arena generation.
  void reset() noexcept;

  // Ensure the shared recipe workspace buffer is at least `min_bytes`.
  // Increments generation if a new workspace buffer is allocated.
  uint64_t ensure_workspace(uint64_t min_bytes);

  // Use caller-owned device memory as workspace before the first allocation.
  // Requests beyond `bytes` fail instead of falling back to the slab.
  void set_borrowed_workspace(uint64_t address, uint64_t bytes);

  // Map/unmap host memory for DMA and execute async H2D / D2H transfers.
  void copy_h2d_sync(synStreamHandle h2d_stream, const void* host_src,
                     uint64_t dev_dst, uint64_t bytes) const;
  void copy_d2h_sync(synStreamHandle d2h_stream, uint64_t dev_src,
                     void* host_dst, uint64_t bytes) const;

  uint64_t base_addr() const noexcept { return slab_base_; }
  uint64_t capacity_bytes() const noexcept { return slab_capacity_; }
  uint64_t used_bytes() const noexcept { return offset_; }
  uint64_t workspace_addr() const noexcept { return workspace_addr_; }
  uint64_t workspace_bytes() const noexcept { return workspace_bytes_; }
  uint64_t generation() const noexcept { return generation_; }

 private:
  synDeviceId device_id_ = SYN_INVALID_DEVICE_ID;
  uint64_t slab_base_ = 0;
  uint64_t slab_capacity_ = 0;
  uint64_t offset_ = 0;
  uint64_t workspace_addr_ = 0;
  uint64_t workspace_bytes_ = 0;
  uint64_t generation_ = 1;
  bool owns_slab_ = true;
  bool owns_workspace_ = true;
  bool workspace_is_borrowed_ = false;
};

}  // namespace gaudirt::runtime
