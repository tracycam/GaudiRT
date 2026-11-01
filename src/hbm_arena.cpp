// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/hbm_arena.hpp"

#include <stdexcept>
#include <string>

namespace gaudirt::runtime {

HbmArena::HbmArena(synDeviceId device_id, uint64_t slab_bytes)
    : device_id_(device_id), slab_capacity_(slab_bytes) {
  if (slab_capacity_ > 0) {
    const synStatus status = synDeviceMalloc(device_id_, slab_capacity_, 0, 0, &slab_base_);
    if (status != synSuccess) {
      throw std::runtime_error("HbmArena: synDeviceMalloc failed for slab, status " +
                               std::to_string(static_cast<int>(status)) +
                               ", device " + std::to_string(device_id_));
    }
  }
}

HbmArena::HbmArena(synDeviceId device_id, uint64_t base_address,
                   uint64_t slab_bytes, BorrowedSlab)
    : device_id_(device_id), slab_base_(base_address),
      slab_capacity_(slab_bytes), owns_slab_(false), owns_workspace_(false) {
  if (slab_base_ == 0 || slab_capacity_ == 0 ||
      slab_base_ % kDefaultAlignment != 0) {
    throw std::invalid_argument("HbmArena: invalid borrowed slab");
  }
}

HbmArena::~HbmArena() {
  if (workspace_addr_ != 0 && owns_workspace_) {
    synDeviceFree(device_id_, workspace_addr_, 0);
    workspace_addr_ = 0;
  }
  if (slab_base_ != 0 && owns_slab_) {
    synDeviceFree(device_id_, slab_base_, 0);
    slab_base_ = 0;
  }
}

uint64_t HbmArena::allocate(uint64_t bytes, uint64_t alignment) {
  if (bytes == 0) {
    return slab_base_ + offset_;
  }
  if (alignment == 0) {
    alignment = kDefaultAlignment;
  }
  const uint64_t rem = offset_ % alignment;
  const uint64_t aligned_offset =
      rem == 0 ? offset_ : (offset_ + (alignment - rem));
  if (aligned_offset + bytes > slab_capacity_) {
    throw std::runtime_error("HbmArena: out of pre-allocated HBM slab memory: requested=" +
                             std::to_string(bytes) + ", used=" + std::to_string(aligned_offset) +
                             ", capacity=" + std::to_string(slab_capacity_));
  }
  const uint64_t addr = slab_base_ + aligned_offset;
  offset_ = aligned_offset + bytes;
  return addr;
}

void HbmArena::reset() noexcept {
  offset_ = 0;
  ++generation_;
}

uint64_t HbmArena::ensure_workspace(uint64_t min_bytes) {
  if (min_bytes <= workspace_bytes_) {
    return workspace_addr_;
  }
  if (workspace_is_borrowed_) {
    throw std::runtime_error("HbmArena: required workspace exceeds borrowed buffer");
  }
  if (!owns_slab_) {
    workspace_addr_ = allocate(min_bytes);
    workspace_bytes_ = min_bytes;
    ++generation_;
    return workspace_addr_;
  }
  if (workspace_addr_ != 0 && owns_workspace_) {
    synDeviceFree(device_id_, workspace_addr_, 0);
    workspace_addr_ = 0;
    workspace_bytes_ = 0;
  }
  if (synDeviceMalloc(device_id_, min_bytes, 0, 0, &workspace_addr_) !=
      synSuccess) {
    throw std::runtime_error("HbmArena: failed to allocate workspace HBM");
  }
  workspace_bytes_ = min_bytes;
  ++generation_;
  return workspace_addr_;
}

void HbmArena::set_borrowed_workspace(uint64_t address, uint64_t bytes) {
  if (address == 0 || bytes == 0) {
    throw std::invalid_argument("HbmArena: invalid borrowed workspace");
  }
  if (workspace_addr_ != 0 || workspace_is_borrowed_) {
    throw std::runtime_error("HbmArena: workspace is already configured");
  }
  workspace_addr_ = address;
  workspace_bytes_ = bytes;
  owns_workspace_ = false;
  workspace_is_borrowed_ = true;
  ++generation_;
}

void HbmArena::copy_h2d_sync(synStreamHandle h2d_stream, const void* host_src,
                             uint64_t dev_dst, uint64_t bytes) const {
  if (bytes == 0) return;
  if (synHostMap(device_id_, bytes, host_src) != synSuccess) {
    throw std::runtime_error("HbmArena: synHostMap failed in copy_h2d_sync");
  }
  synStatus st = synMemCopyAsync(
      h2d_stream, reinterpret_cast<uint64_t>(host_src), bytes, dev_dst,
      HOST_TO_DRAM);
  if (st == synSuccess) {
    st = synStreamSynchronize(h2d_stream);
  }
  synHostUnmap(device_id_, host_src);
  if (st != synSuccess) {
    throw std::runtime_error("HbmArena: H2D DMA failed");
  }
}

void HbmArena::copy_d2h_sync(synStreamHandle d2h_stream, uint64_t dev_src,
                             void* host_dst, uint64_t bytes) const {
  if (bytes == 0) return;
  if (synHostMap(device_id_, bytes, host_dst) != synSuccess) {
    throw std::runtime_error("HbmArena: synHostMap failed in copy_d2h_sync");
  }
  synStatus st = synMemCopyAsync(
      d2h_stream, dev_src, bytes, reinterpret_cast<uint64_t>(host_dst),
      DRAM_TO_HOST);
  if (st == synSuccess) {
    st = synStreamSynchronize(d2h_stream);
  }
  synHostUnmap(device_id_, host_dst);
  if (st != synSuccess) {
    throw std::runtime_error("HbmArena: D2H DMA failed");
  }
}

}  // namespace gaudirt::runtime
