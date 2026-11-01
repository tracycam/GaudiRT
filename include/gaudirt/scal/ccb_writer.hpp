// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "gaudirt/scal/scal_packet_decoder.hpp"

namespace gaudirt::scal {

inline size_t compute_packet_size(const uint8_t* ptr, size_t max_len) {
  if (!ptr || max_len < 4) return 0;
  const auto fits = [max_len](size_t bytes) {
    return bytes >= 4 && bytes % 4 == 0 && bytes <= max_len ? bytes : 0;
  };
  uint32_t dw0 = 0;
  std::memcpy(&dw0, ptr, sizeof(dw0));
  switch (dw0 & 0x1Fu) {
    case SCHED_COMPUTE_ARC_CMD_FENCE_WAIT:
      return fits(sizeof(sched_arc_cmd_fence_wait_t));
    case SCHED_COMPUTE_ARC_CMD_LBW_WRITE:
      return fits(sizeof(sched_arc_cmd_lbw_write_t));
    case SCHED_COMPUTE_ARC_CMD_LBW_BURST_WRITE: {
      const auto* c =
          reinterpret_cast<const sched_arc_cmd_lbw_burst_write_t*>(ptr);
      return fits(sizeof(*c) + 8ULL * c->num_lbw_write);
    }
    case SCHED_COMPUTE_ARC_CMD_DISPATCH_BARRIER:
      return fits(sizeof(sched_arc_cmd_dispatch_barrier_t));
    case SCHED_COMPUTE_ARC_CMD_FENCE_INC_IMMEDIATE: {
      const auto* c =
          reinterpret_cast<const sched_arc_cmd_fence_inc_immediate_t*>(ptr);
      return c->fence_count == 0
                 ? fits(sizeof(sched_arc_cmd_fence_inc_immediate_t))
                 : 0;
    }
    case SCHED_COMPUTE_ARC_CMD_NOP: {
      const auto* c = reinterpret_cast<const sched_arc_cmd_nop_t*>(ptr);
      return fits(sizeof(*c) + 4ULL * c->padding_count);
    }
    case SCHED_COMPUTE_ARC_CMD_ALLOC_BARRIER_V2:
      return fits(sizeof(sched_arc_cmd_alloc_barrier_v2_t));
    case SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3:
      return fits(sizeof(sched_arc_cmd_dispatch_compute_ecb_list_t));
    case SCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER: {
      const auto* c =
          reinterpret_cast<const sched_arc_cmd_pdma_batch_transfer_t*>(ptr);
      return fits(sizeof(*c) +
                  sizeof(sched_arc_pdma_commands_params_t) * c->batch_count);
    }
    case SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2: {
      const auto* c =
          reinterpret_cast<const sched_arc_cmd_update_recipe_base_v2_t*>(ptr);
      if (c->num_recipe_addrs > SCHED_CMD_UPDATE_RECIPE_MAX_ENG_GROUPS) return 0;
      return fits(sizeof(*c) + sizeof(uint64_t) * c->num_recipe_addrs);
    }
    case SCHED_COMPUTE_ARC_CMD_WAIT_FOR_EXT_SIGNAL:
      return fits(sizeof(sched_arc_cmd_wait_for_ext_signal_t));
    case SCHED_COMPUTE_ARC_CMD_MEM_FENCE:
      return fits(sizeof(sched_arc_cmd_mem_fence_t));
    default:
      return 0;
  }
}

inline size_t network_packet_size(const uint8_t* ptr, size_t max_len,
                                  SchedulerKind kind) {
  if (!ptr || max_len < 4 || kind == SchedulerKind::kCompute) return 0;
  const auto fits = [max_len](size_t bytes) {
    return bytes >= 4 && bytes % 4 == 0 && bytes <= max_len ? bytes : 0;
  };
  uint32_t dw0 = 0;
  std::memcpy(&dw0, ptr, sizeof(dw0));
  const uint32_t opcode = dw0 & 0x1Fu;
  if (opcode == 0) return fits(sizeof(sched_arc_cmd_fence_wait_t));
  if (opcode == 1) return fits(sizeof(sched_arc_cmd_lbw_write_t));
  if (opcode == 2) {
    const auto* c = reinterpret_cast<const sched_arc_cmd_lbw_burst_write_t*>(ptr);
    return fits(sizeof(*c) + 8ULL * c->num_lbw_write);
  }
  if (opcode == 3) {
    const auto* c =
        reinterpret_cast<const sched_arc_cmd_fence_inc_immediate_t*>(ptr);
    return c->fence_count == 0 ? fits(sizeof(*c)) : 0;
  }
  if (opcode == 4) {
    const auto* c = reinterpret_cast<const sched_arc_cmd_nop_t*>(ptr);
    return fits(sizeof(*c) + 4ULL * c->padding_count);
  }

  const bool scaleup = kind == SchedulerKind::kScaleUpSend ||
                       kind == SchedulerKind::kScaleUpRecv;
  const bool gc_or_scaleout = kind == SchedulerKind::kGcReduction ||
                              kind == SchedulerKind::kScaleOutSend ||
                              kind == SchedulerKind::kScaleOutRecv;
  if (scaleup && opcode == 5) {
    const auto* c =
        reinterpret_cast<const sched_arc_cmd_update_nic_glbl_ctxt_t*>(ptr);
    return fits(c->cmd_size);
  }
  if (scaleup && opcode == 6) {
    const auto* c =
        reinterpret_cast<const sched_arc_cmd_update_nic_coll_ctxt_t*>(ptr);
    return fits(8ULL + 4ULL * c->cmd_update_coll_ctxt.num_dwords);
  }
  if (scaleup && opcode == 7) {
    const auto* c = reinterpret_cast<const sched_arc_cmd_nic_coll_ops_t*>(ptr);
    return fits(c->cmd_size);
  }
  if ((scaleup && opcode == 8) || (gc_or_scaleout && opcode == 5)) {
    const auto* c =
        reinterpret_cast<const sched_arc_cmd_alloc_nic_barrier_t*>(ptr);
    const size_t fence_dwords = (c->fence_count > 0) + (c->fence_count > 4);
    const size_t expected = 4 + fence_dwords * 4 + 8ULL * c->num_lbw_write;
    return c->cmd_size_dwords * 4ULL == expected ? fits(expected) : 0;
  }
  if (scaleup && opcode == 9) {
    const auto* c = reinterpret_cast<const sched_arc_cmd_nic_passthrough_t*>(ptr);
    return fits(4ULL * c->cmd_dw_size);
  }
  if ((scaleup && opcode == 10) ||
      (kind == SchedulerKind::kGcReduction && opcode == 6)) {
    const auto* c = reinterpret_cast<const sched_arc_cmd_nic_edma_ops_t*>(ptr);
    return fits(c->cmd_size);
  }
  return 0;
}

struct CcbReplayStats {
  uint32_t start_pi = 0;
  uint32_t end_pi = 0;
  uint64_t submit_calls = 0;
  uint64_t boundary_submit_calls = 0;
  uint64_t nop_bytes = 0;

  uint64_t pi_bytes() const { return static_cast<uint32_t>(end_pi - start_pi); }
};

class CcbWriter {
 public:
  CcbWriter(uint8_t* host, uint32_t ccb_bytes, uint32_t start_pi,
            uint32_t command_alignment = 256,
            uint32_t nop_opcode = SCHED_COMPUTE_ARC_CMD_NOP)
      : host_(host),
        ccb_bytes_(ccb_bytes),
        alignment_(command_alignment),
        nop_opcode_(nop_opcode),
        pi_(start_pi) {
    stats_.start_pi = start_pi;
    stats_.end_pi = start_pi;
  }

  bool valid() const {
    return host_ && ccb_bytes_ && alignment_ >= 4 && nop_opcode_ < 32 &&
           alignment_ <= ccb_bytes_ && (ccb_bytes_ & (ccb_bytes_ - 1)) == 0 &&
           (alignment_ & (alignment_ - 1)) == 0;
  }

  template <typename Submit>
  int append(const uint8_t* packet, size_t bytes, bool submit_at_boundary,
             Submit&& submit) {
    uint32_t next_pi = 0;
    uint32_t nop_bytes = 0;
    if (!valid() || !packet ||
        !plan_append(pi_, bytes, alignment_, next_pi, nop_bytes)) {
      return SCAL_INVALID_PARAM;
    }
    const uint64_t start_to_next =
        static_cast<uint64_t>(static_cast<uint32_t>(next_pi - stats_.start_pi));
    if (start_to_next > ccb_bytes_) return SCAL_OUT_OF_MEMORY;

    const uint32_t ccb_off = pi_ & (ccb_bytes_ - 1);
    if (nop_bytes) {
      std::memset(host_ + ccb_off, 0, nop_bytes);
      auto* nop = reinterpret_cast<sched_arc_cmd_nop_t*>(host_ + ccb_off);
      nop->opcode = nop_opcode_;
      nop->padding_count = nop_bytes / 4 - 1;
      pi_ += nop_bytes;
      stats_.nop_bytes += nop_bytes;
      stats_.end_pi = pi_;
      if (submit_at_boundary) {
        const int rc = submit_now(submit, true);
        if (rc != SCAL_SUCCESS) return rc;
      }
    }

    const uint32_t packet_off = pi_ & (ccb_bytes_ - 1);
    std::memcpy(host_ + packet_off, packet, bytes);
    pi_ += static_cast<uint32_t>(bytes);
    stats_.end_pi = pi_;
    return SCAL_SUCCESS;
  }

  template <typename Submit>
  int submit_now(Submit&& submit, bool boundary = false) {
    ++stats_.submit_calls;
    if (boundary) ++stats_.boundary_submit_calls;
    return submit(pi_);
  }

  uint32_t pi() const { return pi_; }
  const CcbReplayStats& stats() const { return stats_; }

 private:
  static bool plan_append(uint32_t pi, size_t bytes, uint32_t alignment,
                          uint32_t& next_pi, uint32_t& nop_bytes) {
    if (!bytes || bytes % 4 || bytes > alignment) return false;
    const uint32_t rem = alignment - (pi & (alignment - 1));
    nop_bytes = bytes > rem ? rem : 0;
    const uint64_t next = static_cast<uint64_t>(pi) + nop_bytes + bytes;
    if (next > std::numeric_limits<uint32_t>::max()) return false;
    next_pi = static_cast<uint32_t>(next);
    return true;
  }

  uint8_t* host_ = nullptr;
  uint32_t ccb_bytes_ = 0;
  uint32_t alignment_ = 256;
  uint32_t nop_opcode_ = SCHED_COMPUTE_ARC_CMD_NOP;
  uint32_t pi_ = 0;
  CcbReplayStats stats_;
};

}  // namespace gaudirt::scal
