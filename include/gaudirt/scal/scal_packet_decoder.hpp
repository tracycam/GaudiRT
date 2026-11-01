// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include "gaudirt/scal/gaudi2_arc_common_packets.h"
#include "gaudirt/scal/gaudi2_arc_eng_packets.h"
#include "gaudirt/scal/gaudi2_arc_sched_packets.h"
#include "gaudirt/scal/scal.h"
#pragma GCC diagnostic pop

namespace gaudirt::scal {

enum class SchedulerKind {
  kCompute = 0,
  kScaleUpSend = 1,
  kScaleUpRecv = 2,
  kScaleOutSend = 3,
  kScaleOutRecv = 4,
  kGcReduction = 5,
  kUnknown = 6
};

inline SchedulerKind classify_scheduler(const std::string& sched_name) {
  if (sched_name.find("scaleup_send") != std::string::npos) {
    return SchedulerKind::kScaleUpSend;
  }
  if (sched_name.find("scaleup_receive") != std::string::npos ||
      sched_name.find("scaleup_recv") != std::string::npos) {
    return SchedulerKind::kScaleUpRecv;
  }
  if (sched_name.find("scaleout_send") != std::string::npos) {
    return SchedulerKind::kScaleOutSend;
  }
  if (sched_name.find("scaleout_receive") != std::string::npos ||
      sched_name.find("scaleout_recv") != std::string::npos) {
    return SchedulerKind::kScaleOutRecv;
  }
  if (sched_name.find("garbage_collector") != std::string::npos ||
      sched_name.find("reduction") != std::string::npos) {
    return SchedulerKind::kGcReduction;
  }
  return SchedulerKind::kCompute;
}

inline const char* compute_group_name(uint32_t group) {
  switch (group) {
    case SCAL_MME_COMPUTE_GROUP:
      return "MME";
    case SCAL_TPC_COMPUTE_GROUP:
      return "TPC";
    case SCAL_EDMA_COMPUTE_GROUP:
      return "EDMA";
    case SCAL_RTR_COMPUTE_GROUP:
      return "RTR";
    case SCAL_PDMA_TX_CMD_GROUP:
      return "PDMA_TX_CMD";
    case SCAL_PDMA_TX_DATA_GROUP:
      return "PDMA_TX_DATA";
    case SCAL_PDMA_RX_GROUP:
      return "PDMA_RX";
    case SCAL_PDMA_RX_DEBUG_GROUP:
      return "PDMA_RX_DBG";
    case SCAL_PDMA_DEV2DEV_DEBUG_GROUP:
      return "PDMA_D2D_DBG";
    case SCAL_CME_GROUP:
      return "CME";
    default:
      return "UNKNOWN_GRP";
  }
}

struct MonitorShadowState {
  uint32_t pay_addrl = 0;
  uint32_t pay_addrh = 0;
  uint32_t pay_data = 0;
  uint32_t config = 0;
  uint32_t arm = 0;
  bool has_config = false;
};

struct PdmaBlockSnapshot {
  std::string phase;
  std::string stream_name;
  uint64_t src_addr = 0;
  uint64_t dst_addr = 0;
  uint32_t size = 0;
  std::vector<uint8_t> payload;
};

// Tracks HBM shadow memory (populated by PDMA_BATCH_TRANSFER), recipe base
// registers, and Sync Manager monitor states so we can decode the entire
// hardware execution chain down to ECB lists and SOB/Monitor/Fence links.
class HardwareStateTracker {
 public:
  static constexpr uint32_t kDcoreSmBase[4] = {0xFC100000u, 0xFC300000u,
                                               0xFC500000u, 0xFC700000u};
  static constexpr uint32_t kDcoreSmMaxOffset = 0x15A00u;

  void register_host_mapping(uint64_t dev_va, const void* host_ptr,
                             uint64_t size, const std::string& tag) {
    if (!host_ptr || size == 0) return;
    host_mappings_[dev_va] =
        HostMapping{dev_va, reinterpret_cast<const uint8_t*>(host_ptr), size, tag};
  }

  void unregister_host_mapping(uint64_t dev_va) { host_mappings_.erase(dev_va); }

  void register_scheduler_dccm(uint32_t dccm_addr_lo,
                               const std::string& sched_name,
                               uint32_t sched_idx) {
    dccm_to_sched_[dccm_addr_lo] =
        sched_name + "(idx=" + std::to_string(sched_idx) + ")";
  }

  const uint8_t* translate_dev_to_host(uint64_t dev_va,
                                       uint64_t size) const {
    for (const auto& kv : host_mappings_) {
      const auto& m = kv.second;
      if (dev_va >= m.dev_va && dev_va + size <= m.dev_va + m.size) {
        return m.host_ptr + (dev_va - m.dev_va);
      }
    }
    return nullptr;
  }

  void record_pdma_to_hbm(const std::string& phase,
                          const std::string& stream_name, uint64_t src_addr,
                          uint64_t dst_addr, uint32_t size) {
    const uint8_t* src_ptr = translate_dev_to_host(src_addr, size);
    if (!src_ptr) return;
    // Write to byte-granular HBM shadow pages (4 KiB pages)
    for (uint32_t off = 0; off < size; ++off) {
      uint64_t addr = dst_addr + off;
      uint64_t page = addr >> 12;
      auto& pg = hbm_pages_[page];
      if (pg.empty()) pg.resize(4096, 0);
      pg[addr & 0xFFFu] = src_ptr[off];
    }
    PdmaBlockSnapshot snap;
    snap.phase = phase;
    snap.stream_name = stream_name;
    snap.src_addr = src_addr;
    snap.dst_addr = dst_addr;
    snap.size = size;
    snap.payload.assign(src_ptr, src_ptr + size);
    pdma_snapshots_.push_back(std::move(snap));
  }

  bool read_hbm_shadow(uint64_t hbm_addr, uint32_t size,
                       std::vector<uint8_t>& out) const {
    out.resize(size);
    for (uint32_t off = 0; off < size; ++off) {
      uint64_t addr = hbm_addr + off;
      uint64_t page = addr >> 12;
      auto it = hbm_pages_.find(page);
      if (it == hbm_pages_.end()) {
        return false;
      }
      out[off] = it->second[addr & 0xFFFu];
    }
    return true;
  }

  void set_recipe_base(uint32_t eng_group, uint32_t slot, uint64_t addr) {
    if (eng_group < 16 && slot < 8) {
      recipe_bases_[eng_group][slot] = addr;
    }
  }

  uint64_t get_recipe_base(uint32_t eng_group, uint32_t slot) const {
    if (eng_group < 16 && slot < 8) {
      return recipe_bases_[eng_group][slot];
    }
    return 0;
  }

  // Decode an LBW address (0xFC10xxxx..0xFC70xxxx or 27-bit 0x410xxxx..0x470xxxx)
  std::string decode_lbw_write(uint32_t raw_addr, uint32_t data,
                               bool update_state = true) {
    uint32_t addr = raw_addr;
    if ((addr & 0xF8000000u) == 0x00000000u && (addr & 0x07000000u) == 0x04000000u) {
      addr |= 0xF8000000u;
    }
    auto dccm_it = dccm_to_sched_.find(addr & 0xFFFFFF00u);
    if (dccm_it != dccm_to_sched_.end()) {
      std::ostringstream oss;
      oss << dccm_it->second << " DCCM_Q";
      if ((data & 0x3u) == 0) {
        oss << ": MON_EXP_FENCE_UPDATE(fence_id=" << ((data >> 2) & 0x3Fu) << ")";
      } else {
        oss << " = 0x" << std::hex << data << std::dec;
      }
      return oss.str();
    }
    for (int dcore = 0; dcore < 4; ++dcore) {
      uint32_t base = kDcoreSmBase[dcore];
      if (addr >= base && addr < base + kDcoreSmMaxOffset) {
        uint32_t off = addr - base;
        std::ostringstream oss;
        if (off < 0x8000u) {
          uint32_t sob_id = off >> 2;
          uint32_t val = data & 0x7FFFu;
          uint32_t long_sob = (data >> 24) & 1u;
          uint32_t inc = (data >> 31) & 1u;
          oss << "DCORE" << dcore << "_SOB[" << sob_id << "] "
              << (inc ? "+=" : "=") << " " << val
              << (long_sob ? " (long_sob)" : "");
          return oss.str();
        } else if (off >= 0x8000u && off < 0xA000u) {
          uint32_t mon_id = (off - 0x8000u) >> 2;
          if (update_state) monitors_[dcore * 2048 + mon_id].pay_addrl = data;
          oss << "DCORE" << dcore << "_MON_PAY_ADDRL[" << mon_id << "] = 0x"
              << std::hex << data << std::dec;
          auto it = dccm_to_sched_.find(data & 0xFFFFFF00u);
          if (it != dccm_to_sched_.end()) {
            oss << " -> " << it->second;
          } else {
            // Check if target is an SOB
            std::string target_sob = decode_sob_only_addr(data);
            if (!target_sob.empty()) oss << " -> " << target_sob;
          }
          return oss.str();
        } else if (off >= 0xA000u && off < 0xC000u) {
          uint32_t mon_id = (off - 0xA000u) >> 2;
          if (update_state) monitors_[dcore * 2048 + mon_id].pay_addrh = data;
          oss << "DCORE" << dcore << "_MON_PAY_ADDRH[" << mon_id << "] = 0x"
              << std::hex << data << std::dec;
          return oss.str();
        } else if (off >= 0xC000u && off < 0xE000u) {
          uint32_t mon_id = (off - 0xC000u) >> 2;
          if (update_state) monitors_[dcore * 2048 + mon_id].pay_data = data;
          oss << "DCORE" << dcore << "_MON_PAY_DATA[" << mon_id << "] = 0x"
              << std::hex << data << std::dec;
          if ((data & 0xFFFFFF00u) == 0 && (data & 0x3u) == 0) {
            uint32_t fence_id = (data >> 2) & 0x3Fu;
            oss << " (MON_EXP_FENCE_UPDATE: fence_id=" << fence_id << ")";
          } else if (data & 0x80000000u) {
            oss << " (SOB_INC +" << (data & 0x7FFFu) << ")";
          }
          return oss.str();
        } else if (off >= 0xE000u && off < 0x10000u) {
          uint32_t mon_id = (off - 0xE000u) >> 2;
          auto& m = monitors_[dcore * 2048 + mon_id];
          if (update_state) m.arm = data;
          uint32_t sid = data & 0xFFu;
          uint32_t mask = (data >> 8) & 0xFFu;
          uint32_t sop = (data >> 16) & 1u;
          uint32_t sod = (data >> 17) & 0x7FFFu;
          uint32_t msb_sid = (m.config >> 16) & 0xFu;
          uint32_t base_sob = ((msb_sid << 8) | sid) * 8u;
          oss << "DCORE" << dcore << "_MON_ARM[" << mon_id << "]: raw=0x"
              << std::hex << std::setw(8) << std::setfill('0') << data
              << " sid=0x" << sid << " mask=0x" << mask << std::dec
              << " sop=" << sop << " target=0x" << std::hex << sod << std::dec
              << "(" << sod << ") watch SOB_group(" << base_sob << ".."
              << (base_sob + 7) << ", mask=0x" << std::hex << mask << std::dec
              << ") " << (sop ? "==" : ">=") << " " << sod;
          if (m.pay_addrl != 0) {
            oss << " => fire [addr=0x" << std::hex << m.pay_addrl
                << ", data=0x" << m.pay_data << std::dec << "]";
          }
          return oss.str();
        } else if (off >= 0x10000u && off < 0x12000u) {
          uint32_t mon_id = (off - 0x10000u) >> 2;
          if (update_state) {
            monitors_[dcore * 2048 + mon_id].config = data;
            monitors_[dcore * 2048 + mon_id].has_config = true;
          }
          uint32_t long_sob = data & 1u;
          uint32_t cq_en = (data >> 4) & 1u;
          uint32_t wr_num = (data >> 5) & 3u;
          uint32_t lbw_en = (data >> 8) & 1u;
          uint32_t msb_sid = (data >> 16) & 0xFu;
          oss << "DCORE" << dcore << "_MON_CONFIG[" << mon_id
              << "]: msb_sid=" << msb_sid << " lbw_en=" << lbw_en
              << " wr_num=" << (wr_num + 1) << " cq_en=" << cq_en
              << " long_sob=" << long_sob;
          return oss.str();
        }
      }
    }
    std::ostringstream oss;
    oss << "LBW[0x" << std::hex << raw_addr << "] = 0x" << data << std::dec;
    return oss.str();
  }

  std::string decode_sob_only_addr(uint32_t raw_addr) const {
    uint32_t addr = raw_addr;
    if ((addr & 0xF8000000u) == 0x00000000u && (addr & 0x07000000u) == 0x04000000u) {
      addr |= 0xF8000000u;
    }
    for (int dcore = 0; dcore < 4; ++dcore) {
      uint32_t base = kDcoreSmBase[dcore];
      if (addr >= base && addr < base + 0x8000u) {
        uint32_t sob_id = (addr - base) >> 2;
        return "DCORE" + std::to_string(dcore) + "_SOB[" +
               std::to_string(sob_id) + "]";
      }
    }
    return "";
  }

  const std::vector<PdmaBlockSnapshot>& pdma_snapshots() const {
    return pdma_snapshots_;
  }

 private:
  struct HostMapping {
    uint64_t dev_va;
    const uint8_t* host_ptr;
    uint64_t size;
    std::string tag;
  };
  std::map<uint64_t, HostMapping> host_mappings_;
  std::unordered_map<uint64_t, std::vector<uint8_t>> hbm_pages_;
  uint64_t recipe_bases_[16][8] = {};
  std::unordered_map<uint32_t, MonitorShadowState> monitors_;
  std::unordered_map<uint32_t, std::string> dccm_to_sched_;
  std::vector<PdmaBlockSnapshot> pdma_snapshots_;
};

// Decodes an Engine ARC ECB command list (Static or Dynamic) from HBM shadow
inline std::string decode_ecb_list(const uint8_t* bytes, uint32_t max_bytes,
                                   uint32_t eng_group, bool is_static,
                                   const HardwareStateTracker& tracker) {
  std::ostringstream oss;
  if (!bytes || max_bytes < 4) return "      <empty ECB list>\n";

  const auto* list_hdr =
      reinterpret_cast<const eng_arc_cmd_list_size_t*>(bytes);
  if (list_hdr->cmd_type != ECB_CMD_LIST_SIZE) {
    oss << "      [WARN] First ECB cmd is not ECB_CMD_LIST_SIZE (cmd_type="
        << list_hdr->cmd_type << ")\n";
    return oss.str();
  }
  uint32_t total_size = list_hdr->list_size;
  if (total_size > max_bytes) total_size = max_bytes;

  uint32_t offset = 0;
  uint32_t cmd_idx = 0;
  while (offset + 4 <= total_size) {
    const auto* gen =
        reinterpret_cast<const eng_arc_cmd_generic_t*>(bytes + offset);
    uint32_t cmd_type = gen->cmd_type;
    uint32_t cmd_bytes = 4;

    oss << "      [" << (is_static ? "STATIC_ECB" : "DYNAMIC_ECB") << " #"
        << cmd_idx << " +0x" << std::hex << offset << std::dec << "] ";

    switch (cmd_type) {
      case ECB_CMD_LIST_SIZE: {
        const auto* c =
            reinterpret_cast<const eng_arc_cmd_list_size_t*>(bytes + offset);
        oss << "ECB_CMD_LIST_SIZE: list_size=" << c->list_size
            << "B topology_start=" << c->topology_start
            << " yield=" << c->yield << "\n";
        cmd_bytes = 4;
        break;
      }
      case ECB_CMD_NOP: {
        const auto* c =
            reinterpret_cast<const eng_arc_cmd_nop_t*>(bytes + offset);
        cmd_bytes = 4 + c->padding * 4;
        oss << "ECB_CMD_NOP: padding_dwords=" << c->padding << " ("
            << cmd_bytes << "B) switch_cq=" << c->switch_cq
            << " dma_comp=" << c->dma_completion << " yield=" << c->yield
            << "\n";
        break;
      }
      case ECB_CMD_WD_FENCE_AND_EXE: {
        const auto* c =
            reinterpret_cast<const eng_arc_cmd_wd_fence_and_exec_t*>(bytes +
                                                                     offset);
        oss << "ECB_CMD_WD_FENCE_AND_EXE: wd_ctxt_id=" << c->wd_ctxt_id
            << " wd_ctxt2_id=" << c->wd_ctxt2_id
            << " patch_addr=" << c->patch_address
            << " patch_addr2=" << c->patch_address2
            << " dma_comp=" << c->dma_completion
            << " dma2_comp=" << c->dma2_completion
            << " signal_arc=" << c->signal_arc << " yield=" << c->yield << "\n";
        cmd_bytes = 4;
        break;
      }
      case ECB_CMD_SCHED_DMA: {
        if (offset + 8 > total_size) break;
        const auto* c =
            reinterpret_cast<const eng_arc_cmd_sched_dma_t*>(bytes + offset);
        uint64_t base = tracker.get_recipe_base(eng_group, c->addr_index);
        uint64_t hbm_src = base + c->addr_offset;
        oss << "ECB_CMD_SCHED_DMA: base_slot=" << c->addr_index << " + 0x"
            << std::hex << c->addr_offset << " (HBM 0x" << hbm_src << std::dec
            << ") -> DCCM_gc_ctxt_off=0x" << std::hex << c->gc_ctxt_offset
            << std::dec << " size=" << c->size << "B yield=" << c->yield
            << "\n";
        std::vector<uint8_t> dma_payload;
        if (base != 0 &&
            tracker.read_hbm_shadow(hbm_src, c->size, dma_payload)) {
          if (eng_group == SCAL_TPC_COMPUTE_GROUP &&
              c->size == sizeof(tpc_wd_ctxt_t)) {
            const auto* tpc =
                reinterpret_cast<const tpc_wd_ctxt_t*>(dma_payload.data());
            oss << "        -> TPC_WD_CTXT: grid=[" << tpc->ist.grid_size[0]
                << "," << tpc->ist.grid_size[1] << "," << tpc->ist.grid_size[2]
                << "] box=[" << tpc->ist.box_size[0] << ","
                << tpc->ist.box_size[1] << "," << tpc->ist.box_size[2]
                << "] switch_bit=" << (int)tpc->switch_bit
                << " virt_sob_bmp=0x" << std::hex
                << (int)tpc->virtual_sob_bitmap << std::dec
                << " sig_inc=" << tpc->sig_inc_value << "\n";
          }
        }
        cmd_bytes = 8;
        break;
      }
      case ECB_CMD_SCHED_DMA_V2: {
        if (offset + 8 > total_size) break;
        const auto* c =
            reinterpret_cast<const eng_arc_cmd_sched_dma_v2_t*>(bytes + offset);
        uint64_t base = tracker.get_recipe_base(eng_group, c->addr_index);
        uint64_t hbm_src = base + c->addr_offset;
        oss << "ECB_CMD_SCHED_DMA_V2: dma_type="
            << (c->dma_type ? "HBM_TENSOR_ADDR" : "EXPERT_MAP")
            << " base_slot=" << c->addr_index << " + 0x" << std::hex
            << c->addr_offset << " (HBM 0x" << hbm_src << std::dec
            << ") wd_ctxt_id=" << c->wd_ctxt_id << " size=" << c->size
            << "B dma_comp=" << c->dma_completion << " yield=" << c->yield
            << "\n";
        cmd_bytes = 8;
        break;
      }
      case ECB_CMD_STATIC_DESC_V2: {
        if (offset + 8 > total_size) break;
        const auto* c =
            reinterpret_cast<const eng_arc_cmd_static_desc_v2_t*>(bytes +
                                                                  offset);
        uint64_t base = tracker.get_recipe_base(eng_group, c->addr_index);
        uint64_t hbm_src = base + c->addr_offset;
        oss << "ECB_CMD_STATIC_DESC_V2: cpu_idx=";
        if (c->cpu_index == CPU_INDEX_ALL) {
          oss << "ALL(0xFE)";
        } else {
          oss << c->cpu_index;
        }
        oss << " base_slot=" << c->addr_index << " + 0x" << std::hex
            << c->addr_offset << " (HBM 0x" << hbm_src << std::dec
            << ") size=" << c->size << "B yield=" << c->yield << "\n";
        // Dump first up to 8 dwords of static/patched QMAN descriptor in HBM
        std::vector<uint8_t> desc_bytes;
        if (base != 0 &&
            tracker.read_hbm_shadow(hbm_src, c->size, desc_bytes)) {
          oss << "        -> QMAN_CP_BLOBS (" << c->size << "B): ";
          uint32_t dwords = c->size / 4;
          uint32_t show = dwords < 8 ? dwords : 8;
          const auto* dw = reinterpret_cast<const uint32_t*>(desc_bytes.data());
          for (uint32_t i = 0; i < show; ++i) {
            oss << "0x" << std::hex << std::setw(8) << std::setfill('0')
                << dw[i] << " " << std::dec;
          }
          if (dwords > show) oss << "...";
          oss << "\n";
        }
        cmd_bytes = 8;
        break;
      }
      case ECB_CMD_SFG: {
        const auto* c =
            reinterpret_cast<const eng_arc_cmd_sfg_t*>(bytes + offset);
        oss << "ECB_CMD_SFG: sob_inc_value=" << c->sob_inc_value
            << " switch_cq=" << c->switch_cq << " yield=" << c->yield << "\n";
        cmd_bytes = 4;
        break;
      }
      case ECB_CMD_RESET_SOSET: {
        const auto* c =
            reinterpret_cast<const eng_arc_cmd_reset_soset_t*>(bytes + offset);
        oss << "ECB_CMD_RESET_SOSET: num_cmpt_engines=" << c->num_cmpt_engines
            << " target=0x" << std::hex << c->target << std::dec
            << " switch_cq=" << c->switch_cq << " yield=" << c->yield << "\n";
        cmd_bytes = 4;
        break;
      }
      default: {
        oss << "ECB_CMD_UNKNOWN(" << cmd_type << ")\n";
        cmd_bytes = 4;
        break;
      }
    }
    offset += cmd_bytes;
    ++cmd_idx;
  }
  return oss.str();
}

// Decodes a slice of scheduler CCB commands submitted via scal_stream_submit
inline std::string decode_ccb_commands(const std::string& phase,
                                       const std::string& stream_name,
                                       const std::string& sched_name,
                                       const uint8_t* buf, uint32_t len,
                                       uint32_t start_pi_offset,
                                       uint64_t global_hbm_base,
                                       HardwareStateTracker& tracker) {
  std::ostringstream oss;
  SchedulerKind kind = classify_scheduler(sched_name);
  uint32_t off = 0;
  uint32_t cmd_idx = 0;

  while (off + 4 <= len) {
    const uint8_t* ptr = buf + off;
    uint32_t dw0 = *reinterpret_cast<const uint32_t*>(ptr);
    uint32_t opcode = dw0 & 0x1Fu;
    uint32_t cmd_size = 4;

    oss << "  [Cmd #" << cmd_idx << " ccb_off=0x" << std::hex
        << (start_pi_offset + off) << " dw0=0x" << std::setw(8)
        << std::setfill('0') << dw0 << std::dec << "] ";

    if (kind == SchedulerKind::kCompute) {
      switch (opcode) {
        case SCHED_COMPUTE_ARC_CMD_FENCE_WAIT: {
          const auto* c =
              reinterpret_cast<const sched_arc_cmd_fence_wait_t*>(ptr);
          oss << "FENCE_WAIT: fence_id=" << c->fence_id
              << " target=" << c->target << "\n";
          cmd_size = sizeof(sched_arc_cmd_fence_wait_t);
          break;
        }
        case SCHED_COMPUTE_ARC_CMD_LBW_WRITE: {
          const auto* c =
              reinterpret_cast<const sched_arc_cmd_lbw_write_t*>(ptr);
          oss << "LBW_WRITE: dst=0x" << std::hex << c->dst_addr << " data=0x"
              << c->src_data << std::dec << " ("
              << tracker.decode_lbw_write(c->dst_addr, c->src_data) << ")";
          if (c->fence) {
            oss << " [wait fence_id=" << c->fence_id << " target=" << c->target
                << "]";
          }
          oss << "\n";
          cmd_size = sizeof(sched_arc_cmd_lbw_write_t);
          break;
        }
        case SCHED_COMPUTE_ARC_CMD_LBW_BURST_WRITE: {
          const auto* c =
              reinterpret_cast<const sched_arc_cmd_lbw_burst_write_t*>(ptr);
          cmd_size = sizeof(sched_arc_cmd_lbw_burst_write_t) +
                     8 * c->num_lbw_write;
          oss << "LBW_BURST_WRITE: count=" << c->num_lbw_write
              << " wait_comp=" << c->wait_for_completion
              << " block_next=" << c->block_next << "\n";
          for (uint32_t i = 0; i < c->num_lbw_write; ++i) {
            oss << "    [" << i << "] 0x" << std::hex << c->addr_data[i].addr
                << " = 0x" << c->addr_data[i].data << std::dec << " ("
                << tracker.decode_lbw_write(c->addr_data[i].addr,
                                            c->addr_data[i].data)
                << ")\n";
          }
          break;
        }
        case SCHED_COMPUTE_ARC_CMD_DISPATCH_BARRIER: {
          const auto* c =
              reinterpret_cast<const sched_arc_cmd_dispatch_barrier_t*>(ptr);
          oss << "DISPATCH_BARRIER: num_eng_groups=" << c->num_engine_group_type
              << " wd_sig=" << c->watch_dog_sig_value << " groups=[";
          for (uint32_t i = 0; i < c->num_engine_group_type && i < 4; ++i) {
            if (i) oss << ",";
            oss << compute_group_name(c->engine_group_type[i]);
          }
          oss << "]\n";
          cmd_size = sizeof(sched_arc_cmd_dispatch_barrier_t);
          break;
        }
        case SCHED_COMPUTE_ARC_CMD_FENCE_INC_IMMEDIATE: {
          const auto* c =
              reinterpret_cast<const sched_arc_cmd_fence_inc_immediate_t*>(ptr);
          oss << "FENCE_INC_IMMEDIATE: fence_index=" << c->fence_index
              << " fence_count=" << c->fence_count << "\n";
          cmd_size = sizeof(sched_arc_cmd_fence_inc_immediate_t);
          break;
        }
        case SCHED_COMPUTE_ARC_CMD_NOP: {
          const auto* c = reinterpret_cast<const sched_arc_cmd_nop_t*>(ptr);
          cmd_size = sizeof(sched_arc_cmd_nop_t) + 4 * c->padding_count;
          oss << "NOP: padding_dwords=" << c->padding_count << " (" << cmd_size
              << "B)\n";
          break;
        }
        case SCHED_COMPUTE_ARC_CMD_ALLOC_BARRIER_V2: {
          const auto* c =
              reinterpret_cast<const sched_arc_cmd_alloc_barrier_v2_t*>(ptr);
          oss << "ALLOC_BARRIER_V2: cg_idx=" << c->comp_group_index
              << " target_val=" << c->target_value
              << " alloc_so_set=" << c->allocate_so_set
              << " rel_so_set=" << c->rel_so_set
              << " no_fence=" << c->b2b_exec_with_no_fence
              << " num_eng_groups=" << c->num_engine_group_type << " groups=[";
          for (uint32_t i = 0; i < c->num_engine_group_type && i < 4; ++i) {
            if (i) oss << ",";
            oss << compute_group_name(c->engine_group_type[i]);
          }
          oss << "]\n";
          cmd_size = sizeof(sched_arc_cmd_alloc_barrier_v2_t);
          break;
        }
        case SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3: {
          const auto* c =
              reinterpret_cast<const sched_arc_cmd_dispatch_compute_ecb_list_t*>(
                  ptr);
          uint64_t dyn_addr =
              (global_hbm_base & 0xFFFFFFFF00000000ULL) |
              (static_cast<uint64_t>(c->dynamic_ecb_list_addr_256) << 8);
          uint64_t stat_addr =
              dyn_addr +
              (static_cast<uint64_t>(c->static_ecb_list_offset) << 8);
          oss << "DISPATCH_COMPUTE_ECB_LIST_V3: group="
              << compute_group_name(c->engine_group_type) << "("
              << c->engine_group_type
              << ") single_stat=" << c->single_static_chunk
              << " single_dyn=" << c->single_dynamic_chunk
              << " dyn_ecb_hbm=0x" << std::hex << dyn_addr
              << " stat_ecb_hbm=0x" << stat_addr << " stat_eng_stride=0x"
              << (c->static_ecb_list_eng_offset << 8) << std::dec << "\n";

          std::vector<uint8_t> ecb_buf;
          if (tracker.read_hbm_shadow(dyn_addr, 512, ecb_buf)) {
            oss << decode_ecb_list(ecb_buf.data(), ecb_buf.size(),
                                   c->engine_group_type, false, tracker);
          }
          if (tracker.read_hbm_shadow(stat_addr, 512, ecb_buf)) {
            oss << decode_ecb_list(ecb_buf.data(), ecb_buf.size(),
                                   c->engine_group_type, true, tracker);
          }
          cmd_size = sizeof(sched_arc_cmd_dispatch_compute_ecb_list_t);
          break;
        }
        case SCHED_COMPUTE_ARC_CMD_PDMA_BATCH_TRANSFER: {
          const auto* c =
              reinterpret_cast<const sched_arc_cmd_pdma_batch_transfer_t*>(ptr);
          cmd_size = sizeof(sched_arc_cmd_pdma_batch_transfer_t) +
                     sizeof(sched_arc_pdma_commands_params_t) * c->batch_count;
          uint32_t dir = c->stream_ctxt_id & 0x3u;
          const char* dir_str =
              (dir == PDMA_DIR_HOST2DEV)
                  ? "HOST2DEV"
                  : (dir == PDMA_DIR_DEV2HOST ? "DEV2HOST" : "DEV2DEV");
          oss << "PDMA_BATCH_TRANSFER: group="
              << compute_group_name(c->engine_group_type)
              << " dir=" << dir_str << " workload=" << c->workload_type
              << " batch_count=" << c->batch_count
              << " has_payload=" << c->has_payload
              << " signal_to_cg=" << c->signal_to_cg;
          if (c->has_payload || c->signal_to_cg) {
            oss << " pay_addr=0x" << std::hex << c->pay_addr << " pay_data=0x"
                << c->pay_data << std::dec << " ("
                << tracker.decode_lbw_write(c->pay_addr, c->pay_data, false)
                << ")";
          }
          oss << "\n";
          for (uint32_t i = 0; i < c->batch_count; ++i) {
            const auto& bp = c->batch_params[i];
            oss << "    [Batch #" << i << "] src=0x" << std::hex << bp.src_addr
                << " -> dst=0x" << bp.dst_addr << std::dec
                << " size=" << bp.transfer_size << "B";
            const uint8_t* host_src =
                tracker.translate_dev_to_host(bp.src_addr, bp.transfer_size);
            if (host_src) {
              tracker.record_pdma_to_hbm(phase, stream_name, bp.src_addr,
                                         bp.dst_addr, bp.transfer_size);
              oss << " [HOST_MAPPED: ";
              uint32_t dwords = bp.transfer_size / 4;
              uint32_t show = dwords < 6 ? dwords : 6;
              const auto* dw = reinterpret_cast<const uint32_t*>(host_src);
              for (uint32_t k = 0; k < show; ++k) {
                oss << "0x" << std::hex << std::setw(8) << std::setfill('0')
                    << dw[k] << " " << std::dec;
              }
              if (dwords > show) oss << "...";
              oss << "]";
            }
            oss << "\n";
          }
          break;
        }
        case SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2: {
          const auto* c =
              reinterpret_cast<const sched_arc_cmd_update_recipe_base_v2_t*>(
                  ptr);
          cmd_size = sizeof(sched_arc_cmd_update_recipe_base_v2_t) +
                     sizeof(uint64_t) * c->num_recipe_addrs;
          uint32_t idx[4] = {c->recipe_base_index0, c->recipe_base_index1,
                             c->recipe_base_index2, c->recipe_base_index3};
          oss << "UPDATE_RECIPE_BASE_V2: num_addrs=" << c->num_recipe_addrs
              << " groups=[";
          for (uint32_t g = 0; g < c->num_engine_group_type && g < 4; ++g) {
            if (g) oss << ",";
            oss << compute_group_name(c->engine_group_type[g]);
          }
          oss << "]";
          for (uint32_t i = 0; i < c->num_recipe_addrs; ++i) {
            oss << " base[" << idx[i] << "]=0x" << std::hex
                << c->recipe_base_addr[i] << std::dec;
            for (uint32_t g = 0; g < c->num_engine_group_type && g < 4; ++g) {
              tracker.set_recipe_base(c->engine_group_type[g], idx[i],
                                      c->recipe_base_addr[i]);
            }
          }
          oss << "\n";
          break;
        }
        case SCHED_COMPUTE_ARC_CMD_WAIT_FOR_EXT_SIGNAL: {
          const auto* c =
              reinterpret_cast<const sched_arc_cmd_wait_for_ext_signal_t*>(ptr);
          const uint64_t target =
              static_cast<uint64_t>(c->target_value0) |
              (static_cast<uint64_t>(c->target_value1) << 15) |
              (static_cast<uint64_t>(c->target_value2) << 30) |
              (static_cast<uint64_t>(c->target_value3) << 45);
          oss << "WAIT_FOR_EXT_SIGNAL: sm_id=" << c->sm_id
              << " sob_id=" << c->sob_id << " raw=[";
          for (uint32_t i = 0; i < sizeof(*c) / sizeof(uint32_t); ++i) {
            if (i) oss << " ";
            oss << "0x" << std::hex << std::setw(8) << std::setfill('0')
                << reinterpret_cast<const uint32_t*>(ptr)[i];
          }
          oss << std::dec << "] target_chunks=[0x" << std::hex
              << c->target_value0 << ",0x" << c->target_value1 << ",0x"
              << c->target_value2 << ",0x" << c->target_value3 << std::dec
              << "] target=0x" << std::hex << target << std::dec << "(" << target
              << ")\n";
          cmd_size = sizeof(sched_arc_cmd_wait_for_ext_signal_t);
          break;
        }
        case SCHED_COMPUTE_ARC_CMD_MEM_FENCE: {
          const auto* c =
              reinterpret_cast<const sched_arc_cmd_mem_fence_t*>(ptr);
          oss << "MEM_FENCE: arc=" << c->arc << " dup_eng=" << c->dup_eng
              << " arc_dma=" << c->arc_dma << "\n";
          cmd_size = sizeof(sched_arc_cmd_mem_fence_t);
          break;
        }
        default: {
          oss << "UNKNOWN_COMPUTE_OPCODE(" << opcode << ")\n";
          cmd_size = 4;
          break;
        }
      }
    } else {
      // Network Scheduler (ScaleUp Send, ScaleUp Recv, GcReduction, ScaleOut)
      // Common opcodes 0..4 across all network schedulers:
      // 0 = FENCE_WAIT, 1 = LBW_WRITE, 2 = LBW_BURST_WRITE,
      // 3 = FENCE_INC_IMMEDIATE, 4 = NOP
      if (opcode == 0) {
        const auto* c =
            reinterpret_cast<const sched_arc_cmd_fence_wait_t*>(ptr);
        oss << "FENCE_WAIT: fence_id=" << c->fence_id
            << " target=" << c->target << "\n";
        cmd_size = sizeof(sched_arc_cmd_fence_wait_t);
      } else if (opcode == 1) {
        const auto* c =
            reinterpret_cast<const sched_arc_cmd_lbw_write_t*>(ptr);
        oss << "LBW_WRITE: dst=0x" << std::hex << c->dst_addr << " data=0x"
            << c->src_data << std::dec << " ("
            << tracker.decode_lbw_write(c->dst_addr, c->src_data) << ")";
        if (c->fence) {
          oss << " [wait fence_id=" << c->fence_id << " target=" << c->target
              << "]";
        }
        oss << "\n";
        cmd_size = sizeof(sched_arc_cmd_lbw_write_t);
      } else if (opcode == 2) {
        const auto* c =
            reinterpret_cast<const sched_arc_cmd_lbw_burst_write_t*>(ptr);
        cmd_size =
            sizeof(sched_arc_cmd_lbw_burst_write_t) + 8 * c->num_lbw_write;
        oss << "LBW_BURST_WRITE: count=" << c->num_lbw_write << "\n";
        for (uint32_t i = 0; i < c->num_lbw_write; ++i) {
          oss << "    [" << i << "] 0x" << std::hex << c->addr_data[i].addr
              << " = 0x" << c->addr_data[i].data << std::dec << " ("
              << tracker.decode_lbw_write(c->addr_data[i].addr,
                                          c->addr_data[i].data)
              << ")\n";
        }
      } else if (opcode == 3) {
        const auto* c =
            reinterpret_cast<const sched_arc_cmd_fence_inc_immediate_t*>(ptr);
        oss << "FENCE_INC_IMMEDIATE: fence_index=" << c->fence_index
            << " fence_count=" << c->fence_count << "\n";
        cmd_size = sizeof(sched_arc_cmd_fence_inc_immediate_t);
      } else if (opcode == 4) {
        const auto* c = reinterpret_cast<const sched_arc_cmd_nop_t*>(ptr);
        cmd_size = sizeof(sched_arc_cmd_nop_t) + 4 * c->padding_count;
        oss << "NOP: padding_dwords=" << c->padding_count << " (" << cmd_size
            << "B)\n";
      } else if ((kind == SchedulerKind::kScaleUpSend ||
                  kind == SchedulerKind::kScaleUpRecv) &&
                 opcode == 5) {
        const auto* c =
            reinterpret_cast<const sched_arc_cmd_update_nic_glbl_ctxt_t*>(ptr);
        cmd_size = c->cmd_size;
        oss << "UPDATE_NIC_GLBL_CTXT: cmd_size=" << cmd_size
            << "B so_lbw_addr=0x" << std::hex << c->so_lbw_address << std::dec
            << " v2=" << c->cmd_update_glbl_ctxt.update_glbl_ctxt_v2
            << " num_dwords=" << c->cmd_update_glbl_ctxt.num_dwords << "\n";
      } else if ((kind == SchedulerKind::kScaleUpSend ||
                  kind == SchedulerKind::kScaleUpRecv) &&
                 opcode == 6) {
        const auto* c =
            reinterpret_cast<const sched_arc_cmd_update_nic_coll_ctxt_t*>(ptr);
        uint32_t ndw = c->cmd_update_coll_ctxt.num_dwords;
        cmd_size = 8 + 4 * ndw;
        oss << "UPDATE_NIC_COLL_CTXT: ctxt_id="
            << c->cmd_update_coll_ctxt.ctxt_id
            << " comm_desc_idx=" << c->cmd_update_coll_ctxt.comm_desc_index
            << " update_bitmap=0x" << std::hex
            << c->cmd_update_coll_ctxt.update_bitmap << std::dec
            << " update_qpn=" << c->cmd_update_coll_ctxt.update_qpn
            << " update_rri_ce=" << c->cmd_update_coll_ctxt.update_rri_ce
            << " num_dwords=" << ndw << " dwords=[";
        for (uint32_t i = 0; i < ndw; ++i) {
          if (i) oss << ", ";
          oss << "0x" << std::hex << c->dwords[i].dword_value << std::dec;
        }
        oss << "]\n";
      } else if ((kind == SchedulerKind::kScaleUpSend ||
                  kind == SchedulerKind::kScaleUpRecv) &&
                 opcode == 7) {
        const auto* c =
            reinterpret_cast<const sched_arc_cmd_nic_coll_ops_t*>(ptr);
        cmd_size = c->cmd_size;
        uint32_t dw1 = *reinterpret_cast<const uint32_t*>(ptr + 4);
        uint32_t nic_op5 = (dw1 >> 27) & 0x1Fu;
        if (nic_op5 == NIC_CMD_COLL_OPS_LONG) {
          const auto* l = &c->cmd_coll_ops_long[0];
          uint64_t addr_low45 =
              (static_cast<uint64_t>(l->addr_msb) << 32) | l->buffer_addr_lsb;
          oss << "NIC_COLL_OPS (LONG): ctxt_id=" << l->coll_ctxt_id
              << " comm_desc=" << l->comm_desc_index
              << " addr_low45=0x" << std::hex << addr_low45 << std::dec
              << " cache_lines=" << l->cache_line_count
              << " cl_rem=" << l->cache_line_remainder
              << " elem_rem=" << l->element_remainder
              << " force_rank0_off=" << l->force_remote_rank_offset
              << " has_size=" << l->has_size
              << " notify_rndv=" << l->notify_rndv_ack
              << " wait_rndv=" << l->wait_for_rndv_acks;
          if (l->has_size && cmd_size >= 20) {
            oss << " buffer_size="
                << *reinterpret_cast<const uint32_t*>(ptr + 16) << "B";
          }
          oss << "\n";
        } else if (nic_op5 == NIC_CMD_COLL_OPS_RECV_INORDER_V2) {
          const auto* r = &c->cmd_coll_ops_short_inorder_v2[0];
          oss << "NIC_COLL_OPS (RECV_INORDER_V2): ctxt_id=" << r->coll_ctxt_id
              << " comm_desc=" << r->comm_desc_index
              << " local_rank=" << r->local_rank_index
              << " cache_lines=" << r->cache_line_count
              << " sibo_idx=" << r->sibo_index << " pool_id=" << r->pool_id
              << " red_op=0x" << std::hex << r->reduction_opcode << std::dec
              << "\n";
        } else {
          const auto* s = &c->cmd_coll_ops_short[0];
          uint64_t lsb_addr = static_cast<uint64_t>(s->buffer_addr_lsb) << 4;
          oss << "NIC_COLL_OPS (SHORT): ctxt_id=" << s->coll_ctxt_id
              << " comm_desc=" << s->comm_desc_index << " addr_lsb=0x"
              << std::hex << lsb_addr << std::dec
              << " cache_lines=" << s->cache_line_count
              << " cl_rem=" << s->cache_line_remainder
              << " elem_rem=" << s->element_remainder
              << " force_rank0_off=" << s->force_remote_rank_offset
              << " has_size=" << s->has_size
              << " notify_rndv=" << s->notify_rndv_ack
              << " wait_rndv=" << s->wait_for_rndv_acks;
          if (s->has_size && cmd_size >= 16) {
            oss << " buffer_size="
                << *reinterpret_cast<const uint32_t*>(ptr + 12) << "B";
          }
          oss << "\n";
        }
      } else if (((kind == SchedulerKind::kScaleUpSend ||
                   kind == SchedulerKind::kScaleUpRecv) &&
                  opcode == 8) ||
                 ((kind == SchedulerKind::kGcReduction ||
                   kind == SchedulerKind::kScaleOutSend ||
                   kind == SchedulerKind::kScaleOutRecv) &&
                  opcode == 5)) {
        const auto* c =
            reinterpret_cast<const sched_arc_cmd_alloc_nic_barrier_t*>(ptr);
        cmd_size = c->cmd_size_dwords * 4;
        uint32_t fence_dwords = (c->fence_count > 0) + (c->fence_count > 4);
        const uint8_t* fence_ids =
            reinterpret_cast<const uint8_t*>(ptr + 4);
        const auto* lbw_arr = reinterpret_cast<const sched_arc_lbw_write_t*>(
            ptr + 4 + fence_dwords * 4);
        oss << "ALLOC_NIC_BARRIER: cg_idx=" << c->comp_group_index
            << " req_sobs=" << c->required_sobs
            << " fence_cnt=" << c->fence_count
            << " lbw_cnt=" << c->num_lbw_write << " fences=[";
        for (uint32_t i = 0; i < c->fence_count; ++i) {
          if (i) oss << ",";
          oss << static_cast<uint32_t>(fence_ids[i]);
        }
        oss << "]\n";
        for (uint32_t i = 0; i < c->num_lbw_write; ++i) {
          oss << "    [Barrier LBW #" << i << "] 0x" << std::hex
              << lbw_arr[i].addr << " = 0x" << lbw_arr[i].data << std::dec
              << " ("
              << tracker.decode_lbw_write(lbw_arr[i].addr, lbw_arr[i].data)
              << ")\n";
        }
      } else if ((kind == SchedulerKind::kScaleUpSend ||
                  kind == SchedulerKind::kScaleUpRecv) &&
                 opcode == 9) {
        const auto* c =
            reinterpret_cast<const sched_arc_cmd_nic_passthrough_t*>(ptr);
        cmd_size = c->cmd_dw_size * 4;
        oss << "NIC_PASSTHROUGH: dwords=" << c->cmd_dw_size
            << " q_credits_bytes=" << c->required_q_credits_inbytes << "\n";
        uint32_t rec_off = 4;
        uint32_t rec_idx = 0;
        while (rec_off + 8 <= cmd_size) {
          const auto* rec =
              reinterpret_cast<const nic_passthrough_data_t*>(ptr + rec_off);
          uint32_t rec_dwords = rec->num_payload_dwords ? 3 : 2;
          oss << "    [PT #" << rec_idx << "] dup_mask=0x" << std::hex
              << rec->dup_mask << std::dec << " is_nop=" << rec->is_nop
              << " is_last=" << rec->is_last_config << " ";
          if (rec->num_payload_dwords == 0) {
            const auto* nop = reinterpret_cast<const arc_cmd_nic_send_recv_nop_t*>(
                &rec->payload0);
            oss << "SEND_RECV_NOP(ctxt=" << nop->coll_ctxt_id
                << ", sob_inc=" << nop->sob_increment
                << ", q_cred=" << nop->queue_credits_bytes << ")\n";
          } else {
            const auto* sr =
                reinterpret_cast<const arc_cmd_send_recv_short_t*>(&rec->payload0);
            uint64_t addr_low50 =
                (static_cast<uint64_t>(sr->addr_msb) << 32) | sr->addr_lsb;
            oss << "SEND_RECV_SHORT(ctxt=" << sr->coll_ctxt_id
                << ", comm_desc=" << sr->comm_desc_index << ", addr_low50=0x"
                << std::hex << addr_low50 << std::dec
                << ", cache_lines=" << sr->cache_line_count
                << ", cl_rem=" << sr->cache_line_remainder
                << ", elem_rem=" << sr->element_remainder
                << ", sob_inc=" << sr->sob_increment << ")\n";
          }
          rec_off += rec_dwords * 4;
          ++rec_idx;
        }
      } else if ((kind == SchedulerKind::kScaleUpSend && opcode == 10) ||
                 (kind == SchedulerKind::kGcReduction && opcode == 6)) {
        const auto* c =
            reinterpret_cast<const sched_arc_cmd_nic_edma_ops_t*>(ptr);
        cmd_size = c->cmd_size;
        uint32_t sub_dw0 = *reinterpret_cast<const uint32_t*>(ptr + 4);
        uint32_t edma_op = sub_dw0 & 0xFu;
        oss << "NIC_EDMA_OPS: eng_group=" << c->engine_group_type
            << " cmd_size=" << cmd_size << "B sub_opcode=" << edma_op << " ";
        if (edma_op == NIC_EDMA_CMD_LIN_OPS_V3) {
          const auto* lin = &c->lin_ops_v3[0];
          uint64_t dst =
              (static_cast<uint64_t>(lin->dst_addr_hi) << 32) | lin->dst_addr_lo;
          uint64_t src_hi =
              (static_cast<uint64_t>(lin->dst_addr_hi & 0xFF000000u)) |
              lin->src_addr_hi;
          uint64_t src = (src_hi << 32) | lin->src_addr_lo;
          uint32_t sob_lbw = 0xF8000000u | lin->sob_address;
          oss << "LIN_OPS_V3: src=0x" << std::hex << src << " -> dst=0x" << dst
              << std::dec << " size=" << lin->transfer_size
              << "B red_ind=" << lin->reduction_ind
              << " sob=" << tracker.decode_sob_only_addr(sob_lbw) << "(0x"
              << std::hex << sob_lbw << std::dec << ")\n";
        } else if (edma_op == NIC_EDMA_CMD_SIBO_OPS_V3) {
          const auto* sibo = &c->sibo_ops_v3[0];
          uint64_t dst = (static_cast<uint64_t>(sibo->dst_addr_hi) << 32) |
                         sibo->dst_addr_lo;
          uint64_t src_hi =
              (static_cast<uint64_t>(sibo->dst_addr_hi & 0xFF000000u)) |
              sibo->src_addr_hi;
          uint64_t src = (src_hi << 32) | sibo->src_addr_lo;
          oss << "SIBO_OPS_V3: src=0x" << std::hex << src << " -> dst=0x" << dst
              << std::dec << " size=" << sibo->transfer_size
              << "B sibo_idx=" << sibo->sibo_index
              << " rank_cnt=" << sibo->rank_count
              << " sob_base=" << sibo->sob_base << " sob_idx=" << sibo->sob_index
              << "\n";
        } else if (edma_op == NIC_EDMA_CMD_LIN_MEMSET_V3_2) {
          const auto* ms = &c->edma_linear_memset[0];
          uint64_t dst =
              (static_cast<uint64_t>(ms->dst_addr_hi) << 32) | ms->dst_addr_lo;
          oss << "LIN_MEMSET_V3_2: dst=0x" << std::hex << dst
              << " val=0x" << ms->memset_value << std::dec
              << " size=" << ms->transfer_size << "B sob_base=" << ms->sob_base
              << " sob_idx=" << ms->sob_index << "\n";
        } else if (edma_op == NIC_EDMA_CMD_SIBO_MEMSET_V3) {
          const auto* sms = &c->sibo_memset_v3[0];
          oss << "SIBO_MEMSET_V3: sibo_idx=" << sms->sibo_index
              << " rank_cnt=" << sms->rank_count
              << " size=" << sms->transfer_size << "B val=0x" << std::hex
              << sms->memset_value << std::dec << "\n";
        } else if (edma_op == NIC_EDMA_CMD_UPDATE_GLBL_CTXT_V3) {
          const auto* gc = &c->edma_ctxt_v3[0];
          oss << "UPDATE_EDMA_GLBL_CTXT_V3: bitmap=0x" << std::hex
              << gc->update_bitmap << " sob=0x" << gc->sob_address << std::dec
              << " dwords=" << gc->num_dwords << "\n";
        } else {
          oss << "\n";
        }
      } else {
        oss << "NET_CMD_OPCODE(" << opcode << ")\n";
        cmd_size = 4;
      }
    }

    if (cmd_size == 0 || cmd_size % 4 != 0) cmd_size = 4;
    off += cmd_size;
    ++cmd_idx;
  }
  return oss.str();
}

}  // namespace gaudirt::scal
