// SPDX-License-Identifier: Apache-2.0
// P0 SCAL & HLTHUNK Interceptor Hook Library (libscal_hook.so)
// Captures and decodes all SCAL CCB command streams, PDMA descriptors,
// ECB lists, and Sync Manager SOB/Monitor/Fence operations.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "gaudirt/scal/ccb_writer.hpp"
#include "gaudirt/scal/scal_packet_decoder.hpp"

namespace {

struct StreamRecord {
  scal_stream_handle_t handle = nullptr;
  std::string name;
  std::string sched_name;
  unsigned sched_idx = 0;
  unsigned stream_idx = 0;
  unsigned submission_alignment = 0;
  unsigned command_alignment = 0;
  scal_buffer_handle_t ccb_buffer = nullptr;
  const uint8_t* ccb_host_ptr = nullptr;
  uint64_t ccb_dev_addr = 0;
  uint64_t ccb_size = 262144;  // default 256 KiB if not found
  unsigned prev_pi = 0;
};

struct CgRecord {
  scal_comp_group_handle_t handle = nullptr;
  std::string name;
  unsigned sm = 0;
  uint64_t sm_base_addr = 0;
  unsigned sos_base = 0;
  unsigned sos_num = 0;
  unsigned long_so_sm = 0;
  unsigned long_so_index = 0;
  bool force_order = false;
  uint64_t last_expected_ctr = 0;
};

struct SubmissionSnapshot {
  std::string phase;
  std::string stream_name;
  std::string sched_name;
  unsigned prev_pi = 0;
  unsigned new_pi = 0;
  std::vector<uint8_t> bytes;
};

class HookState {
 public:
  static HookState& instance() {
    static HookState inst;
    return inst;
  }

  std::mutex mu;
  std::string current_phase = "INIT";
  bool enabled = true;
  uint64_t global_hbm_base = 0x1000000000000000ULL;
  gaudirt::scal::HardwareStateTracker tracker;
  std::unordered_map<const void*, uint64_t> buffer_sizes;
  std::unordered_map<const void*, StreamRecord> streams;
  std::unordered_map<const void*, CgRecord> cgs;
  std::vector<SubmissionSnapshot> submissions;
  std::ofstream log_file;

  HookState() {
    const char* env_enable = std::getenv("P0_HOOK_ENABLE");
    if (env_enable && std::string(env_enable) == "0") {
      enabled = false;
    }
    const char* log_path = std::getenv("P0_LOG_FILE");
    if (log_path && log_path[0] != '\0') {
      log_file.open(log_path, std::ios::out | std::ios::trunc);
    }
  }

  void log(const std::string& msg) {
    if (!enabled) return;
    if (log_file.is_open()) {
      log_file << msg;
      log_file.flush();
    } else {
      std::fprintf(stderr, "%s", msg.c_str());
      std::fflush(stderr);
    }
  }
};

template <typename Fn>
Fn resolve_next(const char* sym) {
  void* p = dlsym(RTLD_NEXT, sym);
  if (!p) {
    std::fprintf(stderr, "[P0_HOOK] FATAL: failed to resolve %s\n", sym);
    std::abort();
  }
  return reinterpret_cast<Fn>(p);
}

}  // namespace

extern "C" {

void p0_hook_set_enabled(int en) {
  auto& st = HookState::instance();
  std::lock_guard<std::mutex> lock(st.mu);
  st.enabled = (en != 0);
}

void p0_hook_set_phase(const char* phase_name) {
  auto& st = HookState::instance();
  std::lock_guard<std::mutex> lock(st.mu);
  st.current_phase = phase_name ? phase_name : "UNKNOWN";
  std::ostringstream oss;
  oss << "\n=================================================================="
         "====\n"
      << "[P0_PHASE] >>> " << st.current_phase << " <<<\n"
      << "======================================================================"
         "\n";
  st.log(oss.str());
}

void p0_hook_dump_phase_diff(const char* phase_a, const char* phase_b) {
  auto& st = HookState::instance();
  std::lock_guard<std::mutex> lock(st.mu);
  if (!st.enabled) return;

  std::ostringstream oss;
  oss << "\n------------------------------------------------------------------"
         "----\n"
      << "[P0_DIFF] Comparing Phase '" << phase_a << "' vs '" << phase_b
      << "'\n"
      << "--------------------------------------------------------------------"
         "--\n";

  // 1. Compare CCB submissions
  std::vector<const SubmissionSnapshot*> subs_a, subs_b;
  for (const auto& s : st.submissions) {
    if (s.phase == phase_a) subs_a.push_back(&s);
    if (s.phase == phase_b) subs_b.push_back(&s);
  }
  oss << "  CCB Submissions count: " << phase_a << "=" << subs_a.size()
      << " vs " << phase_b << "=" << subs_b.size() << "\n";
  size_t min_subs =
      subs_a.size() < subs_b.size() ? subs_a.size() : subs_b.size();
  for (size_t i = 0; i < min_subs; ++i) {
    const auto* a = subs_a[i];
    const auto* b = subs_b[i];
    oss << "  [Sub #" << i << "] stream=" << a->stream_name
        << " size_A=" << a->bytes.size() << "B size_B=" << b->bytes.size()
        << "B\n";
    size_t min_dw = (a->bytes.size() < b->bytes.size() ? a->bytes.size()
                                                       : b->bytes.size()) /
                    4;
    const auto* dwa = reinterpret_cast<const uint32_t*>(a->bytes.data());
    const auto* dwb = reinterpret_cast<const uint32_t*>(b->bytes.data());
    for (size_t d = 0; d < min_dw; ++d) {
      if (dwa[d] != dwb[d]) {
        oss << "    diff at ccb_rel_off +0x" << std::hex << (d * 4)
            << ": A=0x" << std::setw(8) << std::setfill('0') << dwa[d]
            << " -> B=0x" << std::setw(8) << std::setfill('0') << dwb[d]
            << std::dec << "\n";
      }
    }
  }

  // 2. Compare PDMA payload blocks
  std::vector<const gaudirt::scal::PdmaBlockSnapshot*> pdma_a, pdma_b;
  for (const auto& p : st.tracker.pdma_snapshots()) {
    if (p.phase == phase_a) pdma_a.push_back(&p);
    if (p.phase == phase_b) pdma_b.push_back(&p);
  }
  oss << "  PDMA Host->HBM Blocks: " << phase_a << "=" << pdma_a.size()
      << " vs " << phase_b << "=" << pdma_b.size() << "\n";
  size_t min_pdma =
      pdma_a.size() < pdma_b.size() ? pdma_a.size() : pdma_b.size();
  for (size_t i = 0; i < min_pdma; ++i) {
    const auto* a = pdma_a[i];
    const auto* b = pdma_b[i];
    oss << "  [PDMA #" << i << "] size_A=" << a->size
        << "B (dst_A=0x" << std::hex << a->dst_addr << ") vs size_B=" << std::dec
        << b->size << "B (dst_B=0x" << std::hex << b->dst_addr << std::dec
        << ")\n";
    if (a->size == b->size) {
      uint32_t diff_dwords = 0;
      const auto* dwa = reinterpret_cast<const uint32_t*>(a->payload.data());
      const auto* dwb = reinterpret_cast<const uint32_t*>(b->payload.data());
      for (uint32_t d = 0; d < a->size / 4; ++d) {
        if (dwa[d] != dwb[d]) {
          ++diff_dwords;
          oss << "    payload diff at +0x" << std::hex << (d * 4)
              << ": A=0x" << std::setw(8) << std::setfill('0') << dwa[d]
              << " -> B=0x" << std::setw(8) << std::setfill('0') << dwb[d]
              << std::dec << "\n";
        }
      }
      if (diff_dwords == 0) {
        oss << "    -> IDENTICAL BYTE-FOR-BYTE (" << a->size << "B)\n";
      }
    }
  }
  st.log(oss.str());
}

// Intercept hlthunk_host_memory_map
uint64_t hlthunk_host_memory_map(int fd, void* host_virt_addr,
                                 uint64_t hint_addr, uint64_t host_size) {
  using Fn = uint64_t (*)(int, void*, uint64_t, uint64_t);
  static Fn real_fn = resolve_next<Fn>("hlthunk_host_memory_map");
  uint64_t dev_va = real_fn(fd, host_virt_addr, hint_addr, host_size);
  if (dev_va != 0 && host_virt_addr != nullptr) {
    auto& st = HookState::instance();
    std::lock_guard<std::mutex> lock(st.mu);
    st.tracker.register_host_mapping(dev_va, host_virt_addr, host_size,
                                     "hlthunk_host_map");
  }
  return dev_va;
}

uint64_t hlthunk_host_memory_map_flags(int fd, void* host_virt_addr,
                                       uint64_t hint_addr, uint64_t host_size,
                                       uint32_t flags) {
  using Fn = uint64_t (*)(int, void*, uint64_t, uint64_t, uint32_t);
  static Fn real_fn = resolve_next<Fn>("hlthunk_host_memory_map_flags");
  uint64_t dev_va = real_fn(fd, host_virt_addr, hint_addr, host_size, flags);
  if (dev_va != 0 && host_virt_addr != nullptr) {
    auto& st = HookState::instance();
    std::lock_guard<std::mutex> lock(st.mu);
    st.tracker.register_host_mapping(dev_va, host_virt_addr, host_size,
                                     "hlthunk_host_map_flags");
  }
  return dev_va;
}

int hlthunk_memory_unmap(int fd, uint64_t device_virt_addr) {
  using Fn = int (*)(int, uint64_t);
  static Fn real_fn = resolve_next<Fn>("hlthunk_memory_unmap");
  auto& st = HookState::instance();
  std::lock_guard<std::mutex> lock(st.mu);
  const int rc = real_fn(fd, device_virt_addr);
  if (rc == 0) st.tracker.unregister_host_mapping(device_virt_addr);
  return rc;
}

int scal_init(int fd, const char* config_file_path, scal_handle_t* scal,
              scal_arc_fw_config_handle_t fwCfg) {
  using Fn =
      int (*)(int, const char*, scal_handle_t*, scal_arc_fw_config_handle_t);
  static Fn real_fn = resolve_next<Fn>("scal_init");
  int rc = real_fn(fd, config_file_path, scal, fwCfg);
  if (rc == SCAL_SUCCESS && scal && *scal) {
    auto& st = HookState::instance();
    std::lock_guard<std::mutex> lock(st.mu);
    std::ostringstream oss;
    oss << "[P0_SCAL_INIT] fd=" << fd
        << " cfg=" << (config_file_path ? config_file_path : "null")
        << " scal=" << *scal << "\n";

    // Query control cores 0..16 to map DCCM addresses to scheduler names
    using GetCoreByIdFn =
        int (*)(const scal_handle_t, const unsigned, scal_core_handle_t*);
    using GetCoreInfoFn =
        int (*)(const scal_core_handle_t, scal_control_core_info_t*);
    static GetCoreByIdFn get_core_id =
        resolve_next<GetCoreByIdFn>("scal_get_core_handle_by_id");
    static GetCoreInfoFn get_core_info =
        resolve_next<GetCoreInfoFn>("scal_control_core_get_info");

    for (unsigned cid = 0; cid < 16; ++cid) {
      scal_core_handle_t ch = nullptr;
      if (get_core_id(*scal, cid, &ch) == SCAL_SUCCESS && ch) {
        scal_control_core_info_t cinfo{};
        if (get_core_info(ch, &cinfo) == SCAL_SUCCESS && cinfo.name) {
          oss << "  [SCAL_CORE #" << cid << "] name=" << cinfo.name
              << " dccm_msg_q=0x" << std::hex
              << cinfo.dccm_message_queue_address << std::dec << "\n";
          st.tracker.register_scheduler_dccm(
              static_cast<uint32_t>(cinfo.dccm_message_queue_address &
                                    0xFFFFFFFFULL),
              cinfo.name, cinfo.idx);
        }
      }
    }
    st.log(oss.str());
  }
  return rc;
}

int scal_pool_get_info(const scal_pool_handle_t pool,
                       scal_memory_pool_info* info) {
  using Fn = int (*)(const scal_pool_handle_t, scal_memory_pool_info*);
  static Fn real_fn = resolve_next<Fn>("scal_pool_get_info");
  int rc = real_fn(pool, info);
  if (rc == SCAL_SUCCESS && info) {
    auto& st = HookState::instance();
    std::lock_guard<std::mutex> lock(st.mu);
    std::string pname = info->name ? info->name : "unnamed";
    std::ostringstream oss;
    oss << "[P0_POOL_INFO] name=" << pname << " dev_base=0x" << std::hex
        << info->device_base_address << " host_base=" << info->host_base_address
        << " totalSize=0x" << info->totalSize << std::dec << "\n";
    st.log(oss.str());
    if (info->host_base_address && info->totalSize > 0) {
      st.tracker.register_host_mapping(info->device_base_address,
                                       info->host_base_address, info->totalSize,
                                       "pool:" + pname);
    }
    if (pname == "global_hbm") {
      st.global_hbm_base = info->device_base_address;
    }
  }
  return rc;
}

int scal_pool_get_infoV2(const scal_pool_handle_t pool,
                         scal_memory_pool_infoV2* info) {
  using Fn = int (*)(const scal_pool_handle_t, scal_memory_pool_infoV2*);
  static Fn real_fn = resolve_next<Fn>("scal_pool_get_infoV2");
  int rc = real_fn(pool, info);
  if (rc == SCAL_SUCCESS && info) {
    auto& st = HookState::instance();
    std::lock_guard<std::mutex> lock(st.mu);
    std::string pname = info->name ? info->name : "unnamed";
    std::ostringstream oss;
    oss << "[P0_POOL_INFOV2] name=" << pname << " dev_base=0x" << std::hex
        << info->device_base_address << " host_base=" << info->host_base_address
        << " totalSize=0x" << info->totalSize << std::dec << "\n";
    st.log(oss.str());
    if (info->host_base_address && info->totalSize > 0) {
      st.tracker.register_host_mapping(info->device_base_address,
                                       info->host_base_address, info->totalSize,
                                       "pool:" + pname);
    }
    if (pname == "global_hbm") {
      st.global_hbm_base = info->device_base_address;
    }
  }
  return rc;
}

int scal_allocate_buffer(const scal_pool_handle_t pool, const uint64_t size,
                         scal_buffer_handle_t* buff) {
  using Fn =
      int (*)(const scal_pool_handle_t, const uint64_t, scal_buffer_handle_t*);
  static Fn real_fn = resolve_next<Fn>("scal_allocate_buffer");
  int rc = real_fn(pool, size, buff);
  if (rc == SCAL_SUCCESS && buff && *buff) {
    auto& st = HookState::instance();
    std::lock_guard<std::mutex> lock(st.mu);
    st.buffer_sizes[*buff] = size;
  }
  return rc;
}

int scal_allocate_aligned_buffer(const scal_pool_handle_t pool,
                                 const uint64_t size, const uint64_t alignment,
                                 scal_buffer_handle_t* buff) {
  using Fn = int (*)(const scal_pool_handle_t, const uint64_t, const uint64_t,
                     scal_buffer_handle_t*);
  static Fn real_fn = resolve_next<Fn>("scal_allocate_aligned_buffer");
  int rc = real_fn(pool, size, alignment, buff);
  if (rc == SCAL_SUCCESS && buff && *buff) {
    auto& st = HookState::instance();
    std::lock_guard<std::mutex> lock(st.mu);
    st.buffer_sizes[*buff] = size;
  }
  return rc;
}

int scal_stream_set_commands_buffer(const scal_stream_handle_t stream,
                                    const scal_buffer_handle_t buffer) {
  using Fn = int (*)(const scal_stream_handle_t, const scal_buffer_handle_t);
  static Fn real_fn = resolve_next<Fn>("scal_stream_set_commands_buffer");
  int rc = real_fn(stream, buffer);
  if (rc == SCAL_SUCCESS && stream && buffer) {
    using StreamInfoFn =
        int (*)(const scal_stream_handle_t, scal_stream_info_t*);
    using CoreInfoFn =
        int (*)(const scal_core_handle_t, scal_control_core_info_t*);
    using BuffInfoFn = int (*)(const scal_buffer_handle_t, scal_buffer_info_t*);
    static StreamInfoFn get_sinfo =
        resolve_next<StreamInfoFn>("scal_stream_get_info");
    static CoreInfoFn get_cinfo =
        resolve_next<CoreInfoFn>("scal_control_core_get_info");
    static BuffInfoFn get_binfo =
        resolve_next<BuffInfoFn>("scal_buffer_get_info");

    scal_stream_info_t sinfo{};
    scal_control_core_info_t cinfo{};
    scal_buffer_info_t binfo{};
    get_sinfo(stream, &sinfo);
    if (sinfo.scheduler_handle) {
      get_cinfo(sinfo.scheduler_handle, &cinfo);
    }
    get_binfo(buffer, &binfo);

    auto& st = HookState::instance();
    std::lock_guard<std::mutex> lock(st.mu);
    StreamRecord rec;
    rec.handle = stream;
    rec.name = sinfo.name ? sinfo.name : "unnamed_stream";
    rec.sched_name = cinfo.name ? cinfo.name : "unknown_sched";
    rec.sched_idx = cinfo.idx;
    rec.stream_idx = sinfo.index;
    rec.submission_alignment = sinfo.submission_alignment;
    rec.command_alignment = sinfo.command_alignment;
    rec.ccb_buffer = buffer;
    rec.ccb_host_ptr = reinterpret_cast<const uint8_t*>(binfo.host_address);
    rec.ccb_dev_addr = binfo.device_address;
    auto it = st.buffer_sizes.find(buffer);
    if (it != st.buffer_sizes.end()) {
      rec.ccb_size = it->second;
    }
    rec.prev_pi = sinfo.current_pi;
    st.streams[stream] = rec;

    std::ostringstream oss;
    oss << "[P0_STREAM_BIND] stream=" << rec.name << " (idx=" << rec.stream_idx
        << ") sched=" << rec.sched_name << " (idx=" << rec.sched_idx
        << ") ccb_host=" << binfo.host_address << " ccb_dev=0x" << std::hex
        << rec.ccb_dev_addr << " ccb_size=0x" << rec.ccb_size << std::dec
        << " sub_align=" << rec.submission_alignment
        << " cmd_align=" << rec.command_alignment << "\n";
    st.log(oss.str());
  }
  return rc;
}

int scal_stream_submit(const scal_stream_handle_t stream, const unsigned pi,
                       const unsigned submission_alignment) {
  using Fn = int (*)(const scal_stream_handle_t, const unsigned, const unsigned);
  static Fn real_fn = resolve_next<Fn>("scal_stream_submit");

  {
    auto& st = HookState::instance();
    std::lock_guard<std::mutex> lock(st.mu);
    auto it = st.streams.find(stream);
    if (it != st.streams.end() && it->second.ccb_host_ptr != nullptr) {
      auto& rec = it->second;
      uint32_t mask = static_cast<uint32_t>(rec.ccb_size - 1);
      uint32_t start_off = rec.prev_pi & mask;
      uint32_t byte_len = (pi - rec.prev_pi) & mask;
      if (byte_len == 0 && pi != rec.prev_pi) {
        byte_len = static_cast<uint32_t>(rec.ccb_size);
      }

      std::vector<uint8_t> slice(byte_len);
      if (start_off + byte_len <= rec.ccb_size) {
        std::memcpy(slice.data(), rec.ccb_host_ptr + start_off, byte_len);
      } else {
        uint32_t first = static_cast<uint32_t>(rec.ccb_size) - start_off;
        std::memcpy(slice.data(), rec.ccb_host_ptr + start_off, first);
        std::memcpy(slice.data() + first, rec.ccb_host_ptr, byte_len - first);
      }

      if (st.current_phase != "BENCH") {
        std::ostringstream oss;
        oss << "\n[P0_SUBMIT] phase=" << st.current_phase
            << " stream=" << rec.name << " sched=" << rec.sched_name
            << " pi: 0x" << std::hex << rec.prev_pi << " -> 0x" << pi
            << " (" << std::dec << byte_len
            << "B, align=" << submission_alignment << ")\n";
        oss << gaudirt::scal::decode_ccb_commands(
            st.current_phase, rec.name, rec.sched_name, slice.data(), byte_len,
            start_off, st.global_hbm_base, st.tracker);
        st.log(oss.str());
      }

      if (st.current_phase != "INIT" && st.current_phase != "BENCH") {
        SubmissionSnapshot snap;
        snap.phase = st.current_phase;
        snap.stream_name = rec.name;
        snap.sched_name = rec.sched_name;
        snap.prev_pi = rec.prev_pi;
        snap.new_pi = pi;
        snap.bytes = std::move(slice);
        st.submissions.push_back(std::move(snap));
      }
      rec.prev_pi = pi;
    }
  }

  return real_fn(stream, pi, submission_alignment);
}

int scal_get_completion_group_handle_by_name(
    const scal_handle_t scal, const char* cg_name,
    scal_comp_group_handle_t* comp_grp) {
  using Fn =
      int (*)(const scal_handle_t, const char*, scal_comp_group_handle_t*);
  static Fn real_fn =
      resolve_next<Fn>("scal_get_completion_group_handle_by_name");
  int rc = real_fn(scal, cg_name, comp_grp);
  if (rc == SCAL_SUCCESS && comp_grp && *comp_grp) {
    using CgInfoFn = int (*)(const scal_comp_group_handle_t,
                             scal_completion_group_infoV2_t*);
    static CgInfoFn get_cg =
        resolve_next<CgInfoFn>("scal_completion_group_get_infoV2");
    scal_completion_group_infoV2_t info{};
    if (get_cg(*comp_grp, &info) == SCAL_SUCCESS) {
      auto& st = HookState::instance();
      std::lock_guard<std::mutex> lock(st.mu);
      CgRecord rec;
      rec.handle = *comp_grp;
      rec.name = cg_name ? cg_name : "unnamed_cg";
      rec.sm = info.sm;
      rec.sm_base_addr = info.sm_base_addr;
      rec.sos_base = info.sos_base;
      rec.sos_num = info.sos_num;
      rec.long_so_sm = info.long_so_sm;
      rec.long_so_index = info.long_so_index;
      rec.force_order = info.force_order;
      st.cgs[*comp_grp] = rec;

      std::ostringstream oss;
      oss << "[P0_CG_INFO] cg=" << rec.name << " sm=" << rec.sm
          << " sos_base=" << rec.sos_base << " sos_num=" << rec.sos_num
          << " long_so=DCORE" << rec.long_so_sm << "_SOB[" << rec.long_so_index
          << "] force_order=" << rec.force_order << "\n";
      st.log(oss.str());
    }
  }
  return rc;
}

int scal_completion_group_set_expected_ctr(scal_comp_group_handle_t comp_grp,
                                           uint64_t val) {
  using Fn = int (*)(scal_comp_group_handle_t, uint64_t);
  static Fn real_fn =
      resolve_next<Fn>("scal_completion_group_set_expected_ctr");
  {
    auto& st = HookState::instance();
    std::lock_guard<std::mutex> lock(st.mu);
    auto it = st.cgs.find(comp_grp);
    if (it != st.cgs.end()) {
      it->second.last_expected_ctr = val;
    }
    if (st.current_phase != "INIT" && st.current_phase != "BENCH") {
      std::string name = (it != st.cgs.end()) ? it->second.name : "unknown_cg";
      std::ostringstream oss;
      oss << "[P0_CG_SET_EXPECTED] phase=" << st.current_phase << " cg=" << name
          << " val=" << val << "\n";
      st.log(oss.str());
    }
  }
  return real_fn(comp_grp, val);
}

int scal_completion_group_wait(const scal_comp_group_handle_t comp_grp,
                               const uint64_t target, const uint64_t timeout) {
  using Fn =
      int (*)(const scal_comp_group_handle_t, const uint64_t, const uint64_t);
  static Fn real_fn = resolve_next<Fn>("scal_completion_group_wait");
  int rc = real_fn(comp_grp, target, timeout);
  {
    auto& st = HookState::instance();
    std::lock_guard<std::mutex> lock(st.mu);
    if (st.current_phase != "INIT" && st.current_phase != "BENCH" &&
        timeout != 0) {
      auto it = st.cgs.find(comp_grp);
      std::string name = (it != st.cgs.end()) ? it->second.name : "unknown_cg";
      std::ostringstream oss;
      oss << "[P0_CG_WAIT] phase=" << st.current_phase << " cg=" << name
          << " target=" << target;
      if (it != st.cgs.end()) {
        oss << " (long_so=DCORE" << it->second.long_so_sm << "_SOB["
            << it->second.long_so_index << "])";
      }
      oss << " timeout=0x" << std::hex << timeout << std::dec << " rc=" << rc
          << "\n";
      st.log(oss.str());
    }
  }
  return rc;
}

int scal_completion_group_wait_always_interupt(
    const scal_comp_group_handle_t comp_grp, const uint64_t target,
    const uint64_t timeout) {
  using Fn =
      int (*)(const scal_comp_group_handle_t, const uint64_t, const uint64_t);
  static Fn real_fn =
      resolve_next<Fn>("scal_completion_group_wait_always_interupt");
  int rc = real_fn(comp_grp, target, timeout);
  {
    auto& st = HookState::instance();
    std::lock_guard<std::mutex> lock(st.mu);
    if (st.current_phase != "INIT") {
      auto it = st.cgs.find(comp_grp);
      std::string name = (it != st.cgs.end()) ? it->second.name : "unknown_cg";
      std::ostringstream oss;
      oss << "[P0_CG_WAIT_IRQ] phase=" << st.current_phase << " cg=" << name
          << " target=" << target << " rc=" << rc << "\n";
      st.log(oss.str());
    }
  }
  return rc;
}

int p1_extract_recipe_blobs(const char* capture_phase,
                            uint8_t* out_exec_blob, uint32_t* inout_exec_size,
                            uint8_t* out_patch_blob, uint32_t* inout_patch_size,
                            uint8_t* out_dynamic_blob,
                            uint32_t* inout_dynamic_size,
                            uint64_t* out_orig_base0, uint64_t* out_orig_base1,
                            uint64_t* out_orig_base2, uint64_t* out_orig_base7) {
  if (!capture_phase) return -1;
  auto& st = HookState::instance();
  std::lock_guard<std::mutex> lock(st.mu);

  uint64_t bases[4] = {};
  bool found_base = false;
  for (const auto& sub : st.submissions) {
    if (sub.phase != capture_phase || sub.stream_name != "compute0") continue;
    size_t off = 0;
    while (off + 4 <= sub.bytes.size()) {
      const uint8_t* p = sub.bytes.data() + off;
      uint32_t dw0 = 0;
      std::memcpy(&dw0, p, sizeof(dw0));
      const uint32_t opcode = dw0 & 0x1F;
      const size_t sz =
          gaudirt::scal::compute_packet_size(p, sub.bytes.size() - off);
      if (sz == 0) return -6;
      if (opcode == SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2) {
        const auto& b =
            *reinterpret_cast<const sched_arc_cmd_update_recipe_base_v2_t*>(p);
        const uint32_t indices[4] = {b.recipe_base_index0, b.recipe_base_index1,
                                     b.recipe_base_index2, b.recipe_base_index3};
        if (b.num_recipe_addrs == 4 && indices[0] == 0 && indices[1] == 1 &&
            indices[2] == 2 && indices[3] == 7) {
          for (size_t i = 0; i < 4; ++i) bases[i] = b.recipe_base_addr[i];
          found_base = true;
        }
      }
      off += sz;
    }
  }
  if (!found_base) return -2;

  if (out_orig_base0) *out_orig_base0 = bases[0];
  if (out_orig_base1) *out_orig_base1 = bases[1];
  if (out_orig_base2) *out_orig_base2 = bases[2];
  if (out_orig_base7) *out_orig_base7 = bases[3];
  if (bases[3] < bases[2]) return -7;

  const auto copy_blob = [&](uint64_t dst_addr, uint8_t* out,
                             uint32_t* inout_size) {
    bool found = false;
    for (const auto& rec : st.tracker.pdma_snapshots()) {
      if (rec.phase != capture_phase || rec.dst_addr != dst_addr) continue;
      if (found || !out || !inout_size || *inout_size < rec.payload.size()) {
        return -4;
      }
      std::memcpy(out, rec.payload.data(), rec.payload.size());
      *inout_size = static_cast<uint32_t>(rec.payload.size());
      found = true;
    }
    return found ? 0 : -3;
  };

  int rc = copy_blob(bases[1], out_exec_blob, inout_exec_size);
  if (rc != 0) return rc;
  rc = copy_blob(bases[0], out_patch_blob, inout_patch_size);
  if (rc != 0) return rc;
  rc = copy_blob(bases[2], out_dynamic_blob, inout_dynamic_size);
  if (rc != 0) return rc;
  return bases[3] - bases[2] < *inout_dynamic_size ? 0 : -7;
}

int p1_run_autonomous_recipe(const char* capture_phase,
                             uint64_t perm_patch_even_dev_addr,
                             uint64_t perm_patch_odd_dev_addr,
                             uint64_t perm_exec_dev_addr,
                             uint64_t perm_dynamic_dev_addr,
                             uint64_t perm_nop_dev_addr, uint32_t num_iters,
                             int single_submit_batch,
                             gaudirt::scal::CcbReplayStats* out_stats) {
  if (!capture_phase || num_iters == 0 || !perm_patch_even_dev_addr ||
      !perm_exec_dev_addr || !perm_dynamic_dev_addr || !perm_nop_dev_addr) {
    return -1;
  }
  using SubmitFn = int (*)(scal_stream_handle_t, const unsigned, const unsigned);
  using SetExpFn = int (*)(scal_comp_group_handle_t, uint64_t);
  using WaitFn =
      int (*)(const scal_comp_group_handle_t, const uint64_t, const uint64_t);
  static SubmitFn real_submit = resolve_next<SubmitFn>("scal_stream_submit");
  static SetExpFn real_set_exp =
      resolve_next<SetExpFn>("scal_completion_group_set_expected_ctr");
  static WaitFn real_wait = resolve_next<WaitFn>("scal_completion_group_wait");

  auto& st = HookState::instance();
  std::lock_guard<std::mutex> lock(st.mu);

  StreamRecord* comp_stream = nullptr;
  for (auto& kv : st.streams) {
    if (kv.second.name == "compute0") {
      comp_stream = &kv.second;
      break;
    }
  }
  CgRecord* comp_cg = nullptr;
  for (auto& kv : st.cgs) {
    if (kv.second.name == "compute_completion_queue0") {
      comp_cg = &kv.second;
      break;
    }
  }
  if (!comp_stream || !comp_stream->ccb_host_ptr || !comp_cg) return -2;

  std::vector<std::vector<uint8_t>> launch_pkts;
  uint64_t orig_base2 = 0;
  uint64_t orig_base7 = 0;
  uint64_t orig_dyn_ecb = 0;
  size_t payload_bytes = 0;
  for (const auto& sub : st.submissions) {
    if (sub.phase != capture_phase || sub.stream_name != "compute0") continue;
    size_t off = 0;
    while (off + 4 <= sub.bytes.size()) {
      const uint8_t* p = sub.bytes.data() + off;
      uint32_t dw0 = 0;
      std::memcpy(&dw0, p, sizeof(dw0));
      const uint32_t opcode = dw0 & 0x1F;
      const size_t sz =
          gaudirt::scal::compute_packet_size(p, sub.bytes.size() - off);
      if (sz == 0) return -3;

      if (opcode == SCHED_COMPUTE_ARC_CMD_NOP) {
        off += sz;
        continue;
      }
      if (opcode == SCHED_COMPUTE_ARC_CMD_FENCE_WAIT) {
        const auto& fw = *reinterpret_cast<const sched_arc_cmd_fence_wait_t*>(p);
        if (fw.fence_id == 14) {
          off += sz;
          continue;
        }
      }
      if (opcode == SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2) {
        const auto& b =
            *reinterpret_cast<const sched_arc_cmd_update_recipe_base_v2_t*>(p);
        if (b.num_recipe_addrs != 4 || b.recipe_base_index0 != 0 ||
            b.recipe_base_index1 != 1 || b.recipe_base_index2 != 2 ||
            b.recipe_base_index3 != 7) {
          return -4;
        }
        orig_base2 = b.recipe_base_addr[2];
        orig_base7 = b.recipe_base_addr[3];
      }
      if (opcode == SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3) {
        const auto& c =
            *reinterpret_cast<const sched_arc_cmd_dispatch_compute_ecb_list_t*>(
                p);
        orig_dyn_ecb = (st.global_hbm_base & 0xFFFFFFFF00000000ULL) |
                       (static_cast<uint64_t>(c.dynamic_ecb_list_addr_256) << 8);
      }
      launch_pkts.emplace_back(p, p + sz);
      payload_bytes += sz;
      off += sz;
    }
  }
  if (launch_pkts.size() != 6 || payload_bytes != 80 || !orig_base2 ||
      !orig_base7 || !orig_dyn_ecb || orig_base7 < orig_base2 ||
      orig_dyn_ecb < orig_base2 ||
      perm_nop_dev_addr - perm_dynamic_dev_addr != orig_base7 - orig_base2) {
    return -5;
  }

  uint8_t* ccb_base = const_cast<uint8_t*>(comp_stream->ccb_host_ptr);
  const unsigned sub_align =
      comp_stream->submission_alignment ? comp_stream->submission_alignment : 2;
  uint64_t expected_ctr = comp_cg->last_expected_ctr;
  gaudirt::scal::CcbWriter writer(
      ccb_base, static_cast<uint32_t>(comp_stream->ccb_size),
      comp_stream->prev_pi,
      comp_stream->command_alignment ? comp_stream->command_alignment : 256);
  auto submit = [&](unsigned pi) {
    return real_submit(comp_stream->handle, pi, sub_align);
  };

  for (uint32_t it = 0; it < num_iters; ++it) {
    const uint64_t cur_patch_addr =
        (perm_patch_odd_dev_addr != 0 && (it & 1U) != 0)
            ? perm_patch_odd_dev_addr
            : perm_patch_even_dev_addr;
    for (auto pkt : launch_pkts) {
      uint32_t dw0 = 0;
      std::memcpy(&dw0, pkt.data(), sizeof(dw0));
      const uint32_t opcode = dw0 & 0x1F;
      if (opcode == SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2) {
        auto* b =
            reinterpret_cast<sched_arc_cmd_update_recipe_base_v2_t*>(pkt.data());
        b->recipe_base_addr[0] = cur_patch_addr;
        b->recipe_base_addr[1] = perm_exec_dev_addr;
        b->recipe_base_addr[2] = perm_dynamic_dev_addr;
        b->recipe_base_addr[3] = perm_nop_dev_addr;
      } else if (opcode == SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3) {
        auto* c =
            reinterpret_cast<sched_arc_cmd_dispatch_compute_ecb_list_t*>(
                pkt.data());
        const uint64_t new_dyn =
            perm_dynamic_dev_addr + (orig_dyn_ecb - orig_base2);
        const uint64_t encoded = (new_dyn & 0xFFFFFFFFULL) >> 8;
        if ((new_dyn >> 32) != (st.global_hbm_base >> 32) ||
            (new_dyn & 0xFFU) != 0 || encoded > 0xFFFFFFULL) {
          return -6;
        }
        c->dynamic_ecb_list_addr_256 = static_cast<uint32_t>(encoded);
      }
      const int rc = writer.append(pkt.data(), pkt.size(),
                                   !single_submit_batch, submit);
      if (rc != SCAL_SUCCESS) return rc;
    }
    if (!single_submit_batch) {
      expected_ctr += 1;
      int rc = real_set_exp(comp_cg->handle, expected_ctr);
      if (rc != SCAL_SUCCESS) return -11;
      rc = writer.submit_now(submit);
      if (rc != SCAL_SUCCESS) return rc;
      rc = real_wait(comp_cg->handle, expected_ctr, UINT64_MAX);
      if (rc != SCAL_SUCCESS) return -13;
    }
  }

  if (single_submit_batch) {
    expected_ctr += num_iters;
    int rc = real_set_exp(comp_cg->handle, expected_ctr);
    if (rc != SCAL_SUCCESS) return -14;
    rc = writer.submit_now(submit);
    if (rc != SCAL_SUCCESS) return rc;
    rc = real_wait(comp_cg->handle, expected_ctr, UINT64_MAX);
    if (rc != SCAL_SUCCESS) return -16;
  }

  comp_stream->prev_pi = writer.pi();
  comp_cg->last_expected_ctr = expected_ctr;
  if (out_stats) *out_stats = writer.stats();
  return 0;
}

int p3a_extract_compute_phase(const char* capture_phase, uint8_t* out_bytes,
                              uint32_t* inout_bytes,
                              uint32_t* out_submission_ends,
                              uint32_t* inout_submission_count) {
  if (!capture_phase || !out_bytes || !inout_bytes || !out_submission_ends ||
      !inout_submission_count) {
    return -1;
  }
  auto& st = HookState::instance();
  std::lock_guard<std::mutex> lock(st.mu);
  uint32_t total_bytes = 0;
  uint32_t count = 0;
  for (const auto& sub : st.submissions) {
    if (sub.phase != capture_phase || sub.stream_name != "compute0") continue;
    if (sub.bytes.size() > UINT32_MAX - total_bytes) return -5;
    total_bytes += static_cast<uint32_t>(sub.bytes.size());
    ++count;
  }
  if (!count) return -2;
  if (*inout_bytes < total_bytes) return -3;
  if (*inout_submission_count < count) return -4;

  uint32_t offset = 0;
  uint32_t index = 0;
  for (const auto& sub : st.submissions) {
    if (sub.phase != capture_phase || sub.stream_name != "compute0") continue;
    std::memcpy(out_bytes + offset, sub.bytes.data(), sub.bytes.size());
    offset += static_cast<uint32_t>(sub.bytes.size());
    out_submission_ends[index++] = offset;
  }
  *inout_bytes = total_bytes;
  *inout_submission_count = count;
  return 0;
}

int p3a_compute_current_pi(uint32_t* out_pi) {
  if (!out_pi) return -1;
  auto& st = HookState::instance();
  std::lock_guard<std::mutex> lock(st.mu);
  for (const auto& kv : st.streams) {
    if (kv.second.name == "compute0") {
      *out_pi = kv.second.prev_pi;
      return 0;
    }
  }
  return -2;
}

}  // extern "C"
