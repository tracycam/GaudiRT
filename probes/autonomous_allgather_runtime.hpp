// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <hlthunk.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <new>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "gaudirt/common/abba_harness.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/scal/ccb_writer.hpp"

#define P2A_CHECK_SYN(call)                                                   \
  do {                                                                        \
    synStatus _rc = (call);                                                   \
    if (_rc != synSuccess) {                                                  \
      std::fprintf(stderr, "[P2A_FATAL] %s rc=%d at %s:%d\n", #call,         \
                   static_cast<int>(_rc), __FILE__, __LINE__);                \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define P2A_CHECK_SCAL(call)                                                  \
  do {                                                                        \
    int _rc = (call);                                                         \
    if (_rc != SCAL_SUCCESS) {                                                \
      std::fprintf(stderr, "[P2A_FATAL] %s rc=%d at %s:%d\n", #call, _rc,    \
                   __FILE__, __LINE__);                                       \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define P2A_CHECK_HCCL(call)                                                  \
  do {                                                                        \
    hcclResult_t _rc = (call);                                                \
    if (_rc != hcclSuccess) {                                                 \
      std::fprintf(stderr, "[P2A_FATAL] %s rc=%d at %s:%d\n", #call,         \
                   static_cast<int>(_rc), __FILE__, __LINE__);                \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

constexpr uint32_t kRanks = 8;
constexpr uint32_t kCount = 1024;
constexpr uint32_t kSliceBytes = kCount * sizeof(float);
constexpr uint32_t kRecvBytes = kSliceBytes * kRanks;
constexpr uint32_t kGuardBytes = 256;
constexpr uint32_t kRecvRegionBytes = kRecvBytes + 2 * kGuardBytes;
constexpr uint32_t kNetworkCcbBytes = 0x40000;
constexpr uint32_t kCanonicalCalls = 130;

const char* const kActiveStreams[] = {
    "network_garbage_collector_and_reduction2",
    "scaleup_send1",
    "scaleup_send2",
    "scaleup_receive1",
    "scaleup_receive2",
    "scaleout_send1",
    "scaleout_send2",
    "scaleout_receive1",
    "scaleout_receive2",
    "network_garbage_collector_and_reduction5",
};
constexpr size_t kActiveCount = sizeof(kActiveStreams) / sizeof(kActiveStreams[0]);

struct SharedCoord {
  hcclUniqueId uid;
  std::atomic<int> uid_ready{0};
  std::atomic<int> barrier_sense{0};
  std::atomic<int> barrier_count{0};
  std::atomic<int> failed{0};
};

bool host_barrier(SharedCoord* shm, int& local_sense) {
  if (shm->failed.load()) return false;
  local_sense = 1 - local_sense;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(60);
  if (shm->barrier_count.fetch_add(1) == static_cast<int>(kRanks) - 1) {
    shm->barrier_count.store(0);
    shm->barrier_sense.store(local_sense);
  } else {
    while (shm->barrier_sense.load() != local_sense) {
      if (shm->failed.load() || std::chrono::steady_clock::now() > deadline) {
        shm->failed.store(1);
        return false;
      }
      std::this_thread::yield();
    }
  }
  return !shm->failed.load();
}

struct StreamCapture {
  std::string name;
  std::string scheduler;
  gaudirt::scal::SchedulerKind kind = gaudirt::scal::SchedulerKind::kUnknown;
  scal_stream_handle_t handle = nullptr;
  scal_stream_info_t info{};
  scal_buffer_info_t buffer{};
  uint32_t before_pi = 0;
  uint32_t after_pi = 0;
  uint32_t next_pi = 0x164;
  std::vector<uint8_t> bytes;
};

struct CgCapture {
  const char* name = nullptr;
  scal_comp_group_handle_t handle = nullptr;
  uint64_t expected = 0;
};

class NetworkTimestamps {
 public:
  NetworkTimestamps(int fd, CgCapture& cg, std::ofstream& log)
      : fd_(fd), cg_(cg), log_(log) {
    constexpr uint32_t count = 260;
    P2A_CHECK_SCAL(hlthunk_allocate_timestamp_elements(fd_, count, &handle_));
    bytes_ = (count * sizeof(uint64_t) + 0xFFF) & ~0xFFFULL;
    timestamps_ = static_cast<volatile uint64_t*>(mmap(
        nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
        static_cast<off_t>(handle_)));
    if (timestamps_ == MAP_FAILED) {
      std::fprintf(stderr, "[P2A_FATAL] timestamp mmap failed\n");
      std::exit(1);
    }
    for (uint32_t i = 0; i < count; ++i) timestamps_[i] = 0;
    preflight();
  }

  NetworkTimestamps(const NetworkTimestamps&) = delete;
  NetworkTimestamps& operator=(const NetworkTimestamps&) = delete;

  ~NetworkTimestamps() {
    if (timestamps_ && timestamps_ != MAP_FAILED) {
      munmap(const_cast<uint64_t*>(timestamps_), bytes_);
    }
  }

  uint32_t register_target(uint64_t target) {
    if (next_ >= 260) {
      std::fprintf(stderr, "[P2A_FATAL] timestamp slots exhausted\n");
      std::exit(1);
    }
    P2A_CHECK_SCAL(scal_completion_group_register_timestamp(
        cg_.handle, target, handle_, next_));
    return next_++;
  }

  uint64_t at(uint32_t slot) const {
    if (slot >= next_) {
      std::fprintf(stderr, "[P2A_FATAL] invalid timestamp slot %u\n", slot);
      std::exit(1);
    }
    return timestamps_[slot];
  }

  uint64_t wait_at(uint32_t slot) const {
    at(slot);
    const auto start = std::chrono::steady_clock::now();
    while (timestamps_[slot] == 0) {
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start);
      if (elapsed.count() >= 65000) {
        std::fprintf(stderr, "[P2A_FATAL] timestamp slot %u stayed zero\n", slot);
        std::exit(1);
      }
    }
    return timestamps_[slot];
  }

 private:
  void preflight() {
    const uint64_t target = cg_.expected;
    const uint32_t first = register_target(target);
    const uint32_t second = register_target(target);
    P2A_CHECK_SCAL(scal_completion_group_wait(cg_.handle, target, 0));
    const uint64_t first_ns = at(first);
    const uint64_t second_ns = at(second);
    if (!first_ns || !second_ns || second_ns <= first_ns) {
      std::fprintf(stderr,
                   "[P2A_FATAL] timestamp preflight failed first=%llu "
                   "second=%llu\n",
                   static_cast<unsigned long long>(first_ns),
                   static_cast<unsigned long long>(second_ns));
      std::exit(1);
    }
    log_ << "[P2A_TIMESTAMP_PRECHECK] target=" << target
         << " first_ns=" << first_ns << " second_ns=" << second_ns
         << " offset_unit=uint64_element\n"
         << std::flush;
  }

  int fd_ = -1;
  CgCapture& cg_;
  std::ofstream& log_;
  uint64_t handle_ = 0;
  volatile uint64_t* timestamps_ = nullptr;
  size_t bytes_ = 0;
  uint32_t next_ = 0;
};

struct CallCapture {
  uint64_t index = 0;
  uint64_t send_addr = 0;
  uint64_t recv_addr = 0;
  uint64_t external_target = 0;
  uint64_t internal_target = 0;
  std::vector<StreamCapture> streams;
};

struct PerfSample {
  double submit_us = 0;
  double wall_us = 0;
  uint32_t slot = 0;
  uint64_t completion_ns = 0;
};

void check(bool condition, const char* message);
bool is_monitor_arm_addr(uint32_t addr);
bool is_dcore1_sob_addr(uint32_t addr);
uint32_t replay_lbw_addr(uint32_t addr, uint32_t slot);
uint32_t replay_lbw_data(uint32_t addr, uint32_t data, uint32_t target);
uint32_t monitor_target(const CallCapture& call);
void validate_monitor_progress(const std::vector<CallCapture>& calls);
bool same_packets(const StreamCapture& a, const StreamCapture& b,
                  bool ignore_addresses);

class RankRuntime {
 public:
  RankRuntime(uint32_t rank, const hcclUniqueId& uid, std::ofstream& log)
      : rank_(rank), log_(log) {
    gaudirt::runtime::DeviceRuntime::Config cfg;
    cfg.module_id = rank;
    cfg.rank = rank;
    cfg.world_size = kRanks;
    cfg.pin_cpu = static_cast<int>(rank * 4);
    runtime_.emplace(cfg);
    runtime_->init_comm(uid);
    device_ = runtime_->device_id();
    stream_ = runtime_->compute_stream();

    synDeviceInfoV2 info{};
    P2A_CHECK_SYN(synDeviceGetInfoV2(device_, &info));
    fd_ = static_cast<int>(info.fd);
    global_hbm_base_ = info.globalHbmBaseAddress;
    P2A_CHECK_SCAL(scal_get_handle_from_fd(fd_, &scal_));
    for (size_t i = 0; i < kActiveCount; ++i) {
      streams_[i].name = kActiveStreams[i];
      init_stream(streams_[i]);
    }
    init_cg(external_, "network_completion_queue_external_00");
    init_cg(internal_, "network_completion_queue_internal_00");
    init_cg(pdma_tx_, "pdma_tx_completion_queue0");
    timestamps_.emplace(fd_, external_, log_);
    log_ << "[P2A_INIT] rank=" << rank_ << " streams=" << streams_.size()
         << " ccb_bytes=0x" << std::hex << kNetworkCcbBytes << std::dec
         << " global_hbm_base=0x" << std::hex << global_hbm_base_ << std::dec
         << "\n"
         << std::flush;
  }

  RankRuntime(const RankRuntime&) = delete;
  RankRuntime& operator=(const RankRuntime&) = delete;

  uint32_t rank() const { return rank_; }
  synDeviceId device() const { return device_; }
  synStreamHandle stream() const { return stream_; }
  synStreamHandle verification_stream() const { return runtime_->h2d_stream(); }
  hcclComm_t comm() const { return runtime_->comm(); }
  scal_handle_t scal() const { return scal_; }
  uint64_t global_hbm_base() const { return global_hbm_base_; }
  uint64_t next_external_target() const { return external_.expected + 1; }
  gaudirt::scal::HardwareStateTracker& tracker() { return tracker_; }

  CallCapture capture(uint64_t index, uint64_t send_addr, uint64_t recv_addr,
                      bool synchronize = true) {
    CallCapture call;
    call.index = index;
    call.send_addr = send_addr;
    call.recv_addr = recv_addr;
    read_expected(external_);
    read_expected(internal_);
    uint64_t prefix_hashes[kActiveCount];
    std::vector<std::vector<uint8_t>> before_ccbs(kActiveCount);
    for (size_t i = 0; i < kActiveCount; ++i) {
      auto& stream = streams_[i];
      stream.before_pi = stream.next_pi;
      stream.bytes.clear();
      if (stream.before_pi >= kNetworkCcbBytes) {
        fail("CCB capacity exhausted on " + stream.name);
      }
      prefix_hashes[i] = prefix_hash(stream);
      const uint8_t* ccb =
          static_cast<const uint8_t*>(stream.buffer.host_address);
      before_ccbs[i].assign(ccb, ccb + kNetworkCcbBytes);
    }

    P2A_CHECK_HCCL(hcclAllGather(
        reinterpret_cast<const void*>(send_addr), reinterpret_cast<void*>(recv_addr),
        kCount, hcclFloat32, runtime_->comm(), stream_));
    if (synchronize) P2A_CHECK_SYN(synStreamSynchronize(stream_));

    call.external_target = read_expected(external_);
    call.internal_target = read_expected(internal_);
    for (size_t i = 0; i < kActiveCount; ++i) {
      auto& stream = streams_[i];
      if (prefix_hash(stream) != prefix_hashes[i]) {
        fail("pre-existing CCB bytes changed on " + stream.name);
      }
      const uint8_t* ccb =
          static_cast<const uint8_t*>(stream.buffer.host_address);
      uint32_t last_diff = stream.before_pi;
      bool found_diff = false;
      for (uint32_t off = stream.before_pi; off < kNetworkCcbBytes; ++off) {
        if (before_ccbs[i][off] != ccb[off]) {
          last_diff = off;
          found_diff = true;
        }
      }
      if (!found_diff) {
        fail("no new CCB bytes on " + stream.name);
      }
      uint32_t off = stream.before_pi;
      while (off <= last_diff) {
        uint32_t dw0 = 0;
        std::memcpy(&dw0, ccb + off, sizeof(dw0));
        const size_t size = gaudirt::scal::network_packet_size(
            ccb + off, kNetworkCcbBytes - off, stream.kind);
        if (!size) {
          std::fprintf(stderr,
                       "[P2A_FATAL] invalid packet rank=%u index=%llu stream=%s "
                       "offset=0x%x dw0=0x%x\n",
                       rank_, static_cast<unsigned long long>(index),
                       stream.name.c_str(), off, dw0);
          std::exit(1);
        }
        off += static_cast<uint32_t>(size);
      }
      if (off == stream.before_pi || off > kNetworkCcbBytes) {
        fail("invalid parsed submission on " + stream.name);
      }
      stream.after_pi = off;
      stream.next_pi = off;
      stream.bytes.assign(ccb + stream.before_pi, ccb + off);
      call.streams.push_back(stream);
    }
    if (call.streams.size() != kActiveCount) {
      fail("unexpected active stream count: " +
           std::to_string(call.streams.size()));
    }
    return call;
  }

  void native_timed(uint64_t index, uint64_t send_addr, uint64_t recv_addr,
                    PerfSample& sample) {
    const uint64_t external_target = external_.expected + 1;
    const uint64_t internal_target = internal_.expected + 1;
    const double t0 = gaudirt::common::now_us();
    P2A_CHECK_HCCL(hcclAllGather(
        reinterpret_cast<const void*>(send_addr),
        reinterpret_cast<void*>(recv_addr), kCount, hcclFloat32,
        runtime_->comm(), stream_));
    const double t1 = gaudirt::common::now_us();
    check(read_expected(external_) == external_target &&
              read_expected(internal_) == internal_target,
          "native completion target mismatch");
    int wait_rc = wait_for_perf(external_, external_target);
    if (wait_rc != SCAL_SUCCESS) {
      log_sync_state(index);
      P2A_CHECK_SCAL(wait_rc);
    }
    wait_rc = wait_for_perf(internal_, internal_target);
    if (wait_rc != SCAL_SUCCESS) {
      log_sync_state(index);
      P2A_CHECK_SCAL(wait_rc);
    }
    const double t2 = gaudirt::common::now_us();
    sample.submit_us = t1 - t0;
    sample.wall_us = t2 - t0;
  }

  CallCapture replay(const CallCapture& replay_template, uint64_t index,
                     uint64_t send_addr, uint64_t recv_addr,
                     PerfSample* sample = nullptr,
                     std::optional<uint32_t> compute_monitor_target = std::nullopt,
                     bool wait_for_completion = true, bool log_state = true) {
    const double t0 = sample ? gaudirt::common::now_us() : 0;
    check(replay_template.streams.size() == kActiveCount &&
              replay_template.index % 64 == index % 64 && index >= 130 &&
              (send_addr >> 32) == (replay_template.send_addr >> 32) &&
              (recv_addr >> 32) == (replay_template.recv_addr >> 32),
          "unsupported replay template or address range");
    const uint32_t slot = index % 64;
    uint32_t actual_monitor = 0;
    if (compute_monitor_target) {
      actual_monitor = *compute_monitor_target;
      check(actual_monitor <= 0x7FFF, "compute monitor target exceeds 15 bits");
    } else {
      const uint32_t expected_monitor =
          monitor_target(replay_template) + 4 * (index - replay_template.index);
      actual_monitor = read_current(pdma_tx_);
      check(actual_monitor == expected_monitor && actual_monitor <= 0x7FFF,
            "PDMA monitor target mismatch");
    }

    CallCapture result;
    result.index = index;
    result.send_addr = send_addr;
    result.recv_addr = recv_addr;
    std::vector<gaudirt::scal::CcbWriter> writers;
    writers.reserve(kActiveCount);
    std::vector<std::vector<uint8_t>> encoded(kActiveCount);

    for (size_t i = 0; i < kActiveCount; ++i) {
      auto& stream = streams_[i];
      const auto& captured = replay_template.streams[i];
      check(stream.name == captured.name, "replay stream order mismatch");
      writers.emplace_back(
          static_cast<uint8_t*>(stream.buffer.host_address), kNetworkCcbBytes,
          stream.next_pi, stream.info.command_alignment, 4);
      auto& writer = writers.back();
      size_t off = 0;
      while (off < captured.bytes.size()) {
        const uint8_t* ptr = captured.bytes.data() + off;
        const size_t size = gaudirt::scal::network_packet_size(
            ptr, captured.bytes.size() - off, captured.kind);
        check(size != 0, "invalid replay packet");
        uint32_t dw0 = 0;
        std::memcpy(&dw0, ptr, sizeof(dw0));
        const uint32_t opcode = dw0 & 0x1F;
        if (opcode == 4) {
          off += size;
          continue;
        }
        std::vector<uint8_t> packet(ptr, ptr + size);
        if (opcode == 1) {
          auto* c = reinterpret_cast<sched_arc_cmd_lbw_write_t*>(packet.data());
          c->src_data = replay_lbw_data(c->dst_addr, c->src_data, actual_monitor);
          c->dst_addr = replay_lbw_addr(c->dst_addr, slot);
        } else if (opcode == 2) {
          auto* c =
              reinterpret_cast<sched_arc_cmd_lbw_burst_write_t*>(packet.data());
          for (uint32_t j = 0; j < c->num_lbw_write; ++j) {
            c->addr_data[j].data = replay_lbw_data(
                c->addr_data[j].addr, c->addr_data[j].data, actual_monitor);
            c->addr_data[j].addr = replay_lbw_addr(c->addr_data[j].addr, slot);
          }
        }

        const bool scaleup =
            captured.kind == gaudirt::scal::SchedulerKind::kScaleUpSend ||
            captured.kind == gaudirt::scal::SchedulerKind::kScaleUpRecv;
        const bool gc_or_scaleout =
            captured.kind == gaudirt::scal::SchedulerKind::kGcReduction ||
            captured.kind == gaudirt::scal::SchedulerKind::kScaleOutSend ||
            captured.kind == gaudirt::scal::SchedulerKind::kScaleOutRecv;
        if ((scaleup && opcode == 8) || (gc_or_scaleout && opcode == 5)) {
          auto* c =
              reinterpret_cast<sched_arc_cmd_alloc_nic_barrier_t*>(packet.data());
          const size_t fence_dwords = (c->fence_count > 0) + (c->fence_count > 4);
          auto* lbw = reinterpret_cast<sched_arc_lbw_write_t*>(
              packet.data() + 4 + fence_dwords * 4);
          for (uint32_t j = 0; j < c->num_lbw_write; ++j) {
            lbw[j].data = replay_lbw_data(lbw[j].addr, lbw[j].data, actual_monitor);
            lbw[j].addr = replay_lbw_addr(lbw[j].addr, slot);
          }
        } else if (scaleup && opcode == 10) {
          auto* c = reinterpret_cast<sched_arc_cmd_nic_edma_ops_t*>(packet.data());
          auto* lin = c->lin_ops_v3;
          check(lin->opcode == NIC_EDMA_CMD_LIN_OPS_V3,
                "unsupported EDMA replay opcode");
          lin->src_addr_lo = static_cast<uint32_t>(send_addr);
          lin->dst_addr_lo =
              static_cast<uint32_t>(recv_addr + rank_ * kSliceBytes);
          lin->sob_address = 0x04300100U + 4 * slot;
        } else if (scaleup && opcode == 7) {
          auto* c = reinterpret_cast<sched_arc_cmd_nic_coll_ops_t*>(packet.data());
          auto* short_op = &c->cmd_coll_ops_short[0];
          const uint64_t addr = stream.name == "scaleup_send1" ? send_addr
                                                               : recv_addr;
          check((addr & 0xF) == 0, "collective address is not 16B aligned");
          short_op->buffer_addr_lsb = static_cast<uint32_t>(addr) >> 4;
        }
        const int rc = writer.append(packet.data(), packet.size(), false,
                                     [](unsigned) { return SCAL_SUCCESS; });
        P2A_CHECK_SCAL(rc);
        encoded[i].insert(encoded[i].end(), packet.begin(), packet.end());
        off += size;
      }
      StreamCapture actual = captured;
      actual.before_pi = writers[i].stats().start_pi;
      actual.after_pi = writers[i].stats().end_pi;
      actual.bytes = encoded[i];
      result.streams.push_back(std::move(actual));
    }

    for (size_t i = 0; i < kActiveCount; ++i) {
      if (sample == nullptr &&
          !same_packets(result.streams[i], replay_template.streams[i], true)) {
        log_call(replay_template);
        log_call(result);
        fail("planned replay packets differ from template on " + streams_[i].name);
      }
    }

    for (size_t i = 0; i < kActiveCount; ++i) {
      auto& stream = streams_[i];
      auto submit = [&](unsigned pi) {
        return scal_stream_submit(stream.handle, pi,
                                  stream.info.submission_alignment);
      };
      P2A_CHECK_SCAL(writers[i].submit_now(submit));
      const auto& stats = writers[i].stats();
      direct_submit_calls_ += stats.submit_calls;
      direct_boundary_calls_ += stats.boundary_submit_calls;
      direct_nop_bytes_ += stats.nop_bytes;
      stream.next_pi = writers[i].pi();
      result.streams[i].next_pi = writers[i].pi();
      result.streams[i].before_pi = stats.start_pi;
      result.streams[i].after_pi = stats.end_pi;
    }

    result.external_target = external_.expected + 1;
    result.internal_target = internal_.expected + 1;
    P2A_CHECK_SCAL(scal_completion_group_set_expected_ctr(
        external_.handle, result.external_target));
    P2A_CHECK_SCAL(scal_completion_group_set_expected_ctr(
        internal_.handle, result.internal_target));
    read_expected(external_);
    read_expected(internal_);
    const double t1 = sample ? gaudirt::common::now_us() : 0;
    if (wait_for_completion) {
      wait_network_targets(result, index, sample == nullptr);
    }
    if (sample) {
      const double t2 = gaudirt::common::now_us();
      sample->submit_us = t1 - t0;
      sample->wall_us = t2 - t0;
    }
    if (log_state && sample == nullptr) log_sync_state(index);
    return result;
  }

  void wait_network(const CallCapture& call, uint64_t index,
                    bool log_state = false) {
    wait_network_targets(call, index);
    if (log_state) log_sync_state(index);
  }

  void wait_network_targets(const CallCapture& call, uint64_t index,
                            bool trace = true) {
    int wait_rc = trace ? wait_with_trace(external_, call.external_target, index)
                        : wait_for_perf(external_, call.external_target);
    if (wait_rc != SCAL_SUCCESS) {
      log_sync_state(index);
      P2A_CHECK_SCAL(wait_rc);
    }
    wait_rc = trace ? wait_with_trace(internal_, call.internal_target, index)
                    : wait_for_perf(internal_, call.internal_target);
    if (wait_rc != SCAL_SUCCESS) {
      log_sync_state(index);
      P2A_CHECK_SCAL(wait_rc);
    }
  }

  int wait_with_trace(CgCapture& cg, uint64_t target, uint64_t index) {
    const auto start = std::chrono::steady_clock::now();
    uint32_t next_log_ms = 1000;
    for (;;) {
      const int rc = scal_completion_group_wait(cg.handle, target, 0);
      if (rc == SCAL_SUCCESS) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start);
        log_ << "[P2A_WAIT] rank=" << rank_ << " index=" << index
             << " cg=" << cg.name << " target=" << target << " rc=0 elapsed_ms="
             << elapsed.count() << "\n"
             << std::flush;
        return SCAL_SUCCESS;
      }
      if (rc != SCAL_TIMED_OUT) return rc;
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start);
      if (static_cast<uint64_t>(elapsed.count()) >= next_log_ms) {
        log_ << "[P2A_WAIT_PROGRESS] rank=" << rank_ << " index=" << index
             << " cg=" << cg.name << " target=" << target
             << " current=" << read_current(cg) << " elapsed_ms="
             << elapsed.count() << "\n"
             << std::flush;
        next_log_ms = elapsed.count() + 5000;
      }
      if (elapsed.count() >= 65000) return rc;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  int wait_for_perf(CgCapture& cg, uint64_t target) {
    const auto start = std::chrono::steady_clock::now();
    for (;;) {
      const int rc = scal_completion_group_wait(cg.handle, target, 0);
      if (rc == SCAL_SUCCESS || rc != SCAL_TIMED_OUT) return rc;
      const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start);
      if (elapsed.count() >= 65000) return rc;
    }
  }

  uint64_t direct_submit_calls() const { return direct_submit_calls_; }
  uint64_t direct_boundary_calls() const { return direct_boundary_calls_; }
  uint64_t direct_nop_bytes() const { return direct_nop_bytes_; }

  uint32_t register_timestamp(uint64_t target) {
    return timestamps_->register_target(target);
  }

  uint64_t timestamp_at(uint32_t slot) const { return timestamps_->at(slot); }
  uint64_t timestamp_at_when_ready(uint32_t slot) const {
    return timestamps_->wait_at(slot);
  }

  void log_sync_state(uint64_t index) {
    const uint32_t slot = index % 64;
    log_ << "[P2A_SYNC_STATE] rank=" << rank_ << " index=" << index
         << " external_current=" << read_current(external_)
         << " external_expected=" << external_.expected
         << " internal_current=" << read_current(internal_)
         << " internal_expected=" << internal_.expected
         << " pdma_current=" << read_current(pdma_tx_) << "\n";
    scal_sm_info_t sm1{};
    scal_sm_info_t sm3{};
    const int rc1 = scal_get_sm_info(scal_, 1, &sm1);
    const int rc3 = scal_get_sm_info(scal_, 3, &sm3);
    if (rc1 == SCAL_SUCCESS && sm1.objs) {
      const uint32_t previous = (slot + 63) % 64;
      log_ << "[P2A_SOB_STATE] rank=" << rank_ << " index=" << index
           << " external[" << 64 + slot << "]=0x" << std::hex
           << sm1.objs[64 + slot] << " internal[" << slot << "]=0x"
           << sm1.objs[slot] << " previous_external[" << 64 + previous
           << "]=0x" << sm1.objs[64 + previous] << " previous_internal["
           << previous << "]=0x" << sm1.objs[previous] << std::dec << "\n";
    } else {
      log_ << "[P2A_SOB_STATE] rank=" << rank_ << " sm1_unavailable=" << rc1
           << " objs=" << sm1.objs << "\n";
    }
    if (rc3 == SCAL_SUCCESS && sm3.objs) {
      log_ << "[P2A_LONG_SOB_STATE] rank=" << rank_ << " index=" << index
           << " external[1440]=0x" << std::hex << sm3.objs[1440]
           << " internal[1408]=0x" << sm3.objs[1408] << std::dec << "\n";
    } else {
      log_ << "[P2A_LONG_SOB_STATE] rank=" << rank_ << " sm3_unavailable=" << rc3
           << " objs=" << sm3.objs << "\n";
    }
    for (const auto& stream : streams_) {
      const uint32_t ci = scal_debug_stream_get_curr_ci(stream.handle);
      log_ << "[P2A_CI_STATE] rank=" << rank_ << " index=" << index
           << " stream=" << stream.name << " ci=0x" << std::hex << ci
           << " next_pi=0x" << stream.next_pi << std::dec << "\n";
    }
    log_.flush();
  }

  void log_call(const CallCapture& call) {
    log_ << "[P2A_CAPTURE] rank=" << rank_ << " index=" << call.index
         << " send=0x" << std::hex << call.send_addr << " recv=0x"
         << call.recv_addr << std::dec << " external_target="
         << call.external_target << " internal_target=" << call.internal_target
         << " streams=" << call.streams.size() << "\n";
    for (const auto& stream : call.streams) {
      log_ << "[P2A_STREAM] rank=" << rank_ << " index=" << call.index
           << " name=" << stream.name << " pi=0x" << std::hex
           << stream.before_pi << "->0x" << stream.after_pi << std::dec
           << " bytes=" << stream.bytes.size() << "\n";
      log_ << gaudirt::scal::decode_ccb_commands(
          "P2A_CAPTURE", stream.name, stream.scheduler, stream.bytes.data(),
          static_cast<uint32_t>(stream.bytes.size()),
          stream.before_pi & (kNetworkCcbBytes - 1), global_hbm_base_, tracker_);
    }
    log_.flush();
  }

  void log_dynamic_state(const CallCapture& call) {
    for (const size_t index : {size_t{0}, size_t{2}, size_t{4}, size_t{6},
                               size_t{8}}) {
      const auto& stream = call.streams[index];
      log_ << "[P2A_DYNAMIC] rank=" << rank_ << " index=" << call.index
           << " name=" << stream.name << " pi=0x" << std::hex
           << stream.before_pi << "->0x" << stream.after_pi << std::dec << "\n";
      log_ << gaudirt::scal::decode_ccb_commands(
          "P2A_DYNAMIC", stream.name, stream.scheduler, stream.bytes.data(),
          static_cast<uint32_t>(stream.bytes.size()),
          stream.before_pi & (kNetworkCcbBytes - 1), global_hbm_base_, tracker_);
    }
    log_.flush();
  }

 private:
  static void fail(const std::string& message) {
    std::fprintf(stderr, "[P2A_FATAL] %s\n", message.c_str());
    std::exit(1);
  }

  void init_stream(StreamCapture& stream) {
    P2A_CHECK_SCAL(
        scal_get_stream_handle_by_name(scal_, stream.name.c_str(), &stream.handle));
    P2A_CHECK_SCAL(scal_stream_get_info(stream.handle, &stream.info));
    if (!stream.info.control_core_buffer || stream.info.command_alignment != 256 ||
        !stream.info.submission_alignment) {
      fail("invalid stream info for " + stream.name);
    }
    P2A_CHECK_SCAL(
        scal_buffer_get_info(stream.info.control_core_buffer, &stream.buffer));
    if (!stream.buffer.host_address) {
      fail("missing CCB host mapping for " + stream.name);
    }
    stream.next_pi =
        stream.name == "network_garbage_collector_and_reduction2" ? 0x2D8
                                                                  : 0x164;
    scal_control_core_info_t core{};
    P2A_CHECK_SCAL(scal_control_core_get_info(stream.info.scheduler_handle, &core));
    stream.scheduler = core.name ? core.name : "unknown";
    stream.kind = gaudirt::scal::classify_scheduler(stream.scheduler);
    if (stream.kind == gaudirt::scal::SchedulerKind::kUnknown ||
        stream.kind == gaudirt::scal::SchedulerKind::kCompute) {
      fail("invalid scheduler kind for " + stream.name);
    }
  }

  void init_cg(CgCapture& cg, const char* name) {
    cg.name = name;
    P2A_CHECK_SCAL(
        scal_get_completion_group_handle_by_name(scal_, name, &cg.handle));
    read_expected(cg);
  }

  uint64_t read_expected(CgCapture& cg) {
    P2A_CHECK_SCAL(scal_completion_group_get_expected_ctr(cg.handle, &cg.expected));
    return cg.expected;
  }

  uint64_t read_current(CgCapture& cg) {
    scal_completion_group_infoV2_t info{};
    P2A_CHECK_SCAL(scal_completion_group_get_infoV2(cg.handle, &info));
    return info.current_value;
  }

  uint64_t prefix_hash(const StreamCapture& stream) const {
    const uint8_t* bytes =
        static_cast<const uint8_t*>(stream.buffer.host_address);
    uint64_t hash = 1469598103934665603ULL;
    for (uint32_t i = 0; i < stream.before_pi; ++i) {
      hash = (hash ^ bytes[i]) * 1099511628211ULL;
    }
    return hash;
  }

  uint32_t rank_ = 0;
  std::ofstream& log_;
  std::optional<gaudirt::runtime::DeviceRuntime> runtime_;
  std::optional<NetworkTimestamps> timestamps_;
  int fd_ = -1;
  synDeviceId device_ = SYN_INVALID_DEVICE_ID;
  synStreamHandle stream_ = nullptr;
  scal_handle_t scal_ = nullptr;
  uint64_t global_hbm_base_ = 0;
  std::vector<StreamCapture> streams_{kActiveCount};
  CgCapture external_;
  CgCapture internal_;
  CgCapture pdma_tx_;
  gaudirt::scal::HardwareStateTracker tracker_;
  uint64_t direct_submit_calls_ = 0;
  uint64_t direct_boundary_calls_ = 0;
  uint64_t direct_nop_bytes_ = 0;
};

float input_value(uint32_t logical_step, uint32_t source_rank, uint32_t element) {
  return static_cast<float>(logical_step * 100000 + source_rank * 1000 + element);
}

void check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "[P2A_FATAL] %s\n", message);
    std::exit(1);
  }
}

class BufferSet {
 public:
  BufferSet(synDeviceId device, uint32_t index) : index_(index) {
    P2A_CHECK_SYN(synDeviceMalloc(device, kSliceBytes, 0, 0, &send_));
    uint64_t recv_region = 0;
    P2A_CHECK_SYN(synDeviceMalloc(device, kRecvRegionBytes, 0, 0, &recv_region));
    recv_base_ = recv_region;
    recv_ = recv_region + kGuardBytes;
    P2A_CHECK_SYN(synHostMalloc(device, kSliceBytes, 0,
                                reinterpret_cast<void**>(&host_send_)));
    P2A_CHECK_SYN(synHostMalloc(device, kRecvRegionBytes, 0,
                                reinterpret_cast<void**>(&host_recv_region_)));
  }

  ~BufferSet() {
    if (host_send_) synHostFree(device_, host_send_, 0);
    if (host_recv_region_) synHostFree(device_, host_recv_region_, 0);
    if (send_) synDeviceFree(device_, send_, 0);
    if (recv_base_) synDeviceFree(device_, recv_base_, 0);
  }

  BufferSet(const BufferSet&) = delete;
  BufferSet& operator=(const BufferSet&) = delete;

  void initialize(synDeviceId device, synStreamHandle stream, uint32_t rank,
                  uint32_t logical_step) {
    device_ = device;
    for (uint32_t i = 0; i < kCount; ++i) {
      host_send_[i] = input_value(logical_step, rank, i);
    }
    uint8_t* region = reinterpret_cast<uint8_t*>(host_recv_region_);
    std::memset(region, 0xA5, kGuardBytes);
    std::memset(region + kGuardBytes, 0x5A, kRecvBytes);
    std::memset(region + kGuardBytes + kRecvBytes, 0xA5, kGuardBytes);
    P2A_CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_send_),
                                  kSliceBytes, send_, HOST_TO_DRAM));
    P2A_CHECK_SYN(synMemCopyAsync(
        stream, reinterpret_cast<uint64_t>(host_recv_region_), kRecvRegionBytes,
        recv_base_, HOST_TO_DRAM));
  }

  void verify(synStreamHandle stream, uint32_t logical_step,
              uint32_t world_size) {
    P2A_CHECK_SYN(synMemCopyAsync(
        stream, recv_base_, kRecvRegionBytes,
        reinterpret_cast<uint64_t>(host_recv_region_), DRAM_TO_HOST));
    P2A_CHECK_SYN(synStreamSynchronize(stream));
    const uint8_t* region = reinterpret_cast<const uint8_t*>(host_recv_region_);
    for (uint32_t i = 0; i < kGuardBytes; ++i) {
      check(region[i] == 0xA5 && region[kGuardBytes + kRecvBytes + i] == 0xA5,
            "receive guard corruption");
    }
    const float* recv =
        reinterpret_cast<const float*>(region + kGuardBytes);
    for (uint32_t source = 0; source < world_size; ++source) {
      for (uint32_t i = 0; i < kCount; ++i) {
        if (recv[source * kCount + i] != input_value(logical_step, source, i)) {
          std::fprintf(stderr,
                       "[P2A_FATAL] output mismatch step=%u source=%u elem=%u "
                       "actual=%f expected=%f\n",
                       logical_step, source, i, recv[source * kCount + i],
                       input_value(logical_step, source, i));
          std::exit(1);
        }
      }
    }
  }

  uint64_t send() const { return send_; }
  uint64_t recv() const { return recv_; }
  uint32_t index() const { return index_; }

 private:
  uint32_t index_ = 0;
  synDeviceId device_ = SYN_INVALID_DEVICE_ID;
  uint64_t send_ = 0;
  uint64_t recv_base_ = 0;
  uint64_t recv_ = 0;
  float* host_send_ = nullptr;
  void* host_recv_region_ = nullptr;
};

bool is_monitor_arm_addr(uint32_t addr) {
  for (const uint32_t base : {0xFC100000U, 0xFC300000U, 0xFC500000U,
                              0xFC700000U}) {
    if (addr >= base + 0xE000 && addr < base + 0x10000) return true;
  }
  return false;
}

bool is_dcore1_sob_addr(uint32_t addr) {
  return addr >= 0xFC300000 && addr < 0xFC308000;
}

uint32_t replay_lbw_addr(uint32_t addr, uint32_t slot) {
  if (!is_dcore1_sob_addr(addr)) return addr;
  const uint32_t offset = addr - 0xFC300000;
  return offset >= 0x100 ? 0xFC300100 + 4 * slot : 0xFC300000 + 4 * slot;
}

uint32_t replay_lbw_data(uint32_t addr, uint32_t data, uint32_t target) {
  return is_monitor_arm_addr(addr) ? (data & 0x1FFFFU) | (target << 17) : data;
}

void collect_monitor_target(uint32_t addr, uint32_t data, uint32_t& target,
                            bool& found) {
  if (!is_monitor_arm_addr(addr)) return;
  const uint32_t current = (data >> 17) & 0x7FFF;
  if (found && target != current) {
    std::fprintf(stderr, "[P2A_FATAL] inconsistent monitor targets %u/%u\n",
                 target, current);
    std::exit(1);
  }
  target = current;
  found = true;
}

uint32_t monitor_target(const CallCapture& call) {
  uint32_t target = 0;
  bool found = false;
  for (const auto& stream : call.streams) {
    size_t off = 0;
    while (off < stream.bytes.size()) {
      const uint8_t* ptr = stream.bytes.data() + off;
      const size_t size = gaudirt::scal::network_packet_size(
          ptr, stream.bytes.size() - off, stream.kind);
      check(size != 0, "invalid monitor packet");
      uint32_t dw0 = 0;
      std::memcpy(&dw0, ptr, sizeof(dw0));
      const uint32_t opcode = dw0 & 0x1F;
      if (opcode == 1) {
        const auto* c = reinterpret_cast<const sched_arc_cmd_lbw_write_t*>(ptr);
        collect_monitor_target(c->dst_addr, c->src_data, target, found);
      } else if (opcode == 2) {
        const auto* c =
            reinterpret_cast<const sched_arc_cmd_lbw_burst_write_t*>(ptr);
        for (uint32_t i = 0; i < c->num_lbw_write; ++i) {
          collect_monitor_target(c->addr_data[i].addr, c->addr_data[i].data,
                                 target, found);
        }
      }
      const bool scaleup = stream.kind == gaudirt::scal::SchedulerKind::kScaleUpSend ||
                           stream.kind == gaudirt::scal::SchedulerKind::kScaleUpRecv;
      const bool gc_or_scaleout =
          stream.kind == gaudirt::scal::SchedulerKind::kGcReduction ||
          stream.kind == gaudirt::scal::SchedulerKind::kScaleOutSend ||
          stream.kind == gaudirt::scal::SchedulerKind::kScaleOutRecv;
      if ((scaleup && opcode == 8) || (gc_or_scaleout && opcode == 5)) {
        const auto* c = reinterpret_cast<const sched_arc_cmd_alloc_nic_barrier_t*>(ptr);
        const size_t fence_dwords = (c->fence_count > 0) + (c->fence_count > 4);
        const auto* lbw = reinterpret_cast<const sched_arc_lbw_write_t*>(
            ptr + 4 + fence_dwords * 4);
        for (uint32_t i = 0; i < c->num_lbw_write; ++i) {
          collect_monitor_target(lbw[i].addr, lbw[i].data, target, found);
        }
      }
      off += size;
    }
  }
  check(found, "missing monitor target");
  return target;
}

void validate_monitor_progress(const std::vector<CallCapture>& calls) {
  check(calls.size() >= 130, "monitor validation requires 130 calls");
  uint32_t previous = monitor_target(calls[7]);
  for (size_t k = 8; k < 130; ++k) {
    const uint32_t current = monitor_target(calls[k]);
    check(current == previous + 4, "monitor target progression mismatch");
    previous = current;
  }
}

uint32_t normalized_lbw_addr(uint32_t addr) {
  return is_dcore1_sob_addr(addr) ? 0 : addr;
}

uint32_t normalized_lbw_data(uint32_t addr, uint32_t data) {
  return is_monitor_arm_addr(addr) ? data & 0x1FFFFU : data;
}

void normalize_packet(std::vector<uint8_t>& bytes,
                      gaudirt::scal::SchedulerKind kind,
                      bool ignore_addresses = false) {
  uint32_t dw0 = 0;
  std::memcpy(&dw0, bytes.data(), sizeof(dw0));
  const uint32_t opcode = dw0 & 0x1F;
  if (opcode == 1) {
    auto* c = reinterpret_cast<sched_arc_cmd_lbw_write_t*>(bytes.data());
    c->src_data = normalized_lbw_data(c->dst_addr, c->src_data);
    c->dst_addr = normalized_lbw_addr(c->dst_addr);
  } else if (opcode == 2) {
    auto* c = reinterpret_cast<sched_arc_cmd_lbw_burst_write_t*>(bytes.data());
    for (uint32_t i = 0; i < c->num_lbw_write; ++i) {
      c->addr_data[i].data =
          normalized_lbw_data(c->addr_data[i].addr, c->addr_data[i].data);
      c->addr_data[i].addr = normalized_lbw_addr(c->addr_data[i].addr);
    }
  }

  const bool scaleup = kind == gaudirt::scal::SchedulerKind::kScaleUpSend ||
                       kind == gaudirt::scal::SchedulerKind::kScaleUpRecv;
  const bool gc_or_scaleout = kind == gaudirt::scal::SchedulerKind::kGcReduction ||
                              kind == gaudirt::scal::SchedulerKind::kScaleOutSend ||
                              kind == gaudirt::scal::SchedulerKind::kScaleOutRecv;
  if ((scaleup && opcode == 8) || (gc_or_scaleout && opcode == 5)) {
    auto* c = reinterpret_cast<sched_arc_cmd_alloc_nic_barrier_t*>(bytes.data());
    const size_t fence_dwords = (c->fence_count > 0) + (c->fence_count > 4);
    auto* lbw = reinterpret_cast<sched_arc_lbw_write_t*>(
        bytes.data() + 4 + fence_dwords * 4);
    for (uint32_t i = 0; i < c->num_lbw_write; ++i) {
      lbw[i].data = normalized_lbw_data(lbw[i].addr, lbw[i].data);
      lbw[i].addr = normalized_lbw_addr(lbw[i].addr);
    }
  } else if ((scaleup && opcode == 10) ||
             (kind == gaudirt::scal::SchedulerKind::kGcReduction && opcode == 6)) {
    auto* c = reinterpret_cast<sched_arc_cmd_nic_edma_ops_t*>(bytes.data());
    auto* lin = c->lin_ops_v3;
    if (lin->opcode == NIC_EDMA_CMD_LIN_OPS_V3) {
      lin->sob_address = 0;
      if (ignore_addresses) {
        lin->src_addr_lo = 0;
        lin->dst_addr_lo = 0;
      }
    }
  } else if (scaleup && opcode == 7 && ignore_addresses) {
    auto* c = reinterpret_cast<sched_arc_cmd_nic_coll_ops_t*>(bytes.data());
    auto* short_op = c->cmd_coll_ops_short;
    short_op->buffer_addr_lsb = 0;
  }
}

bool same_packets(const StreamCapture& a, const StreamCapture& b,
                  bool ignore_addresses = false) {
  size_t aoff = 0;
  size_t boff = 0;
  size_t count = 0;
  while (aoff < a.bytes.size() && boff < b.bytes.size()) {
    const size_t asz = gaudirt::scal::network_packet_size(
        a.bytes.data() + aoff, a.bytes.size() - aoff, a.kind);
    const size_t bsz = gaudirt::scal::network_packet_size(
        b.bytes.data() + boff, b.bytes.size() - boff, b.kind);
    check(asz && bsz, "invalid network packet during template comparison");
    const uint32_t aop = *reinterpret_cast<const uint32_t*>(a.bytes.data() + aoff) & 0x1F;
    const uint32_t bop = *reinterpret_cast<const uint32_t*>(b.bytes.data() + boff) & 0x1F;
    if (aop == 4) {
      aoff += asz;
      continue;
    }
    if (bop == 4) {
      boff += bsz;
      continue;
    }
    if (aop != bop || asz != bsz) {
      std::fprintf(stderr,
                   "[P2A_SEQUENCE_DIFF] stream=%s aoff=0x%zx boff=0x%zx "
                   "aop=%u bop=%u asz=%zu bsz=%zu\n",
                   a.name.c_str(), aoff, boff, aop, bop, asz, bsz);
      return false;
    }
    if (aop != 4) {
      std::vector<uint8_t> an(a.bytes.begin() + aoff, a.bytes.begin() + aoff + asz);
      std::vector<uint8_t> bn(b.bytes.begin() + boff, b.bytes.begin() + boff + bsz);
      normalize_packet(an, a.kind, ignore_addresses);
      normalize_packet(bn, b.kind, ignore_addresses);
      if (an != bn) {
        for (size_t i = 0; i < asz; i += 4) {
          uint32_t av = 0;
          uint32_t bv = 0;
          std::memcpy(&av, an.data() + i, sizeof(av));
          std::memcpy(&bv, bn.data() + i, sizeof(bv));
          if (av != bv) {
            std::fprintf(stderr,
                         "[P2A_PACKET_DIFF] stream=%s opcode=%u offset=0x%zx "
                         "A=0x%08x B=0x%08x\n",
                         a.name.c_str(), aop, i, av, bv);
            break;
          }
        }
        return false;
      }
      ++count;
    }
    aoff += asz;
    boff += bsz;
  }
  while (aoff < a.bytes.size()) {
    const size_t sz = gaudirt::scal::network_packet_size(
        a.bytes.data() + aoff, a.bytes.size() - aoff, a.kind);
    const uint32_t op = *reinterpret_cast<const uint32_t*>(a.bytes.data() + aoff) & 0x1F;
    if (!sz || op != 4) {
      std::fprintf(stderr,
                   "[P2A_TRAIL_DIFF] stream=%s side=A off=0x%zx op=%u sz=%zu\n",
                   a.name.c_str(), aoff, op, sz);
      return false;
    }
    aoff += sz;
  }
  while (boff < b.bytes.size()) {
    const size_t sz = gaudirt::scal::network_packet_size(
        b.bytes.data() + boff, b.bytes.size() - boff, b.kind);
    const uint32_t op = *reinterpret_cast<const uint32_t*>(b.bytes.data() + boff) & 0x1F;
    if (!sz || op != 4) {
      std::fprintf(stderr,
                   "[P2A_TRAIL_DIFF] stream=%s side=B off=0x%zx op=%u sz=%zu\n",
                   b.name.c_str(), boff, op, sz);
      return false;
    }
    boff += sz;
  }
  return count > 0;
}

bool compare_cycles(const CallCapture& a, const CallCapture& b,
                    std::string& failing_stream) {
  if (a.streams.size() != b.streams.size()) return false;
  for (size_t i = 0; i < a.streams.size(); ++i) {
    if (a.streams[i].name != b.streams[i].name ||
        !same_packets(a.streams[i], b.streams[i])) {
      failing_stream = a.streams[i].name;
      return false;
    }
  }
  return true;
}

void validate_addresses(const CallCapture& call, uint32_t rank) {
  check(call.streams.size() == kActiveCount, "address validation stream count");
  const auto& send1 = call.streams[1];
  const auto& recv1 = call.streams[3];
  bool found_lin = false;
  bool found_send_short = false;
  bool found_recv_short = false;
  for (const auto& packet_stream : {send1, recv1}) {
    size_t off = 0;
    while (off < packet_stream.bytes.size()) {
      const uint8_t* ptr = packet_stream.bytes.data() + off;
      const size_t size = gaudirt::scal::network_packet_size(
          ptr, packet_stream.bytes.size() - off, packet_stream.kind);
      check(size != 0, "invalid address packet");
      const uint32_t opcode = *reinterpret_cast<const uint32_t*>(ptr) & 0x1F;
      if (packet_stream.name == "scaleup_send1" && opcode == 10) {
        const auto* edma = reinterpret_cast<const sched_arc_cmd_nic_edma_ops_t*>(ptr);
        const auto* lin = edma->lin_ops_v3;
        check(lin->src_addr_lo == static_cast<uint32_t>(call.send_addr),
              "LIN source address mismatch");
        check(lin->dst_addr_lo ==
                  static_cast<uint32_t>(call.recv_addr + rank * kSliceBytes),
              "LIN destination address mismatch");
        found_lin = true;
      }
      if (opcode == 7) {
        const auto* coll = reinterpret_cast<const sched_arc_cmd_nic_coll_ops_t*>(ptr);
        const auto* short_op = &coll->cmd_coll_ops_short[0];
        const uint64_t expected = packet_stream.name == "scaleup_send1"
                                      ? call.send_addr
                                      : call.recv_addr;
        const uint64_t actual =
            static_cast<uint64_t>(short_op->buffer_addr_lsb) << 4;
        if (actual != static_cast<uint32_t>(expected)) {
          std::fprintf(stderr,
                       "[P2A_FATAL] SHORT mismatch rank=%u index=%llu "
                       "stream=%s actual=0x%llx expected=0x%llx send=0x%llx "
                       "recv=0x%llx\n",
                       rank, static_cast<unsigned long long>(call.index),
                       packet_stream.name.c_str(),
                       static_cast<unsigned long long>(actual),
                       static_cast<unsigned long long>(expected),
                       static_cast<unsigned long long>(call.send_addr),
                       static_cast<unsigned long long>(call.recv_addr));
          std::exit(1);
        }
        if (packet_stream.name == "scaleup_send1") {
          found_send_short = true;
        } else {
          found_recv_short = true;
        }
      }
      off += size;
    }
  }
  check(found_lin && found_send_short && found_recv_short,
        "missing address-bearing network commands");
}

}  // namespace

#ifdef GAUDIRT_ALLGATHER_P2A_DRIVER
namespace {

void write_perf_samples(const std::string& log_dir, uint32_t rank,
                        const std::string& mode,
                        const std::vector<PerfSample>& samples) {
  check(samples.size() == 130, "performance sample count mismatch");
  std::vector<double> submit_us;
  std::vector<double> wall_us;
  std::vector<double> completion_span_us;
  submit_us.reserve(samples.size());
  wall_us.reserve(samples.size());
  completion_span_us.reserve(samples.size() - 1);
  for (size_t i = 0; i < samples.size(); ++i) {
    check(samples[i].submit_us > 0 && samples[i].wall_us > 0 &&
              samples[i].completion_ns > 0,
            "invalid performance sample");
    submit_us.push_back(samples[i].submit_us);
    wall_us.push_back(samples[i].wall_us);
    if (i > 0) {
      check(samples[i].completion_ns >= samples[i - 1].completion_ns,
            "completion timestamps are not monotonic");
      completion_span_us.push_back(
          (samples[i].completion_ns - samples[i - 1].completion_ns) / 1000.0);
    }
  }
  const auto submit_stats = gaudirt::common::summarize(submit_us);
  const auto wall_stats = gaudirt::common::summarize(wall_us);
  const auto span_stats = gaudirt::common::summarize(completion_span_us);
  std::ofstream json(log_dir + "/rank" + std::to_string(rank) + "_perf.json",
                     std::ios::out | std::ios::trunc);
  json << std::fixed << std::setprecision(6);
  json << "{\"rank\":" << rank << ",\"mode\":\"" << mode
       << "\",\"samples\":[";
  for (size_t i = 0; i < samples.size(); ++i) {
    if (i) json << ',';
    json << "{\"index\":" << 130 + i << ",\"submit_us\":"
         << samples[i].submit_us << ",\"wall_us\":" << samples[i].wall_us
         << ",\"completion_ns\":" << samples[i].completion_ns << '}';
  }
  json << "],\"summary\":{\"submit_p50_us\":" << submit_stats.p50
       << ",\"wall_p50_us\":" << wall_stats.p50
       << ",\"completion_to_completion_p50_us\":" << span_stats.p50
       << ",\"submit_mean_us\":" << submit_stats.mean
       << ",\"wall_mean_us\":" << wall_stats.mean
       << ",\"completion_to_completion_mean_us\":" << span_stats.mean
       << ",\"submit_max_us\":" << submit_stats.max
       << ",\"wall_max_us\":" << wall_stats.max
       << ",\"completion_to_completion_max_us\":" << span_stats.max << "}}\n";
  if (rank == 0) {
    std::fprintf(stderr,
                 "[P2A_PERF_SUMMARY] mode=%s count=130 submit_p50_us=%.3f "
                 "wall_p50_us=%.3f completion_to_completion_p50_us=%.3f\n",
                 mode.c_str(), submit_stats.p50, wall_stats.p50, span_stats.p50);
  }
}

int run_rank(uint32_t rank, SharedCoord* shm, const std::string& log_dir,
             const std::string& mode) {
  int sense = 0;
  if (rank == 0) {
    P2A_CHECK_HCCL(hcclGetUniqueId(&shm->uid));
    shm->uid_ready.store(1);
  } else {
    while (!shm->uid_ready.load()) std::this_thread::yield();
  }
  check(host_barrier(shm, sense), "initial rank barrier failed");

  std::ofstream log(log_dir + "/rank" + std::to_string(rank) + ".log",
                    std::ios::out | std::ios::trunc);
  RankRuntime rt(rank, shm->uid, log);
  check(host_barrier(shm, sense), "communicator barrier failed");

  BufferSet a(rt.device(), 0);
  BufferSet b(rt.device(), 1);
  check(((a.send() ^ b.send()) & 0xFFFFFFFF00000000ULL) == 0 &&
            ((a.recv() ^ b.recv()) & 0xFFFFFFFF00000000ULL) == 0,
        "P2a requires both buffer sets in the same 32-bit address region");

  const char* calls_env = std::getenv("P2A_MAX_CALLS");
  const uint32_t call_limit =
      calls_env ? static_cast<uint32_t>(std::strtoul(calls_env, nullptr, 0))
                : kCanonicalCalls;
  check(call_limit >= 5 && call_limit <= kCanonicalCalls,
        "P2A_MAX_CALLS must be in [5,130]");
  check(mode == "capture" || mode == "native" || mode == "direct",
        "invalid P2a mode");
  check(mode == "capture" || call_limit == kCanonicalCalls,
        "performance modes require 130 capture calls");

  std::vector<CallCapture> calls;
  calls.reserve(call_limit);
  size_t cycle_mismatches = 0;
  std::string failing_stream;
  for (uint32_t k = 0; k < call_limit; ++k) {
    a.initialize(rt.device(), rt.stream(), rank, k);
    b.initialize(rt.device(), rt.stream(), rank, k);
    P2A_CHECK_SYN(synStreamSynchronize(rt.stream()));
    const bool alternate_b = k >= 5 && (k & 1);
    const bool send_b = k == 2 || k == 4 || alternate_b;
    const bool recv_b = k == 3 || k == 4 || alternate_b;
    BufferSet& output = recv_b ? b : a;
    CallCapture call = rt.capture(k, send_b ? b.send() : a.send(),
                                recv_b ? b.recv() : a.recv());
    output.verify(rt.stream(), k, kRanks);
    if (k < 5 || k == 63 || k == 64 || k == 65 || k == 129) {
      rt.log_call(call);
    }
    validate_addresses(call, rank);
    rt.log_dynamic_state(call);
    failing_stream.clear();
    if (k >= 71 && !compare_cycles(calls[k - 64], call, failing_stream)) {
      ++cycle_mismatches;
      if (rank == 0) {
        std::fprintf(stderr,
                     "[P2A_CYCLE_MISMATCH] index=%u/%u stream=%s\n", k - 64, k,
                     failing_stream.c_str());
      }
      if (cycle_mismatches == 1) {
        rt.log_call(calls[k - 64]);
        rt.log_call(call);
      }
    }
    calls.push_back(std::move(call));
    check(host_barrier(shm, sense), "collective barrier failed");
    if (rank == 0 && (k == 0 || k == 64 || k == 129)) {
      std::fprintf(stderr, "[P2A_PROGRESS] native_capture=%u\n", k);
    }
  }

  for (uint32_t k = 66; k < call_limit; ++k) {
    check(calls[k].external_target == k + 1 && calls[k].internal_target == k + 1,
          "completion target mismatch");
  }
  if (cycle_mismatches) {
    log << "[P2A_CAPTURE_VERDICT] rank=" << rank
        << " cycle_checks=FAIL mismatches=" << cycle_mismatches << "\n"
        << std::flush;
    if (rank == 0) {
      std::fprintf(stderr,
                   "[P2A_CAPTURE_VERDICT] cycle_checks=FAIL mismatches=%zu\n",
                   cycle_mismatches);
    }
    return 2;
  }
  if (rank == 0) {
    std::fprintf(stderr,
                 "[P2A_CAPTURE_VERDICT] mode=%s ranks=8 native_calls=%u "
                 "streams=10 address_checks=PASS cycle_checks=PASS "
                 "output_checks=PASS\n",
                 call_limit == kCanonicalCalls ? "FULL" : "SMOKE", call_limit);
  }
  log << "[P2A_CAPTURE_VERDICT] rank=" << rank
      << " native_calls=" << call_limit << " output_bytes=" << kRecvBytes
      << " guard_bytes=" << 2 * kGuardBytes
      << " address_checks=PASS cycle_checks=PASS output_checks=PASS\n"
      << std::flush;

  if (call_limit == kCanonicalCalls && mode != "native") {
    validate_monitor_progress(calls);
    std::vector<PerfSample> samples;
    for (uint32_t step = 0; step < 130; ++step) {
      const uint64_t index = call_limit + step;
      const auto& replay_template = calls[66 + ((index - 130) % 64)];
      a.initialize(rt.device(), rt.stream(), rank, step);
      b.initialize(rt.device(), rt.stream(), rank, step);
      P2A_CHECK_SYN(synStreamSynchronize(rt.stream()));
      BufferSet& output = (step & 1) ? b : a;
      PerfSample sample;
      if (mode == "direct") sample.slot = rt.register_timestamp(index + 1);
      CallCapture actual = rt.replay(replay_template, index, output.send(),
                                     output.recv(),
                                     mode == "direct" ? &sample : nullptr);
      if (mode == "direct") samples.push_back(sample);
      if (mode != "direct") {
        validate_addresses(actual, rank);
        if (step < 2 || step == 63 || step == 64 || step == 128 || step == 129) {
          rt.log_call(actual);
        }
        rt.log_dynamic_state(actual);
      }
      output.verify(rt.stream(), step, kRanks);
      if (mode == "direct") {
        samples.back().completion_ns =
            rt.timestamp_at_when_ready(samples.back().slot);
      }
      check(host_barrier(shm, sense), "direct collective barrier failed");
      if (mode == "capture" && rank == 0 &&
          (step == 0 || step == 63 || step == 64 || step == 129)) {
        std::fprintf(stderr, "[P2A_PROGRESS] direct_replay=%u global=%llu\n",
                     step, static_cast<unsigned long long>(index));
      }
    }
    check(rt.direct_submit_calls() == 1300 && rt.direct_boundary_calls() == 0,
          "direct submit count mismatch");
    if (mode == "direct") write_perf_samples(log_dir, rank, mode, samples);
    log << "[P2A_DIRECT_VERDICT] rank=" << rank
        << " direct_calls=130 streams=10 submit_calls="
        << rt.direct_submit_calls()
        << " boundary_calls=" << rt.direct_boundary_calls()
        << " nop_bytes=" << rt.direct_nop_bytes()
        << " output_checks=PASS guard_checks=PASS\n"
        << std::flush;
    if (rank == 0) {
      std::fprintf(stderr,
                   "[P2A_DIRECT_VERDICT] ranks=8 direct_calls=130 streams=10 "
                   "submit_calls=%llu boundary_calls=%llu output_checks=PASS "
                   "guard_checks=PASS\n",
                   static_cast<unsigned long long>(rt.direct_submit_calls()),
                   static_cast<unsigned long long>(rt.direct_boundary_calls()));
    }
    std::_Exit(0);
  }

  if (call_limit == kCanonicalCalls && mode == "native") {
    std::vector<PerfSample> samples;
    for (uint32_t step = 0; step < 130; ++step) {
      const uint64_t index = call_limit + step;
      a.initialize(rt.device(), rt.stream(), rank, step);
      b.initialize(rt.device(), rt.stream(), rank, step);
      P2A_CHECK_SYN(synStreamSynchronize(rt.stream()));
      BufferSet& output = (step & 1) ? b : a;
      PerfSample sample;
      sample.slot = rt.register_timestamp(index + 1);
      rt.native_timed(index, output.send(), output.recv(), sample);
      samples.push_back(sample);
      output.verify(rt.stream(), step, kRanks);
      samples.back().completion_ns =
          rt.timestamp_at_when_ready(samples.back().slot);
      check(host_barrier(shm, sense), "native timed collective barrier failed");
    }
    write_perf_samples(log_dir, rank, mode, samples);
    if (rank == 0) {
      std::fprintf(stderr,
                   "[P2A_NATIVE_VERDICT] ranks=8 native_calls=130 "
                   "output_checks=PASS guard_checks=PASS\n");
    }
  }
  return 0;
}

int run_all(const std::string& log_dir, const std::string& mode) {
  std::filesystem::create_directories(log_dir);
  void* mem = mmap(nullptr, sizeof(SharedCoord), PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  check(mem != MAP_FAILED, "shared coordination mmap failed");
  auto* shm = new (mem) SharedCoord();
  std::vector<pid_t> pids(kRanks);
  for (uint32_t rank = 0; rank < kRanks; ++rank) {
    const pid_t pid = fork();
    if (pid == 0) {
      const int rc = run_rank(rank, shm, log_dir, mode);
      std::_Exit(rc);
    }
    check(pid > 0, "rank fork failed");
    pids[rank] = pid;
  }

  int result = 0;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(240);
  size_t completed = 0;
  while (completed < kRanks) {
    for (size_t i = 0; i < kRanks; ++i) {
      if (pids[i] <= 0) continue;
      int status = 0;
      const pid_t rc = waitpid(pids[i], &status, WNOHANG);
      if (rc == pids[i]) {
        ++completed;
        pids[i] = -1;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) result = 1;
      }
    }
    if (std::chrono::steady_clock::now() > deadline) {
      shm->failed.store(1);
      for (const pid_t pid : pids) {
        if (pid > 0) kill(pid, SIGTERM);
      }
      result = 1;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  munmap(mem, sizeof(SharedCoord));
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  std::string log_dir = "/tmp/gaudirt-p2a-capture";
  std::string mode = "capture";
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--log-dir") == 0 && i + 1 < argc) {
      log_dir = argv[++i];
    } else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      mode = argv[++i];
      if (mode != "capture" && mode != "native" && mode != "direct") {
        std::fprintf(stderr, "mode must be capture, native, or direct\n");
        return 2;
      }
    } else {
      std::fprintf(stderr, "usage: %s [--mode capture|native|direct] [--log-dir PATH]\n",
                   argv[0]);
      return 2;
    }
  }
  return run_all(log_dir, mode);
}
#endif
