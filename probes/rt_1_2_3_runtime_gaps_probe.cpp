// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/common/abba_harness.hpp"
#include "gaudirt/common/host_policy.hpp"
#include "gaudirt/common/synapse_raii.hpp"
#include "gaudirt/runtime/generation_epoch_tracker.hpp"
#include "gaudirt/runtime/inflight_credit_manager.hpp"
#include "gaudirt/runtime/unified_region_lowerer.hpp"

#include <hccl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#define GAUDIRT_HCCL_CHECK(expr)                                               \
  do {                                                                         \
    hcclResult_t _r = (expr);                                                  \
    if (_r != hcclSuccess) {                                                   \
      std::fprintf(stderr, "[GaudiRT] HCCL error %d (%s) at %s:%d\n",          \
                   static_cast<int>(_r), hcclGetErrorString(_r), __FILE__,     \
                   __LINE__);                                                  \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

using namespace gaudirt::common;
using namespace gaudirt::runtime;

namespace {

struct SharedCoord {
  hcclUniqueId uid;
  std::atomic<int> uid_ready{0};
  std::atomic<int> barrier_sense{0};
  std::atomic<int> barrier_count{0};
  // Per-rank RT-1 / RT-2 / RT-3 results
  double rt1_arm_a_host_us[8]{};
  double rt1_arm_b_host_us[8]{};
  double rt1_arm_a_dev_us[8]{};
  double rt1_arm_b_dev_us[8]{};
  double rt1_early_a_dev_us[8]{};
  double rt1_early_b_dev_us[8]{};
};

void host_barrier(SharedCoord* shm, int nranks, int& local_sense) {
  local_sense = 1 - local_sense;
  if (shm->barrier_count.fetch_add(1) == nranks - 1) {
    shm->barrier_count.store(0);
    shm->barrier_sense.store(local_sense);
  } else {
    while (shm->barrier_sense.load() != local_sense) {
    }
  }
}

struct SimpleRecipe {
  synDeviceId dev = 0;
  synRecipeHandle handle = nullptr;
  std::vector<synLaunchTensorInfoExt> bindings;
  std::vector<std::string> names;
  uint64_t workspace = 0;
  uint64_t workspace_bytes = 0;

  void launch(synStreamHandle stream) const {
    GAUDIRT_SYN_CHECK(synLaunchWithExternalEventsExt(
        stream, bindings.data(), bindings.size(), workspace, handle, nullptr, 0,
        0));
  }

  void destroy() {
    for (auto& b : bindings) {
      if (b.pTensorAddress) synDeviceFree(dev, b.pTensorAddress, 0);
    }
    if (workspace) synDeviceFree(dev, workspace, 0);
    if (handle) synRecipeDestroy(handle);
  }
};

SimpleRecipe make_gemm_recipe(synDeviceId dev, unsigned m, unsigned k,
                              unsigned n, unsigned chain_len, const char* tag) {
  SimpleRecipe r;
  r.dev = dev;
  synGraphHandle graph = nullptr;
  GAUDIRT_SYN_CHECK(synGraphCreate(&graph, synDeviceGaudi2));
  r.names = {std::string(tag) + "_in", std::string(tag) + "_w",
             std::string(tag) + "_out"};
  std::vector<synSectionHandle> sections;
  std::vector<synTensor> tensors;

  auto mk = [&](const char* name, unsigned d0, unsigned d1, bool persistent) {
    synSectionHandle s = nullptr;
    if (persistent) {
      GAUDIRT_SYN_CHECK(synSectionCreate(&s, 0, graph));
      GAUDIRT_SYN_CHECK(synSectionSetPersistent(s, true));
      sections.push_back(s);
    }
    synTensorDescriptor d{};
    d.m_name = name;
    d.m_dataType = syn_type_bf16;
    d.m_dims = 2;
    d.m_sizes[0] = d.m_minSizes[0] = d0;
    d.m_sizes[1] = d.m_minSizes[1] = d1;
    synTensor t = nullptr;
    GAUDIRT_SYN_CHECK(synTensorCreate(&t, &d, s, 0));
    tensors.push_back(t);
    return t;
  };

  synTensor x = mk(r.names[0].c_str(), k, m, true);
  synTensor w = mk(r.names[1].c_str(), n, k, true);
  synTensor out = mk(r.names[2].c_str(), n, m, true);
  synTensor cur = x;
  std::vector<std::string> mid_names;
  for (unsigned i = 0; i < chain_len; ++i) {
    mid_names.push_back(std::string(tag) + "_mid_" + std::to_string(i));
  }
  for (unsigned i = 0; i < chain_len; ++i) {
    synTensor next =
        (i + 1 == chain_len) ? out : mk(mid_names[i].c_str(), n, m, false);
    synTensor in[2] = {cur, w};
    synGEMMParams p{false, false};
    GAUDIRT_SYN_CHECK(synNodeCreate(
        graph, in, &next, 2, 1, &p, sizeof(p), "gemm",
        (std::string(tag) + "_g" + std::to_string(i)).c_str(), nullptr,
        nullptr));
    cur = next;
  }

  GAUDIRT_SYN_CHECK(synGraphCompile(&r.handle, graph, tag, nullptr));
  const char* cn[3] = {r.names[0].c_str(), r.names[1].c_str(),
                       r.names[2].c_str()};
  uint64_t ids[3]{};
  GAUDIRT_SYN_CHECK(synTensorRetrieveIds(r.handle, cn, ids, 3));
  const unsigned dims[3][2] = {{k, m}, {n, k}, {n, m}};
  r.bindings.resize(3);
  for (int i = 0; i < 3; ++i) {
    uint64_t addr = 0;
    GAUDIRT_SYN_CHECK(
        synDeviceMalloc(dev, uint64_t(dims[i][0]) * dims[i][1] * 2, 0, 0, &addr));
    r.bindings[i] = {};
    r.bindings[i].tensorName = r.names[i].c_str();
    r.bindings[i].pTensorAddress = addr;
    r.bindings[i].tensorType = DATA_TENSOR;
    r.bindings[i].tensorId = ids[i];
    r.bindings[i].tensorSize[0] = dims[i][0];
    r.bindings[i].tensorSize[1] = dims[i][1];
  }
  GAUDIRT_SYN_CHECK(synWorkspaceGetSize(&r.workspace_bytes, r.handle));
  if (r.workspace_bytes > 0) {
    GAUDIRT_SYN_CHECK(
        synDeviceMalloc(dev, r.workspace_bytes, 0, 0, &r.workspace));
  }
  GAUDIRT_SYN_CHECK(synGraphDestroy(graph));
  for (auto t : tensors) GAUDIRT_SYN_CHECK(synTensorDestroy(t));
  for (auto s : sections) GAUDIRT_SYN_CHECK(synSectionDestroy(s));
  return r;
}

// Build exact 286-command production70 topology for UnifiedRegionLowerer test
std::vector<CapturedCommandDesc> make_production70_command_topology() {
  std::vector<CapturedCommandDesc> cmds(286);
  const uint64_t comm_id = 0x7777;
  uint64_t next_alloc = 100;
  uint64_t cur_activation = next_alloc++;

  for (size_t i = 0; i < 286; ++i) {
    auto& c = cmds[i];
    c.original_index = i;
    c.complete = true;
    c.external_events = 0;

    if (i == 5) {
      // Command 5: mixed compute + 1 collective (o_proj + AllGather)
      c.compute = true;
      c.collectives = 1;
      c.communicator = comm_id;
      uint64_t mid_alloc = next_alloc++;
      uint64_t out_alloc = next_alloc++;
      c.compute_reads = {cur_activation, next_alloc++};
      c.compute_writes = {mid_alloc};
      c.collective_reads = {mid_alloc};
      c.collective_writes = {out_alloc};
      cur_activation = out_alloc;
    } else if (i == 0 || i == 2 || i == 4 || i == 282 || i == 283 || i == 285 ||
               (i >= 6 && i < 282 && (i % 2 == 0))) {
      // Pure compute command (144 total)
      c.compute = true;
      c.collectives = 0;
      c.communicator = 0;
      uint64_t out_alloc = next_alloc++;
      c.compute_reads = {cur_activation, next_alloc++};
      c.compute_writes = {out_alloc};
      cur_activation = out_alloc;
    } else {
      // Pure collective command (141 total: 1, 3, 7..281 odd, 284)
      c.compute = false;
      c.collectives = 1;
      c.communicator = comm_id;
      uint64_t out_alloc = next_alloc++;
      c.collective_reads = {cur_activation};
      c.collective_writes = {out_alloc};
      cur_activation = out_alloc;
    }
  }
  return cmds;
}

void run_rank(int rank, int nranks, SharedCoord* shm,
              const char* out_json_path) {
  int target_cpu = rank * 4;
  lock_host_policy(target_cpu, 1024);

  std::string mod_id = std::to_string(rank);
  setenv("ID", mod_id.c_str(), 1);
  setenv("HLS_MODULE_ID", mod_id.c_str(), 1);

  SynapseContext ctx(rank);
  synDeviceId dev = ctx.device();

  int local_sense = 0;
  if (rank == 0) {
    GAUDIRT_HCCL_CHECK(hcclGetUniqueId(&shm->uid));
    shm->uid_ready.store(1);
  } else {
    while (shm->uid_ready.load() == 0) {
    }
  }
  host_barrier(shm, nranks, local_sense);

  hcclComm_t comm = nullptr;
  GAUDIRT_HCCL_CHECK(hcclCommInitRank(&comm, nranks, shm->uid, rank));
  host_barrier(shm, nranks, local_sense);

  synStreamHandle compute_stream = nullptr;
  synStreamHandle comm_stream = nullptr;
  GAUDIRT_SYN_CHECK(synStreamCreateGeneric(&compute_stream, dev, 0));
  GAUDIRT_SYN_CHECK(synStreamCreateGeneric(&comm_stream, dev, 0));

  // Compile the 12 distinct recipes matching production70's reuse topology:
  // Recipe 0: 60x (MoE block), Recipe 1: 59x (SWA Attn), Recipe 2: 9x,
  // Recipe 3: 9x, plus 8 boundary recipes (1x each) = 145 compute launches/step
  std::vector<SimpleRecipe> distinct_recipes(12);
  for (int i = 0; i < 12; ++i) {
    std::string tag = "rt_r" + std::to_string(rank) + "_rec" + std::to_string(i);
    distinct_recipes[i] = make_gemm_recipe(dev, 64, 512, 512, 1, tag.c_str());
  }

  // Build the 145-compute-launch step schedule mapping each compute index to one
  // of the 12 distinct recipes:
  // 0->rec4, 1->rec5, 2->rec6, 3(Cmd5)->rec7, 4->rec8,
  // then 60x rec0, 59x rec1, 9x rec2, 9x rec3, plus rec9, rec10, rec11
  std::vector<int> compute_recipe_idx;
  compute_recipe_idx.reserve(145);
  compute_recipe_idx.push_back(4);  // Cmd 0
  compute_recipe_idx.push_back(5);  // Cmd 2
  compute_recipe_idx.push_back(6);  // Cmd 4
  compute_recipe_idx.push_back(7);  // Cmd 5 (mixed)
  compute_recipe_idx.push_back(8);  // Cmd 6
  for (int i = 0; i < 60; ++i) compute_recipe_idx.push_back(0);
  for (int i = 0; i < 59; ++i) compute_recipe_idx.push_back(1);
  for (int i = 0; i < 9; ++i) compute_recipe_idx.push_back(2);
  for (int i = 0; i < 9; ++i) compute_recipe_idx.push_back(3);
  compute_recipe_idx.push_back(9);
  compute_recipe_idx.push_back(10);
  compute_recipe_idx.push_back(11);

  // Allocate collective buffers for 4096 FP32 elements (16 KB/rank -> 128 KB)
  const size_t send_count = 4096;
  const size_t send_bytes = send_count * sizeof(float);
  const size_t recv_bytes = send_bytes * nranks;
  uint64_t send_buf = 0, recv_buf = 0;
  GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, send_bytes, 0, 0, &send_buf));
  GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, recv_bytes, 0, 0, &recv_buf));

  // Build 2,392 synthetic retained tensor bindings and addresses (matching
  // production70's 2,392 leases and 2,391 unique boundaries)
  const size_t kNumBindings = 2392;
  std::vector<BindingSnapshot> captured_bindings(kNumBindings);
  std::vector<uint64_t> footprint_addrs(kNumBindings);
  for (size_t i = 0; i < kNumBindings; ++i) {
    uint64_t fake_addr = 0x100000000ULL + i * 4096ULL;
    footprint_addrs[i] = fake_addr;
    captured_bindings[i] = BindingSnapshot{
        reinterpret_cast<const void*>(0x20000000ULL + i * 64),
        reinterpret_cast<const void*>(0x30000000ULL + i * 64),
        fake_addr,
        4096,
        0,
        6,
        0x9e3779b97f4a7c15ULL ^ i};
  }
  std::vector<BindingSnapshot> live_bindings = captured_bindings;

  LegacyAddressEventMap legacy_sem;
  legacy_sem.reserve(kNumBindings);
  legacy_sem.add_producer(footprint_addrs, 1, 1);

  GenerationEpochTracker epoch_tracker;
  epoch_tracker.initialize(captured_bindings, footprint_addrs, 1);
  // Warm up the first verification so steady-state decode steps hit O(1)
  epoch_tracker.verify_bindings(live_bindings);
  epoch_tracker.needs_entry_wait();

  // Pre-allocate Synapse events for region handoffs & timing
  std::vector<synEventHandle> sync_events(16, nullptr);
  for (auto& ev : sync_events) {
    GAUDIRT_SYN_CHECK(synEventCreate(&ev, dev, 0));
  }
  synEventHandle ev_start = nullptr, ev_early = nullptr, ev_end = nullptr;
  GAUDIRT_SYN_CHECK(synEventCreate(&ev_start, dev, EVENT_COLLECT_TIME));
  GAUDIRT_SYN_CHECK(synEventCreate(&ev_early, dev, EVENT_COLLECT_TIME));
  GAUDIRT_SYN_CHECK(synEventCreate(&ev_end, dev, EVENT_COLLECT_TIME));

  // Warmup collectives and recipes
  for (int w = 0; w < 4; ++w) {
    distinct_recipes[w % 12].launch(compute_stream);
    GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_buf),
                                     reinterpret_cast<void*>(recv_buf),
                                     send_count, hcclFloat32, comm,
                                     compute_stream));
  }
  GAUDIRT_SYN_CHECK(synStreamSynchronize(compute_stream));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(comm_stream));
  host_barrier(shm, nranks, local_sense);

  // =========================================================================
  // Experiment 1 (RT-1 + RT-2 on 8 Ranks):
  // Arm A: Legacy 2-Region Split at Command 5 (with stream switch to
  //        comm_stream for Command 5's collective + 2x 2,392-address SEM scan
  //        + 2,392-binding preflight scan)
  // Arm B: GaudiRT Unified Single-Region [0..285] (Command 5 lowered into
  //        compute + collective on compute_stream + O(1) Epoch Guard)
  // =========================================================================
  uint64_t next_ev_id = 10;
  auto run_step_arm = [&](bool unified_arm, double& host_us,
                          double& early_dev_us, double& full_dev_us) {
    GAUDIRT_SYN_CHECK(synStreamSynchronize(compute_stream));
    GAUDIRT_SYN_CHECK(synStreamSynchronize(comm_stream));
    host_barrier(shm, nranks, local_sense);

    auto t_host_0 = std::chrono::steady_clock::now();
    GAUDIRT_SYN_CHECK(synEventRecord(ev_start, compute_stream));

    if (!unified_arm) {
      // Legacy preflight: 2,392 binding checks
      for (size_t i = 0; i < kNumBindings; ++i) {
        if (!captured_bindings[i].matches(live_bindings[i])) std::abort();
      }
      // Enter Region 0 [0..4]: comm_stream -> compute_stream
      GAUDIRT_SYN_CHECK(synEventRecord(sync_events[0], comm_stream));
      GAUDIRT_SYN_CHECK(synStreamWaitEvent(compute_stream, sync_events[0], 0));
      auto waits0 = legacy_sem.get_wait_events(footprint_addrs);
      if (!waits0.empty()) {
        GAUDIRT_SYN_CHECK(synEventRecord(sync_events[1], compute_stream));
        GAUDIRT_SYN_CHECK(
            synStreamWaitEvent(compute_stream, sync_events[1], 0));
      }
      // Commands 0..4: 3 compute (0, 2, 4) + 2 collectives (1, 3)
      distinct_recipes[compute_recipe_idx[0]].launch(compute_stream);
      GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       send_count, hcclFloat32, comm,
                                       compute_stream));
      distinct_recipes[compute_recipe_idx[1]].launch(compute_stream);
      GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       send_count, hcclFloat32, comm,
                                       compute_stream));
      distinct_recipes[compute_recipe_idx[2]].launch(compute_stream);

      // Leave Region 0 + publish(Region 0)
      GAUDIRT_SYN_CHECK(synEventRecord(sync_events[2], compute_stream));
      GAUDIRT_SYN_CHECK(synStreamWaitEvent(comm_stream, sync_events[2], 0));
      legacy_sem.add_producer(footprint_addrs, 1, ++next_ev_id);
      GAUDIRT_SYN_CHECK(synEventRecord(sync_events[3], compute_stream));

      // Command 5 (legacy fallback): compute on compute_stream, collective on
      // comm_stream!
      distinct_recipes[compute_recipe_idx[3]].launch(compute_stream);
      GAUDIRT_SYN_CHECK(synEventRecord(sync_events[4], compute_stream));
      GAUDIRT_SYN_CHECK(synStreamWaitEvent(comm_stream, sync_events[4], 0));
      GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       send_count, hcclFloat32, comm,
                                       comm_stream));
      GAUDIRT_SYN_CHECK(synEventRecord(sync_events[5], comm_stream));

      // Enter Region 1 [6..285]: comm_stream -> compute_stream + 2,392-addr
      // wait scan
      GAUDIRT_SYN_CHECK(synEventRecord(sync_events[6], comm_stream));
      GAUDIRT_SYN_CHECK(synStreamWaitEvent(compute_stream, sync_events[6], 0));
      auto waits1 = legacy_sem.get_wait_events(footprint_addrs);
      if (!waits1.empty()) {
        GAUDIRT_SYN_CHECK(
            synStreamWaitEvent(compute_stream, sync_events[5], 0));
      }

      // Command 6 (5th compute) + Command 7 (4th collective)
      distinct_recipes[compute_recipe_idx[4]].launch(compute_stream);
      GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       send_count, hcclFloat32, comm,
                                       compute_stream));
      GAUDIRT_SYN_CHECK(synEventRecord(ev_early, compute_stream));

      // Remaining Commands 8..285 (140 compute + 138 collectives)
      size_t comp_cursor = 5;
      for (int pair = 0; pair < 137; ++pair) {
        distinct_recipes[compute_recipe_idx[comp_cursor++]].launch(
            compute_stream);
        GAUDIRT_HCCL_CHECK(hcclAllGather(
            reinterpret_cast<const void*>(send_buf),
            reinterpret_cast<void*>(recv_buf), send_count, hcclFloat32, comm,
            compute_stream));
      }
      // Commands 282, 283 (compute), 284 (collective), 285 (compute)
      distinct_recipes[compute_recipe_idx[comp_cursor++]].launch(
          compute_stream);
      distinct_recipes[compute_recipe_idx[comp_cursor++]].launch(
          compute_stream);
      GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       send_count, hcclFloat32, comm,
                                       compute_stream));
      distinct_recipes[compute_recipe_idx[comp_cursor++]].launch(
          compute_stream);

      // Leave Region 1 + publish(Region 1)
      GAUDIRT_SYN_CHECK(synEventRecord(sync_events[7], compute_stream));
      GAUDIRT_SYN_CHECK(synStreamWaitEvent(comm_stream, sync_events[7], 0));
      legacy_sem.add_producer(footprint_addrs, 1, ++next_ev_id);
      GAUDIRT_SYN_CHECK(synEventRecord(sync_events[8], compute_stream));
    } else {
      // Arm B: Unified Single-Region [0..285] + O(1) Generation-Guarded Epoch
      if (!epoch_tracker.verify_bindings(live_bindings)) std::abort();
      if (epoch_tracker.needs_entry_wait()) {
        GAUDIRT_SYN_CHECK(synEventRecord(sync_events[0], comm_stream));
        GAUDIRT_SYN_CHECK(
            synStreamWaitEvent(compute_stream, sync_events[0], 0));
      }

      // Commands 0..7 (including Command 5 lowered into Compute + Collective on
      // compute_stream!)
      distinct_recipes[compute_recipe_idx[0]].launch(compute_stream);
      GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       send_count, hcclFloat32, comm,
                                       compute_stream));
      distinct_recipes[compute_recipe_idx[1]].launch(compute_stream);
      GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       send_count, hcclFloat32, comm,
                                       compute_stream));
      distinct_recipes[compute_recipe_idx[2]].launch(compute_stream);
      // Command 5a (compute) + Command 5b (collective) inline on compute_stream
      distinct_recipes[compute_recipe_idx[3]].launch(compute_stream);
      GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       send_count, hcclFloat32, comm,
                                       compute_stream));
      // Command 6 (compute) + Command 7 (collective)
      distinct_recipes[compute_recipe_idx[4]].launch(compute_stream);
      GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       send_count, hcclFloat32, comm,
                                       compute_stream));
      GAUDIRT_SYN_CHECK(synEventRecord(ev_early, compute_stream));

      // Remaining Commands 8..285 (140 compute + 138 collectives)
      size_t comp_cursor = 5;
      for (int pair = 0; pair < 137; ++pair) {
        distinct_recipes[compute_recipe_idx[comp_cursor++]].launch(
            compute_stream);
        GAUDIRT_HCCL_CHECK(hcclAllGather(
            reinterpret_cast<const void*>(send_buf),
            reinterpret_cast<void*>(recv_buf), send_count, hcclFloat32, comm,
            compute_stream));
      }
      distinct_recipes[compute_recipe_idx[comp_cursor++]].launch(
          compute_stream);
      distinct_recipes[compute_recipe_idx[comp_cursor++]].launch(
          compute_stream);
      GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_buf),
                                       reinterpret_cast<void*>(recv_buf),
                                       send_count, hcclFloat32, comm,
                                       compute_stream));
      distinct_recipes[compute_recipe_idx[comp_cursor++]].launch(
          compute_stream);

      // Single O(1) region exit epoch fence
      GAUDIRT_SYN_CHECK(synEventRecord(sync_events[7], compute_stream));
      epoch_tracker.publish_region_epoch(++next_ev_id);
    }

    GAUDIRT_SYN_CHECK(synEventRecord(ev_end, compute_stream));
    auto t_host_1 = std::chrono::steady_clock::now();
    host_us =
        std::chrono::duration<double, std::micro>(t_host_1 - t_host_0).count();

    GAUDIRT_SYN_CHECK(synEventSynchronize(ev_end));
    uint64_t early_ns = 0, full_ns = 0;
    GAUDIRT_SYN_CHECK(synEventElapsedTime(&early_ns, ev_start, ev_early));
    GAUDIRT_SYN_CHECK(synEventElapsedTime(&full_ns, ev_start, ev_end));
    early_dev_us = static_cast<double>(early_ns) / 1e3;
    full_dev_us = static_cast<double>(full_ns) / 1e3;
  };

  std::vector<double> arm_a_host, arm_b_host, arm_a_early, arm_b_early,
      arm_a_dev, arm_b_dev;
  const int kAbbaRounds = 8;
  for (int r = 0; r < kAbbaRounds; ++r) {
    double h = 0, e = 0, d = 0;
    run_step_arm(false, h, e, d);
    arm_a_host.push_back(h);
    arm_a_early.push_back(e);
    arm_a_dev.push_back(d);

    run_step_arm(true, h, e, d);
    arm_b_host.push_back(h);
    arm_b_early.push_back(e);
    arm_b_dev.push_back(d);

    run_step_arm(true, h, e, d);
    arm_b_host.push_back(h);
    arm_b_early.push_back(e);
    arm_b_dev.push_back(d);

    run_step_arm(false, h, e, d);
    arm_a_host.push_back(h);
    arm_a_early.push_back(e);
    arm_a_dev.push_back(d);
  }

  shm->rt1_arm_a_host_us[rank] = summarize(arm_a_host).p50;
  shm->rt1_arm_b_host_us[rank] = summarize(arm_b_host).p50;
  shm->rt1_early_a_dev_us[rank] = summarize(arm_a_early).p50;
  shm->rt1_early_b_dev_us[rank] = summarize(arm_b_early).p50;
  shm->rt1_arm_a_dev_us[rank] = summarize(arm_a_dev).p50;
  shm->rt1_arm_b_dev_us[rank] = summarize(arm_b_dev).p50;
  host_barrier(shm, nranks, local_sense);

  if (rank == 0) {
    // =========================================================================
    // Experiment 2 (RT-2 Isolated CPU Preflight + Event Map Scan Breakdown):
    // Measure pure CPU overhead of 2,392-binding preflight + 2,392-address SEM
    // wait/publish (2 regions vs 1 region vs O(1) GenerationEpochTracker).
    // =========================================================================
    auto bench_rt2_2region = [&]() {
      for (size_t i = 0; i < kNumBindings; ++i) {
        if (!captured_bindings[i].matches(live_bindings[i])) std::abort();
      }
      auto w0 = legacy_sem.get_wait_events(footprint_addrs);
      legacy_sem.add_producer(footprint_addrs, 1, ++next_ev_id);
      auto w1 = legacy_sem.get_wait_events(footprint_addrs);
      legacy_sem.add_producer(footprint_addrs, 1, ++next_ev_id);
      GAUDIRT_SYN_CHECK(synEventRecord(sync_events[0], compute_stream));
      GAUDIRT_SYN_CHECK(synEventRecord(sync_events[1], compute_stream));
      (void)w0;
      (void)w1;
    };

    auto bench_rt2_o1_epoch = [&]() {
      if (!epoch_tracker.verify_bindings(live_bindings)) std::abort();
      (void)epoch_tracker.needs_entry_wait();
      GAUDIRT_SYN_CHECK(synEventRecord(sync_events[0], compute_stream));
      epoch_tracker.publish_region_epoch(++next_ev_id);
    };

    std::vector<double> rt2_a_samples, rt2_b_samples;
    for (int i = 0; i < 40; ++i) {
      double t0 = now_us();
      bench_rt2_2region();
      rt2_a_samples.push_back(now_us() - t0);

      t0 = now_us();
      bench_rt2_o1_epoch();
      rt2_b_samples.push_back(now_us() - t0);

      t0 = now_us();
      bench_rt2_o1_epoch();
      rt2_b_samples.push_back(now_us() - t0);

      t0 = now_us();
      bench_rt2_2region();
      rt2_a_samples.push_back(now_us() - t0);
    }
    auto rt2_a_stat = summarize(rt2_a_samples);
    auto rt2_b_stat = summarize(rt2_b_samples);
    GAUDIRT_SYN_CHECK(synStreamSynchronize(compute_stream));

    // =========================================================================
    // Experiment 3 (RT-3: Real-Model 12-Distinct-Recipe Reuse Horizon &
    // InFlightCreditManager Multi-Step Verification at K = 1, 2, 3, 4 steps):
    // =========================================================================
    SimpleRecipe gate_recipe =
        make_gemm_recipe(dev, 4096, 4096, 4096, 24, "rt3_gate");
    // Warm up gate + 12 distinct recipes
    gate_recipe.launch(compute_stream);
    for (int i = 0; i < 12; ++i) distinct_recipes[i].launch(compute_stream);
    GAUDIRT_SYN_CHECK(synStreamSynchronize(compute_stream));

    // 3A: Measure exact unthrottled first_block_idx under 12-distinct-recipe
    // reuse (145 launches/step, expected first_block_idx = 575 - 2*12 = 551)
    const size_t kMaxTestLaunches = 580;  // 4 full steps of 145 launches
    std::vector<double> call_lat_us(kMaxTestLaunches, 0.0);
    gate_recipe.launch(compute_stream);
    for (size_t idx = 0; idx < kMaxTestLaunches; ++idx) {
      int r_id = compute_recipe_idx[idx % 145];
      auto t0 = std::chrono::steady_clock::now();
      distinct_recipes[r_id].launch(compute_stream);
      auto t1 = std::chrono::steady_clock::now();
      call_lat_us[idx] =
          std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    GAUDIRT_SYN_CHECK(synStreamSynchronize(compute_stream));

    int unthrottled_first_block_idx = -1;
    double unthrottled_block_dur_us = 0.0;
    for (size_t idx = 0; idx < kMaxTestLaunches; ++idx) {
      if (call_lat_us[idx] > 1000.0) {
        unthrottled_first_block_idx = static_cast<int>(idx);
        unthrottled_block_dur_us = call_lat_us[idx];
        break;
      }
    }

    // 3B: Test K = 1, 2, 3, 4 full steps (145 launches/step) under
    // InFlightCreditManager (budget = 575, safety_margin = 16 -> effective 559)
    std::vector<synRecipeHandle> step_distinct_handles;
    step_distinct_handles.reserve(12);
    for (int i = 0; i < 12; ++i) {
      step_distinct_handles.push_back(distinct_recipes[i].handle);
    }

    struct StepBurstResult {
      int k_steps = 0;
      size_t total_launches = 0;
      double max_single_launch_us = 0.0;
      double median_launch_us = 0.0;
      size_t mid_step_blocks = 0;
      size_t credit_backpressure_waits = 0;
      double credit_backpressure_wait_us = 0.0;
      size_t peak_used_chunks = 0;
    };
    std::vector<StepBurstResult> burst_results;

    for (int k_steps : {1, 2, 3, 4}) {
      InFlightCreditManager credit_mgr(dev, 575, 16);
      std::vector<double> launch_times;
      launch_times.reserve(k_steps * 145);
      size_t mid_blocks = 0;

      // Launch a shorter ~30ms gate so step 0 finishes cleanly while steps 1..2
      // are still in flight
      SimpleRecipe short_gate =
          make_gemm_recipe(dev, 4096, 4096, 4096, 6, "rt3_short_gate");
      short_gate.launch(compute_stream);
      GAUDIRT_SYN_CHECK(synStreamSynchronize(compute_stream));

      short_gate.launch(compute_stream);
      for (int s = 0; s < k_steps; ++s) {
        credit_mgr.acquire_step_budget(145, step_distinct_handles);
        for (int c = 0; c < 145; ++c) {
          int r_id = compute_recipe_idx[c];
          auto t0 = std::chrono::steady_clock::now();
          distinct_recipes[r_id].launch(compute_stream);
          auto t1 = std::chrono::steady_clock::now();
          double lus =
              std::chrono::duration<double, std::micro>(t1 - t0).count();
          launch_times.push_back(lus);
          if (lus > 1000.0) ++mid_blocks;
        }
        credit_mgr.commit_step(s + 1, compute_stream, 145,
                               step_distinct_handles);
      }
      credit_mgr.drain_all();
      GAUDIRT_SYN_CHECK(synStreamSynchronize(compute_stream));
      short_gate.destroy();

      auto st = credit_mgr.stats();
      auto lstats = summarize(launch_times);
      burst_results.push_back(StepBurstResult{
          k_steps, static_cast<size_t>(k_steps * 145), lstats.max,
          lstats.p50, mid_blocks, st.backpressure_waits,
          st.total_backpressure_wait_us, st.peak_used_chunks});
    }

    gate_recipe.destroy();

    // Verify UnifiedRegionLowerer on the exact 286-command production70 graph
    auto prod70_cmds = make_production70_command_topology();
    auto rep_legacy = UnifiedRegionLowerer::lower(prod70_cmds, false);
    auto rep_unified = UnifiedRegionLowerer::lower(prod70_cmds, true);

    FILE* fp = std::fopen(out_json_path, "w");
    if (fp) {
      std::fprintf(fp, "{\n");
      std::fprintf(fp, "  \"probe\": \"rt_1_2_3_runtime_gaps_probe\",\n");
      std::fprintf(fp, "  \"tp_size\": %d,\n", nranks);
      std::fprintf(fp, "  \"uclamp_locked\": 1024,\n");

      // RT-1 Lowering & 8-Rank Execution
      std::fprintf(fp, "  \"rt1_unified_region_lowering\": {\n");
      std::fprintf(fp, "    \"legacy_lowering\": {\n");
      std::fprintf(fp, "      \"original_commands\": %zu,\n",
                   rep_legacy.original_commands);
      std::fprintf(fp, "      \"subcalls\": %zu,\n",
                   rep_legacy.subcalls.size());
      std::fprintf(fp, "      \"regions\": %zu,\n", rep_legacy.regions.size());
      std::fprintf(fp, "      \"coalesced_collectives\": %zu,\n",
                   rep_legacy.coalesced_collectives);
      std::fprintf(fp, "      \"total_collectives\": %zu\n",
                   rep_legacy.total_collective_subcalls);
      std::fprintf(fp, "    },\n");
      std::fprintf(fp, "    \"gaudirt_unified_lowering\": {\n");
      std::fprintf(fp, "      \"original_commands\": %zu,\n",
                   rep_unified.original_commands);
      std::fprintf(fp, "      \"decomposed_mixed_commands\": %zu,\n",
                   rep_unified.decomposed_mixed_commands);
      std::fprintf(fp, "      \"subcalls\": %zu,\n",
                   rep_unified.subcalls.size());
      std::fprintf(fp, "      \"compute_subcalls\": %zu,\n",
                   rep_unified.total_compute_subcalls);
      std::fprintf(fp, "      \"collective_subcalls\": %zu,\n",
                   rep_unified.total_collective_subcalls);
      std::fprintf(fp, "      \"regions\": %zu,\n", rep_unified.regions.size());
      std::fprintf(fp, "      \"region0_first_original_cmd\": %zu,\n",
                   rep_unified.regions[0].first_original_command);
      std::fprintf(fp, "      \"region0_last_original_cmd\": %zu,\n",
                   rep_unified.regions[0].last_original_command);
      std::fprintf(fp, "      \"coalesced_collectives\": %zu\n",
                   rep_unified.coalesced_collectives);
      std::fprintf(fp, "    },\n");
      std::fprintf(fp, "    \"eight_rank_abba_step_comparison\": {\n");
      std::fprintf(fp, "      \"ranks\": [\n");
      for (int r = 0; r < nranks; ++r) {
        std::fprintf(
            fp,
            "        {\"rank\": %d, \"arm_a_host_us\": %.2f, "
            "\"arm_b_host_us\": %.2f, \"host_saved_us\": %.2f, "
            "\"arm_a_early_dev_us\": %.2f, \"arm_b_early_dev_us\": %.2f, "
            "\"early_dev_saved_us\": %.2f, \"arm_a_full_dev_us\": %.2f, "
            "\"arm_b_full_dev_us\": %.2f, \"full_dev_saved_us\": %.2f}%s\n",
            r, shm->rt1_arm_a_host_us[r], shm->rt1_arm_b_host_us[r],
            shm->rt1_arm_a_host_us[r] - shm->rt1_arm_b_host_us[r],
            shm->rt1_early_a_dev_us[r], shm->rt1_early_b_dev_us[r],
            shm->rt1_early_a_dev_us[r] - shm->rt1_early_b_dev_us[r],
            shm->rt1_arm_a_dev_us[r], shm->rt1_arm_b_dev_us[r],
            shm->rt1_arm_a_dev_us[r] - shm->rt1_arm_b_dev_us[r],
            (r + 1 < nranks) ? "," : "");
      }
      std::fprintf(fp, "      ]\n");
      std::fprintf(fp, "    }\n");
      std::fprintf(fp, "  },\n");

      // RT-2 Isolated CPU Preflight + Event Map Scan Breakdown
      std::fprintf(fp, "  \"rt2_generation_epoch_guard\": {\n");
      std::fprintf(fp, "    \"retained_bindings\": %zu,\n", kNumBindings);
      std::fprintf(fp,
                   "    \"legacy_2region_preflight_and_sem_median_us\": %.2f,\n",
                   rt2_a_stat.p50);
      std::fprintf(fp, "    \"gaudirt_o1_epoch_guard_median_us\": %.2f,\n",
                   rt2_b_stat.p50);
      std::fprintf(fp, "    \"cpu_saved_per_step_us\": %.2f,\n",
                   rt2_a_stat.p50 - rt2_b_stat.p50);
      std::fprintf(fp, "    \"speedup_ratio\": %.2f\n",
                   rt2_a_stat.p50 /
                       std::max(0.01, rt2_b_stat.p50));
      std::fprintf(fp, "  },\n");

      // RT-3 12-Distinct-Recipe Reuse Horizon & Credit Manager
      std::fprintf(fp, "  \"rt3_inflight_credit_manager\": {\n");
      std::fprintf(fp, "    \"distinct_recipes_per_step\": 12,\n");
      std::fprintf(fp, "    \"compute_launches_per_step\": 145,\n");
      std::fprintf(
          fp, "    \"predicted_first_block_idx\": %zu,\n",
          InFlightCreditManager::max_launches_for_distinct(12, 575));
      std::fprintf(fp, "    \"measured_unthrottled_first_block_idx\": %d,\n",
                   unthrottled_first_block_idx);
      std::fprintf(fp,
                   "    \"measured_unthrottled_block_duration_us\": %.2f,\n",
                   unthrottled_block_dur_us);
      std::fprintf(fp,
                   "    \"safe_unthrottled_full_steps\": %.2f,\n",
                   static_cast<double>(unthrottled_first_block_idx) / 145.0);
      std::fprintf(fp, "    \"credit_managed_bursts\": [\n");
      for (size_t i = 0; i < burst_results.size(); ++i) {
        const auto& b = burst_results[i];
        std::fprintf(
            fp,
            "      {\"k_steps\": %d, \"total_launches\": %zu, "
            "\"median_launch_us\": %.2f, \"max_single_launch_us\": %.2f, "
            "\"mid_step_waitForCompletionCopy_blocks\": %zu, "
            "\"credit_backpressure_waits\": %zu, "
            "\"credit_backpressure_wait_us\": %.2f, "
            "\"peak_used_chunks\": %zu}%s\n",
            b.k_steps, b.total_launches, b.median_launch_us,
            b.max_single_launch_us, b.mid_step_blocks,
            b.credit_backpressure_waits, b.credit_backpressure_wait_us,
            b.peak_used_chunks, (i + 1 < burst_results.size()) ? "," : "");
      }
      std::fprintf(fp, "    ]\n");
      std::fprintf(fp, "  }\n");
      std::fprintf(fp, "}\n");
      std::fclose(fp);
    }
    std::printf(
        "[RT-1/2/3] Rank 0 Summary:\n"
        "  RT-1 Lowering: regions %zu -> %zu, coalesced collectives %zu/%zu -> "
        "%zu/%zu\n"
        "  RT-1+2 8-Rank Step (Rank 0): Host %.1f -> %.1f us (saved %.1f us), "
        "Early Dev %.1f -> %.1f us (saved %.1f us), Full Dev %.1f -> %.1f us "
        "(saved %.1f us)\n"
        "  RT-2 Isolated Preflight+SEM: %.2f -> %.2f us (saved %.2f us)\n"
        "  RT-3 12-Recipe Reuse Horizon: predicted=%zu, measured=%d (%.2f "
        "steps); K=4 mid-step blocks under CreditMgr=%zu\n",
        rep_legacy.regions.size(), rep_unified.regions.size(),
        rep_legacy.coalesced_collectives, rep_legacy.total_collective_subcalls,
        rep_unified.coalesced_collectives,
        rep_unified.total_collective_subcalls, shm->rt1_arm_a_host_us[0],
        shm->rt1_arm_b_host_us[0],
        shm->rt1_arm_a_host_us[0] - shm->rt1_arm_b_host_us[0],
        shm->rt1_early_a_dev_us[0], shm->rt1_early_b_dev_us[0],
        shm->rt1_early_a_dev_us[0] - shm->rt1_early_b_dev_us[0],
        shm->rt1_arm_a_dev_us[0], shm->rt1_arm_b_dev_us[0],
        shm->rt1_arm_a_dev_us[0] - shm->rt1_arm_b_dev_us[0],
        rt2_a_stat.p50, rt2_b_stat.p50,
        rt2_a_stat.p50 - rt2_b_stat.p50,
        InFlightCreditManager::max_launches_for_distinct(12, 575),
        unthrottled_first_block_idx,
        static_cast<double>(unthrottled_first_block_idx) / 145.0,
        burst_results.back().mid_step_blocks);
  }

  host_barrier(shm, nranks, local_sense);

  synEventDestroy(ev_start);
  synEventDestroy(ev_early);
  synEventDestroy(ev_end);
  for (auto ev : sync_events) synEventDestroy(ev);
  synDeviceFree(dev, send_buf, 0);
  synDeviceFree(dev, recv_buf, 0);
  for (auto& r : distinct_recipes) r.destroy();
  synStreamDestroy(compute_stream);
  synStreamDestroy(comm_stream);
  hcclCommDestroy(comm);
}

}  // namespace

int main(int argc, char** argv) {
  const char* out_json = (argc > 1) ? argv[1] : "rt-1-2-3-runtime-gaps.json";
  const int nranks = 8;

  void* ptr = mmap(nullptr, sizeof(SharedCoord), PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED) {
    std::perror("mmap");
    return 1;
  }
  auto* shm = new (ptr) SharedCoord();

  std::vector<pid_t> pids;
  for (int r = 0; r < nranks; ++r) {
    pid_t pid = fork();
    if (pid < 0) {
      std::perror("fork");
      return 1;
    }
    if (pid == 0) {
      run_rank(r, nranks, shm, out_json);
      std::_Exit(0);
    }
    pids.push_back(pid);
  }

  int status_all = 0;
  for (pid_t pid : pids) {
    int st = 0;
    waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
      status_all = 1;
    }
  }
  munmap(ptr, sizeof(SharedCoord));
  return status_all;
}
