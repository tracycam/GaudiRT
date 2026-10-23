#include "gaudirt/common/abba_harness.hpp"
#include "gaudirt/common/host_policy.hpp"
#include "gaudirt/common/synapse_raii.hpp"

#include <hccl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

namespace {

struct SharedCoord {
  hcclUniqueId uid;
  std::atomic<int> uid_ready{0};
  std::atomic<int> barrier_sense{0};
  std::atomic<int> barrier_count{0};
};

void host_barrier(SharedCoord* shm, int nranks, int& local_sense) {
  local_sense = 1 - local_sense;
  if (shm->barrier_count.fetch_add(1) == nranks - 1) {
    shm->barrier_count.store(0);
    shm->barrier_sense.store(local_sense);
  } else {
    while (shm->barrier_sense.load() != local_sense) {
      // spin
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
        stream, bindings.data(), bindings.size(), workspace, handle, nullptr, 0, 0));
  }

  void destroy() {
    for (auto& b : bindings) {
      if (b.pTensorAddress) synDeviceFree(dev, b.pTensorAddress, 0);
    }
    if (workspace) synDeviceFree(dev, workspace, 0);
    if (handle) synRecipeDestroy(handle);
  }
};

// Build a BF16 GEMM recipe [M x K] * [K x N] -> [M x N]
SimpleRecipe make_gemm_recipe(synDeviceId dev, unsigned m, unsigned k, unsigned n, unsigned chain_len, const char* tag) {
  SimpleRecipe r;
  r.dev = dev;
  synGraphHandle graph = nullptr;
  GAUDIRT_SYN_CHECK(synGraphCreate(&graph, synDeviceGaudi2));
  r.names = {"in", "w", "out"};
  std::vector<synSectionHandle> sections;

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
    return t;
  };

  synTensor x = mk("in", k, m, true);
  synTensor w = mk("w", n, k, true);
  synTensor out = mk("out", n, m, true);
  synTensor cur = x;
  std::vector<std::string> mid_names;
  for (unsigned i = 0; i < chain_len; ++i) mid_names.push_back("mid_" + std::to_string(i));
  for (unsigned i = 0; i < chain_len; ++i) {
    synTensor next = (i + 1 == chain_len) ? out : mk(mid_names[i].c_str(), n, m, false);
    synTensor in[2] = {cur, w};
    synGEMMParams p{false, false};
    GAUDIRT_SYN_CHECK(synNodeCreate(graph, in, &next, 2, 1, &p, sizeof(p), "gemm",
                                    (std::string(tag) + "_g" + std::to_string(i)).c_str(), nullptr, nullptr));
    cur = next;
  }

  GAUDIRT_SYN_CHECK(synGraphCompile(&r.handle, graph, tag, nullptr));
  const char* cn[3] = {"in", "w", "out"};
  uint64_t ids[3]{};
  GAUDIRT_SYN_CHECK(synTensorRetrieveIds(r.handle, cn, ids, 3));
  const unsigned dims[3][2] = {{k, m}, {n, k}, {n, m}};
  r.bindings.resize(3);
  for (int i = 0; i < 3; ++i) {
    GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, uint64_t(dims[i][0]) * dims[i][1] * 2, 0, 0, &r.bindings[i].pTensorAddress));
    r.bindings[i].tensorId = ids[i];
    r.bindings[i].tensorName = r.names[i].c_str();
    r.bindings[i].tensorType = DATA_TENSOR;
    r.bindings[i].tensorSize[0] = dims[i][0];
    r.bindings[i].tensorSize[1] = dims[i][1];
  }
  GAUDIRT_SYN_CHECK(synWorkspaceGetSize(&r.workspace_bytes, r.handle));
  if (r.workspace_bytes) {
    GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, r.workspace_bytes, 0, 0, &r.workspace));
  }
  GAUDIRT_SYN_CHECK(synGraphDestroy(graph));
  for (auto s : sections) GAUDIRT_SYN_CHECK(synSectionDestroy(s));
  return r;
}

void run_rank(int rank, int nranks, SharedCoord* shm) {
  int target_cpu = rank * 4;
  auto host_st = lock_host_policy(target_cpu, 1024);

  std::string mod_str = std::to_string(rank);
  setenv("ID", mod_str.c_str(), 1);
  setenv("HLS_MODULE_ID", mod_str.c_str(), 1);

  SynapseContext ctx(rank);
  synStreamHandle stream = nullptr;
  GAUDIRT_SYN_CHECK(synStreamCreateGeneric(&stream, ctx.device(), 0));

  int sense = 0;
  if (rank == 0) {
    GAUDIRT_HCCL_CHECK(hcclGetUniqueId(&shm->uid));
    shm->uid_ready.store(1);
  } else {
    while (shm->uid_ready.load() == 0) {
    }
  }
  host_barrier(shm, nranks, sense);

  hcclComm_t comm = nullptr;
  GAUDIRT_HCCL_CHECK(hcclCommInitRank(&comm, nranks, shm->uid, rank));
  host_barrier(shm, nranks, sense);

  // Build:
  // 1) A step compute recipe [M=16, K=2048, N=2048] representing a single decode compute block
  // 2) A 120ms gate recipe [48 x 8192x8192] for pre-queued same-boundary comparison
  SimpleRecipe step_comp = make_gemm_recipe(ctx.device(), 16, 2048, 2048, 1, "g0_5_step_comp");
  SimpleRecipe gate = make_gemm_recipe(ctx.device(), 8192, 8192, 8192, 48, "g0_5_gate");

  // Allocate collective buffers: 6144 BF16 elements = 12 KB (matching production hidden_size=6144 decode token)
  const size_t count_bf16 = 6144;
  DeviceBuffer comm_send(ctx.device(), count_bf16 * 2 * nranks);
  DeviceBuffer comm_recv(ctx.device(), count_bf16 * 2 * nranks);

  // Warmup
  for (int i = 0; i < 16; ++i) {
    step_comp.launch(stream);
    GAUDIRT_HCCL_CHECK(hcclAllReduce(reinterpret_cast<const void*>(comm_send.addr()),
                                     reinterpret_cast<void*>(comm_recv.addr()),
                                     count_bf16, hcclBfloat16, hcclSum, comm, stream));
  }
  GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));
  host_barrier(shm, nranks, sense);

  TimedEventPair ev_span(ctx.device());

  auto run_interleaved = [&](unsigned n_compute, unsigned n_comm, bool use_gate,
                             std::vector<double>& span_us_out,
                             std::vector<double>& submit_us_out,
                             std::vector<double>& wall_us_out) {
    const int reps = 16;
    for (int r = 0; r < reps; ++r) {
      host_barrier(shm, nranks, sense);
      double t0 = now_us();
      if (use_gate) {
        gate.launch(stream);
      }
      // Same start/end boundary around the [first_op_start, last_op_end] of the step
      ev_span.record_start(stream);
      for (unsigned i = 0; i < n_compute; ++i) {
        step_comp.launch(stream);
        if (n_comm > 0 && (i * n_comm) / n_compute != ((i + 1) * n_comm) / n_compute) {
          GAUDIRT_HCCL_CHECK(hcclAllReduce(reinterpret_cast<const void*>(comm_send.addr()),
                                           reinterpret_cast<void*>(comm_recv.addr()),
                                           count_bf16, hcclBfloat16, hcclSum, comm, stream));
        }
      }
      ev_span.record_end(stream);
      double t_sub = now_us() - t0;
      double dev_span = ev_span.elapsed_us();
      double t_wall = now_us() - t0;
      if (r >= 4) { // skip 4 warmup reps
        span_us_out.push_back(dev_span);
        submit_us_out.push_back(t_sub);
        wall_us_out.push_back(t_wall);
      }
    }
  };

  // Experiment 1: Same-boundary comparison of Normal Submit vs Gate-Prequeued on (144 compute + 141 collective)
  // Run in ABBA order
  std::vector<double> norm_span, norm_sub, norm_wall;
  std::vector<double> preq_span, preq_sub, preq_wall;
  run_interleaved(144, 141, false, norm_span, norm_sub, norm_wall);
  run_interleaved(144, 141, true, preq_span, preq_sub, preq_wall);
  run_interleaved(144, 141, true, preq_span, preq_sub, preq_wall);
  run_interleaved(144, 141, false, norm_span, norm_sub, norm_wall);

  // Experiment 2: Marginal collective scaling on fixed 144 compute recipes with N_comm = 0, 35, 70, 141
  const unsigned comm_counts[] = {0, 35, 70, 141};
  SummaryStats scaling_span[4]{};
  SummaryStats scaling_sub[4]{};
  for (int idx = 0; idx < 4; ++idx) {
    std::vector<double> sp, sb, wl;
    run_interleaved(144, comm_counts[idx], false, sp, sb, wl);
    scaling_span[idx] = summarize(sp);
    scaling_sub[idx] = summarize(sb);
  }

  // Experiment 3: Pure back-to-back 141 x 12KB hcclAllReduce (no interleaved compute)
  std::vector<double> pure_comm_span, pure_comm_sub;
  for (int r = 0; r < 20; ++r) {
    host_barrier(shm, nranks, sense);
    double t0 = now_us();
    ev_span.record_start(stream);
    for (unsigned c = 0; c < 141; ++c) {
      GAUDIRT_HCCL_CHECK(hcclAllReduce(reinterpret_cast<const void*>(comm_send.addr()),
                                       reinterpret_cast<void*>(comm_recv.addr()),
                                       count_bf16, hcclBfloat16, hcclSum, comm, stream));
    }
    ev_span.record_end(stream);
    double sub = now_us() - t0;
    double sp = ev_span.elapsed_us();
    if (r >= 4) {
      pure_comm_span.push_back(sp);
      pure_comm_sub.push_back(sub);
    }
  }

  if (rank == 0) {
    auto ns = summarize(norm_span);
    auto nsub = summarize(norm_sub);
    auto nw = summarize(norm_wall);
    auto ps = summarize(preq_span);
    auto psub = summarize(preq_sub);
    auto pcs = summarize(pure_comm_span);
    auto pcsub = summarize(pure_comm_sub);

    std::printf("{\n");
    std::printf("  \"probe\": \"G0-5_collective_and_same_boundary_prequeue\",\n");
    std::printf("  \"nranks\": %d,\n", nranks);
    std::printf("  \"uclamp_applied\": %s,\n", host_st.uclamp_applied ? "true" : "false");
    std::printf("  \"payload_bytes_bf16\": %zu,\n", count_bf16 * 2);
    std::printf("  \"same_boundary_144comp_141comm\": {\n");
    std::printf("    \"normal_device_span_p50_us\": %.2f,\n", ns.p50);
    std::printf("    \"prequeued_device_span_p50_us\": %.2f,\n", ps.p50);
    std::printf("    \"same_boundary_span_diff_us\": %.2f,\n", ns.p50 - ps.p50);
    std::printf("    \"normal_host_submit_p50_us\": %.2f,\n", nsub.p50);
    std::printf("    \"normal_wall_p50_us\": %.2f,\n", nw.p50);
    std::printf("    \"prequeued_host_submit_with_gate_p50_us\": %.2f\n", psub.p50);
    std::printf("  },\n");
    std::printf("  \"interleaved_144comp_comm_scaling\": [\n");
    for (int i = 0; i < 4; ++i) {
      double delta_vs_0 = scaling_span[i].p50 - scaling_span[0].p50;
      double per_comm = (comm_counts[i] > 0) ? (delta_vs_0 / comm_counts[i]) : 0.0;
      std::printf("    {\"n_compute\": 144, \"n_comm\": %u, \"device_span_p50_us\": %.2f, \"host_submit_p50_us\": %.2f, \"marginal_vs_0comm_us\": %.2f, \"marginal_per_comm_us\": %.2f}%s\n",
                  comm_counts[i], scaling_span[i].p50, scaling_sub[i].p50, delta_vs_0, per_comm,
                  (i < 3) ? "," : "");
    }
    std::printf("  ],\n");
    std::printf("  \"pure_141_allreduce_12kb\": {\n");
    std::printf("    \"device_span_p50_us\": %.2f,\n", pcs.p50);
    std::printf("    \"per_allreduce_device_p50_us\": %.2f,\n", pcs.p50 / 141.0);
    std::printf("    \"host_submit_p50_us\": %.2f,\n", pcsub.p50);
    std::printf("    \"per_allreduce_host_submit_p50_us\": %.2f\n", pcsub.p50 / 141.0);
    std::printf("  }\n");
    std::printf("}\n");
    std::fflush(stdout);
  }

  host_barrier(shm, nranks, sense);
  GAUDIRT_HCCL_CHECK(hcclCommDestroy(comm));
  step_comp.destroy();
  gate.destroy();
  synStreamDestroy(stream);
}

} // namespace

int main(int argc, char** argv) {
  const int nranks = (argc > 1) ? std::atoi(argv[1]) : 8;
  if (has_unwanted_instrumentation()) {
    std::fprintf(stderr, "[G0-5] ERROR: Unwanted instrumentation/profiler detected in environment!\n");
    return 1;
  }

  void* mem = mmap(nullptr, sizeof(SharedCoord), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  auto* shm = new (mem) SharedCoord();

  std::vector<pid_t> pids;
  for (int r = 0; r < nranks; ++r) {
    pid_t pid = fork();
    if (pid == 0) {
      run_rank(r, nranks, shm);
      std::_Exit(0);
    }
    pids.push_back(pid);
  }

  int status = 0;
  bool ok = true;
  for (pid_t pid : pids) {
    waitpid(pid, &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      ok = false;
    }
  }
  munmap(mem, sizeof(SharedCoord));
  return ok ? 0 : 1;
}
