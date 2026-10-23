#include "gaudirt/common/abba_harness.hpp"
#include "gaudirt/common/host_policy.hpp"
#include "gaudirt/common/synapse_raii.hpp"

#include <hccl.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cmath>
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

// Exact 144 compute command binding counts (reads + writes = 3068 total) from
// production70-first-principles-a NativeExecutionPlan dependency_diagnostics.
constexpr unsigned kProd144Bindings[144] = {
    29, 34, 16, 4,  26, 16, 26, 16, 26, 16, 26, 16, 26, 16, 26, 16, 36, 15,
    26, 16, 26, 16, 26, 16, 26, 16, 26, 16, 26, 16, 26, 16, 36, 15, 26, 16,
    26, 16, 26, 16, 26, 16, 26, 16, 26, 16, 26, 16, 36, 15, 26, 16, 26, 16,
    26, 16, 26, 16, 26, 16, 26, 16, 26, 16, 36, 15, 26, 16, 26, 16, 26, 16,
    26, 16, 26, 16, 26, 16, 26, 16, 36, 15, 26, 16, 26, 16, 26, 16, 26, 16,
    26, 16, 26, 16, 26, 16, 36, 15, 26, 16, 26, 16, 26, 16, 26, 16, 26, 16,
    26, 16, 26, 16, 36, 15, 26, 16, 26, 16, 26, 16, 26, 16, 26, 16, 26, 16,
    36, 15, 26, 16, 26, 16, 26, 16, 26, 16, 26, 16, 26, 16, 36, 15, 2,  4};

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
    }
  }
}

struct MultiBindingRecipe {
  synDeviceId dev = 0;
  synRecipeHandle handle = nullptr;
  std::vector<synLaunchTensorInfoExt> bindings;
  std::vector<std::string> names;
  uint64_t workspace = 0;
  uint64_t workspace_bytes = 0;

  double launch_timed(synStreamHandle stream) const {
    double t0 = now_us();
    GAUDIRT_SYN_CHECK(synLaunchWithExternalEventsExt(
        stream, bindings.data(), bindings.size(), workspace, handle, nullptr, 0, 0));
    return now_us() - t0;
  }

  void destroy() {
    for (auto& b : bindings) {
      if (b.pTensorAddress) synDeviceFree(dev, b.pTensorAddress, 0);
    }
    if (workspace) synDeviceFree(dev, workspace, 0);
    if (handle) synRecipeDestroy(handle);
  }
};

// Build a recipe with `n_bindings` persistent tensors (minimum 3) and
// enough add_fwd_f32 nodes to consume every persistent input and produce outputs.
MultiBindingRecipe make_binding_matched_recipe(synDeviceId dev, unsigned n_bindings,
                                               unsigned elems, const char* tag) {
  if (n_bindings < 3) n_bindings = 3;
  MultiBindingRecipe r;
  r.dev = dev;
  synGraphHandle graph = nullptr;
  GAUDIRT_SYN_CHECK(synGraphCreate(&graph, synDeviceGaudi2));

  std::vector<synSectionHandle> sections;
  std::vector<synTensor> tensors;
  r.names.reserve(n_bindings);
  for (unsigned i = 0; i < n_bindings; ++i) {
    r.names.push_back(std::string(tag) + "_t" + std::to_string(i));
  }

  for (unsigned i = 0; i < n_bindings; ++i) {
    synSectionHandle s = nullptr;
    GAUDIRT_SYN_CHECK(synSectionCreate(&s, 0, graph));
    GAUDIRT_SYN_CHECK(synSectionSetPersistent(s, true));
    sections.push_back(s);

    synTensorDescriptor d{};
    d.m_name = r.names[i].c_str();
    d.m_dataType = syn_type_single;
    d.m_dims = 1;
    d.m_sizes[0] = d.m_minSizes[0] = elems;
    synTensor t = nullptr;
    GAUDIRT_SYN_CHECK(synTensorCreate(&t, &d, s, 0));
    tensors.push_back(t);
  }

  // Chain additions across all n_bindings persistent tensors so every tensor
  // is a live persistent patch-point in the compiled recipe.
  // Tensors 0 .. n_bindings-2 are inputs; tensor n_bindings-1 is final output,
  // with non-persistent intermediates connecting them.
  unsigned n_inputs = n_bindings - 1;
  synTensor cur = tensors[0];
  std::vector<std::string> mid_names;
  for (unsigned i = 1; i < n_inputs; ++i) {
    mid_names.push_back(std::string(tag) + "_m" + std::to_string(i));
  }
  for (unsigned i = 1; i < n_inputs; ++i) {
    synTensor out_t = nullptr;
    if (i + 1 == n_inputs) {
      out_t = tensors[n_bindings - 1];
    } else {
      synTensorDescriptor md{};
      md.m_name = mid_names[i - 1].c_str();
      md.m_dataType = syn_type_single;
      md.m_dims = 1;
      md.m_sizes[0] = md.m_minSizes[0] = elems;
      GAUDIRT_SYN_CHECK(synTensorCreate(&out_t, &md, nullptr, 0));
      tensors.push_back(out_t);
    }
    synTensor in[2] = {cur, tensors[i]};
    GAUDIRT_SYN_CHECK(synNodeCreate(
        graph, in, &out_t, 2, 1, nullptr, 0, "add_fwd_f32",
        (std::string(tag) + "_add" + std::to_string(i)).c_str(), nullptr, nullptr));
    cur = out_t;
  }

  GAUDIRT_SYN_CHECK(synGraphCompile(&r.handle, graph, tag, nullptr));

  std::vector<const char*> cnames;
  cnames.reserve(n_bindings);
  for (const auto& nm : r.names) cnames.push_back(nm.c_str());
  std::vector<uint64_t> ids(n_bindings, 0);
  GAUDIRT_SYN_CHECK(synTensorRetrieveIds(r.handle, cnames.data(), ids.data(), n_bindings));

  r.bindings.resize(n_bindings);
  for (unsigned i = 0; i < n_bindings; ++i) {
    GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, uint64_t(elems) * sizeof(float), 0, 0,
                                      &r.bindings[i].pTensorAddress));
    r.bindings[i].tensorId = ids[i];
    r.bindings[i].tensorName = r.names[i].c_str();
    r.bindings[i].tensorType = DATA_TENSOR;
    r.bindings[i].tensorSize[0] = elems;
  }

  GAUDIRT_SYN_CHECK(synWorkspaceGetSize(&r.workspace_bytes, r.handle));
  if (r.workspace_bytes) {
    GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, r.workspace_bytes, 0, 0, &r.workspace));
  }
  GAUDIRT_SYN_CHECK(synGraphDestroy(graph));
  for (auto t : tensors) GAUDIRT_SYN_CHECK(synTensorDestroy(t));
  for (auto s : sections) GAUDIRT_SYN_CHECK(synSectionDestroy(s));
  return r;
}

// Build a heavy GEMM gate recipe (~135 ms on Gaudi2: 54 x [8192x8192] BF16 GEMM)
MultiBindingRecipe make_gate_recipe(synDeviceId dev, unsigned count, unsigned dim, const char* tag) {
  MultiBindingRecipe r;
  r.dev = dev;
  synGraphHandle graph = nullptr;
  GAUDIRT_SYN_CHECK(synGraphCreate(&graph, synDeviceGaudi2));
  r.names = {std::string(tag) + "_in", std::string(tag) + "_w", std::string(tag) + "_out"};
  std::vector<synSectionHandle> sections;
  std::vector<synTensor> tensors;

  auto mk = [&](const char* name, bool persistent) {
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
    d.m_sizes[0] = d.m_minSizes[0] = dim;
    d.m_sizes[1] = d.m_minSizes[1] = dim;
    synTensor t = nullptr;
    GAUDIRT_SYN_CHECK(synTensorCreate(&t, &d, s, 0));
    tensors.push_back(t);
    return t;
  };

  synTensor x = mk(r.names[0].c_str(), true);
  synTensor w = mk(r.names[1].c_str(), true);
  synTensor out = mk(r.names[2].c_str(), true);
  synTensor cur = x;
  std::vector<std::string> mid_names;
  for (unsigned i = 0; i < count; ++i) mid_names.push_back(std::string(tag) + "_mid" + std::to_string(i));
  for (unsigned i = 0; i < count; ++i) {
    synTensor next = (i + 1 == count) ? out : mk(mid_names[i].c_str(), false);
    synTensor in[2] = {cur, w};
    synGEMMParams p{false, false};
    GAUDIRT_SYN_CHECK(synNodeCreate(
        graph, in, &next, 2, 1, &p, sizeof(p), "gemm",
        (std::string(tag) + "_g" + std::to_string(i)).c_str(), nullptr, nullptr));
    cur = next;
  }

  GAUDIRT_SYN_CHECK(synGraphCompile(&r.handle, graph, tag, nullptr));
  const char* cn[3] = {r.names[0].c_str(), r.names[1].c_str(), r.names[2].c_str()};
  uint64_t ids[3]{};
  GAUDIRT_SYN_CHECK(synTensorRetrieveIds(r.handle, cn, ids, 3));
  r.bindings.resize(3);
  for (int i = 0; i < 3; ++i) {
    GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, uint64_t(dim) * dim * 2, 0, 0, &r.bindings[i].pTensorAddress));
    r.bindings[i].tensorId = ids[i];
    r.bindings[i].tensorName = r.names[i].c_str();
    r.bindings[i].tensorType = DATA_TENSOR;
    r.bindings[i].tensorSize[0] = dim;
    r.bindings[i].tensorSize[1] = dim;
  }
  GAUDIRT_SYN_CHECK(synWorkspaceGetSize(&r.workspace_bytes, r.handle));
  if (r.workspace_bytes) {
    GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, r.workspace_bytes, 0, 0, &r.workspace));
  }
  GAUDIRT_SYN_CHECK(synGraphDestroy(graph));
  for (auto t : tensors) GAUDIRT_SYN_CHECK(synTensorDestroy(t));
  for (auto s : sections) GAUDIRT_SYN_CHECK(synSectionDestroy(s));
  return r;
}

// Device-side feedback state vector: [token_id, position_id, kv_slot, eos_accum]
// Each step adds [token_delta=7, pos_delta=1, kv_delta=1, eos_check=1] in-place on device
// and writes a snapshot to step_history[step_idx].
struct DeviceFeedbackState {
  synDeviceId dev = 0;
  uint64_t d_state = 0;
  uint64_t d_delta = 0;
  uint64_t d_step_snapshots[4]{};
  MultiBindingRecipe step_update_recipe;

  void init(synDeviceId device, const char* tag) {
    dev = device;
    step_update_recipe = make_binding_matched_recipe(dev, 3, 4, tag);
    d_state = step_update_recipe.bindings[0].pTensorAddress;
    d_delta = step_update_recipe.bindings[1].pTensorAddress;
    for (int s = 0; s < 4; ++s) {
      GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, 4 * sizeof(float), 0, 0, &d_step_snapshots[s]));
    }
  }

  void reset_on_device(synStreamHandle stream, float init_tok, float init_pos, float init_kv) {
    float h_state[4] = {init_tok, init_pos, init_kv, 0.0f};
    float h_delta[4] = {7.0f, 1.0f, 1.0f, 1.0f};
    uint64_t host_buf = 0;
    GAUDIRT_SYN_CHECK(synHostMalloc(dev, 8 * sizeof(float), 0, reinterpret_cast<void**>(&host_buf)));
    std::memcpy(reinterpret_cast<void*>(host_buf), h_state, sizeof(h_state));
    std::memcpy(reinterpret_cast<void*>(host_buf + sizeof(h_state)), h_delta, sizeof(h_delta));
    GAUDIRT_SYN_CHECK(synMemCopyAsync(stream, host_buf, sizeof(h_state), d_state, HOST_TO_DRAM));
    GAUDIRT_SYN_CHECK(synMemCopyAsync(stream, host_buf + sizeof(h_state), sizeof(h_delta), d_delta, HOST_TO_DRAM));
    GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));
    GAUDIRT_SYN_CHECK(synHostFree(dev, reinterpret_cast<void*>(host_buf), 0));
  }

  // Launch in-place device feedback update: state -> state (and snapshot[step_idx])
  double launch_feedback_step(synStreamHandle stream, int step_idx) {
    auto b = step_update_recipe.bindings;
    // Read from d_state (or previous step snapshot if step_idx > 0), add d_delta,
    // write both to d_step_snapshots[step_idx] AND feed into next step's input!
    b[0].pTensorAddress = (step_idx == 0) ? d_state : d_step_snapshots[step_idx - 1];
    b[1].pTensorAddress = d_delta;
    b[2].pTensorAddress = d_step_snapshots[step_idx];
    double t0 = now_us();
    GAUDIRT_SYN_CHECK(synLaunchWithExternalEventsExt(
        stream, b.data(), b.size(), step_update_recipe.workspace,
        step_update_recipe.handle, nullptr, 0, 0));
    return now_us() - t0;
  }

  bool verify_snapshots(synStreamHandle stream, int n_steps, float init_tok, float init_pos, float init_kv) {
    uint64_t host_buf = 0;
    GAUDIRT_SYN_CHECK(synHostMalloc(dev, 4 * 4 * sizeof(float), 0, reinterpret_cast<void**>(&host_buf)));
    for (int s = 0; s < n_steps; ++s) {
      GAUDIRT_SYN_CHECK(synMemCopyAsync(stream, d_step_snapshots[s], 4 * sizeof(float),
                                        host_buf + s * 4 * sizeof(float), DRAM_TO_HOST));
    }
    GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));
    const float* out = reinterpret_cast<const float*>(host_buf);
    bool ok = true;
    for (int s = 0; s < n_steps; ++s) {
      float exp_tok = init_tok + 7.0f * (s + 1);
      float exp_pos = init_pos + 1.0f * (s + 1);
      float exp_kv = init_kv + 1.0f * (s + 1);
      float exp_eos = 1.0f * (s + 1);
      if (std::fabs(out[s * 4 + 0] - exp_tok) > 1e-3f ||
          std::fabs(out[s * 4 + 1] - exp_pos) > 1e-3f ||
          std::fabs(out[s * 4 + 2] - exp_kv) > 1e-3f ||
          std::fabs(out[s * 4 + 3] - exp_eos) > 1e-3f) {
        ok = false;
      }
    }
    GAUDIRT_SYN_CHECK(synHostFree(dev, reinterpret_cast<void*>(host_buf), 0));
    return ok;
  }

  void destroy() {
    for (int s = 0; s < 4; ++s) {
      if (d_step_snapshots[s]) synDeviceFree(dev, d_step_snapshots[s], 0);
    }
    step_update_recipe.destroy();
  }
};

struct PrequeueStepResult {
  int steps = 0;
  bool distinct_recipes = false;
  unsigned total_compute_launches = 0;
  unsigned total_collective_launches = 0;
  unsigned total_bindings_submitted = 0;
  double submit_wall_ms = 0.0;
  double max_compute_launch_us = 0.0;
  double max_collective_launch_us = 0.0;
  int blocked_calls_over_5ms = 0;
  int first_block_global_compute_idx = -1;
  int first_block_step = -1;
  int first_block_recipe_in_step = -1;
  bool gate_still_running_after_submit = false;
  bool feedback_state_verified = false;
};

void run_rank(int rank, int nranks, SharedCoord* shm) {
  auto host_st = lock_host_policy(rank * 4, 1024);

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

  // 1) Compile 144 DISTINCT recipes matching exact production70 binding counts (sum = 3068)
  std::vector<MultiBindingRecipe> distinct_144(144);
  for (unsigned i = 0; i < 144; ++i) {
    std::string tag = "g0_7_r" + mod_str + "_c" + std::to_string(i);
    distinct_144[i] = make_binding_matched_recipe(ctx.device(), kProd144Bindings[i], 256, tag.c_str());
  }

  // 2) Compile 1 shared recipe with median bindings (16) for comparison
  MultiBindingRecipe shared_1 = make_binding_matched_recipe(
      ctx.device(), 16, 256, ("g0_7_r" + mod_str + "_shared").c_str());

  // 3) Compile heavy gate recipe (~135 ms) + gate completion event
  MultiBindingRecipe gate = make_gate_recipe(
      ctx.device(), 54, 8192, ("g0_7_r" + mod_str + "_gate").c_str());
  synEventHandle gate_done_ev = nullptr;
  GAUDIRT_SYN_CHECK(synEventCreate(&gate_done_ev, ctx.device(), 0));

  // 4) Device-side feedback state closure recipe
  DeviceFeedbackState fb;
  fb.init(ctx.device(), ("g0_7_r" + mod_str + "_fb").c_str());

  // 5) Collective buffers (FP32 AllGather: 6144 elems * 4B -> 24 KB send, 192 KB recv)
  const size_t count_elems = 6144;
  DeviceBuffer comm_send(ctx.device(), count_elems * 4 * nranks);
  DeviceBuffer comm_recv(ctx.device(), count_elems * 4 * nranks);

  // Warmup all 144 distinct recipes + collective + gate once so any lazy one-time
  // initialization is completed before measurement.
  gate.launch_timed(stream);
  for (unsigned i = 0; i < 144; ++i) {
    distinct_144[i].launch_timed(stream);
  }
  shared_1.launch_timed(stream);
  GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(comm_send.addr()),
                                   reinterpret_cast<void*>(comm_recv.addr()),
                                   count_elems, hcclFloat32, comm, stream));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));
  host_barrier(shm, nranks, sense);

  auto run_prequeue_test = [&](int n_steps, bool use_distinct) -> PrequeueStepResult {
    PrequeueStepResult res{};
    res.steps = n_steps;
    res.distinct_recipes = use_distinct;

    fb.reset_on_device(stream, 100.0f, 4096.0f, 128.0f);
    GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));
    host_barrier(shm, nranks, sense);

    double t_start = now_us();
    // Enqueue 135ms gate on device and record gate_done_ev
    gate.launch_timed(stream);
    GAUDIRT_SYN_CHECK(synEventRecord(gate_done_ev, stream));

    unsigned global_comp_idx = 0;
    for (int s = 0; s < n_steps; ++s) {
      for (unsigned i = 0; i < 144; ++i) {
        double dt = 0.0;
        if (use_distinct) {
          if (i == 143) {
            // Last compute command in the step performs the device-side token/pos/KV/EOS feedback update
            dt = fb.launch_feedback_step(stream, s);
            res.total_bindings_submitted += 3;
          } else {
            dt = distinct_144[i].launch_timed(stream);
            res.total_bindings_submitted += distinct_144[i].bindings.size();
          }
        } else {
          if (i == 143) {
            dt = fb.launch_feedback_step(stream, s);
            res.total_bindings_submitted += 3;
          } else {
            dt = shared_1.launch_timed(stream);
            res.total_bindings_submitted += shared_1.bindings.size();
          }
        }
        if (dt > res.max_compute_launch_us) res.max_compute_launch_us = dt;
        if (dt > 5000.0) {
          res.blocked_calls_over_5ms++;
          if (res.first_block_global_compute_idx < 0) {
            res.first_block_global_compute_idx = static_cast<int>(global_comp_idx);
            res.first_block_step = s + 1;
            res.first_block_recipe_in_step = static_cast<int>(i);
          }
        }
        res.total_compute_launches++;
        global_comp_idx++;

        // Interleave 141 FP32 AllGather collectives per step
        if ((i * 141) / 144 != ((i + 1) * 141) / 144) {
          double tc0 = now_us();
          GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(comm_send.addr()),
                                           reinterpret_cast<void*>(comm_recv.addr()),
                                           count_elems, hcclFloat32, comm, stream));
          double dt_comm = now_us() - tc0;
          if (dt_comm > res.max_collective_launch_us) res.max_collective_launch_us = dt_comm;
          res.total_collective_launches++;
        }
      }
    }
    res.submit_wall_ms = (now_us() - t_start) / 1000.0;

    // Query whether the gate is STILL executing on the device right after host finished submitting!
    synStatus gate_st = synEventQuery(gate_done_ev);
    res.gate_still_running_after_submit = (gate_st == synBusy);

    GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));
    res.feedback_state_verified = fb.verify_snapshots(stream, n_steps, 100.0f, 4096.0f, 128.0f);
    host_barrier(shm, nranks, sense);
    return res;
  };

  // Run progressive K = 1, 2, 3, 4 for 144 DISTINCT recipes (3068 bindings/step)
  PrequeueStepResult distinct_res[4];
  for (int k = 1; k <= 4; ++k) {
    distinct_res[k - 1] = run_prequeue_test(k, true);
  }

  // Run K = 3, 4 for 1 shared recipe for direct A/B comparison
  PrequeueStepResult shared_res[2];
  shared_res[0] = run_prequeue_test(3, false);
  shared_res[1] = run_prequeue_test(4, false);

  // Law verification: test how first_block_global_compute_idx scales with:
  // (a) 1 shared recipe with 36 bindings (max bindings in production70)
  // (b) 72 distinct recipes (each with 16 bindings) launched across 600 calls
  // (c) 140 distinct recipes (simulating merging the 4 extra split recipes so 1 step = 140 recipes)
  MultiBindingRecipe shared_36 = make_binding_matched_recipe(
      ctx.device(), 36, 256, ("g0_7_r" + mod_str + "_sh36").c_str());
  shared_36.launch_timed(stream);
  GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));
  host_barrier(shm, nranks, sense);

  auto find_block_idx = [&](unsigned n_distinct_used, bool use_sh36) -> int {
    GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));
    host_barrier(shm, nranks, sense);
    gate.launch_timed(stream);
    int first_blk = -1;
    for (unsigned c = 0; c < 600; ++c) {
      double dt = use_sh36 ? shared_36.launch_timed(stream)
                           : distinct_144[c % n_distinct_used].launch_timed(stream);
      if (dt > 5000.0 && first_blk < 0) {
        first_blk = static_cast<int>(c);
      }
    }
    GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));
    host_barrier(shm, nranks, sense);
    return first_blk;
  };
  int blk_sh36 = find_block_idx(1, true);
  int blk_dist36 = find_block_idx(36, false);
  int blk_dist72 = find_block_idx(72, false);
  int blk_dist140 = find_block_idx(140, false);
  int blk_dist144 = find_block_idx(144, false);
  shared_36.destroy();

  // Also measure same-boundary [first_op_start, last_op_end] 1-step span for the 144 distinct recipes
  // (3068 bindings + 141 FP32 AllGathers): Normal vs Gate-Prequeued (ABBA)
  TimedEventPair ev_span(ctx.device());
  auto measure_distinct144_span = [&](bool use_gate, std::vector<double>& sp_out, std::vector<double>& sub_out) {
    for (int r = 0; r < 12; ++r) {
      host_barrier(shm, nranks, sense);
      double t0 = now_us();
      if (use_gate) gate.launch_timed(stream);
      ev_span.record_start(stream);
      for (unsigned i = 0; i < 144; ++i) {
        distinct_144[i].launch_timed(stream);
        if ((i * 141) / 144 != ((i + 1) * 141) / 144) {
          GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(comm_send.addr()),
                                           reinterpret_cast<void*>(comm_recv.addr()),
                                           count_elems, hcclFloat32, comm, stream));
        }
      }
      ev_span.record_end(stream);
      double sub = now_us() - t0;
      double sp = ev_span.elapsed_us();
      if (r >= 4) {
        sp_out.push_back(sp);
        sub_out.push_back(sub);
      }
    }
  };
  std::vector<double> d144_norm_sp, d144_norm_sub, d144_preq_sp, d144_preq_sub;
  measure_distinct144_span(false, d144_norm_sp, d144_norm_sub);
  measure_distinct144_span(true, d144_preq_sp, d144_preq_sub);
  measure_distinct144_span(true, d144_preq_sp, d144_preq_sub);
  measure_distinct144_span(false, d144_norm_sp, d144_norm_sub);

  if (rank == 0) {
    auto print_res = [](const PrequeueStepResult& r, bool comma) {
      std::printf("    {\n");
      std::printf("      \"steps\": %d,\n", r.steps);
      std::printf("      \"distinct_144_recipes\": %s,\n", r.distinct_recipes ? "true" : "false");
      std::printf("      \"total_compute_launches\": %u,\n", r.total_compute_launches);
      std::printf("      \"total_collective_launches\": %u,\n", r.total_collective_launches);
      std::printf("      \"total_bindings_submitted\": %u,\n", r.total_bindings_submitted);
      std::printf("      \"submit_wall_ms\": %.3f,\n", r.submit_wall_ms);
      std::printf("      \"max_compute_launch_us\": %.2f,\n", r.max_compute_launch_us);
      std::printf("      \"max_collective_launch_us\": %.2f,\n", r.max_collective_launch_us);
      std::printf("      \"blocked_calls_over_5ms\": %d,\n", r.blocked_calls_over_5ms);
      std::printf("      \"first_block_global_compute_idx\": %d,\n", r.first_block_global_compute_idx);
      std::printf("      \"first_block_step\": %d,\n", r.first_block_step);
      std::printf("      \"first_block_recipe_in_step\": %d,\n", r.first_block_recipe_in_step);
      std::printf("      \"gate_still_running_after_submit\": %s,\n", r.gate_still_running_after_submit ? "true" : "false");
      std::printf("      \"feedback_state_verified\": %s\n", r.feedback_state_verified ? "true" : "false");
      std::printf("    }%s\n", comma ? "," : "");
    };

    auto dns = summarize(d144_norm_sp);
    auto dnsub = summarize(d144_norm_sub);
    auto dps = summarize(d144_preq_sp);

    std::printf("{\n");
    std::printf("  \"probe\": \"G0-7_progressive_prequeue_144_distinct_recipes_3068_bindings\",\n");
    std::printf("  \"nranks\": %d,\n", nranks);
    std::printf("  \"uclamp_applied\": %s,\n", host_st.uclamp_applied ? "true" : "false");
    std::printf("  \"distinct_144_progressive_1_to_4_steps\": [\n");
    for (int i = 0; i < 4; ++i) print_res(distinct_res[i], i < 3);
    std::printf("  ],\n");
    std::printf("  \"shared_recipe_comparison_3_to_4_steps\": [\n");
    print_res(shared_res[0], true);
    print_res(shared_res[1], false);
    std::printf("  ],\n");
    std::printf("  \"distinct_144_3068_bindings_same_boundary_1step\": {\n");
    std::printf("    \"normal_device_span_p50_us\": %.2f,\n", dns.p50);
    std::printf("    \"prequeued_device_span_p50_us\": %.2f,\n", dps.p50);
    std::printf("    \"same_boundary_span_diff_us\": %.2f,\n", dns.p50 - dps.p50);
    std::printf("    \"normal_host_submit_p50_us\": %.2f\n", dnsub.p50);
    std::printf("  },\n");
    std::printf("  \"mapped_mem_mgr_distinct_recipe_law\": {\n");
    std::printf("    \"n_distinct_1_36bindings_first_block_idx\": %d,\n", blk_sh36);
    std::printf("    \"n_distinct_36_first_block_idx\": %d,\n", blk_dist36);
    std::printf("    \"n_distinct_72_first_block_idx\": %d,\n", blk_dist72);
    std::printf("    \"n_distinct_140_first_block_idx\": %d,\n", blk_dist140);
    std::printf("    \"n_distinct_144_first_block_idx\": %d\n", blk_dist144);
    std::printf("  }\n");
    std::printf("}\n");
    std::fflush(stdout);
  }

  host_barrier(shm, nranks, sense);
  synEventDestroy(gate_done_ev);
  fb.destroy();
  gate.destroy();
  shared_1.destroy();
  for (auto& r : distinct_144) r.destroy();
  GAUDIRT_HCCL_CHECK(hcclCommDestroy(comm));
  synStreamDestroy(stream);
}

} // namespace

int main(int argc, char** argv) {
  const int nranks = (argc > 1) ? std::atoi(argv[1]) : 8;
  if (has_unwanted_instrumentation()) {
    std::fprintf(stderr, "[G0-7] ERROR: Unwanted instrumentation/profiler detected!\n");
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
