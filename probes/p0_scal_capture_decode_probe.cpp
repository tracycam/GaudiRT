// SPDX-License-Identifier: Apache-2.0
// P0 Hardware Protocol Capture & Decode Probe
// Mode 1 (--mode recipe): Acquires Module 7, compiles a minimal add_fwd_f32
//   recipe, serializes it to disk, and executes 3 launches:
//     - RECIPE_LAUNCH_1_COLD
//     - RECIPE_LAUNCH_2_WARM_SAME_ADDR
//     - RECIPE_LAUNCH_3_WARM_MUTATED_ADDR (output C shifted by +256B)
//   Then dumps automatic CCB & PDMA byte diffs between launches.
// Mode 2 (--mode allgather): Forks 8 ranks across Modules 0..7, enables
//   capture only on Rank 7 (Module 7), and executes 3 hcclAllGather calls:
//     - HCCL_ALLGATHER_1_COLD
//     - HCCL_ALLGATHER_2_WARM_SAME_ADDR
//     - HCCL_ALLGATHER_3_WARM_MUTATED_ADDR (recv buffer shifted by +4096B)
//   Then dumps automatic CCB diffs between calls.
// Mode 3 (--mode p3a-sync): Captures three official eight-rank
//   add -> AllGather -> add sequences. Each sequence has one trailing host
//   synchronization, after all three device stages have been enqueued.

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

extern "C" {
void p0_hook_set_enabled(int en);
void p0_hook_set_phase(const char* phase_name);
void p0_hook_dump_phase_diff(const char* phase_a, const char* phase_b);
}

#define GAUDIRT_HCCL_CHECK(expr)                                               \
  do {                                                                         \
    hcclResult_t _r = (expr);                                                  \
    if (_r != hcclSuccess) {                                                   \
      std::fprintf(stderr, "[P0_PROBE] HCCL error %d (%s) at %s:%d\n",         \
                   static_cast<int>(_r), hcclGetErrorString(_r), __FILE__,     \
                   __LINE__);                                                  \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

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
    }
  }
}

int run_recipe_capture() {
  p0_hook_set_enabled(1);
  p0_hook_set_phase("INIT");

  GAUDIRT_SYN_CHECK(synInitialize());
  synDeviceId dev = 0;
  GAUDIRT_SYN_CHECK(synDeviceAcquireByModuleId(&dev, 7));

  synStreamHandle stream = nullptr;
  GAUDIRT_SYN_CHECK(synStreamCreateGeneric(&stream, dev, 0));

  // Build minimal 1D add_fwd_f32 recipe: C[1024] = A[1024] + B[1024]
  constexpr unsigned kElems = 1024;
  constexpr uint64_t kBytes = kElems * sizeof(float);

  synGraphHandle graph = nullptr;
  GAUDIRT_SYN_CHECK(synGraphCreate(&graph, synDeviceGaudi2));

  const char* tnames[3] = {"tensor_a", "tensor_b", "tensor_c"};
  synSectionHandle sections[3]{};
  synTensor tensors[3]{};
  for (int i = 0; i < 3; ++i) {
    GAUDIRT_SYN_CHECK(synSectionCreate(&sections[i], 0, graph));
    GAUDIRT_SYN_CHECK(synSectionSetPersistent(sections[i], true));
    synTensorDescriptor desc{};
    desc.m_name = tnames[i];
    desc.m_dataType = syn_type_single;
    desc.m_dims = 1;
    desc.m_sizes[0] = kElems;
    desc.m_minSizes[0] = kElems;
    GAUDIRT_SYN_CHECK(synTensorCreate(&tensors[i], &desc, sections[i], 0));
  }

  synTensor inputs[2] = {tensors[0], tensors[1]};
  synTensor outputs[1] = {tensors[2]};
  GAUDIRT_SYN_CHECK(synNodeCreate(graph, inputs, outputs, 2, 1, nullptr, 0,
                                  "add_fwd_f32", "p0_add_f32_node", nullptr,
                                  nullptr));

  synRecipeHandle recipe = nullptr;
  GAUDIRT_SYN_CHECK(synGraphCompile(&recipe, graph, "p0_min_add_f32", nullptr));
  GAUDIRT_SYN_CHECK(synRecipeSerialize(recipe, "/tmp/p0_min_add_f32.recipe"));

  uint64_t tensor_ids[3]{};
  GAUDIRT_SYN_CHECK(synTensorRetrieveIds(recipe, tnames, tensor_ids, 3));

  uint64_t workspace_size = 0;
  GAUDIRT_SYN_CHECK(synWorkspaceGetSize(&workspace_size, recipe));
  uint64_t workspace_addr = 0;
  if (workspace_size > 0) {
    GAUDIRT_SYN_CHECK(
        synDeviceMalloc(dev, workspace_size, 0, 0, &workspace_addr));
  }

  uint64_t a_dev = 0, b_dev = 0, c_base_dev = 0;
  GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, kBytes, 0, 0, &a_dev));
  GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, kBytes, 0, 0, &b_dev));
  GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, kBytes + 512, 0, 0, &c_base_dev));
  uint64_t c1_dev = c_base_dev;
  uint64_t c2_dev = c_base_dev + 256;

  std::fprintf(
      stderr,
      "[P0_RECIPE_SETUP] Module 7 HBM Tensor Addresses:\n"
      "  A_dev  = 0x%llx (tensorId=%llu)\n"
      "  B_dev  = 0x%llx (tensorId=%llu)\n"
      "  C1_dev = 0x%llx (tensorId=%llu)\n"
      "  C2_dev = 0x%llx (mutated +0x100)\n"
      "  workspace = 0x%llx (size=%llu)\n",
      static_cast<unsigned long long>(a_dev),
      static_cast<unsigned long long>(tensor_ids[0]),
      static_cast<unsigned long long>(b_dev),
      static_cast<unsigned long long>(tensor_ids[1]),
      static_cast<unsigned long long>(c1_dev),
      static_cast<unsigned long long>(tensor_ids[2]),
      static_cast<unsigned long long>(c2_dev),
      static_cast<unsigned long long>(workspace_addr),
      static_cast<unsigned long long>(workspace_size));

  std::vector<float> h_a(kElems, 10.0f);
  std::vector<float> h_b(kElems, 32.0f);
  std::vector<float> h_c1(kElems, 0.0f);
  std::vector<float> h_c2(kElems, 0.0f);

  uint64_t h_a_ptr = reinterpret_cast<uint64_t>(h_a.data());
  uint64_t h_b_ptr = reinterpret_cast<uint64_t>(h_b.data());
  uint64_t h_c1_ptr = reinterpret_cast<uint64_t>(h_c1.data());
  uint64_t h_c2_ptr = reinterpret_cast<uint64_t>(h_c2.data());
  GAUDIRT_SYN_CHECK(synHostMap(dev, kBytes, reinterpret_cast<void*>(h_a_ptr)));
  GAUDIRT_SYN_CHECK(synHostMap(dev, kBytes, reinterpret_cast<void*>(h_b_ptr)));
  GAUDIRT_SYN_CHECK(synHostMap(dev, kBytes, reinterpret_cast<void*>(h_c1_ptr)));
  GAUDIRT_SYN_CHECK(synHostMap(dev, kBytes, reinterpret_cast<void*>(h_c2_ptr)));

  GAUDIRT_SYN_CHECK(
      synMemCopyAsync(stream, h_a_ptr, kBytes, a_dev, HOST_TO_DRAM));
  GAUDIRT_SYN_CHECK(
      synMemCopyAsync(stream, h_b_ptr, kBytes, b_dev, HOST_TO_DRAM));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));

  synLaunchTensorInfoExt bindings[3]{};
  uint64_t addrs[3] = {a_dev, b_dev, c1_dev};
  for (int i = 0; i < 3; ++i) {
    bindings[i].tensorName = tnames[i];
    bindings[i].pTensorAddress = addrs[i];
    bindings[i].tensorType = DATA_TENSOR;
    bindings[i].tensorSize[0] = kElems;
    bindings[i].tensorId = tensor_ids[i];
  }

  // Launch 1: Cold
  p0_hook_set_phase("RECIPE_LAUNCH_1_COLD");
  GAUDIRT_SYN_CHECK(synLaunchWithExternalEventsExt(
      stream, bindings, 3, workspace_addr, recipe, nullptr, 0, 0));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));

  // Launch 2: Warm, identical addresses
  p0_hook_set_phase("RECIPE_LAUNCH_2_WARM_SAME_ADDR");
  GAUDIRT_SYN_CHECK(synLaunchWithExternalEventsExt(
      stream, bindings, 3, workspace_addr, recipe, nullptr, 0, 0));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));

  // Launch 3: Warm, mutated output tensor C address (+256B)
  bindings[2].pTensorAddress = c2_dev;
  p0_hook_set_phase("RECIPE_LAUNCH_3_WARM_MUTATED_ADDR");
  GAUDIRT_SYN_CHECK(synLaunchWithExternalEventsExt(
      stream, bindings, 3, workspace_addr, recipe, nullptr, 0, 0));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));

  p0_hook_set_phase("CLEANUP");
  p0_hook_dump_phase_diff("RECIPE_LAUNCH_1_COLD",
                          "RECIPE_LAUNCH_2_WARM_SAME_ADDR");
  p0_hook_dump_phase_diff("RECIPE_LAUNCH_2_WARM_SAME_ADDR",
                          "RECIPE_LAUNCH_3_WARM_MUTATED_ADDR");

  GAUDIRT_SYN_CHECK(
      synMemCopyAsync(stream, c1_dev, kBytes, h_c1_ptr, DRAM_TO_HOST));
  GAUDIRT_SYN_CHECK(
      synMemCopyAsync(stream, c2_dev, kBytes, h_c2_ptr, DRAM_TO_HOST));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));

  if (std::fabs(h_c1[0] - 42.0f) > 1e-5f ||
      std::fabs(h_c2[0] - 42.0f) > 1e-5f) {
    std::fprintf(stderr, "[P0_RECIPE] Verification FAILED: c1=%f c2=%f\n",
                 h_c1[0], h_c2[0]);
    return 1;
  }
  std::fprintf(stderr,
               "[P0_RECIPE] Verification PASSED: C1[0]=%.1f, C2[0]=%.1f\n",
               h_c1[0], h_c2[0]);

  GAUDIRT_SYN_CHECK(synHostUnmap(dev, reinterpret_cast<void*>(h_a_ptr)));
  GAUDIRT_SYN_CHECK(synHostUnmap(dev, reinterpret_cast<void*>(h_b_ptr)));
  GAUDIRT_SYN_CHECK(synHostUnmap(dev, reinterpret_cast<void*>(h_c1_ptr)));
  GAUDIRT_SYN_CHECK(synHostUnmap(dev, reinterpret_cast<void*>(h_c2_ptr)));
  GAUDIRT_SYN_CHECK(synDeviceFree(dev, a_dev, 0));
  GAUDIRT_SYN_CHECK(synDeviceFree(dev, b_dev, 0));
  GAUDIRT_SYN_CHECK(synDeviceFree(dev, c_base_dev, 0));
  if (workspace_addr) GAUDIRT_SYN_CHECK(synDeviceFree(dev, workspace_addr, 0));
  GAUDIRT_SYN_CHECK(synRecipeDestroy(recipe));
  GAUDIRT_SYN_CHECK(synGraphDestroy(graph));
  for (int i = 0; i < 3; ++i) {
    GAUDIRT_SYN_CHECK(synTensorDestroy(tensors[i]));
    GAUDIRT_SYN_CHECK(synSectionDestroy(sections[i]));
  }
  GAUDIRT_SYN_CHECK(synStreamDestroy(stream));
  GAUDIRT_SYN_CHECK(synDeviceRelease(dev));
  GAUDIRT_SYN_CHECK(synDestroy());
  return 0;
}

int run_allgather_rank(int rank, int nranks, SharedCoord* shm) {
  // Only enable SCAL capture logging on Rank 7 (Module 7)
  const bool is_capture_rank = (rank == 7);
  p0_hook_set_enabled(is_capture_rank ? 1 : 0);
  if (is_capture_rank) {
    p0_hook_set_phase("INIT");
  }

  int local_sense = 0;
  GAUDIRT_SYN_CHECK(synInitialize());
  synDeviceId dev = 0;
  GAUDIRT_SYN_CHECK(
      synDeviceAcquireByModuleId(&dev, static_cast<synModuleId>(rank)));

  synStreamHandle coll_stream = nullptr;
  GAUDIRT_SYN_CHECK(synStreamCreateGeneric(&coll_stream, dev, 0));

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

  constexpr size_t kCountPerRank = 1024;  // 4096B per rank, 32768B total
  constexpr uint64_t kSendBytes = kCountPerRank * sizeof(float);
  constexpr uint64_t kRecvBytes = kSendBytes * 8;

  uint64_t send_dev = 0, recv_base_dev = 0;
  GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, kSendBytes, 0, 0, &send_dev));
  GAUDIRT_SYN_CHECK(
      synDeviceMalloc(dev, kRecvBytes + 4096, 0, 0, &recv_base_dev));
  uint64_t recv1_dev = recv_base_dev;
  uint64_t recv2_dev = recv_base_dev + 4096;

  if (is_capture_rank) {
    std::fprintf(stderr,
                 "[P0_ALLGATHER_SETUP] Rank 7 (Module 7) HBM Buffers:\n"
                 "  send_dev  = 0x%llx (%llu B)\n"
                 "  recv1_dev = 0x%llx (%llu B)\n"
                 "  recv2_dev = 0x%llx (mutated +0x1000)\n",
                 static_cast<unsigned long long>(send_dev),
                 static_cast<unsigned long long>(kSendBytes),
                 static_cast<unsigned long long>(recv1_dev),
                 static_cast<unsigned long long>(kRecvBytes),
                 static_cast<unsigned long long>(recv2_dev));
  }

  std::vector<float> h_send(kCountPerRank, static_cast<float>(rank + 1));
  std::vector<float> h_recv(kCountPerRank * 8, 0.0f);
  uint64_t h_send_ptr = reinterpret_cast<uint64_t>(h_send.data());
  uint64_t h_recv_ptr = reinterpret_cast<uint64_t>(h_recv.data());
  GAUDIRT_SYN_CHECK(
      synHostMap(dev, kSendBytes, reinterpret_cast<void*>(h_send_ptr)));
  GAUDIRT_SYN_CHECK(
      synHostMap(dev, kRecvBytes, reinterpret_cast<void*>(h_recv_ptr)));
  GAUDIRT_SYN_CHECK(
      synMemCopyAsync(coll_stream, h_send_ptr, kSendBytes, send_dev,
                      HOST_TO_DRAM));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(coll_stream));
  host_barrier(shm, nranks, local_sense);

  // AllGather 1: Cold
  if (is_capture_rank) p0_hook_set_phase("HCCL_ALLGATHER_1_COLD");
  GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_dev),
                                   reinterpret_cast<void*>(recv1_dev),
                                   kCountPerRank, hcclFloat32, comm,
                                   coll_stream));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(coll_stream));
  host_barrier(shm, nranks, local_sense);

  // AllGather 2: Warm, identical addresses
  if (is_capture_rank) p0_hook_set_phase("HCCL_ALLGATHER_2_WARM_SAME_ADDR");
  GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_dev),
                                   reinterpret_cast<void*>(recv1_dev),
                                   kCountPerRank, hcclFloat32, comm,
                                   coll_stream));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(coll_stream));
  host_barrier(shm, nranks, local_sense);

  // AllGather 3: Warm, mutated recv address (+4096B)
  if (is_capture_rank) p0_hook_set_phase("HCCL_ALLGATHER_3_WARM_MUTATED_ADDR");
  GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send_dev),
                                   reinterpret_cast<void*>(recv2_dev),
                                   kCountPerRank, hcclFloat32, comm,
                                   coll_stream));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(coll_stream));
  host_barrier(shm, nranks, local_sense);

  if (is_capture_rank) {
    p0_hook_set_phase("CLEANUP");
    p0_hook_dump_phase_diff("HCCL_ALLGATHER_1_COLD",
                            "HCCL_ALLGATHER_2_WARM_SAME_ADDR");
    p0_hook_dump_phase_diff("HCCL_ALLGATHER_2_WARM_SAME_ADDR",
                            "HCCL_ALLGATHER_3_WARM_MUTATED_ADDR");
  }

  GAUDIRT_SYN_CHECK(synMemCopyAsync(coll_stream, recv2_dev, kRecvBytes,
                                    h_recv_ptr, DRAM_TO_HOST));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(coll_stream));

  for (int r = 0; r < nranks; ++r) {
    float expected = static_cast<float>(r + 1);
    float actual = h_recv[r * kCountPerRank];
    if (std::fabs(actual - expected) > 1e-5f) {
      std::fprintf(
          stderr,
          "[P0_ALLGATHER] Rank %d verification FAILED at chunk %d: got %f\n",
          rank, r, actual);
      return 1;
    }
  }
  if (is_capture_rank) {
    std::fprintf(stderr,
                 "[P0_ALLGATHER] Rank 7 verification PASSED (8x1024 fp32)\n");
  }

  GAUDIRT_SYN_CHECK(synHostUnmap(dev, reinterpret_cast<void*>(h_send_ptr)));
  GAUDIRT_SYN_CHECK(synHostUnmap(dev, reinterpret_cast<void*>(h_recv_ptr)));
  GAUDIRT_SYN_CHECK(synDeviceFree(dev, send_dev, 0));
  GAUDIRT_SYN_CHECK(synDeviceFree(dev, recv_base_dev, 0));
  GAUDIRT_HCCL_CHECK(hcclCommDestroy(comm));
  GAUDIRT_SYN_CHECK(synStreamDestroy(coll_stream));
  GAUDIRT_SYN_CHECK(synDeviceRelease(dev));
  GAUDIRT_SYN_CHECK(synDestroy());
  return 0;
}

class AddRecipe {
 public:
  AddRecipe(synDeviceId dev, uint32_t count, const char* graph_name)
      : dev_(dev), count_(count) {
    GAUDIRT_SYN_CHECK(synGraphCreate(&graph_, synDeviceGaudi2));
    for (int i = 0; i < 3; ++i) {
      GAUDIRT_SYN_CHECK(synSectionCreate(&sections_[i], 0, graph_));
      GAUDIRT_SYN_CHECK(synSectionSetPersistent(sections_[i], true));
      synTensorDescriptor desc{};
      desc.m_name = names_[i];
      desc.m_dataType = syn_type_single;
      desc.m_dims = 1;
      desc.m_sizes[0] = count_;
      desc.m_minSizes[0] = count_;
      GAUDIRT_SYN_CHECK(synTensorCreate(&tensors_[i], &desc, sections_[i], 0));
    }
    synTensor inputs[2] = {tensors_[0], tensors_[1]};
    synTensor outputs[1] = {tensors_[2]};
    GAUDIRT_SYN_CHECK(synNodeCreate(graph_, inputs, outputs, 2, 1, nullptr, 0,
                                    "add_fwd_f32", "p3a_add_node", nullptr,
                                    nullptr));
    GAUDIRT_SYN_CHECK(synGraphCompile(&recipe_, graph_, graph_name, nullptr));
    GAUDIRT_SYN_CHECK(synTensorRetrieveIds(recipe_, names_, tensor_ids_, 3));
    GAUDIRT_SYN_CHECK(synWorkspaceGetSize(&workspace_size_, recipe_));
    if (workspace_size_) {
      GAUDIRT_SYN_CHECK(
          synDeviceMalloc(dev_, workspace_size_, 0, 0, &workspace_));
    }
  }

  ~AddRecipe() { destroy(); }

  void destroy() {
    if (recipe_) synRecipeDestroy(recipe_);
    if (graph_) synGraphDestroy(graph_);
    for (synTensor& tensor : tensors_) {
      if (tensor) synTensorDestroy(tensor);
      tensor = nullptr;
    }
    for (synSectionHandle& section : sections_) {
      if (section) synSectionDestroy(section);
      section = nullptr;
    }
    if (workspace_) synDeviceFree(dev_, workspace_, 0);
    recipe_ = nullptr;
    graph_ = nullptr;
    workspace_ = 0;
  }

  AddRecipe(const AddRecipe&) = delete;
  AddRecipe& operator=(const AddRecipe&) = delete;

  void launch(synStreamHandle stream, uint64_t a, uint64_t b, uint64_t c,
              const char* phase, bool capture) {
    if (capture) p0_hook_set_phase(phase);
    const uint64_t addresses[3] = {a, b, c};
    synLaunchTensorInfoExt bindings[3]{};
    for (int i = 0; i < 3; ++i) {
      bindings[i].tensorName = names_[i];
      bindings[i].pTensorAddress = addresses[i];
      bindings[i].tensorType = DATA_TENSOR;
      bindings[i].tensorSize[0] = count_;
      bindings[i].tensorId = tensor_ids_[i];
    }
    GAUDIRT_SYN_CHECK(synLaunchWithExternalEventsExt(
        stream, bindings, 3, workspace_, recipe_, nullptr, 0, 0));
  }

 private:
  synDeviceId dev_ = SYN_INVALID_DEVICE_ID;
  uint32_t count_ = 0;
  const char* names_[3] = {"tensor_a", "tensor_b", "tensor_c"};
  synGraphHandle graph_ = nullptr;
  synSectionHandle sections_[3]{};
  synTensor tensors_[3]{};
  synRecipeHandle recipe_ = nullptr;
  uint64_t tensor_ids_[3]{};
  uint64_t workspace_size_ = 0;
  uint64_t workspace_ = 0;
};

int run_p3a_sync_rank(int rank, int nranks, SharedCoord* shm) {
  const bool capture = rank == 7;
  p0_hook_set_enabled(capture ? 1 : 0);
  if (capture) p0_hook_set_phase("INIT");

  int local_sense = 0;
  GAUDIRT_SYN_CHECK(synInitialize());
  synDeviceId dev = 0;
  GAUDIRT_SYN_CHECK(
      synDeviceAcquireByModuleId(&dev, static_cast<synModuleId>(rank)));
  synStreamHandle stream = nullptr;
  GAUDIRT_SYN_CHECK(synStreamCreateGeneric(&stream, dev, 0));

  constexpr uint32_t kCount = 1024;
  constexpr uint32_t kRecvCount = kCount * 8;
  constexpr uint64_t kSendBytes = kCount * sizeof(float);
  constexpr uint64_t kRecvBytes = kRecvCount * sizeof(float);
  constexpr uint64_t kGuardBytes = 256;
  constexpr uint64_t kRecvRegionBytes = kRecvBytes + 2 * kGuardBytes;

  AddRecipe pre(dev, kCount, "p3a_pre_add");
  AddRecipe post(dev, kRecvCount, "p3a_post_add");

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

  uint64_t a_base = 0;
  uint64_t b = 0;
  uint64_t b_large = 0;
  uint64_t send = 0;
  uint64_t recv_base = 0;
  uint64_t out_base = 0;
  GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, 2 * kSendBytes, 0, 0, &a_base));
  GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, kSendBytes, 0, 0, &b));
  GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, kRecvBytes, 0, 0, &b_large));
  GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, kSendBytes, 0, 0, &send));
  GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, kRecvRegionBytes, 0, 0, &recv_base));
  GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, 2 * kRecvBytes, 0, 0, &out_base));
  const uint64_t recv = recv_base + kGuardBytes;

  std::vector<float> h_a0(kCount);
  std::vector<float> h_a1(kCount);
  std::vector<float> h_b(kCount);
  std::vector<float> h_b_large(kRecvCount);
  std::vector<float> h_out(kRecvCount);
  std::vector<uint8_t> h_recv(kRecvRegionBytes);
  for (uint32_t i = 0; i < kCount; ++i) {
    h_a0[i] = static_cast<float>(rank * 16 + i % 16);
    h_a1[i] = static_cast<float>(rank * 16 + i % 16 + 1000);
    h_b[i] = static_cast<float>(i % 3);
  }
  for (uint32_t i = 0; i < kRecvCount; ++i) {
    h_b_large[i] = static_cast<float>(i % 5);
  }
  std::memset(h_recv.data(), 0xA5, kGuardBytes);
  std::memset(h_recv.data() + kGuardBytes, 0x5A, kRecvBytes);
  std::memset(h_recv.data() + kGuardBytes + kRecvBytes, 0xA5, kGuardBytes);

  const auto map_host = [&](void* ptr, uint64_t bytes) {
    GAUDIRT_SYN_CHECK(synHostMap(dev, bytes, ptr));
  };
  map_host(h_a0.data(), kSendBytes);
  map_host(h_a1.data(), kSendBytes);
  map_host(h_b.data(), kSendBytes);
  map_host(h_b_large.data(), kRecvBytes);
  map_host(h_out.data(), kRecvBytes);
  map_host(h_recv.data(), kRecvRegionBytes);

  GAUDIRT_SYN_CHECK(synMemCopyAsync(
      stream, reinterpret_cast<uint64_t>(h_a0.data()), kSendBytes, a_base,
      HOST_TO_DRAM));
  GAUDIRT_SYN_CHECK(synMemCopyAsync(
      stream, reinterpret_cast<uint64_t>(h_a1.data()), kSendBytes,
      a_base + kSendBytes, HOST_TO_DRAM));
  GAUDIRT_SYN_CHECK(synMemCopyAsync(stream,
                                    reinterpret_cast<uint64_t>(h_b.data()),
                                    kSendBytes, b, HOST_TO_DRAM));
  GAUDIRT_SYN_CHECK(synMemCopyAsync(
      stream, reinterpret_cast<uint64_t>(h_b_large.data()), kRecvBytes, b_large,
      HOST_TO_DRAM));
  GAUDIRT_SYN_CHECK(synMemCopyAsync(
      stream, reinterpret_cast<uint64_t>(h_recv.data()), kRecvRegionBytes,
      recv_base, HOST_TO_DRAM));
  GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));
  host_barrier(shm, nranks, local_sense);

  for (uint32_t iteration = 0; iteration < 3; ++iteration) {
    const char* label = iteration == 0   ? "P3A_COLD"
                        : iteration == 1 ? "P3A_WARM"
                                         : "P3A_MUTATED";
    const std::string pre_phase = std::string(label) + "_PRE";
    const std::string coll_phase = std::string(label) + "_ALLGATHER";
    const std::string post_phase = std::string(label) + "_POST";
    const uint64_t a = iteration == 2 ? a_base + kSendBytes : a_base;
    const uint64_t out = iteration == 2 ? out_base + kRecvBytes : out_base;

    pre.launch(stream, a, b, send, pre_phase.c_str(), capture);
    if (capture) p0_hook_set_phase(coll_phase.c_str());
    GAUDIRT_HCCL_CHECK(hcclAllGather(reinterpret_cast<const void*>(send),
                                     reinterpret_cast<void*>(recv), kCount,
                                     hcclFloat32, comm, stream));
    post.launch(stream, recv, b_large, out, post_phase.c_str(), capture);
    GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));

    if (capture) p0_hook_set_phase("P3A_VERIFY");
    GAUDIRT_SYN_CHECK(synMemCopyAsync(
        stream, out, kRecvBytes, reinterpret_cast<uint64_t>(h_out.data()),
        DRAM_TO_HOST));
    GAUDIRT_SYN_CHECK(synMemCopyAsync(
        stream, recv_base, kRecvRegionBytes,
        reinterpret_cast<uint64_t>(h_recv.data()), DRAM_TO_HOST));
    GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));

    for (uint32_t i = 0; i < kGuardBytes; ++i) {
      if (h_recv[i] != 0xA5 || h_recv[kGuardBytes + kRecvBytes + i] != 0xA5) {
        std::fprintf(stderr, "[P3A_SYNC] rank=%d iteration=%u guard=FAIL\n",
                     rank, iteration);
        return 1;
      }
    }
    for (int source = 0; source < nranks; ++source) {
      for (uint32_t i = 0; i < kCount; ++i) {
        const uint32_t index = source * kCount + i;
        const float expected =
            static_cast<float>(source * 16 + i % 16 +
                               (iteration == 2 ? 1000 : 0) + i % 3 + index % 5);
        if (h_out[index] != expected) {
          std::fprintf(stderr,
                       "[P3A_SYNC] rank=%d iteration=%u source=%d element=%u "
                       "actual=%f expected=%f\n",
                       rank, iteration, source, i, h_out[index], expected);
          return 1;
        }
      }
    }
    host_barrier(shm, nranks, local_sense);
  }

  if (capture) p0_hook_set_phase("CLEANUP");
  std::fprintf(stderr,
               "[P3A_SYNC_VERDICT] rank=%d native_steps=3 "
               "closed_loop=PASS output_bytes=%llu guard_bytes=%llu\n",
               rank, static_cast<unsigned long long>(kRecvBytes),
               static_cast<unsigned long long>(2 * kGuardBytes));

  GAUDIRT_SYN_CHECK(synHostUnmap(dev, h_a0.data()));
  GAUDIRT_SYN_CHECK(synHostUnmap(dev, h_a1.data()));
  GAUDIRT_SYN_CHECK(synHostUnmap(dev, h_b.data()));
  GAUDIRT_SYN_CHECK(synHostUnmap(dev, h_b_large.data()));
  GAUDIRT_SYN_CHECK(synHostUnmap(dev, h_out.data()));
  GAUDIRT_SYN_CHECK(synHostUnmap(dev, h_recv.data()));
  GAUDIRT_SYN_CHECK(synDeviceFree(dev, a_base, 0));
  GAUDIRT_SYN_CHECK(synDeviceFree(dev, b, 0));
  GAUDIRT_SYN_CHECK(synDeviceFree(dev, b_large, 0));
  GAUDIRT_SYN_CHECK(synDeviceFree(dev, send, 0));
  GAUDIRT_SYN_CHECK(synDeviceFree(dev, recv_base, 0));
  GAUDIRT_SYN_CHECK(synDeviceFree(dev, out_base, 0));
  pre.destroy();
  post.destroy();
  GAUDIRT_HCCL_CHECK(hcclCommDestroy(comm));
  GAUDIRT_SYN_CHECK(synStreamDestroy(stream));
  GAUDIRT_SYN_CHECK(synDeviceRelease(dev));
  GAUDIRT_SYN_CHECK(synDestroy());
  return 0;
}

int run_allgather_capture(bool p3a_sync = false) {
  p0_hook_set_enabled(0);
  constexpr int kNumRanks = 8;
  void* mem = mmap(nullptr, sizeof(SharedCoord), PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  auto* shm = new (mem) SharedCoord();

  std::vector<pid_t> pids(kNumRanks);
  for (int r = 0; r < kNumRanks; ++r) {
    pid_t pid = fork();
    if (pid == 0) {
      int rc = p3a_sync ? run_p3a_sync_rank(r, kNumRanks, shm)
                        : run_allgather_rank(r, kNumRanks, shm);
      std::_Exit(rc);
    }
    pids[r] = pid;
  }

  int overall_rc = 0;
  for (int r = 0; r < kNumRanks; ++r) {
    int status = 0;
    waitpid(pids[r], &status, 0);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      overall_rc = 1;
    }
  }
  munmap(mem, sizeof(SharedCoord));
  return overall_rc;
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode = "recipe";
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      mode = argv[++i];
    }
  }
  if (mode == "recipe") {
    return run_recipe_capture();
  } else if (mode == "allgather") {
    return run_allgather_capture();
  } else if (mode == "p3a-sync") {
    return run_allgather_capture(true);
  } else {
    std::fprintf(stderr, "Usage: %s --mode [recipe|allgather|p3a-sync]\n",
                 argv[0]);
    return 1;
  }
}
