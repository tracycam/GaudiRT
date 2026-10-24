// SPDX-License-Identifier: Apache-2.0
// P1 Autonomous Single-Recipe Replay Probe
// Relocates all recipe blobs into self-owned HBM, destroys and poisons the
// original recipe resources, then checks synchronous and single-submit replay
// against full Synapse reference outputs.

#include <dlfcn.h>
#include <synapse_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "gaudirt/scal/ccb_writer.hpp"

#define CHECK_SYN(call)                                                       \
  do {                                                                        \
    synStatus _s = (call);                                                    \
    if (_s != synSuccess) {                                                   \
      std::fprintf(stderr, "[P1_FATAL] %s failed with status %d at %s:%d\n", \
                   #call, static_cast<int>(_s), __FILE__, __LINE__);          \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

#define CHECK_SCAL(call)                                                      \
  do {                                                                        \
    int _rc = (call);                                                         \
    if (_rc != SCAL_SUCCESS) {                                                \
      std::fprintf(stderr, "[P1_FATAL] %s failed with status %d at %s:%d\n", \
                   #call, _rc, __FILE__, __LINE__);                           \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

namespace {

using SetPhaseFn = void (*)(const char*);
using ExtractBlobsFn = int (*)(const char*, uint8_t*, uint32_t*, uint8_t*,
                               uint32_t*, uint8_t*, uint32_t*, uint64_t*,
                               uint64_t*, uint64_t*, uint64_t*);
using RunAutonFn = int (*)(const char*, uint64_t, uint64_t, uint64_t, uint64_t,
                           uint64_t, uint32_t, int,
                           gaudirt::scal::CcbReplayStats*);

SetPhaseFn g_set_phase = nullptr;
ExtractBlobsFn g_extract_blobs = nullptr;
RunAutonFn g_run_auton = nullptr;

struct RecipeCapture {
  std::vector<uint8_t> exec;
  std::vector<uint8_t> patch;
  std::vector<uint8_t> dynamic;
  uint64_t base0 = 0;
  uint64_t base1 = 0;
  uint64_t base2 = 0;
  uint64_t base7 = 0;
};

void resolve_hook_symbols() {
  g_set_phase =
      reinterpret_cast<SetPhaseFn>(dlsym(RTLD_DEFAULT, "p0_hook_set_phase"));
  g_extract_blobs = reinterpret_cast<ExtractBlobsFn>(
      dlsym(RTLD_DEFAULT, "p1_extract_recipe_blobs"));
  g_run_auton = reinterpret_cast<RunAutonFn>(
      dlsym(RTLD_DEFAULT, "p1_run_autonomous_recipe"));
  if (!g_set_phase || !g_extract_blobs || !g_run_auton) {
    std::fprintf(stderr,
                 "[P1_FATAL] libscal_hook.so symbols not found in RTLD_DEFAULT "
                 "(did you set LD_PRELOAD?)\n");
    std::exit(1);
  }
}

RecipeCapture load_capture(const char* phase) {
  RecipeCapture capture;
  capture.exec.resize(4096);
  capture.patch.resize(4096);
  capture.dynamic.resize(65536);
  uint32_t exec_size = static_cast<uint32_t>(capture.exec.size());
  uint32_t patch_size = static_cast<uint32_t>(capture.patch.size());
  uint32_t dynamic_size = static_cast<uint32_t>(capture.dynamic.size());
  const int rc = g_extract_blobs(
      phase, capture.exec.data(), &exec_size, capture.patch.data(), &patch_size,
      capture.dynamic.data(), &dynamic_size, &capture.base0, &capture.base1,
      &capture.base2, &capture.base7);
  if (rc != 0) {
    std::fprintf(stderr, "[P1_FATAL] extract %s failed: %d\n", phase, rc);
    std::exit(1);
  }
  capture.exec.resize(exec_size);
  capture.patch.resize(patch_size);
  capture.dynamic.resize(dynamic_size);
  return capture;
}

int find_unique_u64_offset(const std::vector<uint8_t>& blob,
                           uint64_t target_val) {
  int result = -1;
  for (size_t off = 0; off + sizeof(uint64_t) <= blob.size(); off += 4) {
    uint64_t val = 0;
    std::memcpy(&val, blob.data() + off, sizeof(val));
    if (val == target_val) {
      if (result >= 0) return -2;
      result = static_cast<int>(off);
    }
  }
  return result;
}

uint32_t read_u32_at(const std::vector<uint8_t>& blob, int offset) {
  uint32_t value = 0;
  std::memcpy(&value, blob.data() + offset, sizeof(value));
  return value;
}

void patch_u64_at(std::vector<uint8_t>& blob, int offset, uint64_t new_val) {
  std::memcpy(blob.data() + offset, &new_val, sizeof(new_val));
}

void patch_u32_at(std::vector<uint8_t>& blob, int offset, uint32_t new_val) {
  std::memcpy(blob.data() + offset, &new_val, sizeof(new_val));
}

void print_stats(const char* gate, const gaudirt::scal::CcbReplayStats& stats) {
  std::printf(
      "[P1_REPLAY_STATS] gate=%s calls=%llu boundary_calls=%llu nop_bytes=%llu "
      "pi_start=0x%x pi_end=0x%x pi_bytes=%llu\n",
      gate, static_cast<unsigned long long>(stats.submit_calls),
      static_cast<unsigned long long>(stats.boundary_submit_calls),
      static_cast<unsigned long long>(stats.nop_bytes), stats.start_pi,
      stats.end_pi, static_cast<unsigned long long>(stats.pi_bytes()));
}

}  // namespace

int main() {
  resolve_hook_symbols();
  g_set_phase("INIT");

  CHECK_SYN(synInitialize());

  synDeviceId dev = 0;
  synModuleId mod = 7;
  if (synDeviceAcquireByModuleId(&dev, mod) != synSuccess) {
    CHECK_SYN(synDeviceAcquire(&dev, nullptr));
  }

  synStreamHandle stream = nullptr;
  CHECK_SYN(synStreamCreateGeneric(&stream, dev, 0));

  // Compile minimal TPC add_fwd_f32 recipe: C[1024] = A[1024] + B[1024]
  const unsigned N = 1024;
  const uint64_t byte_size = N * sizeof(float);
  unsigned dims[1] = {N};

  synGraphHandle graph = nullptr;
  CHECK_SYN(synGraphCreate(&graph, synDeviceGaudi2));

  synSectionHandle sec_a = nullptr, sec_b = nullptr, sec_c = nullptr;
  CHECK_SYN(synSectionCreate(&sec_a, 0, graph));
  CHECK_SYN(synSectionSetPersistent(sec_a, true));
  CHECK_SYN(synSectionCreate(&sec_b, 0, graph));
  CHECK_SYN(synSectionSetPersistent(sec_b, true));
  CHECK_SYN(synSectionCreate(&sec_c, 0, graph));
  CHECK_SYN(synSectionSetPersistent(sec_c, true));

  synTensorDescriptor desc_a{};
  desc_a.m_dataType = syn_type_single;
  desc_a.m_dims = 1;
  desc_a.m_sizes[0] = dims[0];
  desc_a.m_name = "tensor_a";
  synTensor tensor_a = nullptr;
  CHECK_SYN(synTensorCreate(&tensor_a, &desc_a, sec_a, 0));

  synTensorDescriptor desc_b = desc_a;
  desc_b.m_name = "tensor_b";
  synTensor tensor_b = nullptr;
  CHECK_SYN(synTensorCreate(&tensor_b, &desc_b, sec_b, 0));

  synTensorDescriptor desc_c = desc_a;
  desc_c.m_name = "tensor_c";
  synTensor tensor_c = nullptr;
  CHECK_SYN(synTensorCreate(&tensor_c, &desc_c, sec_c, 0));

  synTensor inputs[2] = {tensor_a, tensor_b};
  synTensor outputs[1] = {tensor_c};
  CHECK_SYN(synNodeCreate(graph, inputs, outputs, 2, 1, nullptr, 0,
                          "add_fwd_f32", "add_node", nullptr, nullptr));

  synRecipeHandle recipe = nullptr;
  CHECK_SYN(synGraphCompile(&recipe, graph, "p1_min_add_f32", nullptr));

  uint64_t ws_size = 0;
  CHECK_SYN(synWorkspaceGetSize(&ws_size, recipe));
  uint64_t ws_dev = 0;
  if (ws_size > 0) {
    CHECK_SYN(synDeviceMalloc(dev, ws_size, 0, 0, &ws_dev));
  }
  std::printf("[P1_WORKSPACE] self_owned=1 size=%llu addr=0x%lx\n",
              static_cast<unsigned long long>(ws_size), ws_dev);

  // Allocate capture tensors
  uint64_t cap_a_dev = 0, cap_b_dev = 0, cap_c_dev = 0;
  CHECK_SYN(synDeviceMalloc(dev, byte_size * 4, 0, 0, &cap_a_dev));
  CHECK_SYN(synDeviceMalloc(dev, byte_size * 4, 0, 0, &cap_b_dev));
  CHECK_SYN(synDeviceMalloc(dev, byte_size * 4, 0, 0, &cap_c_dev));
  cap_a_dev += 256;
  cap_b_dev += 512;
  cap_c_dev += 768;

  const char* names[3] = {"tensor_a", "tensor_b", "tensor_c"};
  uint64_t ids[3] = {0, 0, 0};
  CHECK_SYN(synTensorRetrieveIds(recipe, names, ids, 3));
  uint64_t idx_a = ids[0], idx_b = ids[1], idx_c = ids[2];

  synLaunchTensorInfoExt launch_tensors[3]{};
  launch_tensors[0].tensorName = "tensor_a";
  launch_tensors[0].pTensorAddress = cap_a_dev;
  launch_tensors[0].tensorType = DATA_TENSOR;
  launch_tensors[0].tensorId = idx_a;

  launch_tensors[1].tensorName = "tensor_b";
  launch_tensors[1].pTensorAddress = cap_b_dev;
  launch_tensors[1].tensorType = DATA_TENSOR;
  launch_tensors[1].tensorId = idx_b;

  launch_tensors[2].tensorName = "tensor_c";
  launch_tensors[2].pTensorAddress = cap_c_dev;
  launch_tensors[2].tensorType = DATA_TENSOR;
  launch_tensors[2].tensorId = idx_c;

  // Cold launch + Warm capture launch
  g_set_phase("CAPTURE_COLD");
  CHECK_SYN(synLaunchWithExternalEventsExt(stream, launch_tensors, 3, ws_dev,
                                           recipe, nullptr, 0, 0));
  CHECK_SYN(synStreamSynchronize(stream));

  g_set_phase("CAPTURE_WARM");
  CHECK_SYN(synLaunchWithExternalEventsExt(stream, launch_tensors, 3, ws_dev,
                                           recipe, nullptr, 0, 0));
  CHECK_SYN(synStreamSynchronize(stream));
  g_set_phase("BENCH");  // Zero hook formatting overhead during 1,000-step benchmark

  const RecipeCapture cold = load_capture("CAPTURE_COLD");
  const RecipeCapture warm = load_capture("CAPTURE_WARM");
  if (cold.exec != warm.exec || cold.dynamic != warm.dynamic ||
      cold.patch.size() != warm.patch.size()) {
    std::fprintf(stderr, "[P1_FATAL] cold/warm descriptor invariant failed\n");
    return 1;
  }

  std::vector<int> kernel_offsets;
  for (size_t off = 0; off + sizeof(uint32_t) <= warm.patch.size(); off += 4) {
    if (std::memcmp(cold.patch.data() + off, warm.patch.data() + off, 4) == 0) {
      continue;
    }
    const uint32_t cold_base = static_cast<uint32_t>(cold.base2);
    const uint32_t warm_base = static_cast<uint32_t>(warm.base2);
    const uint32_t cold_addr = read_u32_at(cold.patch, static_cast<int>(off));
    const uint32_t warm_addr = read_u32_at(warm.patch, static_cast<int>(off));
    if (cold_addr < cold_base || warm_addr < warm_base ||
        cold_addr - cold_base >= cold.dynamic.size() ||
        warm_addr - warm_base >= warm.dynamic.size() ||
        cold_addr - cold_base != warm_addr - warm_base) {
      std::fprintf(
          stderr,
          "[P1_FATAL] invalid kernel delta at +0x%zx cold=[base=0x%x "
          "addr=0x%x rel=0x%x size=0x%zx] warm=[base=0x%x addr=0x%x "
          "rel=0x%x size=0x%zx]\n",
          off, cold_base, cold_addr, cold_addr - cold_base, cold.dynamic.size(),
          warm_base, warm_addr, warm_addr - warm_base, warm.dynamic.size());
      return 1;
    }
    kernel_offsets.push_back(static_cast<int>(off));
  }
  if (kernel_offsets.size() != 2) {
    std::fprintf(stderr, "[P1_FATAL] expected 2 kernel offsets, found %zu\n",
                 kernel_offsets.size());
    return 1;
  }

  const int off_a = find_unique_u64_offset(warm.patch, cap_a_dev);
  const int off_b = find_unique_u64_offset(warm.patch, cap_b_dev);
  const int off_c = find_unique_u64_offset(warm.patch, cap_c_dev);
  if (off_a < 0 || off_b < 0 || off_c < 0) {
    std::fprintf(stderr, "[P1_FATAL] tensor patch offsets are not unique\n");
    return 1;
  }
  std::printf(
      "[P1_EXTRACT] exec_blob=%zuB patch_blob=%zuB dynamic_blob=%zuB "
      "orig_base=[0:0x%lx, 1:0x%lx, 2:0x%lx, 7:0x%lx]\n",
      warm.exec.size(), warm.patch.size(), warm.dynamic.size(), warm.base0,
      warm.base1, warm.base2, warm.base7);
  std::printf(
      "[P1_PATCH_OFFSETS] tensor_a=+0x%x tensor_b=+0x%x tensor_c=+0x%x "
      "kernel=[+0x%x,+0x%x]\n",
      off_a, off_b, off_c, kernel_offsets[0], kernel_offsets[1]);

  uint64_t raw_perm_hbm = 0;
  CHECK_SYN(synDeviceMalloc(dev, 65536, 0, 0, &raw_perm_hbm));
  const uint64_t perm_exec_hbm = (raw_perm_hbm + 0x1FFFULL) & ~0x1FFFULL;
  const uint64_t perm_patch_single_hbm = perm_exec_hbm + 0x200ULL;
  const uint64_t perm_patch_even_hbm = perm_exec_hbm + 0x400ULL;
  const uint64_t perm_patch_odd_hbm = perm_exec_hbm + 0x600ULL;

  synDeviceInfoV2 device_info{};
  CHECK_SYN(synDeviceGetInfoV2(dev, &device_info));
  scal_handle_t scal = nullptr;
  CHECK_SCAL(scal_get_handle_from_fd(static_cast<int>(device_info.fd), &scal));
  scal_pool_handle_t hbm_shared = nullptr;
  CHECK_SCAL(scal_get_pool_handle_by_name(scal, "hbm_shared", &hbm_shared));
  scal_buffer_handle_t perm_dynamic_buffer = nullptr;
  CHECK_SCAL(scal_allocate_aligned_buffer(hbm_shared, warm.dynamic.size(),
                                          0x2000, &perm_dynamic_buffer));
  scal_buffer_info_t perm_dynamic_info{};
  CHECK_SCAL(scal_buffer_get_info(perm_dynamic_buffer, &perm_dynamic_info));
  const uint64_t perm_dynamic_hbm = perm_dynamic_info.device_address;
  const uint64_t perm_nop_hbm =
      perm_dynamic_hbm + (warm.base7 - warm.base2);
  if (!perm_dynamic_hbm || (perm_dynamic_hbm & 0x1FFFULL) != 0) {
    std::fprintf(stderr, "[P1_FATAL] invalid hbm_shared allocation\n");
    return 1;
  }
  std::printf(
      "[P1_ALLOC] exec=0x%lx dynamic=0x%lx nop=0x%lx dynamic_size=%zu "
      "alignment=0x2000\n",
      perm_exec_hbm, perm_dynamic_hbm, perm_nop_hbm, warm.dynamic.size());

  uint64_t stage_host_ptr = 0;
  uint64_t poison_host_ptr = 0;
  CHECK_SYN(synHostMalloc(dev, 65536, 0,
                          reinterpret_cast<void**>(&stage_host_ptr)));
  CHECK_SYN(synHostMalloc(dev, 65536, 0,
                          reinterpret_cast<void**>(&poison_host_ptr)));

  // Allocate P1 test tensors (brand-new device addresses)
  uint64_t p1_a_dev = 0, p1_b_dev = 0, p1_c_auton_dev = 0, p1_c_ref_dev = 0;
  uint64_t accum_syn0_dev = 0, accum_syn1_dev = 0;
  uint64_t accum_auton0_dev = 0, accum_auton1_dev = 0;
  CHECK_SYN(synDeviceMalloc(dev, byte_size, 0, 0, &p1_a_dev));
  CHECK_SYN(synDeviceMalloc(dev, byte_size, 0, 0, &p1_b_dev));
  CHECK_SYN(synDeviceMalloc(dev, byte_size, 0, 0, &p1_c_auton_dev));
  CHECK_SYN(synDeviceMalloc(dev, byte_size, 0, 0, &p1_c_ref_dev));
  CHECK_SYN(synDeviceMalloc(dev, byte_size, 0, 0, &accum_syn0_dev));
  CHECK_SYN(synDeviceMalloc(dev, byte_size, 0, 0, &accum_syn1_dev));
  CHECK_SYN(synDeviceMalloc(dev, byte_size, 0, 0, &accum_auton0_dev));
  CHECK_SYN(synDeviceMalloc(dev, byte_size, 0, 0, &accum_auton1_dev));

  float* host_a = nullptr;
  float* host_b = nullptr;
  float* host_c_ref = nullptr;
  float* host_c_auton = nullptr;
  float* host_accum_ref = nullptr;
  CHECK_SYN(
      synHostMalloc(dev, byte_size, 0, reinterpret_cast<void**>(&host_a)));
  CHECK_SYN(
      synHostMalloc(dev, byte_size, 0, reinterpret_cast<void**>(&host_b)));
  CHECK_SYN(
      synHostMalloc(dev, byte_size, 0, reinterpret_cast<void**>(&host_c_ref)));
  CHECK_SYN(synHostMalloc(dev, byte_size, 0,
                          reinterpret_cast<void**>(&host_c_auton)));
  CHECK_SYN(synHostMalloc(dev, byte_size, 0,
                          reinterpret_cast<void**>(&host_accum_ref)));

  for (unsigned i = 0; i < N; ++i) {
    host_a[i] = static_cast<float>(i) * 0.25f + 1.5f;
    host_b[i] = static_cast<float>(i % 17) * 0.125f + 0.5f;
    host_c_ref[i] = -1.0f;
    host_c_auton[i] = -2.0f;
    host_accum_ref[i] = -3.0f;
  }

  CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_a),
                            byte_size, p1_a_dev, HOST_TO_DRAM));
  CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_b),
                            byte_size, p1_b_dev, HOST_TO_DRAM));
  CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_c_ref),
                            byte_size, p1_c_ref_dev, HOST_TO_DRAM));
  CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_c_auton),
                            byte_size, p1_c_auton_dev, HOST_TO_DRAM));
  CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_a),
                            byte_size, accum_syn0_dev, HOST_TO_DRAM));
  CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_a),
                            byte_size, accum_auton0_dev, HOST_TO_DRAM));
  CHECK_SYN(synStreamSynchronize(stream));

  // =========================================================================
  // STEP 1: Compute Synapse Ground-Truth References (1 launch + 1,000 ping-pong)
  // =========================================================================
  launch_tensors[0].pTensorAddress = p1_a_dev;
  launch_tensors[1].pTensorAddress = p1_b_dev;
  launch_tensors[2].pTensorAddress = p1_c_ref_dev;
  CHECK_SYN(synLaunchWithExternalEventsExt(stream, launch_tensors, 3, ws_dev,
                                           recipe, nullptr, 0, 0));
  CHECK_SYN(synStreamSynchronize(stream));
  CHECK_SYN(synMemCopyAsync(stream, p1_c_ref_dev, byte_size,
                            reinterpret_cast<uint64_t>(host_c_ref),
                            DRAM_TO_HOST));
  CHECK_SYN(synStreamSynchronize(stream));

  const uint32_t kNumIters = 1000;
  auto t0_syn = std::chrono::steady_clock::now();
  for (uint32_t it = 0; it < kNumIters; ++it) {
    uint64_t src_dev = ((it & 1U) == 0) ? accum_syn0_dev : accum_syn1_dev;
    uint64_t dst_dev = ((it & 1U) == 0) ? accum_syn1_dev : accum_syn0_dev;
    launch_tensors[0].pTensorAddress = src_dev;
    launch_tensors[1].pTensorAddress = p1_b_dev;
    launch_tensors[2].pTensorAddress = dst_dev;
    CHECK_SYN(synLaunchWithExternalEventsExt(stream, launch_tensors, 3, ws_dev,
                                             recipe, nullptr, 0, 0));
    CHECK_SYN(synStreamSynchronize(stream));
  }
  auto t1_syn = std::chrono::steady_clock::now();
  double syn_sync_us =
      std::chrono::duration<double, std::micro>(t1_syn - t0_syn).count() /
      kNumIters;

  CHECK_SYN(synMemCopyAsync(stream, accum_syn0_dev, byte_size,
                            reinterpret_cast<uint64_t>(host_accum_ref),
                            DRAM_TO_HOST));
  CHECK_SYN(synStreamSynchronize(stream));

  if (warm.dynamic.size() > 0x9000 || warm.exec.size() > 0x200 ||
      warm.patch.size() > 0x200) {
    std::fprintf(stderr, "[P1_FATAL] descriptor staging layout is too small\n");
    return 1;
  }

  auto make_patch = [&](uint64_t a, uint64_t b, uint64_t c) {
    std::vector<uint8_t> result = warm.patch;
    for (const int off : kernel_offsets) {
      const uint32_t relative =
          read_u32_at(result, off) - static_cast<uint32_t>(warm.base2);
      const uint64_t relocated =
          static_cast<uint64_t>(static_cast<uint32_t>(perm_dynamic_hbm)) +
          relative;
      if (relocated > 0xFFFFFFFFULL) {
        std::fprintf(stderr, "[P1_FATAL] kernel relocation overflows 32 bits\n");
        std::exit(1);
      }
      patch_u32_at(result, off, static_cast<uint32_t>(relocated));
    }
    patch_u64_at(result, off_a, a);
    patch_u64_at(result, off_b, b);
    patch_u64_at(result, off_c, c);
    return result;
  };

  const std::vector<uint8_t> patch_single =
      make_patch(p1_a_dev, p1_b_dev, p1_c_auton_dev);
  const std::vector<uint8_t> patch_even =
      make_patch(accum_auton0_dev, p1_b_dev, accum_auton1_dev);
  const std::vector<uint8_t> patch_odd =
      make_patch(accum_auton1_dev, p1_b_dev, accum_auton0_dev);

  std::memcpy(reinterpret_cast<void*>(stage_host_ptr), warm.dynamic.data(),
              warm.dynamic.size());
  CHECK_SYN(synMemCopyAsync(stream, stage_host_ptr, warm.dynamic.size(),
                            perm_dynamic_hbm, HOST_TO_DRAM));
  std::memcpy(reinterpret_cast<void*>(stage_host_ptr + 0x9000),
              warm.exec.data(), warm.exec.size());
  CHECK_SYN(synMemCopyAsync(stream, stage_host_ptr + 0x9000, warm.exec.size(),
                            perm_exec_hbm, HOST_TO_DRAM));
  std::memcpy(reinterpret_cast<void*>(stage_host_ptr + 0x9200),
              patch_single.data(), patch_single.size());
  CHECK_SYN(synMemCopyAsync(stream, stage_host_ptr + 0x9200,
                            patch_single.size(), perm_patch_single_hbm,
                            HOST_TO_DRAM));
  std::memcpy(reinterpret_cast<void*>(stage_host_ptr + 0x9400),
              patch_even.data(), patch_even.size());
  CHECK_SYN(synMemCopyAsync(stream, stage_host_ptr + 0x9400, patch_even.size(),
                            perm_patch_even_hbm, HOST_TO_DRAM));
  std::memcpy(reinterpret_cast<void*>(stage_host_ptr + 0x9600),
              patch_odd.data(), patch_odd.size());
  CHECK_SYN(synMemCopyAsync(stream, stage_host_ptr + 0x9600, patch_odd.size(),
                            perm_patch_odd_hbm, HOST_TO_DRAM));
  CHECK_SYN(synStreamSynchronize(stream));

  CHECK_SYN(synRecipeDestroy(recipe));
  recipe = nullptr;

  auto poison = [&](uint64_t addr, size_t bytes) {
    std::memset(reinterpret_cast<void*>(stage_host_ptr), 0, bytes);
    CHECK_SYN(synMemCopyAsync(stream, stage_host_ptr, bytes, addr, HOST_TO_DRAM));
    CHECK_SYN(synStreamSynchronize(stream));
    std::memset(reinterpret_cast<void*>(poison_host_ptr), 0xA5, bytes);
    CHECK_SYN(synMemCopyAsync(stream, addr, bytes, poison_host_ptr, DRAM_TO_HOST));
    CHECK_SYN(synStreamSynchronize(stream));
    for (size_t i = 0; i < bytes; ++i) {
      if (*reinterpret_cast<uint8_t*>(poison_host_ptr + i) != 0) return false;
    }
    return true;
  };
  const bool poisoned = poison(warm.base0, warm.patch.size()) &&
                        poison(warm.base1, warm.exec.size()) &&
                        poison(warm.base2, warm.dynamic.size());
  if (!poisoned) {
    std::fprintf(stderr, "[P1_FATAL] original slot poison verification failed\n");
    return 1;
  }
  std::printf("[P1_LIFETIME] recipe_destroyed=1 poison_blocks=3 verified=1\n");

  // =========================================================================
  // GATE 1: Single Autonomous Replay vs Synapse Reference
  // =========================================================================
  gaudirt::scal::CcbReplayStats stats1;
  const int rc1 = g_run_auton("CAPTURE_WARM", perm_patch_single_hbm, 0,
                              perm_exec_hbm, perm_dynamic_hbm, perm_nop_hbm, 1,
                              0, &stats1);
  if (rc1 != 0) {
    std::fprintf(
        stderr, "[P1_FATAL] Gate 1 p1_run_autonomous_recipe failed: %d\n", rc1);
    return 1;
  }
  print_stats("gate1", stats1);

  CHECK_SYN(synMemCopyAsync(stream, p1_c_auton_dev, byte_size,
                            reinterpret_cast<uint64_t>(host_c_auton),
                            DRAM_TO_HOST));
  CHECK_SYN(synStreamSynchronize(stream));

  const int cmp1 = std::memcmp(host_c_ref, host_c_auton, byte_size);
  std::printf(
      "[P1_GATE1_SINGLE] memcmp=%d ref[0]=%.4f auton[0]=%.4f ref[1023]=%.4f "
      "auton[1023]=%.4f => %s\n",
      cmp1, host_c_ref[0], host_c_auton[0], host_c_ref[1023],
      host_c_auton[1023], (cmp1 == 0) ? "PASS" : "FAIL");
  if (cmp1 != 0) return 1;

  // =========================================================================
  // GATE 2: 1,000 Consecutive Autonomous Replays (Synchronous Loop)
  // =========================================================================
  gaudirt::scal::CcbReplayStats stats2;
  auto t0_auton = std::chrono::steady_clock::now();
  const int rc2 = g_run_auton("CAPTURE_WARM", perm_patch_even_hbm,
                              perm_patch_odd_hbm, perm_exec_hbm,
                              perm_dynamic_hbm, perm_nop_hbm, kNumIters, 0,
                              &stats2);
  auto t1_auton = std::chrono::steady_clock::now();
  if (rc2 != 0) {
    std::fprintf(stderr,
                 "[P1_FATAL] Gate 2 p1_run_autonomous_recipe(1000, sync) "
                 "failed: %d\n",
                 rc2);
    return 1;
  }
  print_stats("gate2", stats2);
  const double auton_sync_us =
      std::chrono::duration<double, std::micro>(t1_auton - t0_auton).count() /
      kNumIters;

  std::memset(host_c_auton, 0, byte_size);
  CHECK_SYN(synMemCopyAsync(stream, accum_auton0_dev, byte_size,
                            reinterpret_cast<uint64_t>(host_c_auton),
                            DRAM_TO_HOST));
  CHECK_SYN(synStreamSynchronize(stream));

  const int cmp2 = std::memcmp(host_accum_ref, host_c_auton, byte_size);
  std::printf(
      "[P1_GATE2_1000_SYNC] memcmp=%d syn_ref[0]=%.4f auton[0]=%.4f "
      "syn_ref[1023]=%.4f auton[1023]=%.4f | diag_syn_us=%.2f us/launch vs "
      "diag_auton_sync_us=%.2f us/launch => %s\n",
      cmp2, host_accum_ref[0], host_c_auton[0], host_accum_ref[1023],
      host_c_auton[1023], syn_sync_us, auton_sync_us,
      (cmp2 == 0) ? "PASS" : "FAIL");
  if (cmp2 != 0) return 1;

  // =========================================================================
  // GATE 3: 1,000 Consecutive Autonomous Replays in a SINGLE scal_stream_submit
  // =========================================================================
  CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_a),
                            byte_size, accum_auton0_dev, HOST_TO_DRAM));
  CHECK_SYN(synStreamSynchronize(stream));

  gaudirt::scal::CcbReplayStats stats3;
  auto t0_batch = std::chrono::steady_clock::now();
  const int rc3 = g_run_auton("CAPTURE_WARM", perm_patch_even_hbm,
                              perm_patch_odd_hbm, perm_exec_hbm,
                              perm_dynamic_hbm, perm_nop_hbm, kNumIters, 1,
                              &stats3);
  auto t1_batch = std::chrono::steady_clock::now();
  if (rc3 != 0) {
    std::fprintf(stderr,
                 "[P1_FATAL] Gate 3 p1_run_autonomous_recipe(1000, batch) "
                 "failed: %d\n",
                 rc3);
    return 1;
  }
  print_stats("gate3", stats3);
  const double auton_batch_us =
      std::chrono::duration<double, std::micro>(t1_batch - t0_batch).count() /
      kNumIters;

  std::memset(host_c_auton, 0, byte_size);
  CHECK_SYN(synMemCopyAsync(stream, accum_auton0_dev, byte_size,
                            reinterpret_cast<uint64_t>(host_c_auton),
                            DRAM_TO_HOST));
  CHECK_SYN(synStreamSynchronize(stream));

  const int cmp3 = std::memcmp(host_accum_ref, host_c_auton, byte_size);
  const bool gate3_pass =
      cmp3 == 0 && stats3.submit_calls == 1 &&
      stats3.boundary_submit_calls == 0 && stats3.pi_bytes() >= 80000;
  std::printf(
      "[P1_GATE3_1000_SINGLE_SUBMIT] memcmp=%d calls=%llu boundary_calls=%llu "
      "diag_batch_us=%.2f us/replay => %s\n",
      cmp3, static_cast<unsigned long long>(stats3.submit_calls),
      static_cast<unsigned long long>(stats3.boundary_submit_calls),
      auton_batch_us, gate3_pass ? "PASS" : "FAIL");
  if (!gate3_pass) return 1;

  std::printf(
      "\n[P1_VERDICT] ALL 3 GATES PASSED (relocated_base012=1 "
      "recipe_destroyed=1 original_slots_poisoned=1 single_submit_calls=1)!\n");
  std::fflush(stdout);

  // Cleanup
  synHostFree(dev, host_a, 0);
  synHostFree(dev, host_b, 0);
  synHostFree(dev, host_c_ref, 0);
  synHostFree(dev, host_c_auton, 0);
  synHostFree(dev, host_accum_ref, 0);
  synHostFree(dev, reinterpret_cast<void*>(stage_host_ptr), 0);
  synHostFree(dev, reinterpret_cast<void*>(poison_host_ptr), 0);
  CHECK_SCAL(scal_free_buffer(perm_dynamic_buffer));
  synDeviceFree(dev, raw_perm_hbm, 0);
  synDeviceFree(dev, p1_a_dev, 0);
  synDeviceFree(dev, p1_b_dev, 0);
  synDeviceFree(dev, p1_c_auton_dev, 0);
  synDeviceFree(dev, p1_c_ref_dev, 0);
  synDeviceFree(dev, accum_syn0_dev, 0);
  synDeviceFree(dev, accum_syn1_dev, 0);
  synDeviceFree(dev, accum_auton0_dev, 0);
  synDeviceFree(dev, accum_auton1_dev, 0);
  synDeviceFree(dev, cap_a_dev - 256, 0);
  synDeviceFree(dev, cap_b_dev - 512, 0);
  synDeviceFree(dev, cap_c_dev - 768, 0);
  if (ws_dev) synDeviceFree(dev, ws_dev, 0);
  synTensorDestroy(tensor_a);
  synTensorDestroy(tensor_b);
  synTensorDestroy(tensor_c);
  synSectionDestroy(sec_a);
  synSectionDestroy(sec_b);
  synSectionDestroy(sec_c);
  synGraphDestroy(graph);
  synStreamDestroy(stream);
  synDeviceRelease(dev);
  synDestroy();
  return 0;
}
