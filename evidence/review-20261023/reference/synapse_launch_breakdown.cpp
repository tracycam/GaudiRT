#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <atomic>

static inline uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

enum StageId {
  ST_SYN_LAUNCH = 0,
  ST_PREPARE_LAUNCH_UTILS,
  ST_QUEUE_COMPUTE_SCAL_LAUNCH,
  ST_CHECK_FOR_COMPLETION,
  ST_RECIPE_LAUNCHER_CTOR,
  ST_RECIPE_LAUNCHER_LAUNCH,
  ST_VALIDATE_EVENTS,
  ST_PREPARE_FOR_DOWNLOAD,
  ST_HBM_GLBL_GET_ADDR,
  ST_ARC_HBM_GET_ADDR,
  ST_GET_MAPPED_MEMORY_INFO,
  ST_ANALYZE_TENSORS,
  ST_RUN_SIF_PRE_DOWNLOAD,
  ST_DOWNLOAD_TO_DEV,
  ST_MEMCPY_TO_MAPPED,
  ST_PATCH,
  ST_PATCH_ALL,
  ST_PDMA_DOWNLOAD,
  ST_UPDATE_SFG_LONG_SOS,
  ST_SCAL_ENQUEUE,
  ST_ADD_ECB_LIST_AND_SEND,
  ST_SCAL_STREAM_SUBMIT,
  ST_VALIDATE_SECTIONS_INFO,
  ST_LAZY_LOG_ACQUIRE_ITEM,
  ST_COUNT
};

static const char* stage_names[ST_COUNT] = {
  "synLaunchWithExternalEventsExt (Total)",
  "  QueueComputeUtils::prepareLaunch",
  "  QueueComputeScal::launch(LaunchInfo)",
  "    LaunchTracker::checkForCompletion",
  "    RecipeLauncher::RecipeLauncher (ctor)",
  "    RecipeLauncher::launch(LaunchInfo)",
  "      validateEvents",
  "      RecipeLauncher::prepareForDownload",
  "        HbmGlblMemMgr::getAddr",
  "        ArcHbmMemMgr::getAddr",
  "        RecipeLauncher::getMappedMemoryInfo",
  "        RecipeLauncher::analyzeTensors",
  "        RecipeLauncher::runSifPreDownload",
  "      RecipeLauncher::downloadToDev",
  "        MappedMemorySectionsUtils::memcpyToMapped",
  "        RecipeLauncher::patch",
  "          RecipeAddrPatcher::patchAll",
  "        RecipeLauncher::pdmaDownload",
  "      RecipeLauncher::updateSfgLongSos",
  "      RecipeLauncher::scalEnqueue",
  "        RecipeLauncher::addEcbListWithBarrierAndSend",
  "          scal_stream_submit",
  "      RecipeLauncher::validateSectionsInfo",
  "  [Logger] ConcurrentRecentElementsQueue::acquireItem"
};

static std::atomic<uint64_t> stage_calls[ST_COUNT];
static std::atomic<uint64_t> stage_ns[ST_COUNT];
static __thread int in_syn_launch = 0;

struct ScopedTimer {
  int stage;
  uint64_t t0;
  bool active;
  ScopedTimer(int s) : stage(s), active(in_syn_launch > 0 || s == ST_SYN_LAUNCH) {
    if (active) t0 = now_ns();
  }
  ~ScopedTimer() {
    if (active) {
      uint64_t dt = now_ns() - t0;
      stage_calls[stage].fetch_add(1, std::memory_order_relaxed);
      stage_ns[stage].fetch_add(dt, std::memory_order_relaxed);
    }
  }
};


static void* resolve_real(const char* sym) {
  void* p = dlsym(RTLD_NEXT, sym);
  if (p) return p;
  static void* h_syn = nullptr;
  if (!h_syn) h_syn = dlopen("/home/ai/gaudi-experiments/framework-validation-20261010/sdk1241/usr/lib/habanalabs/libSynapse.so", RTLD_LAZY | RTLD_NOLOAD);
  if (!h_syn) h_syn = dlopen("libSynapse.so", RTLD_LAZY);
  if (h_syn) {
    p = dlsym(h_syn, sym);
    if (p) return p;
  }
  static void* h_scal = nullptr;
  if (!h_scal) h_scal = dlopen("/home/ai/gaudi-experiments/framework-validation-20261010/sdk1241/usr/lib/habanalabs/libscal.so", RTLD_LAZY | RTLD_NOLOAD);
  if (h_scal) {
    p = dlsym(h_scal, sym);
    if (p) return p;
  }
  fprintf(stderr, "FATAL: resolve_real failed for %s\n", sym);
  abort();
}

extern "C" {

typedef int (*fn_synLaunchWithExternalEventsExt)(void*, const void*, uint32_t, uint64_t, void*, void*, uint32_t, uint32_t);
int synLaunchWithExternalEventsExt(void* stream, const void* tensors, uint32_t num_tensors, uint64_t workspace, void* recipe, void* events, uint32_t num_events, uint32_t flags) {
  static fn_synLaunchWithExternalEventsExt real_fn = (fn_synLaunchWithExternalEventsExt)resolve_real("synLaunchWithExternalEventsExt");
  in_syn_launch++;
  int ret;
  {
    ScopedTimer t(ST_SYN_LAUNCH);
    ret = real_fn(stream, tensors, num_tensors, workspace, recipe, events, num_events, flags);
  }
  in_syn_launch--;
  return ret;
}

typedef int (*fn_prepareLaunch)(void*, const void*, uint32_t, void*, void*, uint32_t);
int _ZN17QueueComputeUtils13prepareLaunchERK14BasicQueueInfoPK22synLaunchTensorInfoExtjP20InternalRecipeHandleRSt6vectorI15SlotMapItemSptrI21EventWithMappedTensorESaISB_EEj(
    void* a, const void* b, uint32_t c, void* d, void* e, uint32_t f) {
  static fn_prepareLaunch real_fn = (fn_prepareLaunch)resolve_real("_ZN17QueueComputeUtils13prepareLaunchERK14BasicQueueInfoPK22synLaunchTensorInfoExtjP20InternalRecipeHandleRSt6vectorI15SlotMapItemSptrI21EventWithMappedTensorESaISB_EEj");
  ScopedTimer t(ST_PREPARE_LAUNCH_UTILS);
  if (getenv("BYPASS_PREPARE_LAUNCH_UTILS") && (f & 1) == 0) {
    return 0;
  }
  return real_fn(a, b, c, d, e, f);
}

typedef int (*fn_qcs_launch)(void*, const void*);
int _ZN16QueueComputeScal6launchERK10LaunchInfo(void* self, const void* info) {
  static fn_qcs_launch real_fn = (fn_qcs_launch)resolve_real("_ZN16QueueComputeScal6launchERK10LaunchInfo");
  ScopedTimer t(ST_QUEUE_COMPUTE_SCAL_LAUNCH);
  return real_fn(self, info);
}

typedef uint64_t (*fn_check_comp)(void*, uint64_t*, uint64_t*);
uint64_t _ZN13LaunchTracker18checkForCompletionERmS0_(void* self, uint64_t* a, uint64_t* b) {
  static fn_check_comp real_fn = (fn_check_comp)resolve_real("_ZN13LaunchTracker18checkForCompletionERmS0_");
  ScopedTimer t(ST_CHECK_FOR_COMPLETION);
  return real_fn(self, a, b);
}

typedef void (*fn_rl_ctor)(void*, void*, const void*, void*, void*, void*, const void*, void*, uint64_t, uint8_t);
void _ZN14RecipeLauncherC1EP26ScalStreamComputeInterfacePK24ComputeCompoundResourcesR23DevMemoryAllocInterfaceR7MemMgrsR22LaunchTrackerInterfacePK20InternalRecipeHandleP13DynamicRecipemh(
    void* self, void* a, const void* b, void* c, void* d, void* e, const void* f, void* g, uint64_t h, uint8_t i) {
  static fn_rl_ctor real_fn = (fn_rl_ctor)resolve_real("_ZN14RecipeLauncherC1EP26ScalStreamComputeInterfacePK24ComputeCompoundResourcesR23DevMemoryAllocInterfaceR7MemMgrsR22LaunchTrackerInterfacePK20InternalRecipeHandleP13DynamicRecipemh");
  ScopedTimer t(ST_RECIPE_LAUNCHER_CTOR);
  real_fn(self, a, b, c, d, e, f, g, h, i);
}

typedef int (*fn_rl_launch)(void*, const void*);
int _ZN14RecipeLauncher6launchERK10LaunchInfo(void* self, const void* info) {
  static fn_rl_launch real_fn = (fn_rl_launch)resolve_real("_ZN14RecipeLauncher6launchERK10LaunchInfo");
  ScopedTimer t(ST_RECIPE_LAUNCHER_LAUNCH);
  return real_fn(self, info);
}

typedef int (*fn_validate_events)(void*);
int _Z14validateEventsRSt6vectorI15SlotMapItemSptrI21EventWithMappedTensorESaIS2_EE(void* events) {
  static fn_validate_events real_fn = (fn_validate_events)resolve_real("_Z14validateEventsRSt6vectorI15SlotMapItemSptrI21EventWithMappedTensorESaIS2_EE");
  ScopedTimer t(ST_VALIDATE_EVENTS);
  return real_fn(events);
}

int _ZN14RecipeLauncher18prepareForDownloadERK10LaunchInfo(void* self, const void* info) {
  static fn_rl_launch real_fn = (fn_rl_launch)resolve_real("_ZN14RecipeLauncher18prepareForDownloadERK10LaunchInfo");
  ScopedTimer t(ST_PREPARE_FOR_DOWNLOAD);
  return real_fn(self, info);
}

typedef void* (*fn_hbm_glbl_get)(void*, void*, uint64_t, uint64_t, uint64_t*);
void* _ZN13HbmGlblMemMgr7getAddrEmmRm(void* sret, void* self, uint64_t a, uint64_t b, uint64_t* out) {
  static fn_hbm_glbl_get real_fn = (fn_hbm_glbl_get)resolve_real("_ZN13HbmGlblMemMgr7getAddrEmmRm");
  ScopedTimer t(ST_HBM_GLBL_GET_ADDR);
  return real_fn(sret, self, a, b, out);
}

typedef void* (*fn_arc_hbm_get)(void*, void*, uint64_t, uint64_t, uint64_t*, uint32_t*);
void* _ZN12ArcHbmMemMgr7getAddrEmmRmRj(void* sret, void* self, uint64_t a, uint64_t b, uint64_t* out, uint32_t* idx) {
  static fn_arc_hbm_get real_fn = (fn_arc_hbm_get)resolve_real("_ZN12ArcHbmMemMgr7getAddrEmmRmRj");
  ScopedTimer t(ST_ARC_HBM_GET_ADDR);
  return real_fn(sret, self, a, b, out, idx);
}

typedef uint64_t (*fn_get_mapped)(void*);
uint64_t _ZN14RecipeLauncher19getMappedMemoryInfoEv(void* self) {
  static fn_get_mapped real_fn = (fn_get_mapped)resolve_real("_ZN14RecipeLauncher19getMappedMemoryInfoEv");
  ScopedTimer t(ST_GET_MAPPED_MEMORY_INFO);
  return real_fn(self);
}

int _ZN14RecipeLauncher14analyzeTensorsERK10LaunchInfo(void* self, const void* info) {
  static fn_rl_launch real_fn = (fn_rl_launch)resolve_real("_ZN14RecipeLauncher14analyzeTensorsERK10LaunchInfo");
  ScopedTimer t(ST_ANALYZE_TENSORS);
  if (getenv("BYPASS_ANALYZE_TENSORS")) return 0;
  return real_fn(self, info);
}

int _ZN14RecipeLauncher17runSifPreDownloadERK10LaunchInfo(void* self, const void* info) {
  static fn_rl_launch real_fn = (fn_rl_launch)resolve_real("_ZN14RecipeLauncher17runSifPreDownloadERK10LaunchInfo");
  ScopedTimer t(ST_RUN_SIF_PRE_DOWNLOAD);
  return real_fn(self, info);
}

int _ZN14RecipeLauncher13downloadToDevERK10LaunchInfo(void* self, const void* info) {
  static fn_rl_launch real_fn = (fn_rl_launch)resolve_real("_ZN14RecipeLauncher13downloadToDevERK10LaunchInfo");
  ScopedTimer t(ST_DOWNLOAD_TO_DEV);
  return real_fn(self, info);
}

typedef uint64_t (*fn_memcpy_mapped)(const void*, const void*, uint8_t, uint8_t, uint8_t);
uint64_t _ZN25MappedMemorySectionsUtils14memcpyToMappedERK18MemorySectionsScalRKN13llvm_vecsmall11SmallVectorI19RecipeSingleSectionLj14EEEbbb(
    const void* a, const void* b, uint8_t c, uint8_t d, uint8_t e) {
  static fn_memcpy_mapped real_fn = (fn_memcpy_mapped)resolve_real("_ZN25MappedMemorySectionsUtils14memcpyToMappedERK18MemorySectionsScalRKN13llvm_vecsmall11SmallVectorI19RecipeSingleSectionLj14EEEbbb");
  ScopedTimer t(ST_MEMCPY_TO_MAPPED);
  return real_fn(a, b, c, d, e);
}

int _ZN14RecipeLauncher5patchERK10LaunchInfo(void* self, const void* info) {
  static fn_rl_launch real_fn = (fn_rl_launch)resolve_real("_ZN14RecipeLauncher5patchERK10LaunchInfo");
  ScopedTimer t(ST_PATCH);
  return real_fn(self, info);
}

typedef int (*fn_patch_all)(const void*, const uint64_t*, void*);
int _ZNK17RecipeAddrPatcher8patchAllEPKmRSt6vectorI16MemoryMappedAddrSaIS3_EE(const void* self, const uint64_t* a, void* b) {
  static fn_patch_all real_fn = (fn_patch_all)resolve_real("_ZNK17RecipeAddrPatcher8patchAllEPKmRSt6vectorI16MemoryMappedAddrSaIS3_EE");
  ScopedTimer t(ST_PATCH_ALL);
  return real_fn(self, a, b);
}

int _ZN14RecipeLauncher12pdmaDownloadERK18MemorySectionsScal(void* self, const void* secs) {
  static fn_rl_launch real_fn = (fn_rl_launch)resolve_real("_ZN14RecipeLauncher12pdmaDownloadERK18MemorySectionsScal");
  ScopedTimer t(ST_PDMA_DOWNLOAD);
  return real_fn(self, secs);
}

typedef int (*fn_update_sfg)(void*, uint64_t, void*);
int _ZN14RecipeLauncher16updateSfgLongSosEmRSt6vectorI15SlotMapItemSptrI21EventWithMappedTensorESaIS3_EE(void* self, uint64_t a, void* b) {
  static fn_update_sfg real_fn = (fn_update_sfg)resolve_real("_ZN14RecipeLauncher16updateSfgLongSosEmRSt6vectorI15SlotMapItemSptrI21EventWithMappedTensorESaIS3_EE");
  ScopedTimer t(ST_UPDATE_SFG_LONG_SOS);
  return real_fn(self, a, b);
}

typedef int (*fn_scal_enq)(void*, uint64_t);
int _ZN14RecipeLauncher11scalEnqueueEm(void* self, uint64_t a) {
  static fn_scal_enq real_fn = (fn_scal_enq)resolve_real("_ZN14RecipeLauncher11scalEnqueueEm");
  ScopedTimer t(ST_SCAL_ENQUEUE);
  return real_fn(self, a);
}

int _ZN14RecipeLauncher28addEcbListWithBarrierAndSendEm(void* self, uint64_t a) {
  static fn_scal_enq real_fn = (fn_scal_enq)resolve_real("_ZN14RecipeLauncher28addEcbListWithBarrierAndSendEm");
  ScopedTimer t(ST_ADD_ECB_LIST_AND_SEND);
  return real_fn(self, a);
}

typedef int (*fn_scal_submit)(void*, const void*, unsigned);
int scal_stream_submit(void* stream, const void* pi, unsigned size) {
  static fn_scal_submit real_fn = (fn_scal_submit)resolve_real("scal_stream_submit");
  ScopedTimer t(ST_SCAL_STREAM_SUBMIT);
  return real_fn(stream, pi, size);
}

int _ZN14RecipeLauncher20validateSectionsInfoERK10LaunchInfo(void* self, const void* info) {
  static fn_rl_launch real_fn = (fn_rl_launch)resolve_real("_ZN14RecipeLauncher20validateSectionsInfoERK10LaunchInfo");
  ScopedTimer t(ST_VALIDATE_SECTIONS_INFO);
  if (getenv("BYPASS_VALIDATE_SECTIONS")) return 0;
  return real_fn(self, info);
}

void reset_synapse_launch_stats() {
  for (int i = 0; i < ST_COUNT; ++i) {
    stage_calls[i].store(0);
    stage_ns[i].store(0);
  }
}

__attribute__((destructor))
static void dump_stats() {
  uint64_t total_launches = stage_calls[ST_SYN_LAUNCH].load();
  if (total_launches == 0) return;
  fprintf(stderr, "\n================ SYNAPSE LAUNCH BREAKDOWN (total launches = %lu) ================\n", total_launches);
  fprintf(stderr, "%-54s | %10s | %10s | %10s | %7s\n", "Stage", "Calls", "Total(ms)", "PerCall(us)", "%Launch");
  fprintf(stderr, "-----------------------------------------------------------------------------------------------------\n");
  uint64_t total_ns = stage_ns[ST_SYN_LAUNCH].load();
  for (int i = 0; i < ST_COUNT; ++i) {
    uint64_t c = stage_calls[i].load();
    uint64_t ns = stage_ns[i].load();
    double ms = ns / 1e6;
    double us_per_call = c ? (ns / 1e3 / c) : 0.0;
    double pct = total_ns ? (100.0 * ns / total_ns) : 0.0;
    fprintf(stderr, "%-54s | %10lu | %10.2f | %10.3f | %6.1f%%\n",
            stage_names[i], c, ms, us_per_call, pct);
  }
  fprintf(stderr, "=====================================================================================================\n");
}

}
