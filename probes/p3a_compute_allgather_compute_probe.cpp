// SPDX-License-Identifier: Apache-2.0
#include "autonomous_allgather_runtime.hpp"

#include <array>
#include <limits>
#include <sstream>

extern "C" {
void p0_hook_set_enabled(int en);
void p0_hook_set_phase(const char* phase_name);
int p1_extract_recipe_blobs(const char* capture_phase, uint8_t* out_exec_blob,
                            uint32_t* inout_exec_size, uint8_t* out_patch_blob,
                            uint32_t* inout_patch_size, uint8_t* out_dynamic_blob,
                            uint32_t* inout_dynamic_size, uint64_t* out_orig_base0,
                            uint64_t* out_orig_base1, uint64_t* out_orig_base2,
                            uint64_t* out_orig_base7);
int p3a_extract_compute_phase(const char* capture_phase, uint8_t* out_bytes,
                              uint32_t* inout_bytes,
                              uint32_t* out_submission_ends,
                              uint32_t* inout_submission_count);
int p3a_compute_current_pi(uint32_t* out_pi);
}

#define P3A_CHECK_SYN(call)                                                    \
  do {                                                                         \
    synStatus _rc = (call);                                                    \
    if (_rc != synSuccess) {                                                   \
      std::fprintf(stderr, "[P3A_FATAL] %s rc=%d at %s:%d\n", #call,          \
                   static_cast<int>(_rc), __FILE__, __LINE__);                 \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define P3A_CHECK_SCAL(call)                                                   \
  do {                                                                         \
    int _rc = (call);                                                          \
    if (_rc != SCAL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[P3A_FATAL] %s rc=%d at %s:%d\n", #call, _rc,     \
                   __FILE__, __LINE__);                                        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

constexpr uint32_t kComputeCcbBytes = 0x20000;
constexpr uint32_t kRecordBytes = kRecvBytes + 2 * kGuardBytes;
constexpr uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
using Record = std::array<uint8_t, kRecordBytes>;

void p3a_check(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "[P3A_FATAL] %s\n", message);
    std::exit(1);
  }
}

void p3a_check_hook(int rc, const char* what) {
  if (rc != 0) {
    std::fprintf(stderr, "[P3A_FATAL] %s rc=%d\n", what, rc);
    std::exit(1);
  }
}

std::string hex64(uint64_t value) {
  std::ostringstream out;
  out << std::hex << std::setw(16) << std::setfill('0') << value;
  return out.str();
}

uint64_t fnv(uint64_t hash, const void* data, size_t bytes) {
  const auto* ptr = static_cast<const uint8_t*>(data);
  for (size_t i = 0; i < bytes; ++i) hash = (hash ^ ptr[i]) * kFnvPrime;
  return hash;
}

uint64_t record_hash(const Record& record) {
  return fnv(kFnvOffset, record.data(), record.size());
}

uint64_t chain_hash(uint64_t chain, uint32_t step, const Record& record) {
  chain = fnv(chain, &step, sizeof(step));
  return fnv(chain, record.data(), record.size());
}

float input_a(uint32_t step, uint32_t rank, uint32_t element) {
  return static_cast<float>((step * 8 + rank) * 16 + element % 16);
}

float input_b(uint32_t step, uint32_t element) {
  return static_cast<float>(step % 11 + element % 3);
}

float input_b_large(uint32_t step, uint32_t element) {
  return static_cast<float>(step % 13 + element % 5);
}

float expected_output(uint32_t step, uint32_t source, uint32_t element) {
  const uint32_t index = source * kCount + element;
  return input_a(step, source, element) + input_b(step, element) +
         input_b_large(step, index);
}

class AddRecipe {
 public:
  AddRecipe(synDeviceId device, uint32_t count, const char* graph_name)
      : device_(device), count_(count) {
    P3A_CHECK_SYN(synGraphCreate(&graph_, synDeviceGaudi2));
    for (uint32_t i = 0; i < 3; ++i) {
      P3A_CHECK_SYN(synSectionCreate(&sections_[i], 0, graph_));
      P3A_CHECK_SYN(synSectionSetPersistent(sections_[i], true));
      synTensorDescriptor desc{};
      desc.m_name = names_[i];
      desc.m_dataType = syn_type_single;
      desc.m_dims = 1;
      desc.m_sizes[0] = count_;
      desc.m_minSizes[0] = count_;
      P3A_CHECK_SYN(synTensorCreate(&tensors_[i], &desc, sections_[i], 0));
    }
    synTensor inputs[2] = {tensors_[0], tensors_[1]};
    synTensor outputs[1] = {tensors_[2]};
    P3A_CHECK_SYN(synNodeCreate(graph_, inputs, outputs, 2, 1, nullptr, 0,
                                "add_fwd_f32", "p3a_add_node", nullptr, nullptr));
    P3A_CHECK_SYN(synGraphCompile(&recipe_, graph_, graph_name, nullptr));
    P3A_CHECK_SYN(synTensorRetrieveIds(recipe_, names_, tensor_ids_, 3));
    P3A_CHECK_SYN(synWorkspaceGetSize(&workspace_size_, recipe_));
    if (workspace_size_) {
      P3A_CHECK_SYN(synDeviceMalloc(device_, workspace_size_, 0, 0, &workspace_));
    }
  }

  ~AddRecipe() {
    if (recipe_) synRecipeDestroy(recipe_);
    if (graph_) synGraphDestroy(graph_);
    for (synTensor& tensor : tensors_) {
      if (tensor) synTensorDestroy(tensor);
    }
    for (synSectionHandle& section : sections_) {
      if (section) synSectionDestroy(section);
    }
    if (workspace_) synDeviceFree(device_, workspace_, 0);
  }

  AddRecipe(const AddRecipe&) = delete;
  AddRecipe& operator=(const AddRecipe&) = delete;

  void launch(synStreamHandle stream, uint64_t a, uint64_t b, uint64_t c) {
    const uint64_t addresses[3] = {a, b, c};
    synLaunchTensorInfoExt bindings[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
      bindings[i].tensorName = names_[i];
      bindings[i].pTensorAddress = addresses[i];
      bindings[i].tensorType = DATA_TENSOR;
      bindings[i].tensorSize[0] = count_;
      bindings[i].tensorId = tensor_ids_[i];
    }
    P3A_CHECK_SYN(synLaunchWithExternalEventsExt(
        stream, bindings, 3, workspace_, recipe_, nullptr, 0, 0));
  }

 private:
  synDeviceId device_ = SYN_INVALID_DEVICE_ID;
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

class P3aBufferSet {
 public:
  explicit P3aBufferSet(synDeviceId device) : device_(device) {
    P3A_CHECK_SYN(synDeviceMalloc(device_, kSliceBytes, 0, 0, &a_));
    P3A_CHECK_SYN(synDeviceMalloc(device_, kSliceBytes, 0, 0, &b_));
    P3A_CHECK_SYN(synDeviceMalloc(device_, kRecvBytes, 0, 0, &b_large_));
    P3A_CHECK_SYN(synDeviceMalloc(device_, kSliceBytes, 0, 0, &send_));
    P3A_CHECK_SYN(synDeviceMalloc(device_, kRecvRegionBytes, 0, 0, &recv_base_));
    P3A_CHECK_SYN(synDeviceMalloc(device_, kRecvBytes, 0, 0, &out_));
    recv_ = recv_base_ + kGuardBytes;
    P3A_CHECK_SYN(synHostMalloc(device_, kSliceBytes, 0,
                                reinterpret_cast<void**>(&host_a_)));
    P3A_CHECK_SYN(synHostMalloc(device_, kSliceBytes, 0,
                                reinterpret_cast<void**>(&host_b_)));
    P3A_CHECK_SYN(synHostMalloc(device_, kRecvBytes, 0,
                                reinterpret_cast<void**>(&host_b_large_)));
    P3A_CHECK_SYN(synHostMalloc(device_, kSliceBytes, 0, &host_send_));
    P3A_CHECK_SYN(synHostMalloc(device_, kRecvRegionBytes, 0, &host_recv_));
    P3A_CHECK_SYN(synHostMalloc(device_, kRecvBytes, 0, &host_out_));
  }

  ~P3aBufferSet() {
    if (host_a_) synHostFree(device_, host_a_, 0);
    if (host_b_) synHostFree(device_, host_b_, 0);
    if (host_b_large_) synHostFree(device_, host_b_large_, 0);
    if (host_send_) synHostFree(device_, host_send_, 0);
    if (host_recv_) synHostFree(device_, host_recv_, 0);
    if (host_out_) synHostFree(device_, host_out_, 0);
    if (a_) synDeviceFree(device_, a_, 0);
    if (b_) synDeviceFree(device_, b_, 0);
    if (b_large_) synDeviceFree(device_, b_large_, 0);
    if (send_) synDeviceFree(device_, send_, 0);
    if (recv_base_) synDeviceFree(device_, recv_base_, 0);
    if (out_) synDeviceFree(device_, out_, 0);
  }

  P3aBufferSet(const P3aBufferSet&) = delete;
  P3aBufferSet& operator=(const P3aBufferSet&) = delete;

  void initialize(synStreamHandle stream, uint32_t rank, uint32_t step) {
    for (uint32_t i = 0; i < kCount; ++i) {
      host_a_[i] = input_a(step, rank, i);
      host_b_[i] = input_b(step, i);
    }
    for (uint32_t i = 0; i < kCount * kRanks; ++i) {
      host_b_large_[i] = input_b_large(step, i);
    }
    std::memset(host_send_, 0x7E, kSliceBytes);
    auto* recv = static_cast<uint8_t*>(host_recv_);
    std::memset(recv, 0xA5, kGuardBytes);
    std::memset(recv + kGuardBytes, 0x5A, kRecvBytes);
    std::memset(recv + kGuardBytes + kRecvBytes, 0xA5, kGuardBytes);
    std::memset(host_out_, 0xC3, kRecvBytes);
    P3A_CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_a_),
                                  kSliceBytes, a_, HOST_TO_DRAM));
    P3A_CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_b_),
                                  kSliceBytes, b_, HOST_TO_DRAM));
    P3A_CHECK_SYN(synMemCopyAsync(stream,
                                  reinterpret_cast<uint64_t>(host_b_large_),
                                  kRecvBytes, b_large_, HOST_TO_DRAM));
    P3A_CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_send_),
                                  kSliceBytes, send_, HOST_TO_DRAM));
    P3A_CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_recv_),
                                  kRecvRegionBytes, recv_base_, HOST_TO_DRAM));
    P3A_CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host_out_),
                                  kRecvBytes, out_, HOST_TO_DRAM));
  }

  Record readback(synStreamHandle stream) {
    P3A_CHECK_SYN(synMemCopyAsync(stream, out_, kRecvBytes,
                                  reinterpret_cast<uint64_t>(host_out_),
                                  DRAM_TO_HOST));
    P3A_CHECK_SYN(synMemCopyAsync(stream, recv_base_, kRecvRegionBytes,
                                  reinterpret_cast<uint64_t>(host_recv_),
                                  DRAM_TO_HOST));
    P3A_CHECK_SYN(synStreamSynchronize(stream));
    Record record{};
    std::memcpy(record.data(), host_out_, kRecvBytes);
    const auto* recv = static_cast<const uint8_t*>(host_recv_);
    std::memcpy(record.data() + kRecvBytes, recv, kGuardBytes);
    std::memcpy(record.data() + kRecvBytes + kGuardBytes,
                recv + kGuardBytes + kRecvBytes, kGuardBytes);
    return record;
  }

  uint64_t a() const { return a_; }
  uint64_t b() const { return b_; }
  uint64_t b_large() const { return b_large_; }
  uint64_t send() const { return send_; }
  uint64_t recv() const { return recv_; }
  uint64_t out() const { return out_; }

 private:
  synDeviceId device_ = SYN_INVALID_DEVICE_ID;
  uint64_t a_ = 0;
  uint64_t b_ = 0;
  uint64_t b_large_ = 0;
  uint64_t send_ = 0;
  uint64_t recv_base_ = 0;
  uint64_t recv_ = 0;
  uint64_t out_ = 0;
  float* host_a_ = nullptr;
  float* host_b_ = nullptr;
  float* host_b_large_ = nullptr;
  void* host_send_ = nullptr;
  void* host_recv_ = nullptr;
  void* host_out_ = nullptr;
};

void verify_expected(const Record& record, uint32_t step, uint32_t rank) {
  for (uint32_t i = 0; i < kGuardBytes; ++i) {
    p3a_check(record[kRecvBytes + i] == 0xA5 &&
                  record[kRecvBytes + kGuardBytes + i] == 0xA5,
              "native receive guard mismatch");
  }
  for (uint32_t source = 0; source < kRanks; ++source) {
    for (uint32_t element = 0; element < kCount; ++element) {
      const uint32_t index = source * kCount + element;
      float actual = 0;
      std::memcpy(&actual, record.data() + index * sizeof(float), sizeof(float));
      const float expected = expected_output(step, source, element);
      if (actual != expected) {
        std::fprintf(stderr,
                     "[P3A_FATAL] native mismatch rank=%u step=%u source=%u "
                     "element=%u actual=%f expected=%f\n",
                     rank, step, source, element, actual, expected);
        std::exit(1);
      }
    }
  }
}

void compare_reference(const Record& expected, const Record& actual,
                       uint32_t step, uint32_t rank) {
  if (std::memcmp(expected.data(), actual.data(), kRecordBytes) == 0) return;
  size_t offset = 0;
  while (offset < kRecordBytes && expected[offset] == actual[offset]) ++offset;
  std::fprintf(stderr,
               "[P3A_FATAL] reference memcmp rank=%u step=%u offset=0x%zx "
               "expected=0x%02x actual=0x%02x output_bytes=%u guard_bytes=%u\n",
               rank, step, offset, expected[offset], actual[offset], kRecvBytes,
               2 * kGuardBytes);
  std::exit(1);
}

struct ManifestEntry {
  uint32_t step = 0;
  uint64_t record = 0;
  uint64_t chain = 0;
};

class ReferenceWriter {
 public:
  ReferenceWriter(const std::string& directory, uint32_t rank) {
    const std::string base = directory + "/rank" + std::to_string(rank);
    data_.open(base + "_reference.bin", std::ios::binary | std::ios::trunc);
    manifest_.open(base + "_manifest.tsv", std::ios::out | std::ios::trunc);
    p3a_check(data_.is_open() && manifest_.is_open(), "reference output open failed");
    manifest_ << "# step record_fnv1a64 chain_fnv1a64\n";
  }

  void write(uint32_t step, const Record& record, uint64_t& chain) {
    const uint64_t hash = record_hash(record);
    chain = chain_hash(chain, step, record);
    data_.write(reinterpret_cast<const char*>(record.data()), record.size());
    manifest_ << step << ' ' << hex64(hash) << ' ' << hex64(chain) << '\n';
    data_.flush();
    manifest_.flush();
    p3a_check(data_.good() && manifest_.good(), "reference output write failed");
  }

 private:
  std::ofstream data_;
  std::ofstream manifest_;
};

class ReferenceReader {
 public:
  ReferenceReader(const std::string& directory, uint32_t rank, uint32_t steps) {
    const std::string base = directory + "/rank" + std::to_string(rank);
    data_.open(base + "_reference.bin", std::ios::binary);
    std::ifstream manifest(base + "_manifest.tsv");
    p3a_check(data_.is_open() && manifest.is_open(), "reference input open failed");
    std::string line;
    std::getline(manifest, line);
    ManifestEntry entry;
    std::string record;
    std::string chain;
    while (manifest >> entry.step >> record >> chain) {
      entry.record = std::stoull(record, nullptr, 16);
      entry.chain = std::stoull(chain, nullptr, 16);
      entries_.push_back(entry);
    }
    p3a_check(entries_.size() >= steps, "reference manifest is too short");
    for (uint32_t i = 0; i < steps; ++i) {
      p3a_check(entries_[i].step == i, "reference manifest step mismatch");
    }
  }

  Record read(uint32_t step) {
    p3a_check(step < entries_.size(), "reference read past manifest");
    Record record{};
    data_.clear();
    data_.seekg(static_cast<std::streamoff>(step) * kRecordBytes);
    data_.read(reinterpret_cast<char*>(record.data()), record.size());
    p3a_check(data_.gcount() == static_cast<std::streamsize>(record.size()),
              "short reference record");
    return record;
  }

  const ManifestEntry& entry(uint32_t step) const { return entries_.at(step); }

 private:
  std::ifstream data_;
  std::vector<ManifestEntry> entries_;
};

void initialize_uid(uint32_t rank, SharedCoord* shm) {
  if (rank == 0) {
    P2A_CHECK_HCCL(hcclGetUniqueId(&shm->uid));
    shm->uid_ready.store(1);
  } else {
    while (!shm->uid_ready.load()) std::this_thread::yield();
  }
}

struct RecipeBlobs {
  std::vector<uint8_t> exec;
  std::vector<uint8_t> patch;
  std::vector<uint8_t> dynamic;
  uint64_t base0 = 0;
  uint64_t base1 = 0;
  uint64_t base2 = 0;
  uint64_t base7 = 0;
};

RecipeBlobs extract_blobs(const char* phase) {
  RecipeBlobs blobs;
  blobs.exec.resize(4096);
  blobs.patch.resize(4096);
  blobs.dynamic.resize(65536);
  uint32_t exec_size = blobs.exec.size();
  uint32_t patch_size = blobs.patch.size();
  uint32_t dynamic_size = blobs.dynamic.size();
  p3a_check_hook(p1_extract_recipe_blobs(
                     phase, blobs.exec.data(), &exec_size, blobs.patch.data(),
                     &patch_size, blobs.dynamic.data(), &dynamic_size,
                     &blobs.base0, &blobs.base1, &blobs.base2, &blobs.base7),
                 "p1_extract_recipe_blobs");
  blobs.exec.resize(exec_size);
  blobs.patch.resize(patch_size);
  blobs.dynamic.resize(dynamic_size);
  return blobs;
}

uint32_t read32(const std::vector<uint8_t>& bytes, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, bytes.data() + offset, sizeof(value));
  return value;
}

void write32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

void write64(std::vector<uint8_t>& bytes, size_t offset, uint64_t value) {
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

int unique_address_offset(const std::vector<uint8_t>& bytes, uint64_t address) {
  int result = -1;
  for (size_t offset = 0; offset + sizeof(uint64_t) <= bytes.size(); offset += 4) {
    uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, sizeof(value));
    if (value != address) continue;
    if (result >= 0) return -2;
    result = static_cast<int>(offset);
  }
  return result;
}

struct NativeComputeCapture {
  std::vector<std::vector<uint8_t>> submissions;
  std::vector<uint8_t> external_wait;
  uint64_t original_dynamic = 0;
};

NativeComputeCapture extract_compute(const char* phase, bool need_external_wait) {
  std::vector<uint8_t> bytes(16384);
  std::vector<uint32_t> ends(64);
  uint32_t byte_count = bytes.size();
  uint32_t submission_count = ends.size();
  p3a_check_hook(p3a_extract_compute_phase(phase, bytes.data(), &byte_count,
                                           ends.data(), &submission_count),
                 "p3a_extract_compute_phase");
  p3a_check(submission_count >= 2 && submission_count <= ends.size(),
            "invalid native compute submission count");

  NativeComputeCapture result;
  size_t packet_count = 0;
  size_t payload_bytes = 0;
  uint32_t submission_begin = 0;
  for (uint32_t submission = 0; submission < submission_count; ++submission) {
    const uint32_t submission_end = ends[submission];
    p3a_check(submission_end > submission_begin && submission_end <= byte_count,
              "invalid native compute submission boundary");
    std::vector<uint8_t> filtered;
    uint32_t offset = submission_begin;
    while (offset < submission_end) {
      const uint8_t* ptr = bytes.data() + offset;
      const size_t size = gaudirt::scal::compute_packet_size(
          ptr, submission_end - offset);
      p3a_check(size != 0, "invalid captured compute packet");
      uint32_t dw0 = 0;
      std::memcpy(&dw0, ptr, sizeof(dw0));
      const uint32_t opcode = dw0 & 0x1F;
      if (opcode == SCHED_COMPUTE_ARC_CMD_WAIT_FOR_EXT_SIGNAL) {
        p3a_check(size == sizeof(sched_arc_cmd_wait_for_ext_signal_t) &&
                      size == 16,
                  "unexpected external wait size");
        const auto* wait =
            reinterpret_cast<const sched_arc_cmd_wait_for_ext_signal_t*>(ptr);
        if (need_external_wait && wait->sm_id == 3 && wait->sob_id == 1440) {
          p3a_check(result.external_wait.empty() && size == submission_end -
                                                           submission_begin,
                    "post external wait is not an independent submission");
          result.external_wait.assign(ptr, ptr + size);
        }
      } else if (opcode == SCHED_COMPUTE_ARC_CMD_FENCE_WAIT) {
        const auto* fence =
            reinterpret_cast<const sched_arc_cmd_fence_wait_t*>(ptr);
        if (fence->fence_id != 14) {
          filtered.insert(filtered.end(), ptr, ptr + size);
          ++packet_count;
          payload_bytes += size;
        }
      } else if (opcode != SCHED_COMPUTE_ARC_CMD_NOP) {
        filtered.insert(filtered.end(), ptr, ptr + size);
        ++packet_count;
        payload_bytes += size;
      }
      if (opcode == SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2) {
        const auto* update =
            reinterpret_cast<const sched_arc_cmd_update_recipe_base_v2_t*>(ptr);
        p3a_check(update->num_recipe_addrs == 4 &&
                      update->recipe_base_index0 == 0 &&
                      update->recipe_base_index1 == 1 &&
                      update->recipe_base_index2 == 2 &&
                      update->recipe_base_index3 == 7,
                  "unsupported recipe base layout");
      }
      offset += size;
    }
    if (!filtered.empty()) result.submissions.push_back(std::move(filtered));
    submission_begin = submission_end;
  }
  p3a_check(packet_count == 6 && payload_bytes == 80 &&
                !result.submissions.empty(),
            "unexpected warm recipe publication shape");
  p3a_check(!need_external_wait || result.external_wait.size() == 16,
            "missing post network external wait");
  return result;
}

struct RelocatedRecipe {
  NativeComputeCapture capture;
  RecipeBlobs blobs;
  std::array<int, 3> tensor_offsets{};
  std::vector<uint32_t> kernel_offsets;
  uint64_t raw_hbm = 0;
  uint64_t exec_hbm = 0;
  uint64_t dynamic_hbm = 0;
  uint64_t nop_hbm = 0;
  std::array<uint64_t, 2> patch_hbm{};
  std::array<std::vector<uint8_t>, 2> patches;
  scal_buffer_handle_t dynamic_buffer = nullptr;
  void* staging = nullptr;
};

RelocatedRecipe prepare_recipe(const char* phase,
                               const NativeComputeCapture& capture,
                               RankRuntime& rt,
                               const std::array<P3aBufferSet*, 2>& sets,
                               bool pre, std::ofstream& log) {
  RelocatedRecipe result;
  result.capture = capture;
  result.blobs = extract_blobs(phase);
  const RecipeBlobs cold =
      extract_blobs(pre ? "P3A_COLD_PRE" : "P3A_COLD_POST");
  p3a_check(result.blobs.exec.size() <= 0x200 &&
                result.blobs.patch.size() <= 0x200 &&
                result.blobs.dynamic.size() <= 0x9000 &&
                result.blobs.base7 >= result.blobs.base2 &&
                cold.exec == result.blobs.exec &&
                cold.dynamic == result.blobs.dynamic &&
                cold.patch.size() == result.blobs.patch.size(),
            "recipe staging capacity or cold/warm invariant mismatch");

  const uint64_t warm_addresses[3] = {
      pre ? sets[1]->a() : sets[1]->recv(),
      pre ? sets[1]->b() : sets[1]->b_large(),
      pre ? sets[1]->send() : sets[1]->out()};
  for (uint32_t i = 0; i < 3; ++i) {
    result.tensor_offsets[i] =
        unique_address_offset(result.blobs.patch, warm_addresses[i]);
    p3a_check(result.tensor_offsets[i] >= 0,
              "recipe tensor patch offset is not unique");
  }
  for (size_t offset = 0; offset + 4 <= result.blobs.patch.size(); offset += 4) {
    if (std::memcmp(cold.patch.data() + offset,
                    result.blobs.patch.data() + offset, 4) == 0 ||
        std::find(result.tensor_offsets.begin(), result.tensor_offsets.end(),
                  static_cast<int>(offset)) != result.tensor_offsets.end()) {
      continue;
    }
    const uint32_t cold_base = static_cast<uint32_t>(cold.base2);
    const uint32_t warm_base = static_cast<uint32_t>(result.blobs.base2);
    const uint32_t cold_address = read32(cold.patch, offset);
    const uint32_t warm_address = read32(result.blobs.patch, offset);
    p3a_check(cold_address >= cold_base && warm_address >= warm_base &&
                  cold_address - cold_base < cold.dynamic.size() &&
                  warm_address - warm_base < result.blobs.dynamic.size() &&
                  cold_address - cold_base == warm_address - warm_base,
              "invalid cold/warm kernel relocation");
    result.kernel_offsets.push_back(offset);
  }
  p3a_check(result.kernel_offsets.size() == 2,
            "unexpected recipe kernel patch offset count");

  P3A_CHECK_SYN(synDeviceMalloc(rt.device(), 65536, 0, 0, &result.raw_hbm));
  result.exec_hbm = (result.raw_hbm + 0x1FFFULL) & ~0x1FFFULL;
  result.patch_hbm[0] = result.exec_hbm + 0x400;
  result.patch_hbm[1] = result.exec_hbm + 0x600;
  scal_pool_handle_t hbm_shared = nullptr;
  P3A_CHECK_SCAL(
      scal_get_pool_handle_by_name(rt.scal(), "hbm_shared", &hbm_shared));
  P3A_CHECK_SCAL(scal_allocate_aligned_buffer(
      hbm_shared, result.blobs.dynamic.size(), 0x2000, &result.dynamic_buffer));
  scal_buffer_info_t dynamic_info{};
  P3A_CHECK_SCAL(scal_buffer_get_info(result.dynamic_buffer, &dynamic_info));
  result.dynamic_hbm = dynamic_info.device_address;
  result.nop_hbm = result.dynamic_hbm +
                   (result.blobs.base7 - result.blobs.base2);
  p3a_check(result.dynamic_hbm && (result.dynamic_hbm & 0x1FFFULL) == 0,
            "invalid recipe dynamic allocation");

  for (uint32_t set = 0; set < 2; ++set) {
    result.patches[set] = result.blobs.patch;
    const uint64_t addresses[3] = {
        pre ? sets[set]->a() : sets[set]->recv(),
        pre ? sets[set]->b() : sets[set]->b_large(),
        pre ? sets[set]->send() : sets[set]->out()};
    for (uint32_t i = 0; i < 3; ++i) {
      write64(result.patches[set], result.tensor_offsets[i], addresses[i]);
    }
    for (const uint32_t offset : result.kernel_offsets) {
      const uint64_t relative =
          read32(result.patches[set], offset) - result.blobs.base2;
      const uint64_t relocated = result.dynamic_hbm + relative;
      p3a_check(relocated <= std::numeric_limits<uint32_t>::max(),
                "recipe kernel relocation exceeds 32 bits");
      write32(result.patches[set], offset, static_cast<uint32_t>(relocated));
    }
  }

  P3A_CHECK_SYN(synHostMalloc(rt.device(), 65536, 0, &result.staging));
  const synStreamHandle stream = rt.verification_stream();
  auto upload = [&](const void* source, size_t bytes, uint64_t destination,
                    size_t staging_offset) {
    auto* staging = static_cast<uint8_t*>(result.staging) + staging_offset;
    std::memcpy(staging, source, bytes);
    P3A_CHECK_SYN(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(staging),
                                  bytes, destination, HOST_TO_DRAM));
  };
  upload(result.blobs.dynamic.data(), result.blobs.dynamic.size(),
         result.dynamic_hbm, 0);
  upload(result.blobs.exec.data(), result.blobs.exec.size(), result.exec_hbm,
         0x9000);
  upload(result.patches[0].data(), result.patches[0].size(),
         result.patch_hbm[0], 0x9200);
  upload(result.patches[1].data(), result.patches[1].size(),
         result.patch_hbm[1], 0x9400);
  P3A_CHECK_SYN(synStreamSynchronize(stream));
  auto verify_upload = [&](const std::vector<uint8_t>& expected,
                           uint64_t source, size_t staging_offset) {
    auto* staging = static_cast<uint8_t*>(result.staging) + staging_offset;
    std::memset(staging, 0, expected.size());
    P3A_CHECK_SYN(synMemCopyAsync(stream, source, expected.size(),
                                  reinterpret_cast<uint64_t>(staging),
                                  DRAM_TO_HOST));
    P3A_CHECK_SYN(synStreamSynchronize(stream));
    p3a_check(std::memcmp(staging, expected.data(), expected.size()) == 0,
              "self-owned recipe upload mismatch");
  };
  verify_upload(result.blobs.dynamic, result.dynamic_hbm, 0);
  verify_upload(result.blobs.exec, result.exec_hbm, 0x9000);
  verify_upload(result.patches[0], result.patch_hbm[0], 0x9200);
  verify_upload(result.patches[1], result.patch_hbm[1], 0x9400);
  log << "[P3A_RECIPE] phase=" << phase << " exec_bytes="
      << result.blobs.exec.size() << " patch_bytes="
      << result.blobs.patch.size() << " dynamic_bytes="
      << result.blobs.dynamic.size() << " publications="
      << result.capture.submissions.size() << " external_wait_bytes="
      << result.capture.external_wait.size() << std::endl;
  return result;
}

class ComputeReplay {
 public:
  ComputeReplay(RankRuntime& rt, std::ofstream& log) : rt_(rt), log_(log) {
    P3A_CHECK_SCAL(scal_get_stream_handle_by_name(rt.scal(), "compute0", &stream_));
    P3A_CHECK_SCAL(scal_stream_get_info(stream_, &info_));
    P3A_CHECK_SCAL(scal_buffer_get_info(info_.control_core_buffer, &buffer_));
    p3a_check(buffer_.host_address && info_.command_alignment == 256 &&
                  info_.submission_alignment == 2,
              "invalid compute0 stream");
    P3A_CHECK_SCAL(scal_get_completion_group_handle_by_name(
        rt.scal(), "compute_completion_queue0", &cg_));
    read_expected();
  }

  void set_pre_recipe(RelocatedRecipe* recipe) { pre_ = recipe; }
  void set_post_recipe(RelocatedRecipe* recipe) { post_ = recipe; }

  void finish_native_capture() {
    P3A_CHECK_SCAL(p3a_compute_current_pi(&next_pi_));
    read_expected();
    log_ << "[P3A_COMPUTE_INIT] next_pi=0x" << std::hex << next_pi_
         << " expected=" << std::dec << expected_ << std::endl;
  }

  uint64_t expected() {
    read_expected();
    return expected_;
  }

  void set_expected(uint64_t target) {
    p3a_check(target <= 0x7FFF, "compute target exceeds 15 bits");
    P3A_CHECK_SCAL(scal_completion_group_set_expected_ctr(cg_, target));
    expected_ = target;
  }

  gaudirt::scal::CcbWriter writer() {
    return gaudirt::scal::CcbWriter(
        static_cast<uint8_t*>(buffer_.host_address), kComputeCcbBytes, next_pi_,
        info_.command_alignment, SCHED_COMPUTE_ARC_CMD_NOP);
  }

  void publish_pre(gaudirt::scal::CcbWriter& writer, uint32_t set) {
    p3a_check(pre_ != nullptr, "missing pre recipe");
    publish_recipe(writer, *pre_, set);
  }

  void publish_external_wait(gaudirt::scal::CcbWriter& writer,
                             const RelocatedRecipe& post, uint64_t target) {
    p3a_check(post.capture.external_wait.size() == 16 && target <= 0x7FFF,
              "invalid post external wait target");
    std::vector<uint8_t> packet = post.capture.external_wait;
    const auto* original =
        reinterpret_cast<const sched_arc_cmd_wait_for_ext_signal_t*>(
            post.capture.external_wait.data());
    auto* wait = reinterpret_cast<sched_arc_cmd_wait_for_ext_signal_t*>(
        packet.data());
    const uint64_t original_target_value = original->target_value;
    const uint8_t original_num_engines = original->num_engine_group_type;
    std::array<uint8_t, 4> original_engines{};
    std::memcpy(original_engines.data(), original->engine_group_type, 4);
    wait->target_value0 = static_cast<uint64_t>(target);
    p3a_check((wait->target_value & ~0x7FFFULL) ==
                      (original_target_value & ~0x7FFFULL) &&
                  wait->num_engine_group_type == original_num_engines &&
                  std::memcmp(wait->engine_group_type, original_engines.data(),
                              4) == 0 &&
                  wait->sm_id == 3 && wait->sob_id == 1440,
              "post external wait fields changed beyond target0");
    auto submit = [&](unsigned pi) {
      return scal_stream_submit(stream_, pi, info_.submission_alignment);
    };
    P3A_CHECK_SCAL(writer.append(packet.data(), packet.size(), true, submit));
    P3A_CHECK_SCAL(writer.submit_now(submit));
  }

  void publish_post(gaudirt::scal::CcbWriter& writer, uint32_t set) {
    p3a_check(post_ != nullptr, "missing post recipe");
    publish_recipe(writer, *post_, set);
  }

  void finish_step(gaudirt::scal::CcbWriter& writer) {
    next_pi_ = writer.pi();
    submit_calls_ += writer.stats().submit_calls;
    boundary_calls_ += writer.stats().boundary_submit_calls;
    nop_bytes_ += writer.stats().nop_bytes;
  }

  void wait(uint64_t target, uint64_t index) {
    const auto start = std::chrono::steady_clock::now();
    uint64_t next_log_seconds = 1;
    for (;;) {
      const int rc = scal_completion_group_wait(cg_, target, 0);
      if (rc == SCAL_SUCCESS) return;
      p3a_check(rc == SCAL_TIMED_OUT, "compute completion wait failed");
      const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
          std::chrono::steady_clock::now() - start);
      if (static_cast<uint64_t>(elapsed.count()) >= next_log_seconds) {
        log_ << "[P3A_WAIT_PROGRESS] index=" << index << " target=" << target
             << " elapsed_s=" << elapsed.count() << std::endl;
        next_log_seconds = elapsed.count() < 5   ? 5
                           : elapsed.count() < 30 ? 30
                                                  : 65;
      }
      p3a_check(elapsed.count() < 65, "compute completion wait timed out");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }

  uint64_t current() {
    scal_completion_group_infoV2_t info{};
    P3A_CHECK_SCAL(scal_completion_group_get_infoV2(cg_, &info));
    return info.current_value;
  }

  uint64_t submit_calls() const { return submit_calls_; }
  uint64_t boundary_calls() const { return boundary_calls_; }
  uint64_t nop_bytes() const { return nop_bytes_; }

 private:
  void read_expected() {
    P3A_CHECK_SCAL(scal_completion_group_get_expected_ctr(cg_, &expected_));
  }

  void publish_recipe(gaudirt::scal::CcbWriter& writer,
                      RelocatedRecipe& recipe, uint32_t set) {
    auto submit = [&](unsigned pi) {
      return scal_stream_submit(stream_, pi, info_.submission_alignment);
    };
    for (const auto& submission : recipe.capture.submissions) {
      size_t offset = 0;
      while (offset < submission.size()) {
        const size_t size = gaudirt::scal::compute_packet_size(
            submission.data() + offset, submission.size() - offset);
        p3a_check(size != 0, "invalid direct recipe packet");
        std::vector<uint8_t> packet(submission.begin() + offset,
                                    submission.begin() + offset + size);
        uint32_t dw0 = 0;
        std::memcpy(&dw0, packet.data(), sizeof(dw0));
        const uint32_t opcode = dw0 & 0x1F;
        if (opcode == SCHED_COMPUTE_ARC_CMD_UPDATE_RECIPE_BASE_V2) {
          auto* update =
              reinterpret_cast<sched_arc_cmd_update_recipe_base_v2_t*>(
                  packet.data());
          update->recipe_base_addr[0] = recipe.patch_hbm[set];
          update->recipe_base_addr[1] = recipe.exec_hbm;
          update->recipe_base_addr[2] = recipe.dynamic_hbm;
          update->recipe_base_addr[3] = recipe.nop_hbm;
        } else if (opcode ==
                   SCHED_COMPUTE_ARC_CMD_DISPATCH_COMPUTE_ECB_LIST_V3) {
          auto* dispatch =
              reinterpret_cast<sched_arc_cmd_dispatch_compute_ecb_list_t*>(
                  packet.data());
          const uint64_t original =
              (rt_.global_hbm_base() & 0xFFFFFFFF00000000ULL) |
              (static_cast<uint64_t>(dispatch->dynamic_ecb_list_addr_256) << 8);
          p3a_check(original >= recipe.blobs.base2,
                    "invalid original dynamic ECB address");
          const uint64_t relocated =
              recipe.dynamic_hbm + (original - recipe.blobs.base2);
          const uint64_t encoded = (relocated & 0xFFFFFFFFULL) >> 8;
          p3a_check((relocated >> 32) == (rt_.global_hbm_base() >> 32) &&
                        (relocated & 0xFF) == 0 && encoded <= 0xFFFFFF,
                    "dynamic ECB relocation cannot be encoded");
          dispatch->dynamic_ecb_list_addr_256 = encoded;
        }
        P3A_CHECK_SCAL(writer.append(packet.data(), packet.size(), true, submit));
        offset += size;
      }
      P3A_CHECK_SCAL(writer.submit_now(submit));
    }
  }

  RankRuntime& rt_;
  std::ofstream& log_;
  scal_stream_handle_t stream_ = nullptr;
  scal_stream_info_t info_{};
  scal_buffer_info_t buffer_{};
  scal_comp_group_handle_t cg_ = nullptr;
  RelocatedRecipe* pre_ = nullptr;
  RelocatedRecipe* post_ = nullptr;
  uint32_t next_pi_ = 0;
  uint64_t expected_ = 0;
  uint64_t submit_calls_ = 0;
  uint64_t boundary_calls_ = 0;
  uint64_t nop_bytes_ = 0;
};

int run_reference_rank(uint32_t rank, SharedCoord* shm,
                       const std::string& data_dir, uint32_t steps) {
  int sense = 0;
  p0_hook_set_enabled(0);
  p0_hook_set_phase("BENCH");
  initialize_uid(rank, shm);
  p3a_check(host_barrier(shm, sense), "reference initial barrier failed");

  gaudirt::runtime::DeviceRuntime::Config cfg;
  cfg.module_id = rank;
  cfg.rank = rank;
  cfg.world_size = kRanks;
  cfg.pin_cpu = static_cast<int>(rank * 4);
  gaudirt::runtime::DeviceRuntime runtime(cfg);
  runtime.init_comm(shm->uid);
  p3a_check(host_barrier(shm, sense), "reference communicator barrier failed");
  AddRecipe pre(runtime.device_id(), kCount, "p3a_reference_pre_add");
  AddRecipe post(runtime.device_id(), kCount * kRanks,
                 "p3a_reference_post_add");
  P3aBufferSet a(runtime.device_id());
  P3aBufferSet b(runtime.device_id());
  ReferenceWriter writer(data_dir, rank);
  uint64_t chain = kFnvOffset;

  for (uint32_t step = 0; step < steps; ++step) {
    a.initialize(runtime.h2d_stream(), rank, step);
    b.initialize(runtime.h2d_stream(), rank, step);
    P3A_CHECK_SYN(synStreamSynchronize(runtime.h2d_stream()));
    P3aBufferSet& set = (step & 1) ? b : a;
    pre.launch(runtime.compute_stream(), set.a(), set.b(), set.send());
    P2A_CHECK_HCCL(hcclAllGather(
        reinterpret_cast<const void*>(set.send()),
        reinterpret_cast<void*>(set.recv()), kCount, hcclFloat32,
        runtime.comm(), runtime.compute_stream()));
    post.launch(runtime.compute_stream(), set.recv(), set.b_large(), set.out());
    P3A_CHECK_SYN(synStreamSynchronize(runtime.compute_stream()));
    const Record record = set.readback(runtime.h2d_stream());
    verify_expected(record, step, rank);
    writer.write(step, record, chain);
    p3a_check(host_barrier(shm, sense), "reference step barrier failed");
    if (rank == 0 && (step == 0 || step == 1 || step + 1 == steps)) {
      std::fprintf(stderr,
                   "[P3A_PROGRESS] mode=reference step=%u chain=%s\n",
                   step, hex64(chain).c_str());
    }
  }
  return 0;
}

int run_direct_rank(uint32_t rank, SharedCoord* shm,
                    const std::string& data_dir, const std::string& log_dir,
                    uint32_t steps) {
  int sense = 0;
  if (rank == 7) {
    const std::string path = log_dir + "/rank7_scal_trace.log";
    setenv("P0_LOG_FILE", path.c_str(), 1);
  } else {
    unsetenv("P0_LOG_FILE");
  }
  p0_hook_set_enabled(rank == 7 ? 1 : 0);
  p0_hook_set_phase("INIT");
  initialize_uid(rank, shm);
  p3a_check(host_barrier(shm, sense), "direct initial barrier failed");

  std::ofstream log(log_dir + "/rank" + std::to_string(rank) + ".log",
                    std::ios::out | std::ios::trunc);
  p3a_check(log.is_open(), "direct rank log open failed");
  RankRuntime rt(rank, shm->uid, log);
  p3a_check(host_barrier(shm, sense), "direct communicator barrier failed");
  AddRecipe pre(rt.device(), kCount, "p3a_direct_pre_add");
  AddRecipe post(rt.device(), kCount * kRanks, "p3a_direct_post_add");
  P3aBufferSet a(rt.device());
  P3aBufferSet b(rt.device());
  std::array<P3aBufferSet*, 2> sets{&a, &b};
  p3a_check(((a.send() ^ b.send()) & 0xFFFFFFFF00000000ULL) == 0 &&
                ((a.recv() ^ b.recv()) & 0xFFFFFFFF00000000ULL) == 0,
            "P3a buffer sets cross the 32-bit address region");
  ComputeReplay compute(rt, log);

  std::vector<CallCapture> calls;
  calls.reserve(kCanonicalCalls);
  for (uint32_t k = 0; k < kCanonicalCalls; ++k) {
    const bool trace = k < 2;
    p0_hook_set_enabled(rank == 7 && trace ? 1 : 0);
    const char* pre_phase = k == 0 ? "P3A_COLD_PRE"
                            : k == 1 ? "P3A_WARM_PRE"
                                     : "BENCH";
    const char* coll_phase = k == 0 ? "P3A_COLD_ALLGATHER"
                             : k == 1 ? "P3A_WARM_ALLGATHER"
                                      : "BENCH";
    const char* post_phase = k == 0 ? "P3A_COLD_POST"
                             : k == 1 ? "P3A_WARM_POST"
                                      : "BENCH";
    a.initialize(rt.verification_stream(), rank, k);
    b.initialize(rt.verification_stream(), rank, k);
    P3A_CHECK_SYN(synStreamSynchronize(rt.verification_stream()));
    P3aBufferSet& set = (k & 1) ? b : a;
    p0_hook_set_phase(pre_phase);
    pre.launch(rt.stream(), set.a(), set.b(), set.send());
    const uint64_t native_pre_target = compute.expected();
    p0_hook_set_phase(coll_phase);
    CallCapture call = rt.capture(k, set.send(), set.recv(), false);
    p0_hook_set_phase(post_phase);
    post.launch(rt.stream(), set.recv(), set.b_large(), set.out());
    P3A_CHECK_SYN(synStreamSynchronize(rt.stream()));
    const uint64_t native_post_target = compute.expected();
    const Record record = set.readback(rt.verification_stream());
    verify_expected(record, k, rank);
    validate_addresses(call, rank);
    if (k >= 1) {
      const uint32_t native_monitor = monitor_target(call);
      p3a_check(native_monitor == native_pre_target,
                "native monitor does not watch pre-compute completion");
      if (k >= 2) {
        p3a_check(monitor_target(calls[k - 1]) + 2 == native_monitor,
                  "native compute monitor progression mismatch");
      }
    }
    p3a_check(native_post_target > native_pre_target,
              "native compute target did not advance");
    if (k >= 71) {
      std::string failing_stream;
      p3a_check(compare_cycles(calls[k - 64], call, failing_stream),
                "native network 64-call template mismatch");
    }
    if (k < 2 || k == 63 || k == 64 || k == 65 || k == 129) {
      log << "[P3A_NATIVE_STEP] rank=" << rank << " step=" << k
          << " pre_target=" << native_pre_target
          << " post_target=" << native_post_target
          << " external_target=" << call.external_target
          << " internal_target=" << call.internal_target << std::endl;
      rt.log_call(call);
    }
    calls.push_back(std::move(call));
    p3a_check(host_barrier(shm, sense), "native capture step barrier failed");
    if (rank == 0 && (k == 0 || k == 1 || k == 63 || k == 64 || k == 129)) {
      std::fprintf(stderr, "[P3A_PROGRESS] mode=native_capture step=%u\n", k);
    }
  }

  NativeComputeCapture pre_capture =
      extract_compute("P3A_WARM_PRE", false);
  NativeComputeCapture post_capture =
      extract_compute("P3A_WARM_POST", true);
  RelocatedRecipe pre_recipe =
      prepare_recipe("P3A_WARM_PRE", pre_capture, rt, sets, true, log);
  RelocatedRecipe post_recipe =
      prepare_recipe("P3A_WARM_POST", post_capture, rt, sets, false, log);
  compute.set_pre_recipe(&pre_recipe);
  compute.set_post_recipe(&post_recipe);
  compute.finish_native_capture();
  p0_hook_set_enabled(0);
  p0_hook_set_phase("BENCH");

  ReferenceReader reader(data_dir, rank, steps);
  uint64_t chain = kFnvOffset;
  uint64_t compute_wraps = 0;
  uint64_t network_wraps = 0;
  for (uint32_t step = 0; step < steps; ++step) {
    a.initialize(rt.verification_stream(), rank, step);
    b.initialize(rt.verification_stream(), rank, step);
    P3A_CHECK_SYN(synStreamSynchronize(rt.verification_stream()));
    P3aBufferSet& set = (step & 1) ? b : a;
    const uint32_t set_index = step & 1;
    const uint64_t global_index = kCanonicalCalls + step;
    const uint64_t base_target = compute.expected();
    const uint64_t pre_target = base_target + 1;
    const uint64_t post_target = base_target + 2;
    const uint64_t external_wait_target = rt.next_external_target();
    p3a_check(post_target <= 0x7FFF && external_wait_target <= 0x7FFF,
              "direct long-SO target exceeds 15 bits");

    auto writer = compute.writer();
    const uint32_t step_start_pi = writer.pi();
    compute.set_expected(post_target);
    const uint32_t pre_start_pi = writer.pi();
    compute.publish_pre(writer, set_index);
    const uint32_t pre_end_pi = writer.pi();

    const uint32_t wait_start_pi = writer.pi();
    compute.publish_external_wait(writer, post_recipe, external_wait_target);
    const uint32_t wait_end_pi = writer.pi();
    p3a_check(wait_end_pi - wait_start_pi >= 16,
              "post external wait was not published");

    const uint32_t post_start_pi = writer.pi();
    compute.publish_post(writer, set_index);
    const uint32_t post_end_pi = writer.pi();

    const auto& replay_template = calls[66 + (step % 64)];
    CallCapture actual = rt.replay(replay_template, global_index, set.send(),
                                   set.recv(), nullptr,
                                   static_cast<uint32_t>(pre_target), false,
                                   false);
    p3a_check(actual.external_target == external_wait_target,
              "network external target changed before post wait");
    validate_addresses(actual, rank);

    const auto step_stats = writer.stats();
    compute.finish_step(writer);

    rt.wait_network(actual, global_index, false);
    compute.wait(post_target, global_index);
    p3a_check(compute.current() == post_target,
              "compute completion current value mismatch");
    const Record actual_record = set.readback(rt.verification_stream());
    const Record expected_record = reader.read(step);
    compare_reference(expected_record, actual_record, step, rank);
    const uint64_t hash = record_hash(actual_record);
    chain = chain_hash(chain, step, actual_record);
    p3a_check(hash == reader.entry(step).record &&
                  chain == reader.entry(step).chain,
              "direct hash chain mismatch");

    if (step_start_pi / kComputeCcbBytes != post_end_pi / kComputeCcbBytes) {
      ++compute_wraps;
    }
    for (const auto& stream : actual.streams) {
      if (stream.before_pi / kNetworkCcbBytes !=
          stream.next_pi / kNetworkCcbBytes) {
        ++network_wraps;
      }
    }
    log << "[P3A_STEP] rank=" << rank << " step=" << step
        << " set=" << set_index << " pre_target=" << pre_target
        << " post_target=" << post_target
        << " post_wait_target=" << external_wait_target
        << " external_target=" << actual.external_target
        << " internal_target=" << actual.internal_target
        << " pre_pi=0x" << std::hex << pre_start_pi << ":0x" << pre_end_pi
        << " wait_pi=0x" << wait_start_pi << ":0x" << wait_end_pi
        << " post_pi=0x" << post_start_pi << ":0x" << post_end_pi << std::dec
        << " compute_submissions=" << step_stats.submit_calls
        << " compute_boundary_submissions=" << step_stats.boundary_submit_calls
        << " network_submissions=" << actual.streams.size()
        << " compute_nop_bytes=" << step_stats.nop_bytes
        << " record_fnv1a64=" << hex64(hash)
        << " chain_fnv1a64=" << hex64(chain)
        << " memcmp=PASS guard=PASS" << std::endl;
    p3a_check(host_barrier(shm, sense), "direct step barrier failed");
    if (rank == 0 && (step == 0 || step == 1 || step == 63 || step == 64 ||
                      step == 128 || step + 1 == steps)) {
      std::fprintf(stderr,
                   "[P3A_PROGRESS] mode=direct step=%u pre_target=%llu "
                   "post_target=%llu chain=%s\n",
                   step, static_cast<unsigned long long>(pre_target),
                   static_cast<unsigned long long>(post_target),
                   hex64(chain).c_str());
    }
  }

  p3a_check(rt.direct_submit_calls() == 10ULL * steps,
            "direct network submit count mismatch");
  p3a_check(compute.submit_calls() >= 5ULL * steps,
            "direct compute publication count mismatch");
  log << "[P3A_DIRECT_VERDICT] rank=" << rank << " direct_steps=" << steps
      << " native_capture_steps=130 compute_submissions="
      << compute.submit_calls() << " compute_boundary_submissions="
      << compute.boundary_calls() << " network_submissions="
      << rt.direct_submit_calls() << " compute_wraps=" << compute_wraps
      << " network_wraps=" << network_wraps << " compute_nop_bytes="
      << compute.nop_bytes() << " network_nop_bytes=" << rt.direct_nop_bytes()
      << " output_memcmp=PASS guard_memcmp=PASS hash_chain=PASS" << std::endl;
  if (rank == 0) {
    std::fprintf(stderr,
                 "[P3A_DIRECT_VERDICT] ranks=8 direct_steps=%u "
                 "compute_submissions=%llu network_submissions=%llu "
                 "output_memcmp=PASS guard_memcmp=PASS hash_chain=PASS\n",
                 steps, static_cast<unsigned long long>(compute.submit_calls()),
                 static_cast<unsigned long long>(rt.direct_submit_calls()));
  }
  log.flush();
  std::_Exit(0);
}

int run_all(const std::string& mode, const std::string& data_dir,
            const std::string& log_dir, uint32_t steps) {
  std::filesystem::create_directories(data_dir);
  std::filesystem::create_directories(log_dir);
  void* mem = mmap(nullptr, sizeof(SharedCoord), PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  p3a_check(mem != MAP_FAILED, "shared coordination mmap failed");
  auto* shm = new (mem) SharedCoord();
  std::vector<pid_t> pids(kRanks);
  for (uint32_t rank = 0; rank < kRanks; ++rank) {
    const pid_t pid = fork();
    if (pid == 0) {
      const int rc = mode == "reference"
                         ? run_reference_rank(rank, shm, data_dir, steps)
                         : run_direct_rank(rank, shm, data_dir, log_dir, steps);
      std::_Exit(rc);
    }
    p3a_check(pid > 0, "rank fork failed");
    pids[rank] = pid;
  }

  int result = 0;
  size_t completed = 0;
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(900);
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
  if (result == 0) {
    std::ofstream verdict(log_dir + "/verdict.json", std::ios::out | std::ios::trunc);
    verdict << "{\"verdict\":\"PASS\",\"mode\":\"" << mode
            << "\",\"ranks\":8,\"steps\":" << steps
            << ",\"native_capture_steps\":" << (mode == "direct" ? 130 : 0)
            << ",\"output_bytes_per_rank_step\":32768"
            << ",\"guard_bytes_per_rank_step\":512}\n";
  }
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  std::string mode;
  std::string data_dir;
  std::string log_dir;
  uint32_t steps = 0;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
      mode = argv[++i];
    } else if (std::strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc) {
      data_dir = argv[++i];
    } else if (std::strcmp(argv[i], "--log-dir") == 0 && i + 1 < argc) {
      log_dir = argv[++i];
    } else if (std::strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
      steps = static_cast<uint32_t>(std::strtoul(argv[++i], nullptr, 0));
    } else {
      std::fprintf(stderr,
                   "usage: %s --mode reference|direct --steps N --data-dir PATH "
                   "--log-dir PATH\n",
                   argv[0]);
      return 2;
    }
  }
  if ((mode != "reference" && mode != "direct") || steps == 0 || steps > 6000 ||
      data_dir.empty() || log_dir.empty()) {
    std::fprintf(stderr, "invalid P3a arguments\n");
    return 2;
  }
  return run_all(mode, data_dir, log_dir, steps);
}
