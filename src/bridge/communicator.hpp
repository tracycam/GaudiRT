// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "bridge_internal.hpp"

#include <hccl.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gaudirt::generic {

enum class CollectiveKind : uint8_t {
  AllReduce = 0,
  AllGather = 1,
  AllGatherDimLast = 2,
  ReduceScatter = 3,
  AllToAll = 4,
};

struct CommGroup {
  CommGroup(uint32_t group_id, int group_rank, int group_size,
            std::string name, synDeviceId device_id)
      : id(group_id),
        rank(group_rank),
        world_size(group_size),
        group_name(std::move(name)),
        compute_to_comm(device_id),
        comm_to_compute(device_id) {}

  uint32_t id = 0;
  int rank = 0;
  int world_size = 1;
  std::string group_name;
  bool initialized = false;
  bool destroyed = false;
  hcclComm_t comm = nullptr;
  synStreamHandle comm_stream = nullptr;
  runtime::StreamEvent compute_to_comm;
  runtime::StreamEvent comm_to_compute;
  uint64_t barrier_address = 0;
  uint64_t scratch_address = 0;
  uint64_t scratch_bytes = 0;
  synStreamHandle scratch_stream = nullptr;
};

struct HcclTransferSpec {
  size_t count = 0;
  hcclDataType_t dtype = hcclBfloat16;
};

class CommunicatorRegistry {
 public:
  static constexpr uint64_t kMaxGatherReduceBytes = 128ULL << 20;

  CommunicatorRegistry() = default;
  ~CommunicatorRegistry() = default;
  CommunicatorRegistry(const CommunicatorRegistry&) = delete;
  CommunicatorRegistry& operator=(const CommunicatorRegistry&) = delete;

  uint32_t register_group(Context& ctx, int rank, int world_size, std::string group_name);
  void init_group(Context& ctx, uint32_t comm_id, const hcclUniqueId& uid);
  CommGroup& group(uint32_t comm_id);

  uint64_t ensure_scratch(Context& ctx, CommGroup& grp, uint64_t bytes, synStreamHandle stream);
  void trim_scratch(Context& ctx);
  void destroy_group(Context& ctx, uint32_t comm_id, bool abort_mode);
  void teardown_all(Context& ctx, bool abort_mode);

  static void record_tensor_on_stream(Context& ctx, const at::Tensor& tensor, synStreamHandle stream);
  static HcclTransferSpec resolve_hccl_transfer(uint64_t numel, at::ScalarType dtype, bool is_reduction);
  static void check_hccl(hcclResult_t status, const char* op);

  static bool should_gather_reduce(uint64_t exact_elements, uint32_t bucket_elements,
                                   int world_size, at::ScalarType dtype);
  static void launch_rank_sum(Context& ctx, uint64_t gathered_addr, uint64_t dst_addr,
                              uint64_t exact_elements, uint32_t bucket_elements,
                              uint32_t world_size, at::ScalarType dtype,
                              synStreamHandle stream);
  static void launch_gather_transpose(Context& ctx, uint64_t gathered_addr, uint64_t dst_addr,
                                      uint64_t world_size, uint64_t batch_rows, uint64_t cols,
                                      at::ScalarType dtype, synStreamHandle stream);

 private:
  std::mutex mutex_;
  std::vector<std::unique_ptr<CommGroup>> groups_;
};

CommunicatorRegistry& communicator_registry(Context& ctx);

}  // namespace gaudirt::generic
