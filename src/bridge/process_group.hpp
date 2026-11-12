// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "communicator.hpp"

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace gaudirt::generic {

class WorkGaudiRT final : public c10d::Work {
 public:
  WorkGaudiRT(int rank, c10d::OpType op_type, std::vector<at::Tensor> outputs,
              std::weak_ptr<Context> owner, uint32_t comm_id,
              bool comm_active, bool host_sync);
  ~WorkGaudiRT() override = default;

  bool isCompleted() override;
  bool isSuccess() const override;
  void synchronize() override;
  bool wait(std::chrono::milliseconds timeout = kNoTimeout) override;
  void blockCurrentStream() override;
  std::vector<at::Tensor> result() override;
  c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

 private:
  std::vector<at::Tensor> outputs_;
  std::weak_ptr<Context> owner_;
  uint32_t comm_id_ = 0;
  bool comm_active_ = false;
  bool host_sync_ = false;
  bool stream_waited_ = false;
};

class ProcessGroupGaudiRT final : public c10d::Backend {
 public:
  ProcessGroupGaudiRT(c10::intrusive_ptr<c10d::Store> store, int rank, int size,
                      std::string group_id);
  ~ProcessGroupGaudiRT() override;

  const std::string getBackendName() const override { return "gaudirt"; }
  bool supportsSplitting() const override { return false; }
  bool supportsCoalescing() const override { return false; }

  uint32_t ensure_initialized();
  uint32_t comm_id() { return ensure_initialized(); }
  int device_index() const { return device_index_; }

  c10::intrusive_ptr<c10d::Work> broadcast(
      std::vector<at::Tensor>& tensors,
      const c10d::BroadcastOptions& opts = c10d::BroadcastOptions()) override;

  c10::intrusive_ptr<c10d::Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceOptions& opts = c10d::AllreduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const c10d::AllreduceCoalescedOptions& opts =
          c10d::AllreduceCoalescedOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce(
      std::vector<at::Tensor>& tensors,
      const c10d::ReduceOptions& opts = c10d::ReduceOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> _allgather_base(
      at::Tensor& output_buffer,
      at::Tensor& input_buffer,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> allgather_into_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const c10d::AllgatherOptions& opts = c10d::AllgatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> gather(
      std::vector<std::vector<at::Tensor>>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::GatherOptions& opts = c10d::GatherOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce_scatter(
      std::vector<at::Tensor>& output_tensors,
      std::vector<std::vector<at::Tensor>>& input_tensors,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> _reduce_scatter_base(
      at::Tensor& output_buffer,
      at::Tensor& input_buffer,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> reduce_scatter_tensor_coalesced(
      std::vector<at::Tensor>& outputs,
      std::vector<at::Tensor>& inputs,
      const c10d::ReduceScatterOptions& opts = c10d::ReduceScatterOptions()) override;

  c10::intrusive_ptr<c10d::Work> alltoall_base(
      at::Tensor& output_buffer,
      at::Tensor& input_buffer,
      std::vector<int64_t>& output_split_sizes,
      std::vector<int64_t>& input_split_sizes,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::Work> alltoall(
      std::vector<at::Tensor>& output_tensors,
      std::vector<at::Tensor>& input_tensors,
      const c10d::AllToAllOptions& opts = c10d::AllToAllOptions()) override;

  c10::intrusive_ptr<c10d::Work> send(
      std::vector<at::Tensor>& tensors, int dst_rank, int tag) override;

  c10::intrusive_ptr<c10d::Work> recv(
      std::vector<at::Tensor>& tensors, int src_rank, int tag) override;

  c10::intrusive_ptr<c10d::Work> barrier(
      const c10d::BarrierOptions& opts = c10d::BarrierOptions()) override;

  void abort() override;
  void shutdown() override;

 private:
  std::shared_ptr<Context> active_context();
  void check_collective_tensor(const at::Tensor& tensor, const Context& ctx, const char* role) const;

  c10::intrusive_ptr<c10d::Store> store_;
  std::string group_id_;
  int device_index_ = 0;
  std::mutex init_mutex_;
  bool initialized_ = false;
  bool closed_ = false;
  uint32_t comm_id_ = 0;
  std::weak_ptr<Context> owner_;
};

c10::intrusive_ptr<c10d::Backend> create_process_group_gaudirt(
    const c10::intrusive_ptr<c10d::Store>& store, int rank, int size,
    const std::string& group_id);

}  // namespace gaudirt::generic
