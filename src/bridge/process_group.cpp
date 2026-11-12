// SPDX-License-Identifier: Apache-2.0
#include "process_group.hpp"

#include <ATen/core/ivalue.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstring>
#include <numeric>

namespace gaudirt::generic {
namespace {

std::atomic<uint64_t> g_local_pg_counter{0};

hcclRedOp_t to_hccl_red_op(const c10d::ReduceOp& op) {
  TORCH_CHECK(!op.supplement_, "GaudiRT collective: supplemented ReduceOp is not supported");
  switch (op.op_) {
    case c10d::ReduceOp::SUM: return hcclSum;
    case c10d::ReduceOp::AVG: return hcclAvg;
    case c10d::ReduceOp::PRODUCT: return hcclProd;
    case c10d::ReduceOp::MIN: return hcclMin;
    case c10d::ReduceOp::MAX: return hcclMax;
    default:
      TORCH_CHECK(false, "GaudiRT collective: unsupported ReduceOp ", static_cast<int>(op.op_));
  }
}

uint32_t round_bucket_elements(uint64_t exact) {
  if (exact == 0 || exact > (1ULL << 30)) return 0;
  const uint64_t padded = std::max<uint64_t>(256, std::bit_ceil(exact));
  return padded <= UINT32_MAX ? static_cast<uint32_t>(padded) : 0;
}

uint64_t tensor_device_addr(const at::Tensor& tensor) {
  return reinterpret_cast<uint64_t>(tensor.data_ptr());
}

void copy_if_distinct(const at::Tensor& dst, const at::Tensor& src, synStreamHandle stream) {
  const uint64_t bytes = static_cast<uint64_t>(src.numel()) * src.element_size();
  if (bytes == 0) return;
  const uint64_t dst_addr = tensor_device_addr(dst);
  const uint64_t src_addr = tensor_device_addr(src);
  if (dst_addr != src_addr) {
    runtime::PinnedHostPool::copy_d2d(dst_addr, src_addr, bytes, stream, true);
  }
}

}  // namespace

WorkGaudiRT::WorkGaudiRT(int rank, c10d::OpType op_type, std::vector<at::Tensor> outputs,
                         std::weak_ptr<Context> owner, uint32_t comm_id,
                         bool comm_active, bool host_sync)
    : c10d::Work(rank, op_type),
      outputs_(std::move(outputs)),
      owner_(std::move(owner)),
      comm_id_(comm_id),
      comm_active_(comm_active),
      host_sync_(host_sync) {
  if (!comm_active_) {
    finish();
  }
}

bool WorkGaudiRT::isCompleted() {
  if (c10d::Work::isCompleted()) return true;
  auto ctx = owner_.lock();
  if (!ctx || !ctx->comm_state) {
    finish();
    return true;
  }
  auto& grp = communicator_registry(*ctx).group(comm_id_);
  if (!grp.comm_stream || grp.comm_to_compute.query()) {
    finish();
    return true;
  }
  return false;
}

bool WorkGaudiRT::isSuccess() const {
  return !exception();
}

void WorkGaudiRT::blockCurrentStream() {
  if (!comm_active_) return;
  std::lock_guard lock(mutex_);
  if (stream_waited_) return;
  if (auto ctx = owner_.lock(); ctx && ctx->comm_state) {
    auto& grp = communicator_registry(*ctx).group(comm_id_);
    if (grp.comm_stream) {
      grp.comm_to_compute.wait(ctx->stream());
    }
  }
  stream_waited_ = true;
}

void WorkGaudiRT::synchronize() {
  blockCurrentStream();
}

bool WorkGaudiRT::wait(std::chrono::milliseconds /*timeout*/) {
  blockCurrentStream();
  if (comm_active_ && host_sync_) {
    if (auto ctx = owner_.lock(); ctx && ctx->comm_state) {
      auto& grp = communicator_registry(*ctx).group(comm_id_);
      if (grp.comm_stream) {
        TORCH_CHECK(synStreamSynchronize(grp.comm_stream) == synSuccess,
                    "GaudiRT collective: comm_stream synchronize failed");
      }
    }
  }
  finish();
  return true;
}

std::vector<at::Tensor> WorkGaudiRT::result() {
  return outputs_;
}

c10::intrusive_ptr<c10::ivalue::Future> WorkGaudiRT::getFuture() {
  blockCurrentStream();
  auto fut = c10::make_intrusive<c10::ivalue::Future>(
      c10::ListType::create(c10::TensorType::get()));
  fut->markCompleted(c10::IValue(outputs_));
  return fut;
}

ProcessGroupGaudiRT::ProcessGroupGaudiRT(c10::intrusive_ptr<c10d::Store> store, int rank, int size,
                                         std::string group_id)
    : c10d::Backend(rank, size),
      store_(std::move(store)),
      group_id_(group_id.empty()
                    ? ("pg_" + std::to_string(g_local_pg_counter.fetch_add(1)))
                    : std::move(group_id)),
      device_index_(current_device_index()) {
  TORCH_CHECK(size >= 1 && rank >= 0 && rank < size,
              "ProcessGroupGaudiRT: invalid rank ", rank, " or size ", size);
  TORCH_CHECK(size == 1 || store_ != nullptr,
              "ProcessGroupGaudiRT: Store is required when world_size > 1");
  setBoundDeviceId(c10::Device(c10::kPrivateUse1, static_cast<c10::DeviceIndex>(device_index_)));
  init();
}

ProcessGroupGaudiRT::~ProcessGroupGaudiRT() {
  if (auto ctx = owner_.lock(); ctx && ctx->comm_state && initialized_ && !closed_) {
    communicator_registry(*ctx).destroy_group(*ctx, comm_id_, std::uncaught_exceptions() > 0);
    closed_ = true;
  }
}

uint32_t ProcessGroupGaudiRT::ensure_initialized() {
  std::lock_guard lock(init_mutex_);
  TORCH_CHECK(!closed_, "ProcessGroupGaudiRT: communicator '", group_id_, "' has been closed");
  if (initialized_) return comm_id_;

  set_device_index(static_cast<c10::DeviceIndex>(device_index_));
  auto ctx = current();
  auto& reg = communicator_registry(*ctx);
  const std::string name = !pg_uid_.empty() ? pg_uid_ : group_id_;
  const uint32_t cid = reg.register_group(*ctx, rank_, size_, name);

  hcclUniqueId uid{};
  if (size_ > 1) {
    const std::string store_key = "gaudirt/hccl_uid/" + group_id_;
    if (rank_ == 0) {
      CommunicatorRegistry::check_hccl(hcclGetUniqueId(&uid), "hcclGetUniqueId");
      std::vector<uint8_t> raw(sizeof(hcclUniqueId));
      std::memcpy(raw.data(), &uid, sizeof(hcclUniqueId));
      store_->set(store_key, raw);
    } else {
      const std::vector<uint8_t> raw = store_->get(store_key);
      TORCH_CHECK(raw.size() == sizeof(hcclUniqueId),
                  "ProcessGroupGaudiRT: corrupted hcclUniqueId in Store (", raw.size(), " bytes)");
      std::memcpy(&uid, raw.data(), sizeof(hcclUniqueId));
    }
  }

  reg.init_group(*ctx, cid, uid);
  owner_ = ctx;
  comm_id_ = cid;
  initialized_ = true;
  return comm_id_;
}

std::shared_ptr<Context> ProcessGroupGaudiRT::active_context() {
  ensure_initialized();
  auto ctx = owner_.lock();
  TORCH_CHECK(ctx != nullptr, "ProcessGroupGaudiRT: owning device Context was released");
  return ctx;
}

void ProcessGroupGaudiRT::check_collective_tensor(const at::Tensor& tensor, const Context& ctx,
                                                  const char* role) const {
  TORCH_CHECK(tensor.defined(), "ProcessGroupGaudiRT: undefined ", role, " tensor");
  TORCH_CHECK(tensor.device().is_privateuseone() && tensor.device().index() == ctx.device_index,
              "ProcessGroupGaudiRT: ", role, " tensor must live on gaudirt:", ctx.device_index,
              ", got ", tensor.device());
  TORCH_CHECK(tensor.is_contiguous(),
              "ProcessGroupGaudiRT: ", role, " tensor must be contiguous");
  if (tensor.numel() > 0) {
    (void)validated_program_input(tensor, ctx);
  }
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::broadcast(
    std::vector<at::Tensor>& tensors, const c10d::BroadcastOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, "ProcessGroupGaudiRT::broadcast expects 1 tensor");
  TORCH_CHECK(opts.rootRank >= 0 && opts.rootRank < size_ && opts.rootTensor == 0,
              "ProcessGroupGaudiRT::broadcast invalid rootRank ", opts.rootRank);
  auto ctx = active_context();
  auto& t = tensors[0];
  check_collective_tensor(t, *ctx, "broadcast");
  if (t.numel() == 0 || size_ == 1) {
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::BROADCAST, tensors, ctx, comm_id_, false, false);
  }
  auto& reg = communicator_registry(*ctx);
  auto& grp = reg.group(comm_id_);
  const uint64_t bytes = static_cast<uint64_t>(t.numel()) * static_cast<uint64_t>(t.element_size());
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, t, grp.comm_stream);
  if (bytes % 2 != 0) {
    const uint64_t scratch = reg.ensure_scratch(*ctx, grp, bytes + 1, grp.comm_stream);
    if (rank_ == static_cast<int>(opts.rootRank)) {
      runtime::PinnedHostPool::copy_d2d(scratch, tensor_device_addr(t), bytes, grp.comm_stream, true);
    }
    CommunicatorRegistry::check_hccl(
        hcclBroadcast(reinterpret_cast<const void*>(scratch), reinterpret_cast<void*>(scratch),
                      static_cast<size_t>((bytes + 1) / 2), hcclBfloat16,
                      static_cast<int>(opts.rootRank), grp.comm, grp.comm_stream),
        "hcclBroadcast(odd)");
    if (rank_ != static_cast<int>(opts.rootRank)) {
      runtime::PinnedHostPool::copy_d2d(tensor_device_addr(t), scratch, bytes, grp.comm_stream, true);
    }
  } else {
    const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
        static_cast<uint64_t>(t.numel()), t.scalar_type(), false);
    CommunicatorRegistry::check_hccl(
        hcclBroadcast(t.data_ptr(), t.data_ptr(), xfer.count, xfer.dtype,
                      static_cast<int>(opts.rootRank), grp.comm, grp.comm_stream),
        "hcclBroadcast");
  }
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::BROADCAST, tensors, ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::allreduce(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, "ProcessGroupGaudiRT::allreduce expects 1 tensor");
  TORCH_CHECK(!opts.sparseIndices.has_value(), "ProcessGroupGaudiRT: sparse allreduce unsupported");
  auto ctx = active_context();
  auto& t = tensors[0];
  check_collective_tensor(t, *ctx, "allreduce");
  const hcclRedOp_t red_op = to_hccl_red_op(opts.reduceOp);
  if (t.numel() == 0 || size_ == 1) {
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLREDUCE, tensors, ctx, comm_id_, false, false);
  }

  auto& reg = communicator_registry(*ctx);
  auto& grp = reg.group(comm_id_);
  const auto count = static_cast<uint64_t>(t.numel());
  const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(count, t.scalar_type(), true);

  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, t, grp.comm_stream);

  const uint32_t bucket_elems = round_bucket_elements(count);
  if (opts.reduceOp.op_ == c10d::ReduceOp::SUM &&
      CommunicatorRegistry::should_gather_reduce(count, bucket_elems, size_, t.scalar_type())) {
    const uint64_t gather_bytes =
        static_cast<uint64_t>(size_) * bucket_elems * static_cast<uint64_t>(t.element_size());
    const uint64_t scratch = reg.ensure_scratch(*ctx, grp, gather_bytes, grp.comm_stream);
    CommunicatorRegistry::check_hccl(
        hcclAllGather(t.data_ptr(), reinterpret_cast<void*>(scratch),
                      xfer.count, xfer.dtype, grp.comm, grp.comm_stream),
        "hcclAllGather(gather_reduce)");
    CommunicatorRegistry::launch_rank_sum(
        *ctx, scratch, tensor_device_addr(t), count, bucket_elems,
        static_cast<uint32_t>(size_), t.scalar_type(), grp.comm_stream);
  } else {
    CommunicatorRegistry::check_hccl(
        hcclAllReduce(t.data_ptr(), t.data_ptr(), xfer.count,
                      xfer.dtype, red_op, grp.comm, grp.comm_stream),
        "hcclAllReduce");
  }

  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLREDUCE, tensors, ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::allreduce_coalesced(
    std::vector<at::Tensor>& tensors, const c10d::AllreduceCoalescedOptions& opts) {
  TORCH_CHECK(!tensors.empty(), "ProcessGroupGaudiRT::allreduce_coalesced expects non-empty tensors");
  auto ctx = active_context();
  const hcclRedOp_t red_op = to_hccl_red_op(opts.reduceOp);
  for (const auto& t : tensors) check_collective_tensor(t, *ctx, "allreduce_coalesced");
  if (size_ == 1) {
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLREDUCE_COALESCED, tensors, ctx, comm_id_, false, false);
  }
  auto& grp = communicator_registry(*ctx).group(comm_id_);
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  CommunicatorRegistry::check_hccl(hcclGroupStart(), "hcclGroupStart");
  for (auto& t : tensors) {
    if (t.numel() == 0) continue;
    const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
        static_cast<uint64_t>(t.numel()), t.scalar_type(), true);
    CommunicatorRegistry::record_tensor_on_stream(*ctx, t, grp.comm_stream);
    CommunicatorRegistry::check_hccl(
        hcclAllReduce(t.data_ptr(), t.data_ptr(), xfer.count,
                      xfer.dtype, red_op, grp.comm, grp.comm_stream),
        "hcclAllReduce(coalesced)");
  }
  CommunicatorRegistry::check_hccl(hcclGroupEnd(), "hcclGroupEnd");
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLREDUCE_COALESCED, tensors, ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::reduce(
    std::vector<at::Tensor>& tensors, const c10d::ReduceOptions& opts) {
  TORCH_CHECK(tensors.size() == 1, "ProcessGroupGaudiRT::reduce expects 1 tensor");
  TORCH_CHECK(opts.rootRank >= 0 && opts.rootRank < size_ && opts.rootTensor == 0,
               "ProcessGroupGaudiRT::reduce invalid rootRank ", opts.rootRank);
  auto ctx = active_context();
  auto& t = tensors[0];
  check_collective_tensor(t, *ctx, "reduce");
  const hcclRedOp_t red_op = to_hccl_red_op(opts.reduceOp);
  if (t.numel() == 0 || size_ == 1) {
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::REDUCE, tensors, ctx, comm_id_, false, false);
  }
  const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
      static_cast<uint64_t>(t.numel()), t.scalar_type(), true);
  auto& grp = communicator_registry(*ctx).group(comm_id_);
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, t, grp.comm_stream);
  CommunicatorRegistry::check_hccl(
      hcclReduce(t.data_ptr(), t.data_ptr(), xfer.count,
                 xfer.dtype, red_op, static_cast<int>(opts.rootRank), grp.comm, grp.comm_stream),
      "hcclReduce");
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::REDUCE, tensors, ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::_allgather_base(
    at::Tensor& output_buffer, at::Tensor& input_buffer, const c10d::AllgatherOptions&) {
  auto ctx = active_context();
  check_collective_tensor(output_buffer, *ctx, "allgather output");
  check_collective_tensor(input_buffer, *ctx, "allgather input");
  TORCH_CHECK(output_buffer.scalar_type() == input_buffer.scalar_type(),
              "ProcessGroupGaudiRT::_allgather_base dtype mismatch");
  TORCH_CHECK(output_buffer.numel() == input_buffer.numel() * size_,
              "ProcessGroupGaudiRT::_allgather_base expected output numel ",
              input_buffer.numel() * size_, ", got ", output_buffer.numel());
  std::vector<at::Tensor> outs{output_buffer};
  if (input_buffer.numel() == 0) {
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::_ALLGATHER_BASE, std::move(outs), ctx, comm_id_, false, false);
  }
  if (size_ == 1) {
    copy_if_distinct(output_buffer, input_buffer, ctx->stream());
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::_ALLGATHER_BASE, std::move(outs), ctx, comm_id_, false, false);
  }
  const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
      static_cast<uint64_t>(input_buffer.numel()), input_buffer.scalar_type(), false);
  auto& grp = communicator_registry(*ctx).group(comm_id_);
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, input_buffer, grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, output_buffer, grp.comm_stream);
  CommunicatorRegistry::check_hccl(
      hcclAllGather(input_buffer.data_ptr(), output_buffer.data_ptr(),
                    xfer.count, xfer.dtype, grp.comm, grp.comm_stream),
      "hcclAllGather");
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::_ALLGATHER_BASE, std::move(outs), ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::allgather_into_tensor_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs, const c10d::AllgatherOptions&) {
  TORCH_CHECK(outputs.size() == inputs.size() && !inputs.empty(),
              "ProcessGroupGaudiRT::allgather_into_tensor_coalesced size mismatch");
  auto ctx = active_context();
  for (size_t i = 0; i < inputs.size(); ++i) {
    check_collective_tensor(outputs[i], *ctx, "allgather_coalesced output");
    check_collective_tensor(inputs[i], *ctx, "allgather_coalesced input");
    TORCH_CHECK(outputs[i].scalar_type() == inputs[i].scalar_type() &&
                    outputs[i].numel() == inputs[i].numel() * size_,
                "ProcessGroupGaudiRT::allgather_into_tensor_coalesced shape/dtype mismatch");
  }
  if (size_ == 1) {
    for (size_t i = 0; i < inputs.size(); ++i) copy_if_distinct(outputs[i], inputs[i], ctx->stream());
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::COALESCED, outputs, ctx, comm_id_, false, false);
  }
  auto& grp = communicator_registry(*ctx).group(comm_id_);
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  CommunicatorRegistry::check_hccl(hcclGroupStart(), "hcclGroupStart");
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].numel() == 0) continue;
    const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
        static_cast<uint64_t>(inputs[i].numel()), inputs[i].scalar_type(), false);
    CommunicatorRegistry::record_tensor_on_stream(*ctx, inputs[i], grp.comm_stream);
    CommunicatorRegistry::record_tensor_on_stream(*ctx, outputs[i], grp.comm_stream);
    CommunicatorRegistry::check_hccl(
        hcclAllGather(inputs[i].data_ptr(), outputs[i].data_ptr(),
                      xfer.count, xfer.dtype, grp.comm, grp.comm_stream),
        "hcclAllGather(coalesced)");
  }
  CommunicatorRegistry::check_hccl(hcclGroupEnd(), "hcclGroupEnd");
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::COALESCED, outputs, ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::allgather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllgatherOptions&) {
  TORCH_CHECK(input_tensors.size() == 1 && output_tensors.size() == 1 &&
                  static_cast<int>(output_tensors[0].size()) == size_,
              "ProcessGroupGaudiRT::allgather expects 1 input and world_size outputs");
  auto ctx = active_context();
  auto& inp = input_tensors[0];
  check_collective_tensor(inp, *ctx, "allgather input");
  for (const auto& out : output_tensors[0]) {
    check_collective_tensor(out, *ctx, "allgather output");
    TORCH_CHECK(out.scalar_type() == inp.scalar_type() && out.numel() == inp.numel(),
                "ProcessGroupGaudiRT::allgather output tensor mismatch");
  }
  if (inp.numel() == 0) {
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLGATHER, output_tensors[0], ctx, comm_id_, false, false);
  }
  if (size_ == 1) {
    copy_if_distinct(output_tensors[0][0], inp, ctx->stream());
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLGATHER, output_tensors[0], ctx, comm_id_, false, false);
  }
  auto& reg = communicator_registry(*ctx);
  auto& grp = reg.group(comm_id_);
  const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
      static_cast<uint64_t>(inp.numel()), inp.scalar_type(), false);
  const uint64_t chunk_bytes = static_cast<uint64_t>(inp.numel()) * inp.element_size();
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  const uint64_t scratch = reg.ensure_scratch(*ctx, grp, chunk_bytes * size_, grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, inp, grp.comm_stream);
  CommunicatorRegistry::check_hccl(
      hcclAllGather(inp.data_ptr(), reinterpret_cast<void*>(scratch),
                    xfer.count, xfer.dtype, grp.comm, grp.comm_stream),
      "hcclAllGather(list)");
  for (int r = 0; r < size_; ++r) {
    CommunicatorRegistry::record_tensor_on_stream(*ctx, output_tensors[0][r], grp.comm_stream);
    runtime::PinnedHostPool::copy_d2d(
        tensor_device_addr(output_tensors[0][r]), scratch + static_cast<uint64_t>(r) * chunk_bytes,
        chunk_bytes, grp.comm_stream, true);
  }
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLGATHER, output_tensors[0], ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::gather(
    std::vector<std::vector<at::Tensor>>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::GatherOptions& opts) {
  TORCH_CHECK(input_tensors.size() == 1, "ProcessGroupGaudiRT::gather expects 1 input tensor");
  TORCH_CHECK(opts.rootRank >= 0 && opts.rootRank < size_,
              "ProcessGroupGaudiRT::gather invalid rootRank ", opts.rootRank);
  auto ctx = active_context();
  auto& inp = input_tensors[0];
  check_collective_tensor(inp, *ctx, "gather input");
  const bool is_root = (rank_ == static_cast<int>(opts.rootRank));
  std::vector<at::Tensor> outs;
  if (is_root) {
    TORCH_CHECK(output_tensors.size() == 1 && static_cast<int>(output_tensors[0].size()) == size_,
                "ProcessGroupGaudiRT::gather root expects world_size output tensors");
    for (const auto& out : output_tensors[0]) {
      check_collective_tensor(out, *ctx, "gather output");
      TORCH_CHECK(out.scalar_type() == inp.scalar_type() && out.numel() == inp.numel(),
                  "ProcessGroupGaudiRT::gather output tensor mismatch");
    }
    outs = output_tensors[0];
  } else {
    TORCH_CHECK(output_tensors.empty(), "ProcessGroupGaudiRT::gather non-root expects empty output_tensors");
  }
  if (inp.numel() == 0) {
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::GATHER, std::move(outs), ctx, comm_id_, false, false);
  }
  if (size_ == 1) {
    copy_if_distinct(outs[0], inp, ctx->stream());
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::GATHER, std::move(outs), ctx, comm_id_, false, false);
  }
  const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
      static_cast<uint64_t>(inp.numel()), inp.scalar_type(), false);
  auto& grp = communicator_registry(*ctx).group(comm_id_);
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, inp, grp.comm_stream);
  CommunicatorRegistry::check_hccl(hcclGroupStart(), "hcclGroupStart");
  if (is_root) {
    for (int r = 0; r < size_; ++r) {
      CommunicatorRegistry::record_tensor_on_stream(*ctx, outs[r], grp.comm_stream);
      if (r == rank_) {
        copy_if_distinct(outs[r], inp, grp.comm_stream);
      } else {
        CommunicatorRegistry::check_hccl(
            hcclRecv(outs[r].data_ptr(), xfer.count, xfer.dtype, r, grp.comm, grp.comm_stream),
            "hcclRecv(gather)");
      }
    }
  } else {
    CommunicatorRegistry::check_hccl(
        hcclSend(inp.data_ptr(), xfer.count, xfer.dtype,
                 static_cast<int>(opts.rootRank), grp.comm, grp.comm_stream),
        "hcclSend(gather)");
  }
  CommunicatorRegistry::check_hccl(hcclGroupEnd(), "hcclGroupEnd");
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::GATHER, std::move(outs), ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::_reduce_scatter_base(
    at::Tensor& output_buffer, at::Tensor& input_buffer, const c10d::ReduceScatterOptions& opts) {
  auto ctx = active_context();
  check_collective_tensor(output_buffer, *ctx, "reduce_scatter output");
  check_collective_tensor(input_buffer, *ctx, "reduce_scatter input");
  TORCH_CHECK(output_buffer.scalar_type() == input_buffer.scalar_type(),
              "ProcessGroupGaudiRT::_reduce_scatter_base dtype mismatch");
  TORCH_CHECK(input_buffer.numel() == output_buffer.numel() * size_,
              "ProcessGroupGaudiRT::_reduce_scatter_base expected input numel ",
              output_buffer.numel() * size_, ", got ", input_buffer.numel());
  const hcclRedOp_t red_op = to_hccl_red_op(opts.reduceOp);
  std::vector<at::Tensor> outs{output_buffer};
  if (output_buffer.numel() == 0) {
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::_REDUCE_SCATTER_BASE, std::move(outs), ctx, comm_id_, false, false);
  }
  if (size_ == 1) {
    copy_if_distinct(output_buffer, input_buffer, ctx->stream());
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::_REDUCE_SCATTER_BASE, std::move(outs), ctx, comm_id_, false, false);
  }
  const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
      static_cast<uint64_t>(output_buffer.numel()), input_buffer.scalar_type(), true);
  auto& grp = communicator_registry(*ctx).group(comm_id_);
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, input_buffer, grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, output_buffer, grp.comm_stream);
  CommunicatorRegistry::check_hccl(
      hcclReduceScatter(input_buffer.data_ptr(), output_buffer.data_ptr(),
                        xfer.count, xfer.dtype, red_op,
                        grp.comm, grp.comm_stream),
      "hcclReduceScatter");
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::_REDUCE_SCATTER_BASE, std::move(outs), ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::reduce_scatter_tensor_coalesced(
    std::vector<at::Tensor>& outputs, std::vector<at::Tensor>& inputs, const c10d::ReduceScatterOptions& opts) {
  TORCH_CHECK(outputs.size() == inputs.size() && !outputs.empty(),
              "ProcessGroupGaudiRT::reduce_scatter_tensor_coalesced size mismatch");
  auto ctx = active_context();
  const hcclRedOp_t red_op = to_hccl_red_op(opts.reduceOp);
  for (size_t i = 0; i < outputs.size(); ++i) {
    check_collective_tensor(outputs[i], *ctx, "reduce_scatter_coalesced output");
    check_collective_tensor(inputs[i], *ctx, "reduce_scatter_coalesced input");
    TORCH_CHECK(outputs[i].scalar_type() == inputs[i].scalar_type() &&
                    inputs[i].numel() == outputs[i].numel() * size_,
                "ProcessGroupGaudiRT::reduce_scatter_tensor_coalesced shape/dtype mismatch");
  }
  if (size_ == 1) {
    for (size_t i = 0; i < outputs.size(); ++i) copy_if_distinct(outputs[i], inputs[i], ctx->stream());
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::COALESCED, outputs, ctx, comm_id_, false, false);
  }
  auto& grp = communicator_registry(*ctx).group(comm_id_);
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  CommunicatorRegistry::check_hccl(hcclGroupStart(), "hcclGroupStart");
  for (size_t i = 0; i < outputs.size(); ++i) {
    if (outputs[i].numel() == 0) continue;
    const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
        static_cast<uint64_t>(outputs[i].numel()), outputs[i].scalar_type(), true);
    CommunicatorRegistry::record_tensor_on_stream(*ctx, inputs[i], grp.comm_stream);
    CommunicatorRegistry::record_tensor_on_stream(*ctx, outputs[i], grp.comm_stream);
    CommunicatorRegistry::check_hccl(
        hcclReduceScatter(inputs[i].data_ptr(), outputs[i].data_ptr(),
                          xfer.count, xfer.dtype, red_op,
                          grp.comm, grp.comm_stream),
        "hcclReduceScatter(coalesced)");
  }
  CommunicatorRegistry::check_hccl(hcclGroupEnd(), "hcclGroupEnd");
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::COALESCED, outputs, ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::reduce_scatter(
    std::vector<at::Tensor>& output_tensors,
    std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10d::ReduceScatterOptions& opts) {
  TORCH_CHECK(output_tensors.size() == 1 && input_tensors.size() == 1 &&
                  static_cast<int>(input_tensors[0].size()) == size_,
              "ProcessGroupGaudiRT::reduce_scatter expects 1 output and world_size inputs");
  auto ctx = active_context();
  auto& out = output_tensors[0];
  check_collective_tensor(out, *ctx, "reduce_scatter output");
  for (const auto& inp : input_tensors[0]) {
    check_collective_tensor(inp, *ctx, "reduce_scatter input");
    TORCH_CHECK(inp.scalar_type() == out.scalar_type() && inp.numel() == out.numel(),
                "ProcessGroupGaudiRT::reduce_scatter input tensor mismatch");
  }
  const hcclRedOp_t red_op = to_hccl_red_op(opts.reduceOp);
  if (out.numel() == 0) {
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::REDUCE_SCATTER, output_tensors, ctx, comm_id_, false, false);
  }
  if (size_ == 1) {
    copy_if_distinct(out, input_tensors[0][0], ctx->stream());
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::REDUCE_SCATTER, output_tensors, ctx, comm_id_, false, false);
  }
  auto& reg = communicator_registry(*ctx);
  auto& grp = reg.group(comm_id_);
  const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
      static_cast<uint64_t>(out.numel()), out.scalar_type(), true);
  const uint64_t chunk_bytes = static_cast<uint64_t>(out.numel()) * out.element_size();
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  const uint64_t scratch = reg.ensure_scratch(*ctx, grp, chunk_bytes * size_, grp.comm_stream);
  for (int r = 0; r < size_; ++r) {
    CommunicatorRegistry::record_tensor_on_stream(*ctx, input_tensors[0][r], grp.comm_stream);
    runtime::PinnedHostPool::copy_d2d(
        scratch + static_cast<uint64_t>(r) * chunk_bytes, tensor_device_addr(input_tensors[0][r]),
        chunk_bytes, grp.comm_stream, true);
  }
  CommunicatorRegistry::record_tensor_on_stream(*ctx, out, grp.comm_stream);
  CommunicatorRegistry::check_hccl(
      hcclReduceScatter(reinterpret_cast<const void*>(scratch), out.data_ptr(),
                        xfer.count, xfer.dtype, red_op, grp.comm, grp.comm_stream),
      "hcclReduceScatter(list)");
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::REDUCE_SCATTER, output_tensors, ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::alltoall_base(
    at::Tensor& output_buffer, at::Tensor& input_buffer,
    std::vector<int64_t>& output_split_sizes,
    std::vector<int64_t>& input_split_sizes,
    const c10d::AllToAllOptions&) {
  auto ctx = active_context();
  check_collective_tensor(output_buffer, *ctx, "alltoall output");
  check_collective_tensor(input_buffer, *ctx, "alltoall input");
  TORCH_CHECK(output_buffer.scalar_type() == input_buffer.scalar_type(),
              "ProcessGroupGaudiRT::alltoall_base dtype mismatch");
  std::vector<at::Tensor> outs{output_buffer};

  const bool uniform = output_split_sizes.empty() && input_split_sizes.empty();
  if (uniform) {
    TORCH_CHECK(output_buffer.numel() == input_buffer.numel() &&
                    input_buffer.numel() % size_ == 0,
                "ProcessGroupGaudiRT::alltoall_base uniform numel must be divisible by world_size");
    if (input_buffer.numel() == 0) {
      return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLTOALL_BASE, std::move(outs), ctx, comm_id_, false, false);
    }
    if (size_ == 1) {
      copy_if_distinct(output_buffer, input_buffer, ctx->stream());
      return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLTOALL_BASE, std::move(outs), ctx, comm_id_, false, false);
    }
    const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
        static_cast<uint64_t>(input_buffer.numel()), input_buffer.scalar_type(), false);
    auto& grp = communicator_registry(*ctx).group(comm_id_);
    grp.compute_to_comm.record(ctx->stream());
    grp.compute_to_comm.wait(grp.comm_stream);
    CommunicatorRegistry::record_tensor_on_stream(*ctx, input_buffer, grp.comm_stream);
    CommunicatorRegistry::record_tensor_on_stream(*ctx, output_buffer, grp.comm_stream);
    CommunicatorRegistry::check_hccl(
        hcclAlltoAll(input_buffer.data_ptr(), output_buffer.data_ptr(),
                     xfer.count, xfer.dtype, grp.comm, grp.comm_stream),
        "hcclAlltoAll");
    grp.comm_to_compute.record(grp.comm_stream);
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLTOALL_BASE, std::move(outs), ctx, comm_id_, true, false);
  }

  TORCH_CHECK((output_split_sizes.empty() || static_cast<int>(output_split_sizes.size()) == size_) &&
                  (input_split_sizes.empty() || static_cast<int>(input_split_sizes.size()) == size_),
              "ProcessGroupGaudiRT::alltoall_base split sizes must match world_size");
  TORCH_CHECK(output_buffer.dim() >= 1 && input_buffer.dim() >= 1,
              "ProcessGroupGaudiRT::alltoall_base variable split requires >= 1D tensors");
  const int64_t row_elems_in = input_buffer.size(0) > 0 ? (input_buffer.numel() / input_buffer.size(0)) : 1;
  const int64_t row_elems_out = output_buffer.size(0) > 0 ? (output_buffer.numel() / output_buffer.size(0)) : 1;
  TORCH_CHECK(row_elems_in == row_elems_out,
              "ProcessGroupGaudiRT::alltoall_base trailing dimensions must match");

  std::vector<int64_t> in_splits = input_split_sizes;
  std::vector<int64_t> out_splits = output_split_sizes;
  if (in_splits.empty()) {
    TORCH_CHECK(input_buffer.size(0) % size_ == 0, "input_buffer.size(0) not divisible by world_size");
    in_splits.assign(size_, input_buffer.size(0) / size_);
  }
  if (out_splits.empty()) {
    TORCH_CHECK(output_buffer.size(0) % size_ == 0, "output_buffer.size(0) not divisible by world_size");
    out_splits.assign(size_, output_buffer.size(0) / size_);
  }
  TORCH_CHECK(std::accumulate(in_splits.begin(), in_splits.end(), int64_t{0}) == input_buffer.size(0) &&
                  std::accumulate(out_splits.begin(), out_splits.end(), int64_t{0}) == output_buffer.size(0),
              "ProcessGroupGaudiRT::alltoall_base split sizes sum mismatch");

  if (size_ == 1) {
    copy_if_distinct(output_buffer, input_buffer, ctx->stream());
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLTOALL_BASE, std::move(outs), ctx, comm_id_, false, false);
  }

  const auto dtype = input_buffer.scalar_type();
  const size_t elem_bytes = input_buffer.element_size();
  auto& grp = communicator_registry(*ctx).group(comm_id_);
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, input_buffer, grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, output_buffer, grp.comm_stream);

  const uint8_t* in_ptr = static_cast<const uint8_t*>(input_buffer.data_ptr());
  uint8_t* out_ptr = static_cast<uint8_t*>(output_buffer.data_ptr());
  int64_t self_in_off = 0;
  int64_t self_out_off = 0;
  for (int r = 0; r < rank_; ++r) {
    self_in_off += in_splits[r] * row_elems_in;
    self_out_off += out_splits[r] * row_elems_out;
  }
  const int64_t self_elems = in_splits[rank_] * row_elems_in;
  if (self_elems > 0 && (in_ptr + self_in_off * elem_bytes) != (out_ptr + self_out_off * elem_bytes)) {
    runtime::PinnedHostPool::copy_d2d(
        reinterpret_cast<uint64_t>(out_ptr + self_out_off * elem_bytes),
        reinterpret_cast<uint64_t>(in_ptr + self_in_off * elem_bytes),
        static_cast<uint64_t>(self_elems) * elem_bytes, grp.comm_stream, true);
  }

  int64_t in_off = 0;
  int64_t out_off = 0;
  CommunicatorRegistry::check_hccl(hcclGroupStart(), "hcclGroupStart");
  for (int r = 0; r < size_; ++r) {
    const int64_t send_elems = in_splits[r] * row_elems_in;
    const int64_t recv_elems = out_splits[r] * row_elems_out;
    if (r != rank_) {
      if (send_elems > 0) {
        const auto sxfer = CommunicatorRegistry::resolve_hccl_transfer(
            static_cast<uint64_t>(send_elems), dtype, false);
        CommunicatorRegistry::check_hccl(
            hcclSend(in_ptr + in_off * elem_bytes, sxfer.count, sxfer.dtype, r, grp.comm, grp.comm_stream),
            "hcclSend(alltoall)");
      }
      if (recv_elems > 0) {
        const auto rxfer = CommunicatorRegistry::resolve_hccl_transfer(
            static_cast<uint64_t>(recv_elems), dtype, false);
        CommunicatorRegistry::check_hccl(
            hcclRecv(out_ptr + out_off * elem_bytes, rxfer.count, rxfer.dtype, r, grp.comm, grp.comm_stream),
            "hcclRecv(alltoall)");
      }
    }
    in_off += send_elems;
    out_off += recv_elems;
  }
  CommunicatorRegistry::check_hccl(hcclGroupEnd(), "hcclGroupEnd");
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLTOALL_BASE, std::move(outs), ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::alltoall(
    std::vector<at::Tensor>& output_tensors,
    std::vector<at::Tensor>& input_tensors,
    const c10d::AllToAllOptions&) {
  TORCH_CHECK(static_cast<int>(output_tensors.size()) == size_ &&
                  static_cast<int>(input_tensors.size()) == size_,
              "ProcessGroupGaudiRT::alltoall expects world_size input and output tensors");
  auto ctx = active_context();
  const auto dtype = input_tensors[0].scalar_type();
  for (int r = 0; r < size_; ++r) {
    check_collective_tensor(input_tensors[r], *ctx, "alltoall input");
    check_collective_tensor(output_tensors[r], *ctx, "alltoall output");
    TORCH_CHECK(input_tensors[r].scalar_type() == dtype && output_tensors[r].scalar_type() == dtype,
                 "ProcessGroupGaudiRT::alltoall dtype mismatch");
  }
  if (size_ == 1) {
    copy_if_distinct(output_tensors[0], input_tensors[0], ctx->stream());
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLTOALL, output_tensors, ctx, comm_id_, false, false);
  }
  auto& grp = communicator_registry(*ctx).group(comm_id_);
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  for (int r = 0; r < size_; ++r) {
    CommunicatorRegistry::record_tensor_on_stream(*ctx, input_tensors[r], grp.comm_stream);
    CommunicatorRegistry::record_tensor_on_stream(*ctx, output_tensors[r], grp.comm_stream);
  }
  TORCH_CHECK(input_tensors[rank_].numel() == output_tensors[rank_].numel(),
              "ProcessGroupGaudiRT::alltoall self-rank numel mismatch");
  copy_if_distinct(output_tensors[rank_], input_tensors[rank_], grp.comm_stream);
  CommunicatorRegistry::check_hccl(hcclGroupStart(), "hcclGroupStart");
  for (int r = 0; r < size_; ++r) {
    if (r == rank_) continue;
    if (input_tensors[r].numel() > 0) {
      const auto sxfer = CommunicatorRegistry::resolve_hccl_transfer(
          static_cast<uint64_t>(input_tensors[r].numel()), dtype, false);
      CommunicatorRegistry::check_hccl(
          hcclSend(input_tensors[r].data_ptr(), sxfer.count, sxfer.dtype, r, grp.comm, grp.comm_stream),
          "hcclSend(alltoall)");
    }
    if (output_tensors[r].numel() > 0) {
      const auto rxfer = CommunicatorRegistry::resolve_hccl_transfer(
          static_cast<uint64_t>(output_tensors[r].numel()), dtype, false);
      CommunicatorRegistry::check_hccl(
          hcclRecv(output_tensors[r].data_ptr(), rxfer.count, rxfer.dtype, r, grp.comm, grp.comm_stream),
          "hcclRecv(alltoall)");
    }
  }
  CommunicatorRegistry::check_hccl(hcclGroupEnd(), "hcclGroupEnd");
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::ALLTOALL, output_tensors, ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::send(
    std::vector<at::Tensor>& tensors, int dst_rank, int /*tag*/) {
  TORCH_CHECK(tensors.size() == 1, "ProcessGroupGaudiRT::send expects 1 tensor");
  TORCH_CHECK(dst_rank >= 0 && dst_rank < size_ && dst_rank != rank_,
              "ProcessGroupGaudiRT::send invalid dst_rank ", dst_rank);
  auto ctx = active_context();
  auto& t = tensors[0];
  check_collective_tensor(t, *ctx, "send");
  if (t.numel() == 0) {
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::SEND, tensors, ctx, comm_id_, false, false);
  }
  auto& reg = communicator_registry(*ctx);
  auto& grp = reg.group(comm_id_);
  const uint64_t bytes = static_cast<uint64_t>(t.numel()) * static_cast<uint64_t>(t.element_size());
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, t, grp.comm_stream);
  if (bytes % 2 != 0) {
    const uint64_t scratch = reg.ensure_scratch(*ctx, grp, bytes + 1, grp.comm_stream);
    runtime::PinnedHostPool::copy_d2d(scratch, tensor_device_addr(t), bytes, grp.comm_stream, true);
    CommunicatorRegistry::check_hccl(
        hcclSend(reinterpret_cast<const void*>(scratch), static_cast<size_t>((bytes + 1) / 2),
                 hcclBfloat16, dst_rank, grp.comm, grp.comm_stream),
        "hcclSend(odd)");
  } else {
    const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
        static_cast<uint64_t>(t.numel()), t.scalar_type(), false);
    CommunicatorRegistry::check_hccl(
        hcclSend(t.data_ptr(), xfer.count, xfer.dtype, dst_rank, grp.comm, grp.comm_stream),
        "hcclSend");
  }
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::SEND, tensors, ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::recv(
    std::vector<at::Tensor>& tensors, int src_rank, int /*tag*/) {
  TORCH_CHECK(tensors.size() == 1, "ProcessGroupGaudiRT::recv expects 1 tensor");
  TORCH_CHECK(src_rank >= 0 && src_rank < size_ && src_rank != rank_,
              "ProcessGroupGaudiRT::recv invalid src_rank ", src_rank);
  auto ctx = active_context();
  auto& t = tensors[0];
  check_collective_tensor(t, *ctx, "recv");
  if (t.numel() == 0) {
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::RECV, tensors, ctx, comm_id_, false, false);
  }
  auto& reg = communicator_registry(*ctx);
  auto& grp = reg.group(comm_id_);
  const uint64_t bytes = static_cast<uint64_t>(t.numel()) * static_cast<uint64_t>(t.element_size());
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  CommunicatorRegistry::record_tensor_on_stream(*ctx, t, grp.comm_stream);
  if (bytes % 2 != 0) {
    const uint64_t scratch = reg.ensure_scratch(*ctx, grp, bytes + 1, grp.comm_stream);
    CommunicatorRegistry::check_hccl(
        hcclRecv(reinterpret_cast<void*>(scratch), static_cast<size_t>((bytes + 1) / 2),
                 hcclBfloat16, src_rank, grp.comm, grp.comm_stream),
        "hcclRecv(odd)");
    runtime::PinnedHostPool::copy_d2d(tensor_device_addr(t), scratch, bytes, grp.comm_stream, true);
  } else {
    const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(
        static_cast<uint64_t>(t.numel()), t.scalar_type(), false);
    CommunicatorRegistry::check_hccl(
        hcclRecv(t.data_ptr(), xfer.count, xfer.dtype, src_rank, grp.comm, grp.comm_stream),
        "hcclRecv");
  }
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::RECV, tensors, ctx, comm_id_, true, false);
}

c10::intrusive_ptr<c10d::Work> ProcessGroupGaudiRT::barrier(const c10d::BarrierOptions&) {
  auto ctx = active_context();
  if (size_ == 1) {
    return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::BARRIER, std::vector<at::Tensor>{}, ctx, comm_id_, false, false);
  }
  auto& grp = communicator_registry(*ctx).group(comm_id_);
  TORCH_CHECK(grp.barrier_address != 0, "ProcessGroupGaudiRT::barrier missing barrier slot");
  grp.compute_to_comm.record(ctx->stream());
  grp.compute_to_comm.wait(grp.comm_stream);
  ctx->allocator.record_stream(grp.barrier_address, grp.comm_stream);
  // Gaudi2 libhcl returns hcclUnsupported for hcclBarrier in scale-up NIC mode;
  // synchronize streams and ranks via a 1-element zeroed FP32 hcclAllReduce.
  CommunicatorRegistry::check_hccl(
      hcclAllReduce(reinterpret_cast<const void*>(grp.barrier_address),
                    reinterpret_cast<void*>(grp.barrier_address),
                    1, hcclFloat32, hcclSum, grp.comm, grp.comm_stream),
      "hcclAllReduce(barrier)");
  grp.comm_to_compute.record(grp.comm_stream);
  return c10::make_intrusive<WorkGaudiRT>(rank_, c10d::OpType::BARRIER, std::vector<at::Tensor>{}, ctx, comm_id_, true, true);
}

void ProcessGroupGaudiRT::abort() {
  std::lock_guard lock(init_mutex_);
  if (initialized_ && !closed_) {
    if (auto ctx = owner_.lock(); ctx && ctx->comm_state) {
      communicator_registry(*ctx).destroy_group(*ctx, comm_id_, true);
    }
    closed_ = true;
  }
}

void ProcessGroupGaudiRT::shutdown() {
  std::lock_guard lock(init_mutex_);
  if (initialized_ && !closed_) {
    if (auto ctx = owner_.lock(); ctx && ctx->comm_state) {
      communicator_registry(*ctx).destroy_group(*ctx, comm_id_, std::uncaught_exceptions() > 0);
    }
    closed_ = true;
  }
}

c10::intrusive_ptr<c10d::Backend> create_process_group_gaudirt(
    const c10::intrusive_ptr<c10d::Store>& store, int rank, int size,
    const std::string& group_id) {
  return c10::make_intrusive<ProcessGroupGaudiRT>(store, rank, size, group_id);
}

}  // namespace gaudirt::generic
