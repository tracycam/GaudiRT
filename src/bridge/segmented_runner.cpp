// SPDX-License-Identifier: Apache-2.0
#include "segmented_runner.hpp"

#include "bridge_internal.hpp"
#include "bucket_program.hpp"
#include "paged_attention.hpp"

#include <ATen/Functions.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <bit>
#include <chrono>

namespace gaudirt::generic {
namespace {

uint32_t round_static_bucket_elements(uint64_t exact) {
  if (exact == 0 || exact > (1ULL << 30)) return 0;
  const uint64_t padded = std::max<uint64_t>(256, std::bit_ceil(exact));
  return padded <= UINT32_MAX ? static_cast<uint32_t>(padded) : 0;
}

at::Tensor allocate_collective_output(
    const at::Tensor& src, CollectiveKind kind, int world_size,
    bool dynamic_output, bool allow_reusable_pool, SegmentOutputPool::Frame* frame,
    uint32_t* out_bucket_elems) {
  const auto dtype = src.scalar_type();
  *out_bucket_elems = 0;

  std::vector<int64_t> out_sizes(src.sizes().begin(), src.sizes().end());
  TORCH_CHECK(!out_sizes.empty(), "GaudiRT collective: scalar tensor unsupported in SegmentedRunner");

  switch (kind) {
    case CollectiveKind::AllReduce:
      break;
    case CollectiveKind::AllGather:
      out_sizes[0] *= world_size;
      break;
    case CollectiveKind::AllGatherDimLast:
      out_sizes.back() *= world_size;
      break;
    case CollectiveKind::ReduceScatter:
      TORCH_CHECK(out_sizes[0] % world_size == 0,
                  "GaudiRT collective: ReduceScatter dim 0 (", out_sizes[0],
                  ") not divisible by world_size ", world_size);
      out_sizes[0] /= world_size;
      break;
    case CollectiveKind::AllToAll:
      TORCH_CHECK(out_sizes[0] % world_size == 0,
                  "GaudiRT collective: AllToAll dim 0 (", out_sizes[0],
                  ") not divisible by world_size ", world_size);
      break;
  }

  const int64_t out_m = out_sizes[0];
  if (dynamic_output && out_m > 0 && out_m <= 4096) {
    int64_t width = 1;
    for (size_t d = 1; d < out_sizes.size(); ++d) width *= out_sizes[d];
    const uint32_t bucket = std::bit_ceil(static_cast<uint32_t>(out_m));
    const uint64_t total_bucket_elems = static_cast<uint64_t>(bucket) * static_cast<uint64_t>(width);
    if (total_bucket_elems <= UINT32_MAX) {
      *out_bucket_elems = static_cast<uint32_t>(total_bucket_elems);
    }

    const bool same_m_as_src = (kind == CollectiveKind::AllReduce || kind == CollectiveKind::AllToAll);
    if (frame && allow_reusable_pool && same_m_as_src) {
      frame->outputs.prepare(bucket);
      auto full_sizes = out_sizes;
      full_sizes[0] = bucket;
      auto stored = frame->outputs.acquire(src, dtype, full_sizes, true, width);
      return out_m == static_cast<int64_t>(bucket) ? stored : stored.narrow(0, 0, out_m);
    }
    auto raw = bucket_output(src, dtype, bucket, out_m, width);
    return out_sizes.size() == 2 ? raw : raw.view(out_sizes);
  }

  const uint64_t exact_out_elems = [&]() {
    uint64_t n = 1;
    for (int64_t s : out_sizes) n *= static_cast<uint64_t>(s);
    return n;
  }();
  *out_bucket_elems = round_static_bucket_elements(exact_out_elems);

  if (frame && allow_reusable_pool) {
    return frame->outputs.acquire(src, dtype, out_sizes, false, 0);
  }
  return at::empty(out_sizes, src.options());
}

void launch_segment_collective(
    Context& ctx, CommGroup& grp, CollectiveKind kind,
    const at::Tensor& src, const at::Tensor& dst, bool src_trusted,
    uint32_t bucket_elems, bool use_comm_stream) {
  TORCH_CHECK(src.defined() && src.device().is_privateuseone() &&
                  src.device().index() == ctx.device_index && src.is_contiguous(),
              "GaudiRT collective: input must be contiguous on gaudirt:", ctx.device_index);

  synStreamHandle compute_stream = ctx.stream();
  const int W = grp.world_size;
  const uint64_t src_bytes = static_cast<uint64_t>(src.numel()) * src.element_size();
  const uint64_t src_addr = reinterpret_cast<uint64_t>(src.data_ptr());
  const uint64_t dst_addr = reinterpret_cast<uint64_t>(dst.data_ptr());

  if (!src_trusted && src_bytes > 0) {
    const auto& alloc = validated_program_input(src, ctx);
    if (alloc.stream != compute_stream) {
      if (alloc.kind == AllocationKind::graph) ctx.graph_pool->record_stream(src_addr, compute_stream);
      else ctx.allocator.record_stream(src_addr, compute_stream);
    }
  }

  if (src_bytes == 0) return;
  if (W == 1) {
    if (dst_addr != src_addr) {
      runtime::PinnedHostPool::copy_d2d(dst_addr, src_addr, src_bytes, compute_stream, true);
    }
    return;
  }

  synStreamHandle exec_stream = use_comm_stream ? grp.comm_stream : compute_stream;
  if (use_comm_stream) {
    grp.compute_to_comm.record(compute_stream);
    grp.compute_to_comm.wait(grp.comm_stream);
    CommunicatorRegistry::record_tensor_on_stream(ctx, src, grp.comm_stream);
    CommunicatorRegistry::record_tensor_on_stream(ctx, dst, grp.comm_stream);
  }

  auto& reg = communicator_registry(ctx);
  const auto dtype = src.scalar_type();
  const auto src_elems = static_cast<uint64_t>(src.numel());
  const auto dst_elems = static_cast<uint64_t>(dst.numel());

  switch (kind) {
    case CollectiveKind::AllReduce: {
      const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(src_elems, dtype, true);
      if (CommunicatorRegistry::should_gather_reduce(src_elems, bucket_elems, W, dtype)) {
        const uint64_t gather_bytes =
            static_cast<uint64_t>(W) * bucket_elems * static_cast<uint64_t>(src.element_size());
        const uint64_t scratch = reg.ensure_scratch(ctx, grp, gather_bytes, exec_stream);
        CommunicatorRegistry::check_hccl(
            hcclAllGather(src.data_ptr(), reinterpret_cast<void*>(scratch),
                          xfer.count, xfer.dtype, grp.comm, exec_stream),
            "hcclAllGather(segment_gather_reduce)");
        CommunicatorRegistry::launch_rank_sum(
            ctx, scratch, dst_addr, src_elems, bucket_elems,
            static_cast<uint32_t>(W), dtype, exec_stream);
      } else {
        CommunicatorRegistry::check_hccl(
            hcclAllReduce(src.data_ptr(), dst.data_ptr(), xfer.count,
                          xfer.dtype, hcclSum, grp.comm, exec_stream),
            "hcclAllReduce(segment)");
      }
      break;
    }
    case CollectiveKind::AllGather: {
      const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(src_elems, dtype, false);
      CommunicatorRegistry::check_hccl(
          hcclAllGather(src.data_ptr(), dst.data_ptr(), xfer.count,
                        xfer.dtype, grp.comm, exec_stream),
          "hcclAllGather(segment)");
      break;
    }
    case CollectiveKind::AllGatherDimLast: {
      const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(src_elems, dtype, false);
      const uint64_t cols = static_cast<uint64_t>(src.size(-1));
      const uint64_t batch_rows = cols > 0 ? (src_elems / cols) : 0;
      if (batch_rows <= 1) {
        // When B == 1, [W, 1, cols] and [1, W * cols] have identical memory layout.
        CommunicatorRegistry::check_hccl(
            hcclAllGather(src.data_ptr(), dst.data_ptr(), xfer.count,
                          xfer.dtype, grp.comm, exec_stream),
            "hcclAllGather(dim_last_b1)");
      } else {
        const uint64_t gather_bytes = dst_elems * static_cast<uint64_t>(src.element_size());
        const uint64_t scratch = reg.ensure_scratch(ctx, grp, gather_bytes, exec_stream);
        CommunicatorRegistry::check_hccl(
            hcclAllGather(src.data_ptr(), reinterpret_cast<void*>(scratch),
                          xfer.count, xfer.dtype, grp.comm, exec_stream),
            "hcclAllGather(dim_last)");
        CommunicatorRegistry::launch_gather_transpose(
            ctx, scratch, dst_addr, static_cast<uint64_t>(W), batch_rows, cols, dtype, exec_stream);
      }
      break;
    }
    case CollectiveKind::ReduceScatter: {
      const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(dst_elems, dtype, true);
      CommunicatorRegistry::check_hccl(
          hcclReduceScatter(src.data_ptr(), dst.data_ptr(), xfer.count,
                            xfer.dtype, hcclSum, grp.comm, exec_stream),
          "hcclReduceScatter(segment)");
      break;
    }
    case CollectiveKind::AllToAll: {
      const auto xfer = CommunicatorRegistry::resolve_hccl_transfer(src_elems, dtype, false);
      CommunicatorRegistry::check_hccl(
          hcclAlltoAll(src.data_ptr(), dst.data_ptr(), xfer.count,
                       xfer.dtype, grp.comm, exec_stream),
          "hcclAlltoAll(segment)");
      break;
    }
  }

  if (use_comm_stream) {
    grp.comm_to_compute.record(grp.comm_stream);
  }
}

}  // namespace

SegmentOutputPool::Frame* SegmentOutputPool::acquire(uint64_t generation, synStreamHandle stream) {
  std::lock_guard lock(mutex_);
  for (const auto& frame : frames_) {
    if (frame->busy || frame->generation != generation || frame->stream != stream) continue;
    frame->outputs.reclaim();
    frame->busy = true;
    return frame.get();
  }
  auto frame = std::make_unique<Frame>();
  frame->generation = generation;
  frame->stream = stream;
  frame->busy = true;
  auto* result = frame.get();
  frames_.push_back(std::move(frame));
  return result;
}

void SegmentOutputPool::release(Frame* frame) {
  std::lock_guard lock(mutex_);
  frame->outputs.reclaim();
  frame->busy = false;
}

bool SegmentOutputPool::trim() {
  std::lock_guard lock(mutex_);
  for (const auto& frame : frames_) if (frame->busy) return false;
  frames_.clear();
  return true;
}

SegmentedRunner::SegmentedRunner(size_t input_count)
    : input_count_(input_count), slots_(input_count), trusted_slots_(input_count, false) {}

std::vector<size_t> SegmentedRunner::add_program(std::shared_ptr<BucketProgram> program,
                                                  std::vector<size_t> inputs,
                                                  std::vector<SegmentRoute> outputs) {
  TORCH_CHECK(uses_.empty() && !outputs.empty(), "GaudiRT segments: invalid program plan");
  for (auto input : inputs) TORCH_CHECK(input < slots_, "GaudiRT segments: unknown program input");
  for (const auto& output : outputs) {
    TORCH_CHECK(!output.input || output.index < inputs.size(), "GaudiRT segments: invalid input route");
    for (const auto& view : output.views)
      TORCH_CHECK((view.kind == SegmentViewKind::reshape && !view.arguments.empty()) ||
                  (view.kind != SegmentViewKind::reshape && view.arguments.size() == 2),
                  "GaudiRT segments: invalid view route");
  }
  Instruction step;
  step.kind = StepKind::Program;
  step.program = std::move(program);
  step.inputs = std::move(inputs);
  step.routes = std::move(outputs);
  for (auto input : step.inputs) step.trusted.push_back(trusted_slots_[input]);
  for (const auto& route : step.routes) {
    step.outputs.push_back(slots_++);
    trusted_slots_.push_back(step.program || (route.input && trusted_slots_[step.inputs[route.index]]));
  }
  auto result = step.outputs;
  instructions_.push_back(std::move(step));
  return result;
}

size_t SegmentedRunner::add_attention(std::vector<size_t> inputs, int64_t head_dim, double scale) {
  TORCH_CHECK(uses_.empty() && inputs.size() == 10, "GaudiRT segments: invalid attention plan");
  for (auto input : inputs) TORCH_CHECK(input < slots_, "GaudiRT segments: unknown attention input");
  Instruction step;
  step.kind = StepKind::Attention;
  step.inputs = std::move(inputs);
  for (auto input : step.inputs) step.trusted.push_back(trusted_slots_[input]);
  step.head_dim = head_dim;
  step.scale = scale;
  step.outputs.push_back(slots_++);
  trusted_slots_.push_back(true);
  auto result = step.outputs[0];
  instructions_.push_back(std::move(step));
  return result;
}

uint32_t SegmentedRunner::add_collective_start(uint32_t comm_id, CollectiveKind collective_kind,
                                               size_t input_slot, bool dynamic_output) {
  TORCH_CHECK(uses_.empty() && input_slot < slots_,
              "GaudiRT segments: invalid collective_start input slot ", input_slot);
  const auto coll_id = static_cast<uint32_t>(collectives_.size());
  Instruction step;
  step.kind = StepKind::CollectiveStart;
  step.inputs = {input_slot};
  step.trusted = {trusted_slots_[input_slot]};
  step.coll_id = coll_id;
  step.comm_id = comm_id;
  step.collective_kind = collective_kind;
  step.dynamic_output = dynamic_output;

  collectives_.push_back(CollectiveDescriptor{
      comm_id, collective_kind, input_slot, instructions_.size(), dynamic_output, false});
  instructions_.push_back(std::move(step));
  return coll_id;
}

size_t SegmentedRunner::add_collective_wait(uint32_t coll_id) {
  TORCH_CHECK(uses_.empty() && coll_id < collectives_.size(),
              "GaudiRT segments: unknown collective handle ", coll_id);
  auto& desc = collectives_[coll_id];
  TORCH_CHECK(!desc.waited, "GaudiRT segments: duplicate collective_wait for ", coll_id);
  desc.waited = true;

  Instruction step;
  step.kind = StepKind::CollectiveWait;
  // Pin src_slot through CollectiveWait so SegmentOutputPool never recycles
  // src_slot while hccl* is still reading it on comm_stream.
  step.inputs = {desc.src_slot};
  step.trusted = {trusted_slots_[desc.src_slot]};
  step.coll_id = coll_id;
  step.comm_id = desc.comm_id;
  step.collective_kind = desc.kind;
  step.dynamic_output = desc.dynamic_output;
  step.outputs.push_back(slots_++);
  trusted_slots_.push_back(true);
  const size_t out_slot = step.outputs[0];
  instructions_.push_back(std::move(step));
  return out_slot;
}

void SegmentedRunner::set_outputs(std::vector<size_t> outputs) {
  TORCH_CHECK(uses_.empty() && !outputs.empty(), "GaudiRT segments: invalid output plan");
  for (const auto& desc : collectives_) {
    TORCH_CHECK(desc.waited, "GaudiRT segments: unwaited collective ", desc.comm_id);
  }
  outputs_ = std::move(outputs);
  uses_.resize(slots_, 0);
  for (const auto& step : instructions_)
    for (auto slot : step.inputs) ++uses_[slot];
  for (auto slot : outputs_) {
    TORCH_CHECK(slot < slots_, "GaudiRT segments: unknown output");
    ++uses_[slot];
  }

  std::vector<bool> escapes(slots_, false);
  for (auto slot : outputs_) escapes[slot] = true;
  for (auto step = instructions_.rbegin(); step != instructions_.rend(); ++step) {
    if (step->kind != StepKind::Program) continue;
    for (size_t i = 0; i < step->routes.size(); ++i)
      if (escapes[step->outputs[i]] && step->routes[i].input)
        escapes[step->inputs[step->routes[i].index]] = true;
  }

  for (size_t idx = 0; idx < instructions_.size(); ++idx) {
    auto& step = instructions_[idx];
    if (step.kind == StepKind::Program && step.program) {
      for (const auto& route : step.routes)
        if (!route.input && route.index >= step.reusable_outputs.size())
          step.reusable_outputs.resize(route.index + 1, false);
      for (size_t i = 0; i < step.routes.size(); ++i)
        if (!step.routes[i].input && !escapes[step.outputs[i]])
          step.reusable_outputs[step.routes[i].index] = true;
      for (size_t i = 0; i < step.routes.size(); ++i)
        if (!step.routes[i].input && escapes[step.outputs[i]])
          step.reusable_outputs[step.routes[i].index] = false;
    } else if (step.kind == StepKind::CollectiveWait) {
      auto& start_step = instructions_[collectives_[step.coll_id].start_step_index];
      const size_t start_idx = collectives_[step.coll_id].start_step_index;
      bool has_intervening_work = false;
      for (size_t k = start_idx + 1; k < idx; ++k) {
        const auto& mid = instructions_[k];
        if (mid.kind == StepKind::Attention ||
            (mid.kind == StepKind::Program && mid.program != nullptr) ||
            (mid.kind == StepKind::CollectiveStart && mid.comm_id != step.comm_id)) {
          has_intervening_work = true;
          break;
        }
      }
      start_step.use_comm_stream = has_intervening_work;
      step.use_comm_stream = has_intervening_work;
      start_step.reusable_collective = !escapes[step.outputs[0]];
    }
  }
}

std::vector<SegmentedRunner::ScheduleItem> SegmentedRunner::schedule_summary() const {
  std::vector<ScheduleItem> items;
  items.reserve(instructions_.size());
  for (const auto& step : instructions_) {
    switch (step.kind) {
      case StepKind::Program:
        items.emplace_back(step.program ? "program" : "view", 0U, false);
        break;
      case StepKind::Attention:
        items.emplace_back("attention", 0U, false);
        break;
      case StepKind::CollectiveStart:
        items.emplace_back("collective_start", step.coll_id, step.use_comm_stream);
        break;
      case StepKind::CollectiveWait:
        items.emplace_back("collective_wait", step.coll_id, step.use_comm_stream);
        break;
    }
  }
  return items;
}

void SegmentedRunner::bind_static(const std::vector<size_t>& slots, const std::vector<at::Tensor>& inputs) {
  TORCH_CHECK(!uses_.empty() && static_inputs_.empty() &&
                  slots.size() == inputs.size() && slots.size() <= input_count_,
              "GaudiRT segments: invalid static bindings");
  std::vector<bool> seen(input_count_, false);
  static_slots_ = slots;
  static_inputs_.reserve(slots.size());
  for (size_t i = 0; i < slots.size(); ++i) {
    TORCH_CHECK(slots[i] < input_count_ && !seen[slots[i]] && inputs[i].defined(),
                "GaudiRT segments: invalid static slot");
    seen[slots[i]] = true;
    static_inputs_.emplace_back(inputs[i].getIntrusivePtr());
  }
  for (size_t i = 0; i < input_count_; ++i)
    if (!seen[i]) dynamic_slots_.push_back(i);
  output_pool_ = std::make_shared<SegmentOutputPool>();
  current()->register_segment_outputs(output_pool_);
}

SegmentedRunner::TracedResult SegmentedRunner::execute(const std::vector<at::Tensor>& inputs,
                                                        const std::vector<bool>& trace,
                                                        const std::vector<synEventHandle>* events,
                                                        std::vector<double>* host_ms, bool bound) const {
  TORCH_CHECK(!uses_.empty() && (bound ? output_pool_ != nullptr && inputs.size() == dynamic_slots_.size()
                                       : inputs.size() == input_count_),
              "GaudiRT segments: invalid input count");
  auto ctx = current();
  ContextScope scope(ctx);
  auto* frame = bound ? output_pool_->acquire(ctx->generation, ctx->stream()) : nullptr;
  const auto release_frame = [this](SegmentOutputPool::Frame* value) {
    if (value) output_pool_->release(value);
  };
  std::unique_ptr<SegmentOutputPool::Frame, decltype(release_frame)> frame_guard(frame, release_frame);
  std::vector<at::Tensor> values(slots_);
  std::vector<at::Tensor> pending_collectives(collectives_.size());
  if (bound) {
    for (size_t i = 0; i < static_slots_.size(); ++i) {
      auto tensor = static_inputs_[i].lock();
      TORCH_CHECK(tensor, "GaudiRT segments: static tensor expired");
      values[static_slots_[i]] = at::Tensor(std::move(tensor));
    }
    for (size_t i = 0; i < inputs.size(); ++i) values[dynamic_slots_[i]] = inputs[i];
  } else {
    for (size_t i = 0; i < inputs.size(); ++i) values[i] = inputs[i];
  }
  auto uses = uses_;
  std::vector<SegmentTrace> recorded(trace.empty() ? 0 : instructions_.size());
  std::vector<at::Tensor> arguments;
  for (size_t step_index = 0; step_index < instructions_.size(); ++step_index) {
    const auto& step = instructions_[step_index];
    const bool recording = !trace.empty() && trace[step_index];
    if (events) {
      TORCH_CHECK(synEventRecord((*events)[2 * step_index], ctx->stream()) == synSuccess,
                  "GaudiRT segments: profiling start event failed");
    }
    const auto host_start = events ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto compiled_before = recording ? recipe_cache_stats()[1] : 0;
    arguments.clear();
    arguments.reserve(step.inputs.size());
    for (auto slot : step.inputs) arguments.push_back(values[slot]);

    if (step.kind == StepKind::Attention) {
      values[step.outputs[0]] = paged_attn_internal(arguments[0], arguments[1], arguments[2], arguments[3],
                                                    arguments[4], arguments[5], arguments[6], arguments[7],
                                                    arguments[8], arguments[9], step.head_dim, step.scale,
                                                    step.trusted[0], step.trusted[1], step.trusted[2]);
    } else if (step.kind == StepKind::CollectiveStart) {
      auto& grp = communicator_registry(*ctx).group(step.comm_id);
      uint32_t bucket_elems = 0;
      auto dst = allocate_collective_output(
          arguments[0], step.collective_kind, grp.world_size,
          step.dynamic_output, step.reusable_collective, frame, &bucket_elems);
      launch_segment_collective(
          *ctx, grp, step.collective_kind, arguments[0], dst,
          step.trusted[0], bucket_elems, step.use_comm_stream);
      pending_collectives[step.coll_id] = std::move(dst);
    } else if (step.kind == StepKind::CollectiveWait) {
      if (step.use_comm_stream) {
        auto& grp = communicator_registry(*ctx).group(step.comm_id);
        if (grp.world_size > 1 && grp.comm_stream) {
          grp.comm_to_compute.wait(ctx->stream());
        }
      }
      values[step.outputs[0]] = std::move(pending_collectives[step.coll_id]);
    } else {
      auto computed = step.program ? step.program->run_internal(arguments, step.trusted,
          frame ? &frame->outputs : nullptr, &step.reusable_outputs) : std::vector<at::Tensor>{};
      for (size_t route_index = 0; route_index < step.routes.size(); ++route_index) {
        const auto& route = step.routes[route_index];
        TORCH_CHECK(route.input || route.index < computed.size(), "GaudiRT segments: invalid program output route");
        auto value = route.input ? arguments[route.index] : computed[route.index];
        for (const auto& view : route.views) {
          if (view.kind == SegmentViewKind::feature) value = value.narrow(1, view.arguments[0], view.arguments[1]);
          else if (view.kind == SegmentViewKind::reshape) value = value.view(view.arguments);
          else value = value.transpose(view.arguments[0], view.arguments[1]);
        }
        values[step.outputs[route_index]] = std::move(value);
      }
    }

    if (events) {
      (*host_ms)[step_index] = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - host_start).count();
      TORCH_CHECK(synEventRecord((*events)[2 * step_index + 1], ctx->stream()) == synSuccess,
                  "GaudiRT segments: profiling end event failed");
    }
    if (recording) {
      std::vector<at::Tensor> result;
      if (step.kind == StepKind::CollectiveStart) {
        result.push_back(pending_collectives[step.coll_id]);
      } else {
        result.reserve(step.outputs.size());
        for (auto slot : step.outputs) result.push_back(values[slot]);
      }
      recorded[step_index] = {arguments, std::move(result), recipe_cache_stats()[1] - compiled_before};
    }
    for (auto slot : step.inputs)
      if (--uses[slot] == 0) values[slot] = at::Tensor{};
    for (auto slot : step.outputs)
      if (uses[slot] == 0) values[slot] = at::Tensor{};
    arguments.clear();
    if (frame) frame->outputs.reclaim();
  }
  std::vector<at::Tensor> result;
  result.reserve(outputs_.size());
  for (auto slot : outputs_) result.push_back(values[slot]);
  return {std::move(result), std::move(recorded)};
}

std::vector<at::Tensor> SegmentedRunner::run(const std::vector<at::Tensor>& inputs) const {
  return execute(inputs, {}).first;
}

std::vector<at::Tensor> SegmentedRunner::run_bound(const std::vector<at::Tensor>& inputs) const {
  return execute(inputs, {}, nullptr, nullptr, true).first;
}

SegmentedRunner::TracedResult SegmentedRunner::run_traced(const std::vector<at::Tensor>& inputs,
                                                           const std::vector<size_t>& traced) const {
  return this->traced(inputs, traced, false);
}

SegmentedRunner::TracedResult SegmentedRunner::run_bound_traced(const std::vector<at::Tensor>& inputs,
                                                                 const std::vector<size_t>& traced) const {
  return this->traced(inputs, traced, true);
}

SegmentedRunner::TracedResult SegmentedRunner::traced(const std::vector<at::Tensor>& inputs,
                                                       const std::vector<size_t>& traced, bool bound) const {
  std::vector<bool> mask(instructions_.size(), false);
  for (auto index : traced) {
    TORCH_CHECK(index < mask.size(), "GaudiRT segments: invalid trace index");
    mask[index] = true;
  }
  return execute(inputs, mask, nullptr, nullptr, bound);
}

SegmentedRunner::ProfiledResult SegmentedRunner::run_profiled(const std::vector<at::Tensor>& inputs,
                                                             const std::vector<uint64_t>& event_ids) const {
  return profiled(inputs, event_ids, false);
}

SegmentedRunner::ProfiledResult SegmentedRunner::run_bound_profiled(
    const std::vector<at::Tensor>& inputs, const std::vector<uint64_t>& event_ids) const {
  return profiled(inputs, event_ids, true);
}

SegmentedRunner::ProfiledResult SegmentedRunner::profiled(const std::vector<at::Tensor>& inputs,
                                                           const std::vector<uint64_t>& event_ids, bool bound) const {
  TORCH_CHECK(event_ids.size() == instructions_.size() * 2, "GaudiRT segments: wrong profiling event count");
  auto ctx = current();
  std::vector<synEventHandle> events;
  events.reserve(event_ids.size());
  {
    std::lock_guard lock(event_mutex);
    for (auto id : event_ids) {
      auto* event = reinterpret_cast<EventState*>(id);
      TORCH_CHECK(ctx->events.count(event) && event->owner == ctx.get() && event->timing && event->handle,
                  "GaudiRT segments: invalid profiling event");
      events.push_back(event->handle);
    }
  }
  std::vector<double> host_ms(instructions_.size());
  auto result = execute(inputs, {}, &events, &host_ms, bound);
  return {std::move(result.first), std::move(host_ms)};
}

}  // namespace gaudirt::generic
