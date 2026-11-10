// SPDX-License-Identifier: Apache-2.0
#include "segmented_runner.hpp"

#include "bucket_program.hpp"
#include "bridge_internal.hpp"
#include "paged_attention.hpp"

#include <c10/util/Exception.h>
#include <chrono>

namespace gaudirt::generic {

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
  step.attention = true;
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

void SegmentedRunner::set_outputs(std::vector<size_t> outputs) {
  TORCH_CHECK(uses_.empty() && !outputs.empty(), "GaudiRT segments: invalid output plan");
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
    if (step->attention) continue;
    for (size_t i = 0; i < step->routes.size(); ++i)
      if (escapes[step->outputs[i]] && step->routes[i].input)
        escapes[step->inputs[step->routes[i].index]] = true;
  }
  for (auto& step : instructions_) {
    if (!step.program) continue;
    for (const auto& route : step.routes)
      if (!route.input && route.index >= step.reusable_outputs.size())
        step.reusable_outputs.resize(route.index + 1, false);
    for (size_t i = 0; i < step.routes.size(); ++i)
      if (!step.routes[i].input && !escapes[step.outputs[i]])
        step.reusable_outputs[step.routes[i].index] = true;
    for (size_t i = 0; i < step.routes.size(); ++i)
      if (!step.routes[i].input && escapes[step.outputs[i]])
        step.reusable_outputs[step.routes[i].index] = false;
  }
}

void SegmentedRunner::bind_static(const std::vector<size_t>& slots, const std::vector<at::Tensor>& inputs) {
  TORCH_CHECK(!uses_.empty() && static_inputs_.empty() && !slots.empty() && slots.size() == inputs.size(),
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
  TORCH_CHECK(!uses_.empty() && (bound ? !static_inputs_.empty() && inputs.size() == dynamic_slots_.size()
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
      TORCH_CHECK(synEventRecord((*events)[2 * step_index], current()->stream()) == synSuccess,
                  "GaudiRT segments: profiling start event failed");
    }
    const auto host_start = events ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    const auto compiled_before = recording ? recipe_cache_stats()[1] : 0;
    arguments.clear();
    arguments.reserve(step.inputs.size());
    for (auto slot : step.inputs) arguments.push_back(values[slot]);
    if (step.attention) {
      values[step.outputs[0]] = paged_attn_internal(arguments[0], arguments[1], arguments[2], arguments[3],
                                                     arguments[4], arguments[5], arguments[6], arguments[7],
                                                     arguments[8], arguments[9], step.head_dim, step.scale,
                                                     step.trusted[0], step.trusted[1], step.trusted[2]);
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
      TORCH_CHECK(synEventRecord((*events)[2 * step_index + 1], current()->stream()) == synSuccess,
                  "GaudiRT segments: profiling end event failed");
    }
    if (recording) {
      std::vector<at::Tensor> result;
      result.reserve(step.outputs.size());
      for (auto slot : step.outputs) result.push_back(values[slot]);
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
