// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "bucket_program.hpp"
#include "communicator.hpp"

#include <ATen/core/Tensor.h>
#include <c10/util/intrusive_ptr.h>
#include <synapse_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace gaudirt::generic {

class SegmentOutputPool {
 public:
  struct Frame {
    uint64_t generation;
    synStreamHandle stream;
    bool busy = false;
    BucketProgram::ReusableOutputs outputs;
  };
  Frame* acquire(uint64_t generation, synStreamHandle stream);
  void release(Frame* frame);
  bool trim();

 private:
  std::mutex mutex_;
  std::vector<std::unique_ptr<Frame>> frames_;
};

enum class SegmentViewKind { feature, reshape, transpose };

struct SegmentView {
  SegmentView(SegmentViewKind kind, std::vector<int64_t> arguments)
      : kind(kind), arguments(std::move(arguments)) {}
  SegmentViewKind kind;
  std::vector<int64_t> arguments;
};

struct SegmentRoute {
  SegmentRoute(bool input, size_t index, std::vector<SegmentView> views)
      : input(input), index(index), views(std::move(views)) {}
  bool input;
  size_t index;
  std::vector<SegmentView> views;
};

// Runs a DAG-scheduled sequence of independent FX programs, stateful attention
// operators, and split-phase HCCL collectives (CollectiveStart / CollectiveWait).
class SegmentedRunner {
 public:
  explicit SegmentedRunner(size_t input_count);
  std::vector<size_t> add_program(std::shared_ptr<BucketProgram> program, std::vector<size_t> inputs,
                                  std::vector<SegmentRoute> outputs);
  size_t add_attention(std::vector<size_t> inputs, int64_t head_dim, double scale,
                       double softcap = 0.0, int64_t sliding_window = -1);
  uint32_t add_collective_start(uint32_t comm_id, CollectiveKind collective_kind,
                                size_t input_slot, bool dynamic_output);
  size_t add_collective_wait(uint32_t coll_id);
  void set_outputs(std::vector<size_t> outputs);
  void bind_static(const std::vector<size_t>& slots, const std::vector<at::Tensor>& inputs);
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs) const;
  std::vector<at::Tensor> run_bound(const std::vector<at::Tensor>& inputs) const;
  using SegmentTrace = std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, uint64_t>;
  using TracedResult = std::pair<std::vector<at::Tensor>, std::vector<SegmentTrace>>;
  TracedResult run_traced(const std::vector<at::Tensor>& inputs, const std::vector<size_t>& traced) const;
  TracedResult run_bound_traced(const std::vector<at::Tensor>& inputs, const std::vector<size_t>& traced) const;
  using ProfiledResult = std::pair<std::vector<at::Tensor>, std::vector<double>>;
  ProfiledResult run_profiled(const std::vector<at::Tensor>& inputs, const std::vector<uint64_t>& events) const;
  ProfiledResult run_bound_profiled(const std::vector<at::Tensor>& inputs, const std::vector<uint64_t>& events) const;

  using ScheduleItem = std::tuple<std::string, uint32_t, bool>;
  std::vector<ScheduleItem> schedule_summary() const;

 private:
  enum class StepKind : uint8_t {
    Program = 0,
    Attention = 1,
    CollectiveStart = 2,
    CollectiveWait = 3,
  };

  struct CollectiveDescriptor {
    uint32_t comm_id = 0;
    CollectiveKind kind = CollectiveKind::AllReduce;
    size_t src_slot = 0;
    size_t start_step_index = 0;
    bool dynamic_output = false;
    bool waited = false;
  };

  struct Instruction {
    StepKind kind = StepKind::Program;
    std::shared_ptr<BucketProgram> program;
    std::vector<size_t> inputs;
    std::vector<bool> trusted;
    std::vector<SegmentRoute> routes;
    std::vector<bool> reusable_outputs;
    std::vector<size_t> outputs;
    int64_t head_dim = 0;
    double scale = 0;
    double softcap = 0.0;
    int64_t sliding_window = -1;
    uint32_t coll_id = 0;
    uint32_t comm_id = 0;
    CollectiveKind collective_kind = CollectiveKind::AllReduce;
    bool dynamic_output = false;
    bool reusable_collective = false;
    bool use_comm_stream = false;
  };

  TracedResult execute(const std::vector<at::Tensor>& inputs, const std::vector<bool>& trace,
                       const std::vector<synEventHandle>* events = nullptr, std::vector<double>* host_ms = nullptr,
                       bool bound = false) const;
  TracedResult traced(const std::vector<at::Tensor>& inputs, const std::vector<size_t>& traced, bool bound) const;
  ProfiledResult profiled(const std::vector<at::Tensor>& inputs, const std::vector<uint64_t>& events, bool bound) const;

  size_t input_count_;
  size_t slots_;
  std::vector<Instruction> instructions_;
  std::vector<CollectiveDescriptor> collectives_;
  std::vector<bool> trusted_slots_;
  std::vector<size_t> outputs_;
  std::vector<size_t> uses_;
  std::vector<size_t> static_slots_;
  std::vector<c10::weak_intrusive_ptr<at::TensorImpl, at::UndefinedTensorImpl>> static_inputs_;
  std::vector<size_t> dynamic_slots_;
  std::shared_ptr<SegmentOutputPool> output_pool_;
};

}  // namespace gaudirt::generic
