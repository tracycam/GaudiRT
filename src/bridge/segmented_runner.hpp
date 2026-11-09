// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>
#include <synapse_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

namespace gaudirt::generic {

class BucketProgram;

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

// Runs a sequence of independent FX programs and stateful attention operators.
// Slots are assigned from the captured FX graph; no model geometry is embedded.
class SegmentedRunner {
 public:
  explicit SegmentedRunner(size_t input_count);
  std::vector<size_t> add_program(std::shared_ptr<BucketProgram> program, std::vector<size_t> inputs,
                                  std::vector<SegmentRoute> outputs);
  size_t add_attention(std::vector<size_t> inputs, int64_t head_dim, double scale);
  void set_outputs(std::vector<size_t> outputs);
  std::vector<at::Tensor> run(const std::vector<at::Tensor>& inputs) const;
  using SegmentTrace = std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>, uint64_t>;
  using TracedResult = std::pair<std::vector<at::Tensor>, std::vector<SegmentTrace>>;
  TracedResult run_traced(const std::vector<at::Tensor>& inputs, const std::vector<size_t>& traced) const;
  using ProfiledResult = std::pair<std::vector<at::Tensor>, std::vector<double>>;
  ProfiledResult run_profiled(const std::vector<at::Tensor>& inputs, const std::vector<uint64_t>& events) const;

 private:
  struct Instruction {
    std::shared_ptr<BucketProgram> program;
    std::vector<size_t> inputs;
    std::vector<bool> trusted;
    std::vector<SegmentRoute> routes;
    std::vector<size_t> outputs;
    int64_t head_dim = 0;
    double scale = 0;
    bool attention = false;
  };
  TracedResult execute(const std::vector<at::Tensor>& inputs, const std::vector<bool>& trace,
                       const std::vector<synEventHandle>* events = nullptr, std::vector<double>* host_ms = nullptr) const;
  size_t input_count_;
  size_t slots_;
  std::vector<Instruction> instructions_;
  std::vector<bool> trusted_slots_;
  std::vector<size_t> outputs_;
  std::vector<size_t> uses_;
};

}  // namespace gaudirt::generic
