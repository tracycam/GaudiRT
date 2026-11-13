// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>
#include <c10/util/Optional.h>
#include <torch/custom_class.h>

#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

namespace gaudirt::generic {

inline constexpr uint32_t kTileKvBucket = 512;

class PagedMetadata : public torch::CustomClassHolder {
 public:
  static constexpr int64_t kMaxRows = 8192;
  PagedMetadata(const at::Tensor& locations, const at::Tensor& requests,
                const at::Tensor& positions, const at::Tensor& mapping, int64_t slots);
  PagedMetadata(const at::Tensor& locations, const at::Tensor& batch_requests,
                const at::Tensor& positions, const at::Tensor& mapping,
                std::vector<int64_t> prefixes, std::vector<int64_t> lengths, int64_t slots,
                const at::Tensor& token_ids = {}, int64_t vocabulary = 0,
                bool scheduler_owned = false, bool fuse_scheduler_plan = false);
  static c10::intrusive_ptr<PagedMetadata> from_device(
      const at::Tensor& locations, const at::Tensor& batch_requests,
      const at::Tensor& positions, const at::Tensor& mapping,
      std::vector<int64_t> prefixes, std::vector<int64_t> lengths, int64_t slots) {
    return c10::make_intrusive<PagedMetadata>(locations, batch_requests, positions, mapping,
                                             std::move(prefixes), std::move(lengths), slots);
  }
  static c10::intrusive_ptr<PagedMetadata> from_device_with_tokens(
      const at::Tensor& locations, const at::Tensor& batch_requests,
      const at::Tensor& positions, const at::Tensor& mapping,
      std::vector<int64_t> prefixes, std::vector<int64_t> lengths, int64_t slots,
      const at::Tensor& token_ids, int64_t vocabulary) {
    return c10::make_intrusive<PagedMetadata>(locations, batch_requests, positions, mapping,
                                             std::move(prefixes), std::move(lengths), slots,
                                             token_ids, vocabulary);
  }
  // Internal scheduler inputs have already been checked at request entry;
  // slots and request mappings are produced by the scheduler's own pools.
  static c10::intrusive_ptr<PagedMetadata> from_scheduler(
      const at::Tensor& locations, const at::Tensor& batch_requests,
      const at::Tensor& positions, const at::Tensor& mapping,
      std::vector<int64_t> prefixes, std::vector<int64_t> lengths, int64_t slots,
      const c10::optional<at::Tensor>& token_ids = c10::nullopt, int64_t vocabulary = 0) {
    return c10::make_intrusive<PagedMetadata>(locations, batch_requests, positions, mapping,
                                             std::move(prefixes), std::move(lengths), slots,
                                             token_ids.value_or(at::Tensor{}), vocabulary, true);
  }
  static c10::intrusive_ptr<PagedMetadata> from_scheduler_fused(
      const at::Tensor& locations, const at::Tensor& batch_requests,
      const at::Tensor& positions, const at::Tensor& mapping,
      std::vector<int64_t> prefixes, std::vector<int64_t> lengths, int64_t slots,
      const c10::optional<at::Tensor>& token_ids, int64_t vocabulary,
      int64_t query_heads, int64_t kv_heads, int64_t head_dim) {
    auto metadata = c10::make_intrusive<PagedMetadata>(locations, batch_requests, positions, mapping,
        std::move(prefixes), std::move(lengths), slots, token_ids.value_or(at::Tensor{}), vocabulary, true, true);
    metadata->attention_plan(query_heads, kv_heads, head_dim);
    return metadata;
  }
  int64_t rows() const { return rows_; }
  int64_t slots() const { return slots_; }
  int64_t context() const { return context_; }
  at::Tensor rotary_positions() const { return positions_.narrow(0, 0, rows_); }
  at::Tensor model_positions() const {
    TORCH_CHECK(model_positions_.defined(), "GaudiRT: model positions require fused scheduler metadata");
    return model_positions_;
  }
  at::Tensor rotary_header() const { return header_; }
  at::Tensor row_locations() const { return locations_.narrow(0, 0, rows_); }
  at::Tensor row_requests() const { return requests_.narrow(0, 0, rows_); }
  at::Tensor request_mapping() const { return mapping_; }
  at::Tensor attention_plan(int64_t query_heads, int64_t kv_heads, int64_t head_dim);
  at::Tensor attention_plan_for(int64_t query_heads, int64_t kv_heads, int64_t head_dim, int64_t sliding_window);
  at::Tensor last_token_indices() const {
    TORCH_CHECK(last_token_indices_.defined(), "GaudiRT: last-token indices require batched metadata");
    return last_token_indices_;
  }
  at::Tensor token_ids() const {
    TORCH_CHECK(token_ids_.defined(), "GaudiRT: token IDs were not validated in this metadata");
    return token_ids_.narrow(0, 0, rows_);
  }
  int64_t vocabulary() const { return vocabulary_; }
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, int64_t>
  forward_tensors(int64_t query_heads, int64_t kv_heads, int64_t head_dim) {
    return {token_ids(), last_token_indices(), rotary_positions(), rotary_header(), row_locations(),
            row_requests(), attention_plan(query_heads, kv_heads, head_dim), context_};
  }

  int64_t rows_ = 0, slots_ = 0, request_count_ = 0, context_ = 0, vocabulary_ = 0;
  uint32_t bucket_ = 0;
  at::Tensor locations_, requests_, positions_, mapping_, header_, token_ids_, last_token_indices_;
  struct Sequence { int32_t row, length, prefix; };
  std::vector<Sequence> sequences_;
  at::Tensor attention_plan_;
  std::vector<std::pair<int64_t, at::Tensor>> swa_attention_plans_;
  std::vector<int32_t> pending_host_;
  at::Tensor pending_locations_, pending_requests_, pending_tokens_, pending_info_, pending_status_,
      pending_token_status_, model_positions_;
  void launch_device_metadata(const at::Tensor& locations, const at::Tensor& batch_requests,
                              const at::Tensor& positions, const at::Tensor& token_ids,
                              const at::Tensor& device_info, const at::Tensor& status,
                              const at::Tensor& token_status, bool scheduler_owned);
  void finish_pending_metadata();
};

}  // namespace gaudirt::generic
