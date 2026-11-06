// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>
#include <torch/custom_class.h>

#include <cstdint>
#include <utility>
#include <vector>

namespace gaudirt::generic {

class PagedMetadata : public torch::CustomClassHolder {
 public:
  PagedMetadata(const at::Tensor& locations, const at::Tensor& requests,
                const at::Tensor& positions, const at::Tensor& mapping, int64_t slots);
  PagedMetadata(const at::Tensor& locations, const at::Tensor& batch_requests,
                const at::Tensor& positions, const at::Tensor& mapping,
                std::vector<int64_t> prefixes, std::vector<int64_t> lengths, int64_t slots,
                const at::Tensor& token_ids = {}, int64_t vocabulary = 0,
                bool scheduler_owned = false);
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
      const at::Tensor& token_ids, int64_t vocabulary) {
    return c10::make_intrusive<PagedMetadata>(locations, batch_requests, positions, mapping,
                                             std::move(prefixes), std::move(lengths), slots,
                                             token_ids, vocabulary, true);
  }
  int64_t rows() const { return rows_; }
  int64_t slots() const { return slots_; }
  int64_t context() const { return context_; }
  at::Tensor rotary_positions() const { return positions_.narrow(0, 0, rows_); }
  at::Tensor rotary_header() const { return header_; }
  at::Tensor row_locations() const { return locations_.narrow(0, 0, rows_); }
  at::Tensor row_requests() const { return requests_.narrow(0, 0, rows_); }
  at::Tensor request_mapping() const { return mapping_; }
  at::Tensor last_token_indices() const {
    TORCH_CHECK(last_token_indices_.defined(), "GaudiRT: last-token indices require batched metadata");
    return last_token_indices_;
  }
  at::Tensor token_ids() const {
    TORCH_CHECK(token_ids_.defined(), "GaudiRT: token IDs were not validated in this metadata");
    return token_ids_.narrow(0, 0, rows_);
  }
  int64_t vocabulary() const { return vocabulary_; }

  int64_t rows_ = 0, slots_ = 0, request_count_ = 0, context_ = 0, vocabulary_ = 0;
  uint32_t bucket_ = 0;
  at::Tensor locations_, requests_, positions_, mapping_, header_, token_ids_, last_token_indices_;
};

}  // namespace gaudirt::generic
