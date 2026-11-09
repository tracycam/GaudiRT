// SPDX-License-Identifier: Apache-2.0
#include "paged_metadata.hpp"

#include "bridge_internal.hpp"
#include "layout_copy.hpp"
#include "temporaries.hpp"
#include "gaudirt/common/sha256.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <array>
#include <bit>
#include <climits>
#include <cstring>
#include <memory>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gaudirt::generic {
using namespace runtime;

static ColdRecipe& device_metadata_recipe(Context& ctx, uint32_t bucket, uint64_t requests, uint64_t context,
                                   int32_t slots, const std::array<int, 4>& lanes, int32_t vocabulary,
                                   bool scheduler_owned) {
  std::ostringstream signature;
  signature << "device_metadata_v2;" << bucket;
  if (!scheduler_owned) signature << ":" << requests << ":" << context << ":" << slots;
  for (auto lane : lanes) signature << ":" << lane;
  signature << ":" << vocabulary << ":" << scheduler_owned;
  const auto structure = signature.str();
  const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const auto prefix = "g" + digest.substr(0, 12) + "_b" + std::to_string(bucket);
  const auto key = "device_metadata_" + digest;
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  std::vector<std::vector<uint64_t>> shapes{
      {static_cast<uint64_t>(lanes[0]), bucket}, {static_cast<uint64_t>(lanes[1]), bucket},
      {static_cast<uint64_t>(lanes[2]), bucket},
      scheduler_owned ? std::vector<uint64_t>{1, 1} : std::vector<uint64_t>{context, requests},
      {1}, {1 + 4 * bucket},
      {bucket}, {bucket}, {bucket}, {bucket}};
  if (vocabulary) {
    shapes.push_back({static_cast<uint64_t>(lanes[3]), bucket});
    shapes.push_back({bucket});
    shapes.push_back({bucket});
  }
  auto compiled = std::make_unique<ColdRecipe>(shapes.size() - static_cast<size_t>(scheduler_owned));
  synGraphHandle graph = nullptr;
  std::vector<synSectionHandle> sections(shapes.size(), nullptr);
  std::vector<synTensor> tensors(shapes.size(), nullptr);
  const auto binding_index = [scheduler_owned](size_t index) {
    return index - static_cast<size_t>(scheduler_owned && index > 3);
  };
  const auto dummy_name = prefix + "_t" + std::to_string(shapes.size());
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT device metadata: ", message,
                " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "graph creation failed");
    for (size_t i = 0; i < shapes.size(); ++i) {
      const bool dummy = scheduler_owned && i == 3;
      const auto slot = binding_index(i);
      if (!dummy) {
        compiled->names[slot] = prefix + "_t" + std::to_string(i);
        check(synSectionCreate(&sections[i], 0, graph), "section creation failed");
        check(synSectionSetPersistent(sections[i], true), "section setup failed");
      }
      synTensorDescriptor desc{};
      desc.m_name = dummy ? dummy_name.c_str() : compiled->names[slot].c_str();
      desc.m_dataType = syn_type_int32;
      desc.m_dims = shapes[i].size();
      for (size_t dim = 0; dim < shapes[i].size(); ++dim) {
        desc.m_sizes[dim] = desc.m_minSizes[dim] = shapes[i][dim];
        if (!dummy) compiled->tensors[slot].tensorSize[dim] = shapes[i][dim];
      }
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "tensor creation failed");
      if (!dummy) {
        compiled->tensors[slot].tensorName = compiled->names[slot].c_str();
        compiled->tensors[slot].tensorType = DATA_TENSOR;
      }
    }
    unsigned next_node = 0;
    if (scheduler_owned) {
      const auto reshape = prefix + "_n" + std::to_string(next_node++);
      check(synNodeCreate(graph, &tensors[4], &tensors[3], 1, 1, nullptr, 0,
                          "reshape", reshape.c_str(), nullptr, nullptr), "metadata reshape failed");
    }
    const auto node = prefix + "_n" + std::to_string(next_node++);
    const int32_t kernel_slots = scheduler_owned ? 1 : slots;
    check(synNodeCreate(graph, tensors.data(), tensors.data() + 6, 6, 4, &kernel_slots, sizeof(kernel_slots),
                        scheduler_owned ? "gk_scheduler_metadata_i32_v1" : "gk_paged_metadata_i32_v1",
                        node.c_str(), nullptr, nullptr), "node creation failed");
    if (vocabulary) {
      const std::array<synTensor, 2> token_inputs{tensors[10], tensors[4]};
      const std::array<synTensor, 2> token_outputs{tensors[11], tensors[12]};
      const auto token_node = prefix + "_n" + std::to_string(next_node++);
      check(synNodeCreate(graph, token_inputs.data(), token_outputs.data(), token_inputs.size(), token_outputs.size(),
                          &vocabulary, sizeof(vocabulary),
                          scheduler_owned ? "gk_scheduler_tokens_i32_v1" : "gk_token_indices_i32_v1",
                          token_node.c_str(),
                          nullptr, nullptr), "token validation node creation failed");
    }
    compiled->recipe = compile_cached(graph, prefix, key, &ctx.cache);
    check(synWorkspaceGetSize(&compiled->workspace, compiled->recipe), "workspace query failed");
    initialize_launch_tensor_ids(compiled->recipe, compiled->tensors.data(), compiled->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(compiled));
  TORCH_CHECK(inserted, "GaudiRT device metadata: duplicate recipe");
  return *it->second;
}

PagedMetadata::PagedMetadata(const at::Tensor& locations, const at::Tensor& requests,
                             const at::Tensor& positions, const at::Tensor& mapping, int64_t slots) {
  for (const auto& tensor : {locations, requests, positions, mapping})
    TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous() &&
                (tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong),
                "GaudiRT: paged metadata must be contiguous CPU int32 or int64");
  TORCH_CHECK(locations.dim() == 1 && requests.dim() == 1 && positions.dim() == 1 && mapping.dim() == 2 &&
              locations.size(0) == requests.size(0) && locations.size(0) == positions.size(0),
              "GaudiRT: invalid paged metadata ranks or row counts");
  const int64_t m = locations.size(0), request_count = mapping.size(0), context = mapping.size(1);
  TORCH_CHECK(m > 0 && m <= 4096 && slots > 0 && slots <= INT32_MAX &&
              request_count > 0 && request_count <= INT32_MAX && context > 0 && context <= INT32_MAX &&
              mapping.numel() <= INT32_MAX, "GaudiRT: paged metadata dimensions outside supported range");
  const auto read = [](const at::Tensor& tensor, int64_t index) -> int64_t {
    return tensor.scalar_type() == at::kInt ? tensor.data_ptr<int32_t>()[index] : tensor.data_ptr<int64_t>()[index];
  };
  const uint32_t bucket = std::bit_ceil(static_cast<uint32_t>(m));
  auto ctx = current();
  const uint64_t total = (static_cast<uint64_t>(bucket) * 3 + mapping.numel() + 1) * sizeof(int32_t);
  TORCH_CHECK(total <= ctx->pinned.largest_free_block(), "GaudiRT: pinned pool cannot hold paged metadata");
  std::vector<int32_t> host_locations(bucket), host_requests(bucket), host_positions(bucket),
      host_mapping(mapping.numel()), host_header{static_cast<int32_t>(m)};
  for (int64_t i = 0; i < mapping.numel(); ++i) {
    const auto slot = read(mapping, i);
    TORCH_CHECK(slot >= -1 && slot < slots, "GaudiRT: request-to-token slot is out of range");
    host_mapping[i] = static_cast<int32_t>(slot);
  }
  std::vector<int32_t> max_position(request_count, -1);
  std::unordered_map<int32_t, std::pair<int32_t, int32_t>> targets;
  for (int64_t i = 0; i < m; ++i) {
    const auto slot = read(locations, i), request = read(requests, i), position = read(positions, i);
    TORCH_CHECK(slot >= 0 && slot < slots && request >= 0 && request < request_count &&
                position >= 0 && position < context, "GaudiRT: paged metadata index is out of range");
    TORCH_CHECK(host_mapping[request * context + position] == slot,
                "GaudiRT: output slot does not match request-to-token mapping");
    const auto [it, inserted] = targets.emplace(static_cast<int32_t>(slot),
                                                std::make_pair(static_cast<int32_t>(request),
                                                               static_cast<int32_t>(position)));
    (void)it;
    TORCH_CHECK(inserted, "GaudiRT: duplicate output KV slot");
    host_locations[i] = static_cast<int32_t>(slot);
    host_requests[i] = static_cast<int32_t>(request);
    host_positions[i] = static_cast<int32_t>(position);
    if (i == 0 || host_requests[i - 1] != request || host_positions[i - 1] + 1 != position)
      sequences_.push_back({static_cast<int32_t>(i), 1, static_cast<int32_t>(position)});
    else
      ++sequences_.back().length;
    max_position[request] = std::max(max_position[request], static_cast<int32_t>(position));
  }
  for (int64_t request = 0; request < request_count; ++request) {
    for (int32_t position = 0; position <= max_position[request]; ++position) {
      const auto slot = host_mapping[request * context + position];
      TORCH_CHECK(slot >= 0, "GaudiRT: missing KV slot in a visible prefix");
      auto target = targets.find(slot);
      TORCH_CHECK(target == targets.end() ||
                  (target->second.first == request && target->second.second == position),
                  "GaudiRT: output KV slot aliases another visible prefix position");
    }
  }
  const auto options = at::TensorOptions().device(c10::Device(c10::kPrivateUse1, 0)).dtype(at::kInt);
  locations_ = at::empty({bucket}, options);
  requests_ = at::empty({bucket}, options);
  positions_ = at::empty({bucket}, options);
  mapping_ = at::empty({request_count, context}, options);
  header_ = at::empty({1}, options);
  const std::array<const std::vector<int32_t>*, 5> sources{
      &host_locations, &host_requests, &host_positions, &host_mapping, &host_header};
  const std::array<at::Tensor*, 5> destinations{&locations_, &requests_, &positions_, &mapping_, &header_};
  TemporaryHost staging(*ctx, total);
  auto* host = static_cast<uint8_t*>(staging.address);
  auto& stream_state = ctx->streams.state(active_stream_id);
  auto compute = stream_state.handle;
  auto stream = ctx->device.h2d_stream();
  auto& reusable = stream_state.metadata_reusable;
  auto& ready = stream_state.metadata_ready;
  reusable.record_and_wait(compute, stream);
  uint64_t offset = 0;
  for (size_t i = 0; i < sources.size(); ++i) {
    const uint64_t bytes = sources[i]->size() * sizeof(int32_t);
    std::memcpy(host + offset, sources[i]->data(), bytes);
    const auto destination = address_of(*destinations[i], *ctx);
    ctx->allocator.record_stream(destination, stream);
    ctx->allocator.record_stream(destination, compute);
    ctx->pinned.copy_h2d(destination, host + offset, bytes, stream, true);
    offset += bytes;
  }
  ready.record_and_wait(stream, compute);
  rows_ = m;
  slots_ = slots;
  request_count_ = request_count;
  context_ = context;
  bucket_ = bucket;
}

PagedMetadata::PagedMetadata(const at::Tensor& locations, const at::Tensor& batch_requests,
                             const at::Tensor& positions, const at::Tensor& mapping,
                             std::vector<int64_t> prefixes, std::vector<int64_t> lengths, int64_t slots,
                             const at::Tensor& token_ids, int64_t vocabulary, bool scheduler_owned,
                             bool fuse_scheduler_plan) {
  TORCH_CHECK(!fuse_scheduler_plan || scheduler_owned,
              "GaudiRT device metadata: fused plan requires scheduler-owned inputs");
  for (const auto& tensor : {locations, batch_requests})
    TORCH_CHECK(tensor.device().type() == c10::kPrivateUse1 && tensor.device().index() == 0 &&
                tensor.dim() == 1 && (tensor.scalar_type() == at::kInt || tensor.scalar_type() == at::kLong),
                "GaudiRT device metadata: expected gaudirt int32 or int64 vectors");
  TORCH_CHECK(positions.dim() == 1 &&
              ((fuse_scheduler_plan && positions.device().is_cpu() && positions.scalar_type() == at::kLong &&
                positions.is_contiguous()) ||
               (!fuse_scheduler_plan && positions.device().type() == c10::kPrivateUse1 &&
                positions.device().index() == 0 &&
                (positions.scalar_type() == at::kInt || positions.scalar_type() == at::kLong))),
              "GaudiRT device metadata: invalid position input");
  TORCH_CHECK(mapping.device().type() == c10::kPrivateUse1 && mapping.device().index() == 0 &&
              mapping.scalar_type() == at::kInt && mapping.dim() == 2 && mapping.is_contiguous(),
              "GaudiRT device metadata: expected a contiguous gaudirt int32 request map");
  const int64_t m = locations.numel(), batch = batch_requests.numel();
  const int64_t request_count = mapping.size(0), context = mapping.size(1);
  const bool has_tokens = token_ids.defined();
  TORCH_CHECK(!scheduler_owned || has_tokens, "GaudiRT: scheduler metadata requires token IDs");
  TORCH_CHECK((has_tokens && token_ids.device().type() == c10::kPrivateUse1 && token_ids.device().index() == 0 &&
               token_ids.dim() == 1 && token_ids.numel() == m &&
               (token_ids.scalar_type() == at::kInt || token_ids.scalar_type() == at::kLong) &&
               vocabulary > 0 && vocabulary <= INT32_MAX) || (!has_tokens && vocabulary == 0),
              "GaudiRT device metadata: invalid token ID inputs or vocabulary");
  TORCH_CHECK(m > 0 && m <= 4096 && positions.numel() == m && batch > 0 && batch <= m &&
              static_cast<int64_t>(prefixes.size()) == batch && static_cast<int64_t>(lengths.size()) == batch &&
              slots > 0 && slots <= INT32_MAX && request_count > 0 && request_count <= INT32_MAX &&
              context > 0 && context <= INT32_MAX && mapping.numel() <= INT32_MAX,
              "GaudiRT device metadata: dimensions outside supported range");
  const uint32_t bucket = std::bit_ceil(static_cast<uint32_t>(m));
  std::vector<int32_t> info(1 + 4 * bucket, 0), last_token_indices(batch);
  info[0] = static_cast<int32_t>(batch);
  int64_t cursor = 0;
  for (int64_t index = 0; index < batch; ++index) {
    const auto prefix = prefixes[index], length = lengths[index];
    TORCH_CHECK(prefix >= 0 && prefix < context && length > 0 && length <= context - prefix &&
                length <= m - cursor, "GaudiRT device metadata: invalid sequence lengths");
    sequences_.push_back({static_cast<int32_t>(cursor), static_cast<int32_t>(length),
                          static_cast<int32_t>(prefix)});
    info[1 + 2 * bucket + index] = static_cast<int32_t>(prefix);
    info[1 + 3 * bucket + index] = static_cast<int32_t>(length);
    for (int64_t offset = 0; offset < length; ++offset, ++cursor) {
      info[1 + cursor] = static_cast<int32_t>(index);
      info[1 + bucket + cursor] = static_cast<int32_t>(prefix + offset);
    }
    last_token_indices[index] = static_cast<int32_t>(cursor - 1);
  }
  TORCH_CHECK(cursor == m, "GaudiRT device metadata: sequence lengths do not cover token rows");
  if (fuse_scheduler_plan) {
    const auto* host_positions = positions.data_ptr<int64_t>();
    for (int64_t row = 0; row < m; ++row)
      TORCH_CHECK(host_positions[row] == info[1 + bucket + row],
                  "GaudiRT device metadata: host position does not match scheduler lengths");
  }
  auto ctx = current();
  auto& stream_state = ctx->streams.state(active_stream_id);
  auto compute = stream_state.handle;
  auto h2d = ctx->device.h2d_stream();
  const auto options = at::TensorOptions().device(c10::Device(c10::kPrivateUse1, 0)).dtype(at::kInt);
  // Give each graph input/output its own aligned range while sharing one
  // allocation and one lifetime across the metadata streams.
  constexpr int64_t alignment = BlockCachingAllocator::kAlignment / sizeof(int32_t);
  const auto aligned = [](int64_t count) { return (count + alignment - 1) / alignment * alignment; };
  int64_t cursor_words = 0;
  const auto reserve = [&](int64_t count) {
    const auto start = cursor_words;
    cursor_words += aligned(count);
    return start;
  };
  const auto header_offset = reserve(1);
  const auto info_offset = reserve(info.size());
  const auto last_offset = reserve(batch);
  const auto model_positions_offset = fuse_scheduler_plan ? reserve(2 * bucket) : 0;
  const auto staged_words = cursor_words;
  const auto locations_offset = reserve(bucket);
  const auto requests_offset = reserve(bucket);
  const auto positions_offset = reserve(bucket);
  const auto status_offset = reserve(bucket);
  const auto token_offset = has_tokens ? reserve(bucket) : 0;
  const auto token_status_offset = has_tokens ? reserve(bucket) : 0;
  auto storage = at::empty({cursor_words}, options);
  const auto view = [&](int64_t offset, int64_t count) { return storage.narrow(0, offset, count); };
  header_ = view(header_offset, 1);
  auto device_info = view(info_offset, info.size());
  last_token_indices_ = view(last_offset, batch);
  if (fuse_scheduler_plan)
    model_positions_ = view(model_positions_offset, 2 * bucket).view(at::kLong).narrow(0, 0, m);
  locations_ = view(locations_offset, bucket);
  requests_ = view(requests_offset, bucket);
  positions_ = view(positions_offset, bucket);
  mapping_ = mapping;
  auto status = view(status_offset, bucket);
  at::Tensor token_status;
  if (has_tokens) {
    token_ids_ = view(token_offset, bucket);
    token_status = view(token_status_offset, bucket);
  }
  rows_ = m;
  slots_ = slots;
  request_count_ = request_count;
  context_ = context;
  vocabulary_ = vocabulary;
  bucket_ = bucket;
  if (fuse_scheduler_plan) {
    pending_host_.assign(staged_words, 0);
    pending_host_[header_offset] = static_cast<int32_t>(m);
    std::memcpy(pending_host_.data() + info_offset, info.data(), info.size() * sizeof(int32_t));
    std::memcpy(pending_host_.data() + last_offset, last_token_indices.data(),
                last_token_indices.size() * sizeof(int32_t));
    std::memcpy(pending_host_.data() + model_positions_offset, positions.data_ptr<int64_t>(),
                m * sizeof(int64_t));
    pending_locations_ = locations;
    pending_requests_ = batch_requests;
    pending_tokens_ = token_ids;
    pending_info_ = device_info;
    pending_status_ = status;
    pending_token_status_ = token_status;
    return;
  }
  const auto header_address = address_of(header_, *ctx);
  {
    const uint64_t bytes = staged_words * sizeof(int32_t);
    TemporaryHost staging(*ctx, bytes);
    auto* host = static_cast<int32_t*>(staging.address);
    std::memset(host, 0, bytes);
    host[header_offset] = static_cast<int32_t>(m);
    std::memcpy(host + info_offset, info.data(), info.size() * sizeof(int32_t));
    std::memcpy(host + last_offset, last_token_indices.data(), last_token_indices.size() * sizeof(int32_t));
    auto& reusable = stream_state.metadata_reusable;
    auto& ready = stream_state.metadata_ready;
    reusable.record_and_wait(compute, h2d);
    ctx->allocator.record_stream(header_address, h2d);
    ctx->allocator.record_stream(header_address, compute);
    ctx->pinned.copy_h2d(header_address, host, bytes, h2d, true);
    ready.record_and_wait(h2d, compute);
  }
  launch_device_metadata(locations, batch_requests, positions, token_ids, device_info,
                         status, token_status, scheduler_owned);
}

void PagedMetadata::launch_device_metadata(const at::Tensor& locations, const at::Tensor& batch_requests,
    const at::Tensor& positions, const at::Tensor& token_ids, const at::Tensor& device_info,
    const at::Tensor& status, const at::Tensor& token_status, bool scheduler_owned) {
  auto ctx = current();
  auto& stream_state = ctx->streams.state(active_stream_id);
  const auto compute = stream_state.handle;
  const auto m = rows_, request_count = request_count_, context = context_, slots = slots_, vocabulary = vocabulary_;
  const auto bucket = bucket_;
  const bool has_tokens = token_ids.defined();
  const auto mapping_address = scheduler_owned ? 0 : address_of(mapping_, *ctx);
  TORCH_CHECK(scheduler_owned || mapping_address % BlockCachingAllocator::kAlignment == 0,
              "GaudiRT device metadata: request map must be 256-byte aligned");
  const auto header_address = address_of(header_, *ctx), info_address = address_of(device_info, *ctx);
  const auto location_address = address_of(locations_, *ctx), request_address = address_of(requests_, *ctx);
  const auto position_address = address_of(positions_, *ctx), status_address = address_of(status, *ctx);
  const std::array<at::Tensor, 4> sources{locations, batch_requests, positions, token_ids};
  const std::array<int, 4> lanes{locations.element_size() == 8 ? 2 : 1,
                                  batch_requests.element_size() == 8 ? 2 : 1,
                                  positions.element_size() == 8 ? 2 : 1,
                                  has_tokens ? (token_ids.element_size() == 8 ? 2 : 1) : 0};
  std::vector<std::unique_ptr<GraphTemporary>> staged;
  std::unique_ptr<GraphTemporary> workspace;
  {
    std::lock_guard lock(ctx->cold_mutex);
    auto& recipe = device_metadata_recipe(*ctx, bucket, request_count, context, static_cast<int32_t>(slots),
                                          lanes, static_cast<int32_t>(vocabulary), scheduler_owned);
    auto graph_pool = [&]() -> GraphPool& {
      if (!ctx->graph_pool) {
        const uint64_t budget = std::min<uint64_t>(ctx->allocator.capacity_bytes() / 8, uint64_t{1} << 30);
        TORCH_CHECK(budget >= 256, "GaudiRT device metadata: HBM budget is too small for graph temporaries");
        ctx->graph_pool = std::make_unique<GraphPool>(ctx->allocator, budget, compute);
      }
      return *ctx->graph_pool;
    };
    std::vector<std::pair<uint64_t, uint64_t>> reserved;
    if (!scheduler_owned) reserved.emplace_back(mapping_address, mapping_.nbytes());
    std::array<uint64_t, 4> input_addresses{};
    for (size_t i = 0; i < (has_tokens ? sources.size() : sources.size() - 1); ++i) {
      const auto& input = sources[i];
      const uint64_t address = address_of(input, *ctx);
      ctx->allocator.record_stream(address, compute);
      const uint64_t bytes = static_cast<uint64_t>(bucket) * input.element_size();
      const uint64_t offset = static_cast<uint64_t>(input.storage_offset()) * input.element_size();
      const uint64_t storage = input.storage().nbytes();
      bool direct = input.is_contiguous() && address % BlockCachingAllocator::kAlignment == 0 &&
                    offset <= storage && bytes <= storage - offset;
      for (const auto& [other, length] : reserved)
        if (address < other + length && other < address + bytes) direct = false;
      input_addresses[i] = address;
      if (!direct) {
        auto temporary = std::make_unique<GraphTemporary>(graph_pool(), compute, bytes);
        input_addresses[i] = temporary->address;
        const Layout layout(input);
        dma_copy(temporary->address, nullptr, address, &layout, 0, input.numel(), input.element_size(),
                 compute, DRAM_TO_DRAM);
        staged.push_back(std::move(temporary));
      }
      reserved.emplace_back(input_addresses[i], bytes);
    }
    if (!scheduler_owned) ctx->allocator.record_stream(mapping_address, compute);
    if (recipe.workspace) workspace = std::make_unique<GraphTemporary>(graph_pool(), compute, recipe.workspace);
    auto bindings = recipe.tensors;
    const uint64_t addresses[] = {input_addresses[0], input_addresses[1], input_addresses[2], mapping_address,
                                  header_address, info_address, location_address, request_address,
                                  position_address, status_address};
    const auto binding_index = [scheduler_owned](size_t index) {
      return index - static_cast<size_t>(scheduler_owned && index > 3);
    };
    for (size_t i = 0; i < std::size(addresses); ++i) {
      if (scheduler_owned && i == 3) continue;
      bindings[binding_index(i)].pTensorAddress = addresses[i];
    }
    if (has_tokens) {
      const auto token_address = address_of(token_ids_, *ctx);
      const auto token_status_address = address_of(token_status, *ctx);
      bindings[binding_index(10)].pTensorAddress = input_addresses[3];
      bindings[binding_index(11)].pTensorAddress = token_address;
      bindings[binding_index(12)].pTensorAddress = token_status_address;
    }
    TORCH_CHECK(synLaunch(compute, bindings.data(), bindings.size(), workspace ? workspace->address : 0,
                          recipe.recipe, 0) == synSuccess, "GaudiRT device metadata: validation launch failed");
  }
  if (!scheduler_owned) {
    auto d2h = ctx->device.d2h_stream();
    auto& ready = stream_state.validation_ready;
    auto& completed = stream_state.validation_completed;
    ready.record_and_wait(compute, d2h);
    ctx->allocator.record_stream(status_address, d2h);
    TemporaryHost host_status(*ctx, static_cast<uint64_t>(m) * sizeof(int32_t) * (has_tokens ? 2 : 1));
    ctx->pinned.copy_d2h(host_status.address, status_address, static_cast<uint64_t>(m) * sizeof(int32_t), d2h, true);
    if (has_tokens) {
      const auto token_status_address = address_of(token_status, *ctx);
      ctx->pinned.copy_d2h(static_cast<int32_t*>(host_status.address) + m, token_status_address,
                           static_cast<uint64_t>(m) * sizeof(int32_t), d2h, true);
    }
    completed.record_and_synchronize(d2h);
    const auto* flags = static_cast<const int32_t*>(host_status.address);
    for (int64_t row = 0; row < m; ++row)
      TORCH_CHECK(flags[row] == 0, "GaudiRT device metadata: invalid scheduler metadata at row ", row,
                  ", code ", flags[row]);
    if (has_tokens)
      for (int64_t row = 0; row < m; ++row)
        TORCH_CHECK(flags[m + row] == 0, "GaudiRT device metadata: token ID outside vocabulary at row ", row);
  }
}

void PagedMetadata::finish_pending_metadata() {
  TORCH_CHECK(!pending_host_.empty(), "GaudiRT device metadata: no pending scheduler metadata");
  launch_device_metadata(pending_locations_, pending_requests_, model_positions_, pending_tokens_,
                         pending_info_, pending_status_, pending_token_status_, true);
  pending_host_.clear();
  pending_locations_ = pending_requests_ = pending_tokens_ = pending_info_ = pending_status_ =
      pending_token_status_ = at::Tensor();
}

}  // namespace gaudirt::generic
