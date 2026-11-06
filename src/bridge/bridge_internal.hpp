// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "context.hpp"

#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace gaudirt::generic {

enum class AllocationKind { device, pinned, graph };
struct AllocationContext {
  std::weak_ptr<Context> owner;
  uint64_t address;
  uint64_t bytes;
  synStreamHandle stream;
  AllocationKind kind;
};

struct CastType { synDataType syn; const char* name; };

std::shared_ptr<Context> current();
uint64_t span_bytes(const at::Tensor& tensor);
uint64_t address_of(const at::Tensor& tensor, const Context& ctx);
c10::DataPtr allocation_data_ptr(const std::shared_ptr<Context>& ctx, uint64_t address,
                                 uint64_t bytes, synStreamHandle stream, AllocationKind kind);
at::Tensor tensor_from_allocation(c10::DataPtr data, uint64_t bytes, at::IntArrayRef shape,
                                  const at::TensorOptions& options);
void record_stream(at::Tensor& tensor, c10::Stream stream);
void register_backend();
void initialize(uint32_t module, uint64_t hbm_bytes, uint64_t pinned_bytes);
bool is_initialized();
std::tuple<uint64_t, uint64_t> memory_info();
std::tuple<uint64_t, uint64_t> physical_memory_info();
void shutdown();
void shutdown_at_exit();
uint64_t memory_allocated();
uint64_t max_memory_allocated();
std::vector<uint64_t> recipe_cache_stats();
std::vector<std::string> recipe_cache_keys();
void empty_cache();
void synchronize();
const AllocationContext& validated_program_input(const at::Tensor& tensor, const Context& ctx);
CastType cast_type(at::ScalarType type);
at::Tensor bucket_output(const at::Tensor& reference, at::ScalarType dtype,
                         uint32_t bucket, int64_t rows, int64_t width);

}  // namespace gaudirt::generic
