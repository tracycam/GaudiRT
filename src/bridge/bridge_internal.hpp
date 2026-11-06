// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "context.hpp"

#include <ATen/core/Tensor.h>
#include <c10/core/ScalarType.h>

#include <cstdint>
#include <memory>

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
const AllocationContext& validated_program_input(const at::Tensor& tensor, const Context& ctx);
CastType cast_type(at::ScalarType type);
at::Tensor bucket_output(const at::Tensor& reference, at::ScalarType dtype,
                         uint32_t bucket, int64_t rows, int64_t width);

}  // namespace gaudirt::generic
