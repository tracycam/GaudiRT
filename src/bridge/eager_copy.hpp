// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/Tensor.h>

#include <optional>

namespace gaudirt::generic {

at::Tensor empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                 std::optional<at::Layout> layout, std::optional<at::Device> device,
                 std::optional<bool> pin_memory, std::optional<at::MemoryFormat> format);
at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                         std::optional<at::Device> device, std::optional<bool> pin_memory);
at::Tensor& set_storage(at::Tensor& tensor, at::Storage storage, int64_t storage_offset,
                        at::IntArrayRef size, at::IntArrayRef stride);
at::Tensor& fill(at::Tensor& tensor, const at::Scalar& scalar);
at::Tensor& zero(at::Tensor& tensor);
at::Tensor& copy(at::Tensor& dst, const at::Tensor& src, bool non_blocking);

}  // namespace gaudirt::generic
