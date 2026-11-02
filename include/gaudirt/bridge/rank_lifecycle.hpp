// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <hccl.h>

#include <cstdint>
#include <filesystem>

namespace gaudirt::runtime { class DeviceRuntime; }

namespace gaudirt::bridge {

hcclUniqueId exchange_unique_id(uint32_t rank, const std::filesystem::path& path);
void synchronize_shutdown(runtime::DeviceRuntime& device, uint64_t barrier_buffer);

}  // namespace gaudirt::bridge
