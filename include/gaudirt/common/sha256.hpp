// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace gaudirt::common {

std::string sha256_bytes(const uint8_t* data, size_t size);
std::string sha256_file(const std::filesystem::path& path);

}  // namespace gaudirt::common
