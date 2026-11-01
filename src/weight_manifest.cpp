// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/weight_manifest.hpp"
#include "gaudirt/common/sha256.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace gaudirt::runtime {

namespace {

uint64_t parse_u64(const std::string& text, size_t line) {
  try {
    size_t used = 0;
    const uint64_t value = std::stoull(text, &used, 10);
    if (used != text.size()) throw std::invalid_argument("trailing characters");
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error("invalid integer on weight manifest line " +
                             std::to_string(line));
  }
}

synDataType parse_dtype(const std::string& text, size_t line) {
  if (text == "bf16") return syn_type_bf16;
  if (text == "fp32") return syn_type_float;
  if (text == "fp8_e4m3") return syn_type_fp8_143;
  if (text == "fp8_e5m2") return syn_type_fp8_152;
  if (text == "packed_mxfp4") return syn_type_packed_mxfp4;
  if (text == "int32") return syn_type_int32;
  if (text == "uint8") return syn_type_uint8;
  throw std::runtime_error("unsupported dtype on weight manifest line " +
                           std::to_string(line));
}

std::vector<uint64_t> parse_shape(const std::string& text, size_t line) {
  std::vector<uint64_t> shape;
  size_t begin = 0;
  for (;;) {
    const size_t comma = text.find(',', begin);
    shape.push_back(parse_u64(text.substr(begin, comma - begin), line));
    if (comma == std::string::npos) break;
    begin = comma + 1;
  }
  return shape;
}

std::string manifest_error(size_t line, const std::string& reason) {
  return "weight manifest line " + std::to_string(line) + ": " + reason;
}

}  // namespace

std::vector<WeightTensor> WeightManifest::read_manifest(
    const std::filesystem::path& directory) {
  std::ifstream input(directory / "weights.manifest");
  if (!input) throw std::runtime_error("cannot open weights.manifest");
  std::string line;
  size_t line_number = 0;
  do {
    std::getline(input, line);
    ++line_number;
  } while (input && line.empty());
  if (line != "GAUDIRT_WEIGHTS_V1") {
    throw std::runtime_error("unsupported weight manifest version");
  }

  std::vector<WeightTensor> tensors;
  std::unordered_set<std::string> names;
  while (std::getline(input, line)) {
    ++line_number;
    if (line.empty() || line[0] == '#') continue;
    std::istringstream fields(line);
    std::string tag;
    WeightTensor tensor;
    std::string dtype;
    std::string shape;
    std::string elements;
    std::string rank;
    if (!(fields >> tag >> tensor.name >> dtype >> elements >> rank >> shape >>
          tensor.file >> tensor.sha256) ||
        tag != "T") {
      throw std::runtime_error(manifest_error(line_number, "invalid tensor record"));
    }
    std::string extra;
    if (fields >> extra) {
      throw std::runtime_error(manifest_error(line_number, "trailing fields"));
    }
    if (!names.insert(tensor.name).second) {
      throw std::runtime_error(manifest_error(line_number, "duplicate tensor"));
    }
    tensor.dtype = parse_dtype(dtype, line_number);
    tensor.elements = parse_u64(elements, line_number);
    const uint64_t rank_value = parse_u64(rank, line_number);
    tensor.shape = parse_shape(shape, line_number);
    if (rank_value == 0 || rank_value > HABANA_DIM_MAX ||
        tensor.shape.size() != rank_value || tensor.elements == 0) {
      throw std::runtime_error(manifest_error(line_number, "invalid shape"));
    }
    uint64_t product = 1;
    for (const uint64_t size : tensor.shape) {
      if (size == 0 || product > UINT64_MAX / size) {
        throw std::runtime_error(manifest_error(line_number, "shape overflow"));
      }
      product *= size;
    }
    if (product != tensor.elements) {
      throw std::runtime_error(manifest_error(line_number, "element count mismatch"));
    }
    try {
      tensor.bytes = tensor_byte_size(tensor.dtype, tensor.elements);
    } catch (const std::exception& error) {
      throw std::runtime_error(manifest_error(line_number, error.what()));
    }
    const std::filesystem::path relative(tensor.file);
    if (relative.empty() || relative.is_absolute() ||
        std::find(relative.begin(), relative.end(), "..") != relative.end()) {
      throw std::runtime_error(manifest_error(line_number, "unsafe file path"));
    }
    const std::filesystem::path file = directory / relative;
    if (!std::filesystem::is_regular_file(file) ||
        std::filesystem::file_size(file) != tensor.bytes) {
      throw std::runtime_error(manifest_error(line_number, "file size mismatch"));
    }
    if (tensor.sha256.size() != 64 ||
        !std::all_of(tensor.sha256.begin(), tensor.sha256.end(), [](unsigned char c) {
          return std::isxdigit(c) != 0;
        })) {
      throw std::runtime_error(manifest_error(line_number, "invalid SHA256"));
    }
    std::transform(tensor.sha256.begin(), tensor.sha256.end(),
                   tensor.sha256.begin(), [](unsigned char c) {
                     return std::tolower(c);
                   });
    if (common::sha256_file(file) != tensor.sha256) {
      throw std::runtime_error(manifest_error(line_number, "file content mismatch"));
    }
    tensors.push_back(std::move(tensor));
  }
  if (!input.eof() || tensors.empty()) {
    throw std::runtime_error("invalid or empty weight manifest");
  }
  return tensors;
}

WeightManifest::WeightManifest(std::vector<WeightTensor> tensors)
    : tensors_(std::move(tensors)) {
  if (tensors_.empty()) {
    throw std::invalid_argument("borrowed weight manifest must not be empty");
  }
  std::unordered_set<std::string> names;
  for (const auto& tensor : tensors_) {
    if (tensor.name.empty() || tensor.device_address == 0 ||
        tensor.shape.empty() || tensor.shape.size() > HABANA_DIM_MAX ||
        tensor.elements == 0) {
      throw std::invalid_argument("invalid borrowed weight tensor " + tensor.name);
    }
    uint64_t elements = 1;
    for (const uint64_t size : tensor.shape) {
      if (size == 0 || elements > UINT64_MAX / size) {
        throw std::invalid_argument("borrowed weight shape overflow " + tensor.name);
      }
      elements *= size;
    }
    if (elements != tensor.elements ||
        tensor.bytes != tensor_byte_size(tensor.dtype, tensor.elements)) {
      throw std::invalid_argument("borrowed weight geometry mismatch " + tensor.name);
    }
    if (!names.insert(tensor.name).second) {
      throw std::invalid_argument("duplicate borrowed weight tensor " + tensor.name);
    }
  }
}

WeightManifest::WeightManifest(const std::filesystem::path& directory,
                                   DeviceRuntime& runtime, HbmArena& arena)
    : tensors_(read_manifest(directory)) {
  for (auto& tensor : tensors_) {
    std::vector<uint8_t> host(tensor.bytes);
    std::ifstream input(directory / tensor.file, std::ios::binary);
    if (!input.read(reinterpret_cast<char*>(host.data()), host.size())) {
      throw std::runtime_error("failed to read weight tensor " + tensor.name);
    }
    if (common::sha256_bytes(host.data(), host.size()) != tensor.sha256) {
      throw std::runtime_error("weight content changed before upload " + tensor.name);
    }
    tensor.device_address = arena.allocate(tensor.bytes);
    arena.copy_h2d_sync(runtime.h2d_stream(), host.data(),
                        tensor.device_address, tensor.bytes);
  }
}

const WeightTensor& WeightManifest::get(const std::string& name) const {
  const auto it = std::find_if(tensors_.begin(), tensors_.end(),
                               [&](const WeightTensor& tensor) {
                                 return tensor.name == name;
                               });
  if (it == tensors_.end()) {
    throw std::out_of_range("unknown weight tensor " + name);
  }
  return *it;
}

}  // namespace gaudirt::runtime
