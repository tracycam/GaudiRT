// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gaudirt::runtime {

struct WeightTensor {
  std::string name;
  synDataType dtype = syn_type_na;
  std::vector<uint64_t> shape;
  uint64_t elements = 0;
  uint64_t bytes = 0;
  std::string file;
  std::string sha256;
  uint64_t device_address = 0;
};

class WeightManifest {
 public:
  WeightManifest(const std::filesystem::path& directory,
                   DeviceRuntime& runtime, HbmArena& arena);
  // Register caller-owned device tensors without file I/O or upload. The caller
  // must keep their storage alive for every graph launch using these addresses.
  explicit WeightManifest(std::vector<WeightTensor> tensors);

  static std::vector<WeightTensor> read_manifest(
      const std::filesystem::path& directory);

  const WeightTensor& get(const std::string& name) const;
  const std::vector<WeightTensor>& tensors() const { return tensors_; }

 private:
  std::vector<WeightTensor> tensors_;
};

}  // namespace gaudirt::runtime
