// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <utility>

namespace gaudirt::runtime {

// Cache paths automatically include the compiler, providers, and calling graph
// builder binaries. A present but invalid entry fails; only a missing one compiles.
class RecipeCache {
 public:
  RecipeCache() = default;
  explicit RecipeCache(std::filesystem::path directory);

  bool enabled() const noexcept { return !directory_.empty(); }
  const std::filesystem::path& directory() const noexcept { return directory_; }
  synRecipeHandle load(const std::string& key);
  void save(synRecipeHandle recipe, const std::string& key);
  std::pair<uint64_t, uint64_t> stats() const noexcept {
    return {loaded_, saved_};
  }

 private:
  std::filesystem::path file_for(const std::string& key) const;

  std::filesystem::path directory_;
  uint64_t loaded_ = 0;
  uint64_t saved_ = 0;
};

synRecipeHandle compile_cached(synGraphHandle graph, const std::string& name,
                               const std::string& key, RecipeCache* cache);

void initialize_launch_tensor_ids(synRecipeHandle recipe,
                                  synLaunchTensorInfo* launch_infos,
                                  size_t count);

}  // namespace gaudirt::runtime
