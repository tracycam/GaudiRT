// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/recipe_cache.hpp"
#include "gaudirt/common/sha256.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gaudirt::runtime {

namespace {

std::string digest_text(const std::string& text) {
  return common::sha256_bytes(reinterpret_cast<const uint8_t*>(text.data()), text.size());
}

std::string builder_fingerprint(const char* filename) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::string> hashes;
  std::lock_guard lock(mutex);
  const auto found = hashes.find(filename);
  if (found != hashes.end()) return found->second;
  const auto digest = common::sha256_file(filename);
  hashes.emplace(filename, digest);
  return digest;
}

std::string environment_fingerprint() {
  // GC_KERNEL_PATH is a process-start contract; compute its ordered content
  // hashes once, before any recipe load or serialization.
  struct Fingerprint {
    std::string providers;
    std::string digest;
  };
  static const Fingerprint fingerprint = [] {
    Dl_info compiler{};
    if (!dladdr(reinterpret_cast<void*>(&synGraphCompile), &compiler) || !compiler.dli_fname)
      throw std::runtime_error("RecipeCache: cannot identify libSynapse");
    const char* kernels = std::getenv("GC_KERNEL_PATH");
    if (!kernels || !*kernels)
      throw std::invalid_argument("RecipeCache: GC_KERNEL_PATH is required");
    std::string input = "gaudirt-environment-v1\n";
    const auto add = [&](const std::filesystem::path& path) {
      input += common::sha256_file(path);
      input += '\n';
    };
    add(compiler.dli_fname);
    add(std::filesystem::path(compiler.dli_fname).parent_path() / "libSynapseMme.so");
    const std::string paths(kernels);
    size_t start = 0;
    size_t count = 0;
    while (true) {
      const auto end = paths.find(':', start);
      const auto path = paths.substr(start, end - start);
      if (path.empty()) throw std::invalid_argument("RecipeCache: empty provider path");
      if (++count > 15) throw std::invalid_argument("RecipeCache: too many TPC providers");
      add(path);
      if (end == std::string::npos) break;
      start = end + 1;
    }
    return Fingerprint{paths, digest_text(input)};
  }();
  const char* current = std::getenv("GC_KERNEL_PATH");
  if (!current || fingerprint.providers != current)
    throw std::runtime_error("RecipeCache: GC_KERNEL_PATH changed after initialization");
  return fingerprint.digest;
}

}  // namespace

RecipeCache::RecipeCache(std::filesystem::path root) {
  if (root.empty()) return;
  Dl_info builder{};
  if (!dladdr(__builtin_return_address(0), &builder) || !builder.dli_fname)
    throw std::runtime_error("RecipeCache: cannot identify graph builder binary");
  const std::string input = "gaudirt-recipe-cache-v2\n" + environment_fingerprint() +
                            "\n" + builder_fingerprint(builder.dli_fname) + "\n";
  directory_ = std::move(root) / digest_text(input);
}

std::filesystem::path RecipeCache::file_for(const std::string& key) const {
  if (key.empty() ||
      !std::all_of(key.begin(), key.end(), [](unsigned char c) {
        return std::isalnum(c) != 0 || c == '_' || c == '-' || c == '.';
      })) {
    throw std::invalid_argument("RecipeCache: invalid key");
  }
  return directory_ / (key + ".recipe");
}

synRecipeHandle RecipeCache::load(const std::string& key) {
  if (!enabled()) return nullptr;
  const auto path = file_for(key);
  if (!std::filesystem::exists(path)) return nullptr;
  synRecipeHandle recipe = nullptr;
  const synStatus status = synRecipeDeSerialize(&recipe, path.c_str());
  if (status != synSuccess || recipe == nullptr) {
    throw std::runtime_error("RecipeCache: deserialization failed for " +
                             path.filename().string());
  }
  ++loaded_;
  return recipe;
}

void RecipeCache::save(synRecipeHandle recipe, const std::string& key) {
  if (!enabled()) return;
  if (recipe == nullptr) {
    throw std::invalid_argument("RecipeCache: cannot serialize a null recipe");
  }
  std::filesystem::create_directories(directory_);
  const auto path = file_for(key);
  static std::atomic<uint64_t> temp_seq{0};
  auto temporary = path;
  temporary += ".tmp." + std::to_string(static_cast<long>(::getpid())) + "." +
               std::to_string(temp_seq.fetch_add(1, std::memory_order_relaxed));
  std::filesystem::remove(temporary);
  if (synRecipeSerialize(recipe, temporary.c_str()) != synSuccess) {
    std::filesystem::remove(temporary);
    throw std::runtime_error("RecipeCache: serialization failed for " +
                             path.filename().string());
  }
  std::filesystem::rename(temporary, path);
  ++saved_;
}

synRecipeHandle compile_cached(synGraphHandle graph, const std::string& name,
                               const std::string& key, RecipeCache* cache) {
  synRecipeHandle recipe = cache != nullptr ? cache->load(key) : nullptr;
  if (recipe != nullptr) return recipe;
  if (synGraphCompile(&recipe, graph, name.c_str(), nullptr) != synSuccess ||
      recipe == nullptr) {
    throw std::runtime_error("RecipeCache: graph compilation failed for " + name);
  }
  if (cache != nullptr) cache->save(recipe, key);
  return recipe;
}

void initialize_launch_tensor_ids(synRecipeHandle recipe,
                                  synLaunchTensorInfo* launch_infos,
                                  size_t count) {
  if (recipe == nullptr || (count != 0 && launch_infos == nullptr)) {
    throw std::invalid_argument("initialize_launch_tensor_ids: invalid arguments");
  }
  uint32_t recipe_count = 0;
  if (synTensorRetrieveLaunchAmount(recipe, &recipe_count) != synSuccess ||
      recipe_count != count) {
    throw std::runtime_error("initialize_launch_tensor_ids: tensor count mismatch");
  }
  if (count == 0) return;
  std::vector<uint64_t> ids(count, 0);
  if (synTensorRetrieveLaunchIds(recipe, ids.data(), recipe_count) != synSuccess) {
    throw std::runtime_error("initialize_launch_tensor_ids: id query failed");
  }
  std::vector<synRetrievedLaunchTensorInfoExt> retrieved(count);
  for (size_t i = 0; i < count; ++i) {
    std::memset(&retrieved[i], 0, sizeof(retrieved[i]));
    retrieved[i].tensorId = ids[i];
  }
  if (synTensorRetrieveLaunchInfoByIdExt(recipe, recipe_count, retrieved.data()) !=
      synSuccess) {
    throw std::runtime_error("initialize_launch_tensor_ids: metadata query failed");
  }
  std::unordered_map<std::string, uint64_t> name_to_id;
  for (const auto& info : retrieved) {
    if (!name_to_id.emplace(info.tensorName, info.tensorId).second) {
      throw std::runtime_error("initialize_launch_tensor_ids: duplicate tensor name");
    }
  }
  for (size_t i = 0; i < count; ++i) {
    if (launch_infos[i].tensorName == nullptr) {
      throw std::invalid_argument("initialize_launch_tensor_ids: null tensor name");
    }
    const auto it = name_to_id.find(launch_infos[i].tensorName);
    if (it == name_to_id.end()) {
      throw std::runtime_error("initialize_launch_tensor_ids: unknown tensor " +
                               std::string(launch_infos[i].tensorName));
    }
    launch_infos[i].tensorId = it->second;
  }
}

}  // namespace gaudirt::runtime
