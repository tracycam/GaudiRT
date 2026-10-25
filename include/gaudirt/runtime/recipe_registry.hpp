// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <synapse_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace gaudirt::runtime {

// Metadata for a single persistent launch tensor extracted once at recipe load.
struct TensorSlotMeta {
  uint32_t slot_index = 0;
  std::string name;
  uint64_t tensor_id = 0;
  synTensorType tensor_type = DATA_TENSOR;
  synDataType data_type = syn_type_na;
  uint32_t dims = 0;
  TSize max_size[HABANA_DIM_MAX]{};
  bool is_input = false;
  uint64_t byte_size = 0;
};

// Pre-built contiguous binding array for a registered recipe.
// Supports O(1) slot rebinding and tracks a monotonic binding generation.
class BindingTable {
 public:
  explicit BindingTable(const std::vector<TensorSlotMeta>& slots);

  // Bind HBM address by slot index (O(1)) or by tensor name.
  void bind_slot(size_t slot_index, uint64_t hbm_addr);
  void bind_name(const std::string& name, uint64_t hbm_addr);

  uint64_t address_at(size_t slot_index) const;
  size_t find_slot(const std::string& name) const;
  size_t size() const noexcept { return launch_infos_.size(); }
  uint64_t generation() const noexcept { return generation_; }

  const synLaunchTensorInfoExt* data() const noexcept {
    return launch_infos_.data();
  }
  const std::vector<synLaunchTensorInfoExt>& launch_infos() const noexcept {
    return launch_infos_;
  }

 private:
  std::vector<std::string> names_;
  std::unordered_map<std::string, size_t> name_to_slot_;
  std::vector<synLaunchTensorInfoExt> launch_infos_;
  uint64_t generation_ = 1;
};

// Manages compiled Synapse recipes: registration, disk serialization
// (.recipe), deserialization, workspace query, and persistent tensor metadata.
class RecipeRegistry {
 public:
  using RecipeId = uint32_t;

  struct Entry {
    RecipeId id = 0;
    std::string name;
    synRecipeHandle handle = nullptr;
    uint64_t workspace_bytes = 0;
    std::vector<TensorSlotMeta> slots;
  };

  RecipeRegistry() = default;
  ~RecipeRegistry();

  RecipeRegistry(const RecipeRegistry&) = delete;
  RecipeRegistry& operator=(const RecipeRegistry&) = delete;

  // Take ownership of an already-compiled synRecipeHandle and introspect it.
  RecipeId register_compiled(const std::string& name, synRecipeHandle handle);

  // Load a serialized recipe from disk via synRecipeDeSerialize.
  RecipeId load_from_file(const std::string& name, const std::string& path);

  // Save a registered recipe to disk via synRecipeSerialize.
  void save_to_file(RecipeId id, const std::string& path) const;

  const Entry& get(RecipeId id) const;
  BindingTable create_binding_table(RecipeId id) const;
  size_t size() const noexcept { return entries_.size(); }

 private:
  static std::vector<TensorSlotMeta> introspect_recipe(synRecipeHandle handle);
  std::vector<Entry> entries_;
};

}  // namespace gaudirt::runtime
