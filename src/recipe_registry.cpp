// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/recipe_registry.hpp"

#include <cstring>
#include <stdexcept>

namespace gaudirt::runtime {

namespace {

uint64_t dtype_bytes(synDataType dt) noexcept {
  switch (dt) {
    case syn_type_int8:
    case syn_type_uint8:
    case syn_type_fp8_152:
    case syn_type_fp8_143:
      return 1;
    case syn_type_bf16:
    case syn_type_fp16:
    case syn_type_int16:
    case syn_type_uint16:
      return 2;
    case syn_type_float:
    case syn_type_int32:
    case syn_type_uint32:
      return 4;
    case syn_type_int64:
    case syn_type_uint64:
      return 8;
    default:
      return 4;
  }
}

}  // namespace

BindingTable::BindingTable(const std::vector<TensorSlotMeta>& slots) {
  names_.reserve(slots.size());
  launch_infos_.resize(slots.size());
  for (size_t i = 0; i < slots.size(); ++i) {
    names_.push_back(slots[i].name);
    name_to_slot_.emplace(names_.back(), i);

    synLaunchTensorInfoExt info{};
    info.tensorName = names_[i].c_str();
    info.pTensorAddress = 0;
    info.tensorType = slots[i].tensor_type;
    for (int d = 0; d < HABANA_DIM_MAX; ++d) {
      info.tensorSize[d] = slots[i].max_size[d];
    }
    info.tensorId = slots[i].tensor_id;
    launch_infos_[i] = info;
  }
}

void BindingTable::bind_slot(size_t slot_index, uint64_t hbm_addr) {
  if (slot_index >= launch_infos_.size()) {
    throw std::out_of_range("BindingTable::bind_slot: index out of range");
  }
  if (launch_infos_[slot_index].pTensorAddress != hbm_addr) {
    launch_infos_[slot_index].pTensorAddress = hbm_addr;
    ++generation_;
  }
}

void BindingTable::bind_name(const std::string& name, uint64_t hbm_addr) {
  bind_slot(find_slot(name), hbm_addr);
}

uint64_t BindingTable::address_at(size_t slot_index) const {
  if (slot_index >= launch_infos_.size()) {
    throw std::out_of_range("BindingTable::address_at: index out of range");
  }
  return launch_infos_[slot_index].pTensorAddress;
}

size_t BindingTable::find_slot(const std::string& name) const {
  auto it = name_to_slot_.find(name);
  if (it == name_to_slot_.end()) {
    throw std::out_of_range("BindingTable::find_slot: tensor name not found: " +
                            name);
  }
  return it->second;
}

std::vector<BindingSnapshot> BindingTable::snapshots() const {
  std::vector<BindingSnapshot> out(launch_infos_.size());
  for (size_t i = 0; i < launch_infos_.size(); ++i) {
    out[i].device_address = launch_infos_[i].pTensorAddress;
    out[i].shape_stride_hash = launch_infos_[i].tensorId;
  }
  return out;
}

RecipeRegistry::~RecipeRegistry() {
  for (auto& entry : entries_) {
    if (entry.handle != nullptr) {
      synRecipeDestroy(entry.handle);
      entry.handle = nullptr;
    }
  }
}

std::vector<TensorSlotMeta> RecipeRegistry::introspect_recipe(
    synRecipeHandle handle) {
  uint32_t num_tensors = 0;
  if (synTensorRetrieveLaunchAmount(handle, &num_tensors) != synSuccess) {
    throw std::runtime_error(
        "RecipeRegistry: synTensorRetrieveLaunchAmount failed");
  }
  if (num_tensors == 0) {
    return {};
  }

  std::vector<uint64_t> ids(num_tensors, 0);
  if (synTensorRetrieveLaunchIds(handle, ids.data(), num_tensors) !=
      synSuccess) {
    throw std::runtime_error(
        "RecipeRegistry: synTensorRetrieveLaunchIds failed");
  }

  std::vector<synRetrievedLaunchTensorInfoExt> infos(num_tensors);
  for (uint32_t i = 0; i < num_tensors; ++i) {
    std::memset(&infos[i], 0, sizeof(synRetrievedLaunchTensorInfoExt));
    infos[i].tensorId = ids[i];
  }
  if (synTensorRetrieveLaunchInfoByIdExt(handle, num_tensors, infos.data()) !=
      synSuccess) {
    throw std::runtime_error(
        "RecipeRegistry: synTensorRetrieveLaunchInfoByIdExt failed");
  }

  std::vector<TensorSlotMeta> slots(num_tensors);
  for (uint32_t i = 0; i < num_tensors; ++i) {
    TensorSlotMeta meta{};
    meta.slot_index = i;
    meta.name = infos[i].tensorName;
    meta.tensor_id = infos[i].tensorId;
    meta.tensor_type = infos[i].tensorType;
    meta.data_type = infos[i].tensorDataType;
    meta.dims = infos[i].tensorDims;
    meta.is_input = (infos[i].isInput != 0);

    uint64_t elems = (meta.dims > 0) ? 1 : 0;
    for (int d = 0; d < HABANA_DIM_MAX; ++d) {
      meta.max_size[d] = infos[i].tensorMaxSize[d];
      if (static_cast<uint32_t>(d) < meta.dims) {
        elems *= static_cast<uint64_t>(infos[i].tensorMaxSize[d]);
      }
    }
    meta.byte_size = elems * dtype_bytes(meta.data_type);
    slots[i] = std::move(meta);
  }
  return slots;
}

RecipeRegistry::RecipeId RecipeRegistry::register_compiled(
    const std::string& name, synRecipeHandle handle) {
  if (handle == nullptr) {
    throw std::invalid_argument(
        "RecipeRegistry::register_compiled: null recipe handle");
  }
  uint64_t ws_bytes = 0;
  if (synWorkspaceGetSize(&ws_bytes, handle) != synSuccess) {
    throw std::runtime_error("RecipeRegistry: synWorkspaceGetSize failed");
  }

  Entry entry{};
  entry.id = static_cast<RecipeId>(entries_.size());
  entry.name = name;
  entry.handle = handle;
  entry.workspace_bytes = ws_bytes;
  entry.slots = introspect_recipe(handle);
  entries_.push_back(std::move(entry));
  return entries_.back().id;
}

RecipeRegistry::RecipeId RecipeRegistry::load_from_file(
    const std::string& name, const std::string& path) {
  synRecipeHandle handle = nullptr;
  if (synRecipeDeSerialize(&handle, path.c_str()) != synSuccess ||
      handle == nullptr) {
    throw std::runtime_error("RecipeRegistry: synRecipeDeSerialize failed for " +
                             path);
  }
  return register_compiled(name, handle);
}

void RecipeRegistry::save_to_file(RecipeId id, const std::string& path) const {
  const Entry& e = get(id);
  if (synRecipeSerialize(e.handle, path.c_str()) != synSuccess) {
    throw std::runtime_error("RecipeRegistry: synRecipeSerialize failed for " +
                             path);
  }
}

const RecipeRegistry::Entry& RecipeRegistry::get(RecipeId id) const {
  if (id >= entries_.size()) {
    throw std::out_of_range("RecipeRegistry::get: invalid recipe id");
  }
  return entries_[id];
}

BindingTable RecipeRegistry::create_binding_table(RecipeId id) const {
  const Entry& e = get(id);
  return BindingTable(e.slots);
}

}  // namespace gaudirt::runtime
