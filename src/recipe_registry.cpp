// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/recipe_registry.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

namespace gaudirt::runtime {

namespace {

class RecipeHandleGuard {
 public:
  explicit RecipeHandleGuard(synRecipeHandle handle) : handle_(handle) {}
  ~RecipeHandleGuard() {
    if (handle_) synRecipeDestroy(handle_);
  }
  RecipeHandleGuard(const RecipeHandleGuard&) = delete;
  RecipeHandleGuard& operator=(const RecipeHandleGuard&) = delete;

  synRecipeHandle get() const { return handle_; }
  synRecipeHandle release() {
    synRecipeHandle handle = handle_;
    handle_ = nullptr;
    return handle;
  }

 private:
  synRecipeHandle handle_;
};

uint64_t dtype_bits(synDataType dtype) {
  switch (dtype) {
    case syn_type_int4:
    case syn_type_uint4:
    case syn_type_packed_nf4:
    case syn_type_packed_mxfp4:
      return 4;
    case syn_type_int8:
    case syn_type_uint8:
    case syn_type_fp8_152:
    case syn_type_fp8_143:
      return 8;
    case syn_type_bf16:
    case syn_type_fp16:
    case syn_type_int16:
    case syn_type_uint16:
    case syn_type_ufp16:
      return 16;
    case syn_type_float:
    case syn_type_int32:
    case syn_type_uint32:
    case syn_type_tf32:
    case syn_type_hb_float:
      return 32;
    case syn_type_int64:
    case syn_type_uint64:
      return 64;
    default:
      throw std::invalid_argument("RecipeRegistry: unsupported synDataType " +
                                  std::to_string(static_cast<int>(dtype)));
  }
}

}  // namespace

uint64_t tensor_byte_size(synDataType dtype, uint64_t elements) {
  const uint64_t bits = dtype_bits(dtype);
  if (bits == 4) return elements / 2 + elements % 2;
  const uint64_t bytes = bits / 8;
  if (elements > UINT64_MAX / bytes) {
    throw std::overflow_error("RecipeRegistry: tensor byte size overflow");
  }
  return elements * bytes;
}

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

BindingTable::BindingTable(const BindingTable& other)
    : names_(other.names_),
      name_to_slot_(other.name_to_slot_),
      launch_infos_(other.launch_infos_),
      generation_(other.generation_) {
  relink_tensor_names();
}

BindingTable& BindingTable::operator=(const BindingTable& other) {
  if (this != &other) {
    BindingTable copy(other);
    *this = std::move(copy);
  }
  return *this;
}

BindingTable::BindingTable(BindingTable&& other) noexcept
    : names_(std::move(other.names_)),
      name_to_slot_(std::move(other.name_to_slot_)),
      launch_infos_(std::move(other.launch_infos_)),
      generation_(other.generation_) {
  relink_tensor_names();
}

BindingTable& BindingTable::operator=(BindingTable&& other) noexcept {
  if (this != &other) {
    names_ = std::move(other.names_);
    name_to_slot_ = std::move(other.name_to_slot_);
    launch_infos_ = std::move(other.launch_infos_);
    generation_ = other.generation_;
    relink_tensor_names();
  }
  return *this;
}

void BindingTable::relink_tensor_names() {
  for (size_t i = 0; i < launch_infos_.size(); ++i) {
    launch_infos_[i].tensorName = names_[i].c_str();
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
        const uint64_t size = infos[i].tensorMaxSize[d];
        if (size != 0 && elems > UINT64_MAX / size) {
          throw std::overflow_error("RecipeRegistry: tensor element count overflow");
        }
        elems *= size;
      }
    }
    meta.byte_size = tensor_byte_size(meta.data_type, elems);
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
  RecipeHandleGuard owner(handle);
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
  owner.release();
  return entries_.back().id;
}

RecipeRegistry::RecipeId RecipeRegistry::load_from_file(
    const std::string& name, const std::string& path) {
  synRecipeHandle handle = nullptr;
  const synStatus status = synRecipeDeSerialize(&handle, path.c_str());
  RecipeHandleGuard owner(handle);
  if (status != synSuccess || handle == nullptr) {
    throw std::runtime_error("RecipeRegistry: synRecipeDeSerialize failed for " +
                             path);
  }
  return register_compiled(name, owner.release());
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
