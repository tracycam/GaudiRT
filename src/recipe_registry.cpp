// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/recipe_registry.hpp"
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
  struct Fingerprint { std::string providers; std::string digest; };
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

uint64_t tensor_byte_size(synDataType dtype, uint64_t elements) {
  const uint64_t bits = dtype_bits(dtype);
  if (bits == 4) return elements / 2 + elements % 2;
  const uint64_t bytes = bits / 8;
  if (elements > UINT64_MAX / bytes) {
    throw std::overflow_error("RecipeRegistry: tensor byte size overflow");
  }
  return elements * bytes;
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
