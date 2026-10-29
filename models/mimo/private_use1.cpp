// SPDX-License-Identifier: Apache-2.0
#include "layer0.hpp"
#include "layer_b.hpp"
#include "layer_c.hpp"

#include "gaudirt/runtime/weight_manifest.hpp"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/EmptyTensor.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace gaudirt::mimo {
namespace {

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
  const uint64_t bytes = std::filesystem::file_size(path);
  std::vector<uint8_t> data(bytes);
  std::ifstream input(path, std::ios::binary);
  if (!input.read(reinterpret_cast<char*>(data.data()), bytes)) {
    throw std::runtime_error("PrivateUse1: cannot read HCCL unique ID");
  }
  return data;
}

void write_file(const std::filesystem::path& path,
                const std::vector<uint8_t>& data) {
  std::ofstream output(path, std::ios::binary);
  if (!output.write(reinterpret_cast<const char*>(data.data()), data.size())) {
    throw std::runtime_error("PrivateUse1: cannot write HCCL unique ID");
  }
}

hcclUniqueId exchange_unique_id(uint32_t rank,
                                const std::filesystem::path& path) {
  hcclUniqueId id;
  if (rank == 0) {
    if (hcclGetUniqueId(&id) != hcclSuccess) {
      throw std::runtime_error("PrivateUse1: hcclGetUniqueId failed");
    }
    auto temporary = path;
    temporary += ".tmp";
    write_file(temporary,
               std::vector<uint8_t>(reinterpret_cast<uint8_t*>(&id),
                                    reinterpret_cast<uint8_t*>(&id) + sizeof(id)));
    std::filesystem::rename(temporary, path);
    return id;
  }
  for (unsigned attempt = 0; attempt < 2400; ++attempt) {
    if (std::filesystem::is_regular_file(path) &&
        std::filesystem::file_size(path) == sizeof(id)) {
      const auto data = read_file(path);
      std::copy(data.begin(), data.end(), reinterpret_cast<uint8_t*>(&id));
      return id;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  throw std::runtime_error("PrivateUse1: HCCL unique ID timeout");
}

enum class Kind : int { A = 0, B = 1, C = 2 };

Kind checked_kind(int64_t value) {
  if (value < 0 || value > 2) {
    throw std::invalid_argument("PrivateUse1: layer kind must be 0(Type A), 1(Type B), or 2(Type C)");
  }
  return static_cast<Kind>(value);
}

struct LayerEntry {
  Kind kind;
  std::shared_ptr<runtime::WeightManifest> attention_weights;
  std::shared_ptr<runtime::WeightManifest> dense_weights;
  uint64_t residual_output = 0;
  uint64_t layer_output = 0;
};

struct Instance {
  Kind kind;
  uint32_t cache_slots = 0;
  uint32_t page_count = 0;
  std::unique_ptr<Layer0> a;
  std::unique_ptr<LayerB> b;
  std::unique_ptr<LayerC> c;
  Layer0::Io a_io{};
  LayerB::Io b_io{};
  LayerC::Io c_io{};

  std::vector<uint64_t> recipe_identity() const {
    if (a) return a->recipe_identity();
    if (b) return b->recipe_identity();
    if (c) return c->recipe_identity();
    throw std::runtime_error("PrivateUse1: empty layer instance");
  }
};

class PrivateProgram final : public std::enable_shared_from_this<PrivateProgram> {
 public:
  PrivateProgram(uint32_t rank, uint32_t world,
                 const std::filesystem::path& unique_id,
                 const std::filesystem::path& recipe_cache)
      : rank_(rank), runtime_([&] {
          runtime::DeviceRuntime::Config config;
          config.module_id = rank;
          config.rank = rank;
          config.world_size = world;
          config.pin_cpu = rank * 4;
          return config;
        }()),
        arena_(runtime_.device_id(), 4 * 1024 * 1024 * 1024ULL),
        recipe_cache_(recipe_cache.empty()
                          ? std::filesystem::path{}
                          : recipe_cache / ("rank" + std::to_string(rank))) {
    if (world != 8 || rank >= world) {
      throw std::invalid_argument("PrivateUse1: only TP8 is supported");
    }
    runtime_.init_comm(exchange_unique_id(rank, unique_id));
  }

  void register_layer(uint32_t layer_id, int64_t kind_value,
                      const std::string& attention_weights,
                      const std::string& dense_weights,
                      uint32_t cache_slots, uint32_t page_count) {
    const Kind kind = checked_kind(kind_value);
    if (layer_id >= 70) {
      throw std::invalid_argument("PrivateUse1: layer_id must be in [0,70)");
    }
    if (entries_.count(layer_id)) {
      throw std::invalid_argument("PrivateUse1: duplicate layer_id");
    }
    if (cache_slots == 0 || cache_slots % 128 != 0 || page_count == 0 ||
        (kind == Kind::C && page_count != 2) ||
        (kind != Kind::C && page_count > 64)) {
      throw std::invalid_argument("PrivateUse1: invalid cache or page geometry");
    }
    auto attention = std::make_shared<runtime::WeightManifest>(
        std::filesystem::path(attention_weights) / ("rank" + std::to_string(rank_)),
        runtime_, arena_);
    std::shared_ptr<runtime::WeightManifest> dense;
    if (kind == Kind::A) {
      if (dense_weights.empty()) {
        throw std::invalid_argument("PrivateUse1: Type A requires dense weights");
      }
      dense = std::make_shared<runtime::WeightManifest>(
          std::filesystem::path(dense_weights) / ("rank" + std::to_string(rank_)),
          runtime_, arena_);
    } else if (!dense_weights.empty()) {
      throw std::invalid_argument("PrivateUse1: dense weights are only valid for Type A");
    }
    LayerEntry entry{kind, std::move(attention), std::move(dense),
                     arena_.allocate(12288), arena_.allocate(12288)};
    auto [entry_it, inserted] = entries_.emplace(layer_id, std::move(entry));
    if (!inserted) {
      throw std::invalid_argument("PrivateUse1: duplicate layer_id");
    }
    if (!instances_.count(kind)) {
      instances_.emplace(kind, create_instance(kind, cache_slots, page_count,
                                               *entry_it->second.attention_weights,
                                               entry_it->second.dense_weights.get()));
    } else {
      const auto& instance = instances_.at(kind);
      if (instance->cache_slots != cache_slots || instance->page_count != page_count) {
        throw std::invalid_argument("PrivateUse1: isomorphic layers must share cache geometry");
      }
    }
  }

  std::vector<at::Tensor> forward(uint32_t layer_id,
                                  const std::vector<at::Tensor>& inputs) {
    auto& entry = entries_.at(layer_id);
    auto& instance = *instances_.at(entry.kind);
    const auto addresses = validate_inputs(instance, inputs);
    if (entry.kind == Kind::A) {
      instance.a->rebind_weights(*entry.attention_weights, *entry.dense_weights);
      auto& io = instance.a_io;
      set_common_attention(io.attention, addresses);
      io.attention.residual = addresses[0];
      io.dense_outputs.residual_out_bf16 = entry.residual_output;
      io.layer_output_bf16 = entry.layer_output;
      instance.a->execute(io, runtime_.compute_stream());
    } else if (entry.kind == Kind::B) {
      instance.b->rebind_weights(*entry.attention_weights);
      auto& io = instance.b_io;
      set_common_attention(io.attention, addresses);
      io.moe_outputs.residual_out_bf16 = entry.residual_output;
      io.layer_output_bf16 = entry.layer_output;
      instance.b->execute(io, runtime_.compute_stream());
    } else {
      instance.c->rebind_weights(*entry.attention_weights);
      auto& io = instance.c_io;
      set_common_attention(io.attention, addresses);
      io.attention.window_pages = addresses[7];
      io.attention.window_groups = addresses[9];
      io.moe_outputs.residual_out_bf16 = entry.residual_output;
      io.layer_output_bf16 = entry.layer_output;
      instance.c->execute(io, runtime_.compute_stream());
    }
    return {wrap(entry.residual_output, {1, 6144}, at::kBFloat16),
            wrap(entry.layer_output, {1, 6144}, at::kBFloat16)};
  }

  std::vector<uint64_t> recipe_identity(int64_t kind_value) const {
    return instances_.at(checked_kind(kind_value))->recipe_identity();
  }

  std::vector<uint64_t> recipe_cache_stats() const {
    const auto [loaded, saved] = recipe_cache_.stats();
    return {loaded, saved};
  }

  at::Tensor wrap(uint64_t address, const std::vector<int64_t>& shape,
                  at::ScalarType dtype) {
    if (!owns_address(address)) {
      throw std::invalid_argument("PrivateUse1: address is outside the owned arena");
    }
    auto owner = shared_from_this();
    return at::from_blob(reinterpret_cast<void*>(address), shape,
                         [owner](void*) {},
                         at::TensorOptions().device(c10::Device(c10::kPrivateUse1, rank_))
                             .dtype(dtype).requires_grad(false));
  }

  uint64_t address_of(const at::Tensor& tensor) const {
    if (tensor.device().type() != c10::kPrivateUse1 || !tensor.is_contiguous()) {
      throw std::invalid_argument("PrivateUse1: expected a contiguous GaudiRT tensor");
    }
    const uint64_t address = reinterpret_cast<uint64_t>(tensor.data_ptr());
    if (!owns_address(address)) {
      throw std::invalid_argument("PrivateUse1: tensor does not belong to this rank");
    }
    return address;
  }

  bool owns_address(const void* pointer) const {
    return owns_address(reinterpret_cast<uint64_t>(pointer));
  }

  bool owns_address(uint64_t address) const {
    return address != 0 && address >= arena_.base_addr() &&
           address < arena_.base_addr() + arena_.used_bytes();
  }

  uint32_t rank() const noexcept { return rank_; }
  uint64_t allocate(size_t bytes) { return arena_.allocate(bytes); }

  void copy_h2d(uint64_t destination, const void* source, size_t bytes) {
    arena_.copy_h2d_sync(runtime_.h2d_stream(), source, destination, bytes);
  }

  void copy_d2h(void* destination, uint64_t source, size_t bytes) {
    runtime_.synchronize_compute();
    arena_.copy_d2h_sync(runtime_.d2h_stream(), source, destination, bytes);
  }

  void synchronize_compute() const { runtime_.synchronize_compute(); }
  void synchronize_all() const { runtime_.synchronize_all(); }

 private:
  std::unique_ptr<Instance> create_instance(Kind kind, uint32_t cache_slots,
                                            uint32_t page_count,
                                            runtime::WeightManifest& attention_weights,
                                            runtime::WeightManifest* dense_weights) {
    auto instance = std::make_unique<Instance>();
    instance->kind = kind;
    instance->cache_slots = cache_slots;
    instance->page_count = page_count;
    if (kind == Kind::A) {
      instance->a = std::make_unique<Layer0>(arena_, runtime_, attention_weights,
                                             *dense_weights, cache_slots, page_count,
                                             &recipe_cache_);
      auto& io = instance->a_io;
      io.attention.normalized = arena_.allocate(12288);
      io.attention.qkv = arena_.allocate(3392 * 2);
      io.attention.query = arena_.allocate(16 * 192 * 2);
      io.attention.new_key = arena_.allocate(192 * 2);
      io.attention.new_value = arena_.allocate(128 * 2);
      io.attention.context = arena_.allocate(16 * 128 * 2);
      io.attention.output_local_fp32 = arena_.allocate(6144 * 4);
      io.dense_outputs.activation_fp32 = arena_.allocate(2048 * 4);
      io.dense_outputs.output_local_fp32 = arena_.allocate(6144 * 4);
      io.attention_output_bf16 = arena_.allocate(12288);
      io.attention_gathered_fp32 = arena_.allocate(8 * 6144 * 4);
      io.down_gathered_fp32 = arena_.allocate(8 * 6144 * 4);
    } else if (kind == Kind::B) {
      instance->b = std::make_unique<LayerB>(arena_, runtime_, attention_weights,
                                             cache_slots, page_count, &recipe_cache_);
      auto& io = instance->b_io;
      io.attention.residual_out = arena_.allocate(12288);
      io.attention.normalized = arena_.allocate(12288);
      io.attention.qkv = arena_.allocate(3392 * 2);
      io.attention.query = arena_.allocate(16 * 192 * 2);
      io.attention.new_key = arena_.allocate(192 * 2);
      io.attention.new_value = arena_.allocate(128 * 2);
      io.attention.context = arena_.allocate(16 * 128 * 2);
      io.attention.output_local_fp32 = arena_.allocate(6144 * 4);
      io.moe_outputs.normalized_bf16 = arena_.allocate(12288);
      io.moe_outputs.topk_ids_i32 = arena_.allocate(32);
      io.moe_outputs.topk_weights_fp32 = arena_.allocate(32);
      io.moe_outputs.moe_local_fp32 = arena_.allocate(6144 * 4);
      io.attention_output_bf16 = arena_.allocate(12288);
      io.attention_gathered_fp32 = arena_.allocate(8 * 6144 * 4);
      io.moe_gathered_fp32 = arena_.allocate(8 * 6144 * 4);
    } else {
      instance->c = std::make_unique<LayerC>(arena_, runtime_, attention_weights,
                                             cache_slots, &recipe_cache_);
      auto& io = instance->c_io;
      io.attention.residual_out = arena_.allocate(12288);
      io.attention.normalized = arena_.allocate(12288);
      io.attention.qkv = arena_.allocate(3392 * 2);
      io.attention.query = arena_.allocate(16 * 192 * 2);
      io.attention.new_key = arena_.allocate(192 * 2);
      io.attention.new_value = arena_.allocate(128 * 2);
      io.attention.context = arena_.allocate(16 * 128 * 2);
      io.attention.output_local = arena_.allocate(12288);
      io.moe_outputs.normalized_bf16 = arena_.allocate(12288);
      io.moe_outputs.topk_ids_i32 = arena_.allocate(32);
      io.moe_outputs.topk_weights_fp32 = arena_.allocate(32);
      io.moe_outputs.moe_local_fp32 = arena_.allocate(6144 * 4);
      io.attention_output_bf16 = arena_.allocate(12288);
      io.attention_gathered_bf16 = arena_.allocate(8 * 6144 * 2);
      io.moe_gathered_fp32 = arena_.allocate(8 * 6144 * 4);
    }
    return instance;
  }

  std::array<uint64_t, 11> validate_inputs(const Instance& instance,
                                           const std::vector<at::Tensor>& inputs) const {
    if (inputs.size() != 11) {
      throw std::invalid_argument("PrivateUse1: expected eleven explicit dynamic inputs");
    }
    const std::array<at::ScalarType, 11> dtypes = {
        at::kBFloat16, at::kBFloat16, at::kBFloat16, at::kBFloat16,
        at::kBFloat16, at::kBFloat16, at::kInt, at::kInt, at::kInt,
        at::kInt, at::kInt};
    const std::array<int64_t, 11> elements = {
        6144, 6144, 64, 64, static_cast<int64_t>(instance.cache_slots * 192),
        static_cast<int64_t>(instance.cache_slots * 128), 1,
        static_cast<int64_t>(instance.page_count),
        static_cast<int64_t>(instance.page_count),
        static_cast<int64_t>(instance.page_count), 1};
    std::array<uint64_t, 11> addresses = {};
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].scalar_type() != dtypes[i] || inputs[i].numel() != elements[i]) {
        throw std::invalid_argument("PrivateUse1: invalid input dtype or geometry at index " +
                                    std::to_string(i));
      }
      addresses[i] = address_of(inputs[i]);
    }
    return addresses;
  }

  static void set_common_attention(runtime::MimoAttentionIo& io,
                                   const std::array<uint64_t, 11>& addresses) {
    io.hidden_states = addresses[0];
    io.residual = addresses[1];
    io.rope_cos = addresses[2];
    io.rope_sin = addresses[3];
    io.key_cache = addresses[4];
    io.value_cache = addresses[5];
    io.slot_mapping = addresses[6];
    io.full_block_list = addresses[7];
    io.full_block_usage = addresses[8];
    io.full_block_groups = addresses[9];
    io.position = addresses[10];
  }

  uint32_t rank_;
  runtime::DeviceRuntime runtime_;
  runtime::HbmArena arena_;
  runtime::RecipeCache recipe_cache_;
  std::unordered_map<Kind, std::unique_ptr<Instance>> instances_;
  std::unordered_map<uint32_t, LayerEntry> entries_;
};

std::mutex registry_mutex;
std::unordered_map<uint32_t, std::shared_ptr<PrivateProgram>> registry;

std::shared_ptr<PrivateProgram> get(uint32_t rank) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  const auto it = registry.find(rank);
  if (it == registry.end()) {
    throw std::runtime_error("PrivateUse1: rank is not initialized");
  }
  return it->second;
}

std::shared_ptr<PrivateProgram> get_sole_instance() {
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (registry.size() != 1) {
    throw std::runtime_error("PrivateUse1: expected exactly one initialized rank in this process");
  }
  return registry.begin()->second;
}

class Hooks final : public at::PrivateUse1HooksInterface {
 public:
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return true; }

  at::Device getDeviceFromPtr(void* data) const override {
    std::lock_guard<std::mutex> lock(registry_mutex);
    for (const auto& [rank, instance] : registry) {
      if (instance->owns_address(data)) {
        return at::Device(c10::kPrivateUse1, rank);
      }
    }
    throw std::invalid_argument("PrivateUse1: pointer has no owning rank");
  }

  bool hasPrimaryContext(c10::DeviceIndex device_index) const override {
    std::lock_guard<std::mutex> lock(registry_mutex);
    return device_index >= 0 && registry.count(device_index) != 0;
  }
};

class Allocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t bytes) override {
    auto instance = get_sole_instance();
    if (bytes == 0) {
      return {nullptr, c10::Device(c10::kPrivateUse1, instance->rank())};
    }
    void* pointer = reinterpret_cast<void*>(instance->allocate(bytes));
    return c10::InefficientStdFunctionContext::makeDataPtr(
        pointer, [instance](void*) {}, c10::Device(c10::kPrivateUse1, instance->rank()));
  }

  void copy_data(void*, const void*, std::size_t) const override {
    TORCH_CHECK(false, "PrivateUse1: device-to-device allocator copies are not supported");
  }
};

class Guard final : public c10::impl::DeviceGuardImplInterface {
 public:
  c10::DeviceType type() const override { return c10::kPrivateUse1; }

  c10::Device exchangeDevice(c10::Device device) const override {
    auto instance = checked(device);
    return c10::Device(c10::kPrivateUse1, instance->rank());
  }

  c10::Device getDevice() const override {
    return c10::Device(c10::kPrivateUse1, get_sole_instance()->rank());
  }

  void setDevice(c10::Device device) const override { checked(device); }
  void uncheckedSetDevice(c10::Device) const noexcept override {}

  c10::Stream getStream(c10::Device device) const override {
    auto instance = checked(device);
    return c10::Stream(c10::Stream::DEFAULT,
                       c10::Device(c10::kPrivateUse1, instance->rank()));
  }

  c10::Stream exchangeStream(c10::Stream stream) const override {
    auto instance = checked(stream.device());
    return c10::Stream(c10::Stream::DEFAULT,
                       c10::Device(c10::kPrivateUse1, instance->rank()));
  }

  c10::DeviceIndex deviceCount() const noexcept override { return 8; }

  bool queryStream(const c10::Stream& stream) const override {
    checked(stream.device())->synchronize_compute();
    return true;
  }

  void synchronizeStream(const c10::Stream& stream) const override {
    checked(stream.device())->synchronize_compute();
  }

  void synchronizeDevice(c10::DeviceIndex device_index) const override {
    checked(c10::Device(c10::kPrivateUse1, device_index))->synchronize_all();
  }

 private:
  static std::shared_ptr<PrivateProgram> checked(c10::Device device) {
    auto instance = get_sole_instance();
    if (device.type() != c10::kPrivateUse1 ||
        (device.index() >= 0 &&
         static_cast<uint32_t>(device.index()) != instance->rank())) {
      throw std::invalid_argument("PrivateUse1: device does not belong to this process");
    }
    return instance;
  }
};

void initialize_backend() {
  static std::once_flag flag;
  static Hooks hooks;
  static Allocator allocator;
  static Guard guard;
  std::call_once(flag, [] {
    at::RegisterPrivateUse1HooksInterface(&hooks);
    c10::SetAllocator(c10::kPrivateUse1, &allocator, 1);
    c10::impl::DeviceGuardImplRegistrar registrar(c10::kPrivateUse1, &guard);
  });
}

at::Tensor& copy_tensor(at::Tensor& destination, const at::Tensor& source,
                        bool non_blocking) {
  TORCH_CHECK(!non_blocking, "PrivateUse1: non-blocking copies are not supported");
  TORCH_CHECK(destination.is_contiguous() && source.is_contiguous(),
              "PrivateUse1: copies require contiguous tensors");
  TORCH_CHECK(destination.scalar_type() == source.scalar_type() &&
                  destination.sizes() == source.sizes(),
              "PrivateUse1: copies require identical dtype and shape");
  if (destination.nbytes() == 0) return destination;
  auto instance = get_sole_instance();
  if (destination.device().type() == c10::kPrivateUse1 &&
      source.device().type() == c10::kCPU) {
    instance->copy_h2d(instance->address_of(destination), source.data_ptr(),
                       source.nbytes());
    return destination;
  }
  if (destination.device().type() == c10::kCPU &&
      source.device().type() == c10::kPrivateUse1) {
    instance->copy_d2h(destination.data_ptr(), instance->address_of(source),
                       source.nbytes());
    return destination;
  }
  TORCH_CHECK(false, "PrivateUse1: only CPU-to-device and device-to-CPU copies are supported");
}

void check_factory_options(const std::optional<at::Layout>& layout,
                           const std::optional<at::Device>& device,
                           const std::optional<bool>& pin_memory) {
  TORCH_CHECK(!layout || *layout == at::kStrided,
              "PrivateUse1: only strided tensors are supported");
  TORCH_CHECK(!pin_memory.value_or(false),
              "PrivateUse1: pinned device allocation is not supported");
  if (device) {
    auto instance = get_sole_instance();
    TORCH_CHECK(device->type() == c10::kPrivateUse1 &&
                    (device->index() < 0 ||
                     static_cast<uint32_t>(device->index()) == instance->rank()),
                "PrivateUse1: invalid allocation device");
  }
}

at::Tensor empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                 std::optional<at::Layout> layout, std::optional<at::Device> device,
                 std::optional<bool> pin_memory,
                 std::optional<at::MemoryFormat> memory_format) {
  check_factory_options(layout, device, pin_memory);
  return at::detail::empty_generic(size, c10::GetAllocator(c10::kPrivateUse1),
                                   c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                   dtype.value_or(at::kFloat), memory_format);
}

at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         std::optional<at::ScalarType> dtype,
                         std::optional<at::Layout> layout,
                         std::optional<at::Device> device,
                         std::optional<bool> pin_memory) {
  check_factory_options(layout, device, pin_memory);
  return at::detail::empty_strided_generic(
      size, stride, c10::GetAllocator(c10::kPrivateUse1),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), dtype.value_or(at::kFloat));
}

void initialize(uint32_t rank, uint32_t world, const std::string& unique_id,
                const std::string& recipe_cache) {
  initialize_backend();
  auto instance = std::make_shared<PrivateProgram>(rank, world, unique_id,
                                                   recipe_cache);
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (!registry.empty()) {
    throw std::runtime_error("PrivateUse1: only one rank may be initialized per process");
  }
  registry.emplace(rank, std::move(instance));
}

void register_layer(uint32_t rank, uint32_t layer_id, int64_t kind,
                    const std::string& attention_weights,
                    const std::string& dense_weights,
                    uint32_t cache_slots, uint32_t page_count) {
  try {
    get(rank)->register_layer(layer_id, kind, attention_weights, dense_weights,
                              cache_slots, page_count);
  } catch (const std::exception& error) {
    throw std::runtime_error("PrivateUse1: register layer " +
                             std::to_string(layer_id) + " with weights " +
                             attention_weights + ": " + error.what());
  }
}

std::vector<uint64_t> recipe_identity(uint32_t rank, int64_t kind) {
  return get(rank)->recipe_identity(kind);
}

std::vector<uint64_t> recipe_cache_stats(uint32_t rank) {
  return get(rank)->recipe_cache_stats();
}

void shutdown(uint32_t rank) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (registry.erase(rank) != 1) {
    throw std::runtime_error("PrivateUse1: rank is not initialized");
  }
}

std::vector<at::Tensor> forward(int64_t layer_id, const std::vector<at::Tensor>& inputs) {
  if (layer_id < 0 || layer_id >= 70) {
    throw std::invalid_argument("PrivateUse1: invalid layer_id");
  }
  auto instance = get_sole_instance();
  return instance->forward(static_cast<uint32_t>(layer_id), inputs);
}

std::vector<at::Tensor> forward_meta(int64_t layer_id,
                                     const std::vector<at::Tensor>& inputs) {
  if (layer_id < 0 || layer_id >= 70 || inputs.size() != 11) {
    throw std::invalid_argument("PrivateUse1: invalid meta inputs");
  }
  const auto options = at::TensorOptions().device(at::kMeta).dtype(at::kBFloat16);
  return {at::empty({1, 6144}, options), at::empty({1, 6144}, options)};
}

}  // namespace
}  // namespace gaudirt::mimo

TORCH_LIBRARY(gaudirt, m) {
  m.def("layer(int layer_id, Tensor[] inputs) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(gaudirt, PrivateUse1, m) {
  m.impl("layer", &gaudirt::mimo::forward);
}

TORCH_LIBRARY_IMPL(gaudirt, Meta, m) {
  m.impl("layer", &gaudirt::mimo::forward_meta);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("copy_", TORCH_FN(gaudirt::mimo::copy_tensor));
  m.impl("empty.memory_format", TORCH_FN(gaudirt::mimo::empty));
  m.impl("empty_strided", TORCH_FN(gaudirt::mimo::empty_strided));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  gaudirt::mimo::initialize_backend();
  m.def("initialize", &gaudirt::mimo::initialize);
  m.def("register_layer", &gaudirt::mimo::register_layer);
  m.def("recipe_identity", &gaudirt::mimo::recipe_identity);
  m.def("recipe_cache_stats", &gaudirt::mimo::recipe_cache_stats);
  m.def("shutdown", &gaudirt::mimo::shutdown);
}
