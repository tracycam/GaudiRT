// SPDX-License-Identifier: Apache-2.0
#include "layer0.hpp"

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
    throw std::runtime_error("PrivateUse1: cannot read fixture");
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

class PrivateLayer0 final : public std::enable_shared_from_this<PrivateLayer0> {
 public:
  PrivateLayer0(uint32_t rank, uint32_t world,
                const std::filesystem::path& fixtures,
                const std::filesystem::path& attention_weights,
                const std::filesystem::path& dense_weights,
                const std::filesystem::path& unique_id)
      : rank_(rank), runtime_([&] {
          runtime::DeviceRuntime::Config config;
          config.module_id = rank;
          config.rank = rank;
          config.world_size = world;
          config.pin_cpu = rank * 4;
          return config;
        }()),
        arena_(runtime_.device_id(), 512 * 1024 * 1024ULL) {
    if (world != 8 || rank >= world) {
      throw std::invalid_argument("PrivateUse1: only the TP8 layer0 slice is supported");
    }
    runtime_.init_comm(exchange_unique_id(rank, unique_id));
    const auto fixture = fixtures / ("rank" + std::to_string(rank));
    cache_slots_ = std::filesystem::file_size(fixture / "key_cache_before.bin") / (192 * 2);
    pages_ = std::filesystem::file_size(fixture / "block_list.bin") / sizeof(int32_t);
    attention_weights_ = std::make_unique<runtime::WeightManifest>(
        attention_weights / ("rank" + std::to_string(rank)), runtime_, arena_);
    dense_weights_ = std::make_unique<runtime::WeightManifest>(
        dense_weights / ("rank" + std::to_string(rank)), runtime_, arena_);
    layer_ = std::make_unique<Layer0>(arena_, runtime_, *attention_weights_,
                                      *dense_weights_, cache_slots_, pages_);

    io_.attention.hidden_states = upload_file(fixture / "residual_in.bin");
    io_.attention.residual = io_.attention.hidden_states;
    io_.attention.normalized = arena_.allocate(12288);
    io_.attention.rope_cos = upload_file(fixture / "rope_cos.bin");
    io_.attention.rope_sin = upload_file(fixture / "rope_sin.bin");
    io_.attention.key_cache = upload_file(fixture / "key_cache_before.bin");
    io_.attention.value_cache = upload_file(fixture / "value_cache_before.bin");
    io_.attention.slot_mapping = upload_file(fixture / "slot_mapping.bin");
    io_.attention.full_block_list = upload_file(fixture / "block_list.bin");
    io_.attention.full_block_usage = upload_file(fixture / "block_usage.bin");
    io_.attention.full_block_groups = upload_file(fixture / "block_groups.bin");
    io_.attention.qkv = arena_.allocate(3392 * 2);
    io_.attention.query = arena_.allocate(16 * 192 * 2);
    io_.attention.new_key = arena_.allocate(192 * 2);
    io_.attention.new_value = arena_.allocate(128 * 2);
    io_.attention.context = arena_.allocate(16 * 128 * 2);
    io_.attention.output_local_fp32 = arena_.allocate(6144 * 4);
    io_.dense_outputs.residual_out_bf16 = arena_.allocate(12288);
    io_.dense_outputs.activation_fp32 = arena_.allocate(2048 * 4);
    io_.dense_outputs.output_local_fp32 = arena_.allocate(6144 * 4);
    io_.attention_output_bf16 = arena_.allocate(12288);
    io_.attention_gathered_fp32 = arena_.allocate(world * 6144 * 4);
    io_.down_gathered_fp32 = arena_.allocate(world * 6144 * 4);
    io_.layer_output_bf16 = arena_.allocate(12288);
  }

  std::vector<at::Tensor> forward(const std::vector<at::Tensor>& inputs) {
    if (inputs.size() != 8) {
      throw std::invalid_argument("PrivateUse1: expected eight layer inputs");
    }
    const std::array<at::ScalarType, 8> dtypes = {
        at::kBFloat16, at::kBFloat16, at::kBFloat16, at::kBFloat16,
        at::kBFloat16, at::kInt, at::kInt, at::kInt};
    const std::array<int64_t, 8> elements = {
        6144, 64, 64, static_cast<int64_t>(cache_slots_ * 192),
        static_cast<int64_t>(cache_slots_ * 128), 1,
        static_cast<int64_t>(pages_), static_cast<int64_t>(pages_)};
    std::array<uint64_t, 8> addresses = {};
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].scalar_type() != dtypes[i] || inputs[i].numel() != elements[i]) {
        throw std::invalid_argument("PrivateUse1: invalid input dtype or geometry at index " +
                                    std::to_string(i));
      }
      addresses[i] = address_of(inputs[i]);
    }
    io_.attention.hidden_states = addresses[0];
    io_.attention.residual = addresses[0];
    io_.attention.rope_cos = addresses[1];
    io_.attention.rope_sin = addresses[2];
    io_.attention.key_cache = addresses[3];
    io_.attention.value_cache = addresses[4];
    io_.attention.slot_mapping = addresses[5];
    io_.attention.full_block_list = addresses[6];
    io_.attention.full_block_usage = addresses[7];
    layer_->execute(io_, runtime_.compute_stream());
    return {wrap(io_.dense_outputs.residual_out_bf16, {1, 6144}, at::kBFloat16),
            wrap(io_.layer_output_bf16, {1, 6144}, at::kBFloat16)};
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
  uint64_t upload_file(const std::filesystem::path& path) {
    const auto bytes = read_file(path);
    const uint64_t address = arena_.allocate(bytes.size());
    arena_.copy_h2d_sync(runtime_.h2d_stream(), bytes.data(), address, bytes.size());
    return address;
  }

  uint32_t rank_;
  uint64_t cache_slots_ = 0;
  uint64_t pages_ = 0;
  runtime::DeviceRuntime runtime_;
  runtime::HbmArena arena_;
  std::unique_ptr<runtime::WeightManifest> attention_weights_;
  std::unique_ptr<runtime::WeightManifest> dense_weights_;
  std::unique_ptr<Layer0> layer_;
  Layer0::Io io_{};
};

std::mutex registry_mutex;
std::unordered_map<uint32_t, std::shared_ptr<PrivateLayer0>> registry;

std::shared_ptr<PrivateLayer0> get(uint32_t rank) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  const auto it = registry.find(rank);
  if (it == registry.end()) {
    throw std::runtime_error("PrivateUse1: rank is not initialized");
  }
  return it->second;
}

std::shared_ptr<PrivateLayer0> get_sole_instance() {
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
  static std::shared_ptr<PrivateLayer0> checked(c10::Device device) {
    auto instance = get_sole_instance();
    if (device.type() != c10::kPrivateUse1 ||
        (device.index() >= 0 && device.index() != instance->rank())) {
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
                    (device->index() < 0 || device->index() == instance->rank()),
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

void initialize(uint32_t rank, uint32_t world, const std::string& fixtures,
                const std::string& attention_weights, const std::string& dense_weights,
                const std::string& unique_id) {
  initialize_backend();
  auto instance = std::make_shared<PrivateLayer0>(rank, world, fixtures, attention_weights,
                                                  dense_weights, unique_id);
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (!registry.empty()) {
    throw std::runtime_error("PrivateUse1: only one rank may be initialized per process");
  }
  registry.emplace(rank, std::move(instance));
}

void shutdown(uint32_t rank) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (registry.erase(rank) != 1) {
    throw std::runtime_error("PrivateUse1: rank is not initialized");
  }
}

std::vector<at::Tensor> forward(int64_t rank, const std::vector<at::Tensor>& inputs) {
  if (rank < 0 || rank >= 8) {
    throw std::invalid_argument("PrivateUse1: invalid rank");
  }
  return get(static_cast<uint32_t>(rank))->forward(inputs);
}

std::vector<at::Tensor> forward_meta(int64_t rank,
                                     const std::vector<at::Tensor>& inputs) {
  if (rank < 0 || rank >= 8 || inputs.size() != 8) {
    throw std::invalid_argument("PrivateUse1: invalid meta inputs");
  }
  const auto options = at::TensorOptions().device(at::kMeta).dtype(at::kBFloat16);
  return {at::empty({1, 6144}, options), at::empty({1, 6144}, options)};
}

}  // namespace
}  // namespace gaudirt::mimo

TORCH_LIBRARY(gaudirt, m) {
  m.def("layer0(int rank, Tensor[] inputs) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(gaudirt, PrivateUse1, m) {
  m.impl("layer0", &gaudirt::mimo::forward);
}

TORCH_LIBRARY_IMPL(gaudirt, Meta, m) {
  m.impl("layer0", &gaudirt::mimo::forward_meta);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("copy_", TORCH_FN(gaudirt::mimo::copy_tensor));
  m.impl("empty.memory_format", TORCH_FN(gaudirt::mimo::empty));
  m.impl("empty_strided", TORCH_FN(gaudirt::mimo::empty_strided));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  gaudirt::mimo::initialize_backend();
  m.def("initialize", &gaudirt::mimo::initialize);
  m.def("shutdown", &gaudirt::mimo::shutdown);
}
