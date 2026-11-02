// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/hbm_arena.hpp"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/EmptyTensor.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/extension.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace gaudirt::bridge {

// The owner keeps the model program, its arena and device alive until the
// last PrivateUse1 tensor allocated from this rank has been destroyed.
struct RankContext {
  uint32_t rank_id;
  runtime::DeviceRuntime* device;
  runtime::HbmArena* arena;
  std::shared_ptr<void> owner;

  uint32_t rank() const noexcept { return rank_id; }
  bool owns_address(const void* pointer) const {
    const auto address = reinterpret_cast<uint64_t>(pointer);
    return address != 0 && address >= arena->base_addr() &&
           address < arena->base_addr() + arena->used_bytes();
  }
  uint64_t address_of(const at::Tensor& tensor) const {
    if (tensor.device().type() != c10::kPrivateUse1 || !tensor.is_contiguous())
      throw std::invalid_argument("PrivateUse1: expected a contiguous GaudiRT tensor");
    if (!owns_address(tensor.data_ptr()))
      throw std::invalid_argument("PrivateUse1: tensor does not belong to this rank");
    return reinterpret_cast<uint64_t>(tensor.data_ptr());
  }
  uint64_t allocate(size_t bytes) { return arena->allocate(bytes); }
  void copy_h2d(uint64_t dst, const void* src, size_t bytes) {
    arena->copy_h2d_sync(device->h2d_stream(), src, dst, bytes);
  }
  void copy_d2h(void* dst, uint64_t src, size_t bytes) {
    device->synchronize_compute();
    arena->copy_d2h_sync(device->d2h_stream(), src, dst, bytes);
  }
  void synchronize_compute() const { device->synchronize_compute(); }
  void synchronize_all() const { device->synchronize_all(); }
};

inline std::mutex registry_mutex;
inline std::unordered_map<uint32_t, std::shared_ptr<RankContext>> registry;

inline std::shared_ptr<RankContext> get_rank(uint32_t rank) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  const auto it = registry.find(rank);
  if (it == registry.end())
    throw std::runtime_error("PrivateUse1: rank is not initialized");
  return it->second;
}

inline std::shared_ptr<RankContext> get_sole_instance() {
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (registry.size() != 1)
    throw std::runtime_error("PrivateUse1: expected exactly one initialized rank in this process");
  return registry.begin()->second;
}

inline void register_rank(std::shared_ptr<RankContext> instance) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (!registry.empty())
    throw std::runtime_error("PrivateUse1: only one rank may be initialized per process");
  const uint32_t rank = instance->rank();
  registry.emplace(rank, std::move(instance));
}

inline void unregister_rank(uint32_t rank) {
  std::lock_guard<std::mutex> lock(registry_mutex);
  if (registry.erase(rank) != 1)
    throw std::runtime_error("PrivateUse1: rank is not initialized");
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
  static std::shared_ptr<RankContext> checked(c10::Device device) {
    auto instance = get_sole_instance();
    if (device.type() != c10::kPrivateUse1 ||
        (device.index() >= 0 &&
         static_cast<uint32_t>(device.index()) != instance->rank())) {
      throw std::invalid_argument("PrivateUse1: device does not belong to this process");
    }
    return instance;
  }
};

inline void initialize_backend() {
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

inline at::Tensor& copy_tensor(at::Tensor& destination, const at::Tensor& source,
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

inline void check_factory_options(const std::optional<at::Layout>& layout,
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

inline at::Tensor empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                 std::optional<at::Layout> layout, std::optional<at::Device> device,
                 std::optional<bool> pin_memory,
                 std::optional<at::MemoryFormat> memory_format) {
  check_factory_options(layout, device, pin_memory);
  return at::detail::empty_generic(size, c10::GetAllocator(c10::kPrivateUse1),
                                   c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                   dtype.value_or(at::kFloat), memory_format);
}

inline at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         std::optional<at::ScalarType> dtype,
                         std::optional<at::Layout> layout,
                         std::optional<at::Device> device,
                         std::optional<bool> pin_memory) {
  check_factory_options(layout, device, pin_memory);
  return at::detail::empty_strided_generic(
      size, stride, c10::GetAllocator(c10::kPrivateUse1),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), dtype.value_or(at::kFloat));
}


}  // namespace gaudirt::bridge
