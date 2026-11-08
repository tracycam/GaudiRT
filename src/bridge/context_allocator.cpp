// SPDX-License-Identifier: Apache-2.0
#include "bridge_internal.hpp"

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace gaudirt::generic {
using namespace runtime;

void free_allocation(Context& ctx, uint64_t address, AllocationKind kind) {
  if (kind == AllocationKind::graph) ctx.graph_pool->free(address);
  else if (kind == AllocationKind::pinned) ctx.pinned.free(reinterpret_cast<void*>(address));
  else ctx.allocator.free(address);
}

void release_allocation(void* pointer) {
  std::unique_ptr<AllocationContext> allocation(static_cast<AllocationContext*>(pointer));
  if (auto ctx = allocation->owner.lock()) free_allocation(*ctx, allocation->address, allocation->kind);
}

c10::DataPtr allocation_data_ptr(const std::shared_ptr<Context>& ctx, uint64_t address,
                                 uint64_t bytes, synStreamHandle stream, AllocationKind kind) {
  AllocationContext* allocation = nullptr;
  try {
    allocation = new AllocationContext{ctx, address, bytes, stream, kind};
  } catch (...) {
    free_allocation(*ctx, address, kind);
    throw;
  }
  const auto device = kind == AllocationKind::pinned ? c10::Device(c10::kCPU) :
                                                      c10::Device(c10::kPrivateUse1, 0);
  return {reinterpret_cast<void*>(address), allocation, release_allocation, device};
}

at::Tensor tensor_from_allocation(c10::DataPtr data, uint64_t bytes, at::IntArrayRef shape,
                                  const at::TensorOptions& options) {
  c10::Storage storage(c10::Storage::use_byte_size_t{}, bytes, std::move(data), nullptr, false);
  auto result = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage), c10::DispatchKey::PrivateUse1, options.dtype());
  result.unsafeGetTensorImpl()->set_sizes_contiguous(shape);
  return result;
}

std::mutex context_mutex;
std::shared_ptr<Context> context;

std::shared_ptr<Context> current() {
  std::lock_guard lock(context_mutex);
  TORCH_CHECK(context, "GaudiRT: device is not initialized");
  return context;
}
uint64_t span_bytes(const at::Tensor& tensor) {
  uint64_t extent = 0;
  for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
    const auto stride = tensor.stride(dim);
    TORCH_CHECK(stride >= 0, "GaudiRT: negative strides are unsupported");
    const auto count = static_cast<uint64_t>(tensor.size(dim) - 1);
    TORCH_CHECK(!count || static_cast<uint64_t>(stride) <=
                (std::numeric_limits<uint64_t>::max() - extent) / count,
                "GaudiRT: tensor stride overflow");
    extent += count * static_cast<uint64_t>(stride);
  }
  const auto itemsize = tensor.element_size();
  TORCH_CHECK(extent < std::numeric_limits<uint64_t>::max() / itemsize,
              "GaudiRT: tensor extent overflow");
  return (extent + 1) * itemsize;
}
uint64_t address_of(const at::Tensor& tensor, const Context& ctx) {
  TORCH_CHECK(tensor.device().type() == c10::kPrivateUse1 && tensor.device().index() == 0,
              "GaudiRT: expected gaudirt:0 tensor");
  auto address = reinterpret_cast<uint64_t>(tensor.data_ptr());
  TORCH_CHECK(ctx.allocator.contains_live_range(address, span_bytes(tensor)),
              "GaudiRT: tensor is outside a live allocation");
  return address;
}

const AllocationContext& validated_program_input(const at::Tensor& tensor, const Context& ctx) {
  const auto& storage = tensor.storage();
  const auto* allocation = storage.data_ptr().cast_context<AllocationContext>(release_allocation);
  TORCH_CHECK(allocation && allocation->kind != AllocationKind::pinned &&
              allocation->owner.lock().get() == &ctx,
              "GaudiRT FX: tensor must belong to the active allocator");
  const auto base = reinterpret_cast<uint64_t>(storage.data());
  const auto address = reinterpret_cast<uint64_t>(tensor.data_ptr());
  const auto bytes = storage.nbytes();
  TORCH_CHECK(base == allocation->address && bytes <= allocation->bytes && address >= base &&
              address - base <= bytes && span_bytes(tensor) <= bytes - (address - base),
              "GaudiRT FX: tensor is outside its storage");
  return *allocation;
}

void record_stream(at::Tensor& tensor, c10::Stream stream) {
  auto ctx = current();
  TORCH_CHECK(stream.device_type() == c10::kPrivateUse1 && stream.device_index() == 0,
              "GaudiRT: record_stream requires a gaudirt:0 stream");
  if (tensor.numel()) {
    const auto address = address_of(tensor, *ctx);
    auto handle = ctx->streams.get(stream.id());
    std::lock_guard lock(ctx->cold_mutex);
    if (ctx->graph_pool && ctx->graph_pool->contains_live_range(address, 1))
      ctx->graph_pool->record_stream(address, handle);
    ctx->allocator.record_stream(address, handle);
  }
}

c10::Allocator* pinned_allocator();

class Hooks final : public at::PrivateUse1HooksInterface {
 public:
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return true; }
  at::Device getDeviceFromPtr(void* ptr) const override {
    TORCH_CHECK(current()->allocator.contains_live_range(reinterpret_cast<uint64_t>(ptr), 1),
                "GaudiRT: unknown device pointer");
    return {c10::kPrivateUse1, 0};
  }
  bool hasPrimaryContext(c10::DeviceIndex index) const override {
    std::lock_guard lock(context_mutex);
    return index == 0 && static_cast<bool>(context);
  }
  bool isPinnedPtr(const void* ptr) const override {
    std::lock_guard lock(context_mutex);
    return context && context->pinned.contains_live_range(ptr, 1);
  }
  c10::Allocator* getPinnedMemoryAllocator() const override { return pinned_allocator(); }
};

class Allocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t bytes) override {
    auto ctx = current();
    if (!bytes) return {nullptr, c10::Device(c10::kPrivateUse1, 0)};
    const auto stream = ctx->stream();
    const uint64_t address = ctx->allocator.allocate(bytes, stream);
    return allocation_data_ptr(ctx, address, bytes, stream, AllocationKind::device);
  }
  void copy_data(void* dst, const void* src, size_t bytes) const override {
    if (!bytes) return;
    auto ctx = current();
    ctx->allocator.record_stream(reinterpret_cast<uint64_t>(src), ctx->stream());
    ctx->allocator.record_stream(reinterpret_cast<uint64_t>(dst), ctx->stream());
    PinnedHostPool::copy_d2d(reinterpret_cast<uint64_t>(dst), reinterpret_cast<uint64_t>(src),
                             bytes, ctx->stream(), false);
  }
};

class PinnedAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t bytes) override {
    auto ctx = current();
    if (!bytes) return {nullptr, c10::Device(c10::kCPU)};
    void* address = ctx->pinned.allocate(bytes);
    return allocation_data_ptr(ctx, reinterpret_cast<uint64_t>(address), bytes, nullptr, AllocationKind::pinned);
  }
  void copy_data(void* dst, const void* src, size_t bytes) const override {
    std::memcpy(dst, src, bytes);
  }
};
c10::Allocator* pinned_allocator() {
  static PinnedAllocator allocator;
  return &allocator;
}

class Guard final : public c10::impl::DeviceGuardImplInterface {
 public:
  c10::DeviceType type() const override { return c10::kPrivateUse1; }
  c10::Device exchangeDevice(c10::Device device) const override { checked(device); return {type(), 0}; }
  c10::Device getDevice() const override { current(); return {type(), 0}; }
  void setDevice(c10::Device device) const override { checked(device); }
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device device) const override {
    auto ctx = checked(device);
    ctx->streams.get(active_stream_id);
    return c10::Stream(c10::Stream::UNSAFE, c10::Device(type(), 0), active_stream_id);
  }
  c10::Stream getDefaultStream(c10::Device device) const override {
    checked(device);
    return c10::Stream(c10::Stream::DEFAULT, c10::Device(type(), 0));
  }
  c10::Stream getNewStream(c10::Device device, int priority) const override {
    TORCH_CHECK(priority == 0, "GaudiRT: stream priorities are unsupported");
    auto ctx = checked(device);
    return c10::Stream(c10::Stream::UNSAFE, c10::Device(type(), 0), ctx->streams.create());
  }
  c10::Stream getStreamFromGlobalPool(c10::Device device, bool high_priority) const override {
    TORCH_CHECK(!high_priority, "GaudiRT: stream priorities are unsupported");
    return getNewStream(device, 0);
  }
  c10::Stream exchangeStream(c10::Stream stream) const override {
    auto ctx = checked(stream.device());
    ctx->streams.get(stream.id());
    const auto previous = active_stream_id;
    active_stream_id = stream.id();
    return c10::Stream(c10::Stream::UNSAFE, c10::Device(type(), 0), previous);
  }
  void* getStreamNativeHandle(const c10::Stream stream) const override {
    return checked(stream.device())->streams.get(stream.id());
  }
  void record(void** event, const c10::Stream& stream, c10::DeviceIndex index,
              c10::EventFlag flag) const override {
    TORCH_CHECK(index < 0 || index == 0, "GaudiRT: invalid event device");
    TORCH_CHECK(flag == c10::EventFlag::PYTORCH_DEFAULT || flag == c10::EventFlag::BACKEND_DEFAULT,
                "GaudiRT: unsupported event flags");
    auto ctx = checked(stream.device());
    auto native = ctx->streams.get(stream.id());
    std::lock_guard lock(event_mutex);
    if (!*event) {
      auto state = std::make_unique<EventState>();
      state->owner = ctx.get();
      state->timing = flag == c10::EventFlag::BACKEND_DEFAULT;
      state->handle = nullptr;
      TORCH_CHECK(synEventCreate(&state->handle, ctx->device.device_id(),
                                 state->timing ? EVENT_COLLECT_TIME : 0) == synSuccess,
                  "GaudiRT: event creation failed");
      try { ctx->events.insert(state.get()); }
      catch (...) { synEventDestroy(state->handle); throw; }
      *event = state.release();
    }
    auto* state = static_cast<EventState*>(*event);
    TORCH_CHECK(state->owner == ctx.get() && state->handle,
                "GaudiRT: event belongs to a different or shut down runtime");
    TORCH_CHECK(synEventRecord(state->handle, native) == synSuccess, "GaudiRT: event record failed");
  }
  void destroyEvent(void* event, c10::DeviceIndex) const noexcept override {
    if (!event) return;
    std::lock_guard lock(event_mutex);
    auto* state = static_cast<EventState*>(event);
    if (state->owner) {
      state->owner->events.erase(state);
      if (state->handle) synEventDestroy(state->handle);
    }
    delete state;
  }
  void block(void* event, const c10::Stream& stream) const override {
    if (!event) return;
    auto ctx = checked(stream.device());
    auto native = ctx->streams.get(stream.id());
    std::lock_guard lock(event_mutex);
    auto* state = static_cast<EventState*>(event);
    TORCH_CHECK(state->owner == ctx.get() && state->handle,
                "GaudiRT: event belongs to a different or shut down runtime");
    TORCH_CHECK(synStreamWaitEvent(native, state->handle, 0) == synSuccess,
                "GaudiRT: stream event wait failed");
  }
  bool queryEvent(void* event) const override {
    if (!event) return true;
    std::lock_guard lock(event_mutex);
    auto* state = static_cast<EventState*>(event);
    TORCH_CHECK(state->owner && state->handle, "GaudiRT: event belongs to a shut down runtime");
    const auto status = synEventQuery(state->handle);
    TORCH_CHECK(status == synSuccess || status == synBusy, "GaudiRT: event query failed");
    return status == synSuccess;
  }
  void synchronizeEvent(void* event) const override {
    if (!event) return;
    std::lock_guard lock(event_mutex);
    auto* state = static_cast<EventState*>(event);
    TORCH_CHECK(state->owner && state->handle, "GaudiRT: event belongs to a shut down runtime");
    TORCH_CHECK(synEventSynchronize(state->handle) == synSuccess, "GaudiRT: event synchronization failed");
  }
  double elapsedTime(void* first, void* second, c10::DeviceIndex index) const override {
    TORCH_CHECK((index < 0 || index == 0) && first && second, "GaudiRT: invalid timing events");
    std::lock_guard lock(event_mutex);
    auto* start = static_cast<EventState*>(first);
    auto* end = static_cast<EventState*>(second);
    TORCH_CHECK(start->owner && start->owner == end->owner && start->handle && end->handle &&
                start->timing && end->timing, "GaudiRT: events do not support timing");
    uint64_t ns = 0;
    TORCH_CHECK(synEventElapsedTime(&ns, start->handle, end->handle) == synSuccess,
                "GaudiRT: event elapsed time failed");
    return static_cast<double>(ns) * 1e-6;
  }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
  bool queryStream(const c10::Stream& stream) const override {
    auto ctx = checked(stream.device());
    auto status = synStreamQuery(ctx->streams.get(stream.id()));
    TORCH_CHECK(status == synSuccess || status == synBusy, "GaudiRT: stream query failed");
    return status == synSuccess;
  }
  void synchronizeStream(const c10::Stream& stream) const override {
    TORCH_CHECK(synStreamSynchronize(checked(stream.device())->streams.get(stream.id())) == synSuccess,
                "GaudiRT: stream synchronization failed");
  }
  void synchronizeDevice(c10::DeviceIndex index) const override {
    checked({type(), index})->synchronize_all();
  }
  void recordDataPtrOnStream(const c10::DataPtr& data, const c10::Stream& stream) const override {
    auto ctx = checked(stream.device());
    if (data.get()) ctx->allocator.record_stream(reinterpret_cast<uint64_t>(data.get()), ctx->streams.get(stream.id()));
  }
 private:
  static std::shared_ptr<Context> checked(c10::Device device) {
    TORCH_CHECK(device.type() == c10::kPrivateUse1 && (device.index() < 0 || device.index() == 0),
                "GaudiRT: invalid device");
    return current();
  }
};

void register_backend() {
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

void initialize(uint32_t module, uint64_t hbm_bytes, uint64_t pinned_bytes) {
  TORCH_CHECK(hbm_bytes && pinned_bytes >= 256, "GaudiRT: invalid allocation budget");
  std::lock_guard lock(context_mutex);
  TORCH_CHECK(!context, "GaudiRT: device already initialized");
  context = std::make_shared<Context>(module, hbm_bytes, pinned_bytes);
  active_stream_id = 0;
}
bool is_initialized() {
  std::lock_guard lock(context_mutex);
  return static_cast<bool>(context);
}
std::tuple<uint64_t, uint64_t> memory_info() {
  auto ctx = current();
  const auto capacity = ctx->allocator.capacity_bytes();
  return {capacity - ctx->allocator.memory_allocated(), capacity};
}
std::tuple<uint64_t, uint64_t> physical_memory_info() {
  auto ctx = current();
  uint64_t free = 0, total = 0;
  TORCH_CHECK(synDeviceGetMemoryInfo(ctx->device.device_id(), &free, &total) == synSuccess,
              "GaudiRT: physical device memory query failed");
  return {free, total};
}
void shutdown() {
  std::shared_ptr<Context> old;
  {
    std::lock_guard lock(context_mutex);
    TORCH_CHECK(context, "GaudiRT: device is not initialized");
    const uint64_t graph_reservation = context->graph_pool ?
        context->graph_pool->capacity_bytes() : 0;
    TORCH_CHECK(!context->graph_pool || context->graph_pool->memory_allocated() == 0,
                "GaudiRT: graph tensors must be destroyed before shutdown");
    const uint64_t row_counts = context->row_count_table ? Context::kRowCountTableBytes : 0;
    uint64_t attention_zero_bytes = 0;
    for (const auto& [key, allocation] : context->attention_zero_states) {
      (void)key;
      attention_zero_bytes += allocation.second;
    }
    TORCH_CHECK(context->allocator.memory_allocated() == graph_reservation + row_counts + attention_zero_bytes,
                "GaudiRT: device tensors must be destroyed before shutdown");
    TORCH_CHECK(context->pinned.memory_allocated() == 0,
                "GaudiRT: pinned tensors must be destroyed before shutdown");
    old = std::move(context);
  }
  old->synchronize_all();
  old.reset();
}
void shutdown_at_exit() {
  std::shared_ptr<Context> old;
  {
    std::lock_guard lock(context_mutex);
    old = std::move(context);
  }
  if (!old) return;
  old->synchronize_all();
  old.reset();
}
uint64_t memory_allocated() { return current()->allocator.memory_allocated(); }
uint64_t max_memory_allocated() { return current()->allocator.max_memory_allocated(); }
std::vector<uint64_t> recipe_cache_stats() {
  auto ctx = current();
  std::lock_guard lock(ctx->cold_mutex);
  const auto [loaded, saved] = ctx->cache.stats();
  return {loaded, saved, ctx->cold_recipes.size()};
}
std::vector<std::string> recipe_cache_keys() {
  auto ctx = current();
  std::lock_guard lock(ctx->cold_mutex);
  std::vector<std::string> keys;
  keys.reserve(ctx->cold_recipes.size());
  for (const auto& [key, recipe] : ctx->cold_recipes) {
    (void)recipe;
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}
void empty_cache() {
  auto ctx = current();
  std::lock_guard lock(ctx->cold_mutex);
  if (ctx->graph_pool && !ctx->graph_pool->memory_allocated()) ctx->graph_pool.reset();
  ctx->allocator.empty_cache();
}
void synchronize() { current()->synchronize_all(); }

}  // namespace gaudirt::generic
