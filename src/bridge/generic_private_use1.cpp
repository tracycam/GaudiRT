// SPDX-License-Identifier: Apache-2.0
#include "gaudirt/runtime/block_caching_allocator.hpp"
#include "gaudirt/runtime/device_runtime.hpp"
#include "gaudirt/runtime/pinned_host_pool.hpp"
#include "gaudirt/runtime/recipe_registry.hpp"
#include "gaudirt/common/sha256.hpp"

#include <ATen/EmptyTensor.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <perf_lib_layer_params.h>
#include <c10/core/Allocator.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <unordered_map>

namespace gaudirt::generic {
using namespace runtime;

struct AllocationLease;

struct ColdRecipe {
  explicit ColdRecipe(size_t count = 2) : names(count), tensors(count) {}
  synRecipeHandle recipe = nullptr;
  uint64_t workspace = 0;
  std::vector<std::string> names;
  std::vector<synLaunchTensorInfo> tensors;
  ~ColdRecipe() { if (recipe) synRecipeDestroy(recipe); }
};

struct Context {
  Context(uint32_t module, uint64_t hbm_bytes, uint64_t pinned_bytes)
      : device(config(module)), allocator(device.device_id(), hbm_bytes),
        pinned(device.device_id(), pinned_bytes), cache(cache_path()) {}
  ~Context() {
    synStreamSynchronize(device.compute_stream());
    synStreamSynchronize(device.h2d_stream());
    synStreamSynchronize(device.d2h_stream());
  }
  static std::filesystem::path cache_path() {
    const char* path = std::getenv("GAUDIRT_RECIPE_CACHE");
    return path ? std::filesystem::path(path) : std::filesystem::path{};
  }
  static DeviceRuntime::Config config(uint32_t module) {
    DeviceRuntime::Config cfg{};
    cfg.module_id = module;
    cfg.uclamp_min = 0;
    return cfg;
  }
  DeviceRuntime device;
  BlockCachingAllocator allocator;
  std::unique_ptr<GraphPool> graph_pool;
  PinnedHostPool pinned;
  RecipeCache cache;
  std::mutex cold_mutex;
  std::unordered_map<std::string, std::unique_ptr<ColdRecipe>> cold_recipes;
  std::mutex leases_mutex;
  std::unordered_map<uint64_t, std::shared_ptr<AllocationLease>> leases;
  uint64_t next_lease = 1;
  bool closing = false;
};

struct AllocationLease {
  AllocationLease(std::shared_ptr<Context> owner, uint64_t address, bool pinned)
      : owner(std::move(owner)), address(address), pinned(pinned) {}
  ~AllocationLease() { release(); }
  void release() {
    std::lock_guard lock(mutex);
    auto ctx = owner;
    if (!ctx) return;
    if (pinned) ctx->pinned.free(reinterpret_cast<void*>(address));
    else ctx->allocator.free(address);
    {
      std::lock_guard leases_lock(ctx->leases_mutex);
      ctx->leases.erase(id);
    }
    owner.reset();
  }
  std::mutex mutex;
  std::shared_ptr<Context> owner;
  uint64_t address;
  uint64_t id = 0;
  bool pinned;
};

std::shared_ptr<AllocationLease> make_lease(const std::shared_ptr<Context>& ctx,
                                             uint64_t address, bool pinned) {
  std::shared_ptr<AllocationLease> lease;
  try {
    lease = std::make_shared<AllocationLease>(ctx, address, pinned);
  } catch (...) {
    if (pinned) ctx->pinned.free(reinterpret_cast<void*>(address));
    else ctx->allocator.free(address);
    throw;
  }
  std::lock_guard lock(ctx->leases_mutex);
  TORCH_CHECK(!ctx->closing, "GaudiRT: device is shutting down");
  lease->id = ctx->next_lease++;
  ctx->leases.emplace(lease->id, lease);
  return lease;
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
    const uint64_t address = ctx->allocator.allocate(bytes, ctx->device.compute_stream());
    auto lease = make_lease(ctx, address, false);
    try {
      return c10::InefficientStdFunctionContext::makeDataPtr(
          reinterpret_cast<void*>(address), [lease](void*) { lease->release(); },
          c10::Device(c10::kPrivateUse1, 0));
    } catch (...) {
      lease->release();
      throw;
    }
  }
  void copy_data(void* dst, const void* src, size_t bytes) const override {
    if (!bytes) return;
    auto ctx = current();
    ctx->allocator.record_stream(reinterpret_cast<uint64_t>(src), ctx->device.compute_stream());
    ctx->allocator.record_stream(reinterpret_cast<uint64_t>(dst), ctx->device.compute_stream());
    PinnedHostPool::copy_d2d(reinterpret_cast<uint64_t>(dst), reinterpret_cast<uint64_t>(src),
                             bytes, ctx->device.compute_stream(), false);
  }
};

class PinnedAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t bytes) override {
    auto ctx = current();
    if (!bytes) return {nullptr, c10::Device(c10::kCPU)};
    void* address = ctx->pinned.allocate(bytes);
    auto lease = make_lease(ctx, reinterpret_cast<uint64_t>(address), true);
    try {
      return c10::InefficientStdFunctionContext::makeDataPtr(
          address, [lease](void*) { lease->release(); }, c10::Device(c10::kCPU));
    } catch (...) {
      lease->release();
      throw;
    }
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
    checked(device); return c10::Stream(c10::Stream::DEFAULT, c10::Device(type(), 0));
  }
  c10::Stream exchangeStream(c10::Stream stream) const override {
    checked(stream.device());
    TORCH_CHECK(stream.id() == 0, "GaudiRT: only the default stream is supported");
    return c10::Stream(c10::Stream::DEFAULT, c10::Device(type(), 0));
  }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
  bool queryStream(const c10::Stream& stream) const override {
    auto ctx = checked(stream.device());
    auto status = synStreamQuery(ctx->device.compute_stream());
    TORCH_CHECK(status == synSuccess || status == synBusy, "GaudiRT: stream query failed");
    return status == synSuccess;
  }
  void synchronizeStream(const c10::Stream& stream) const override {
    checked(stream.device())->device.synchronize_compute();
  }
  void synchronizeDevice(c10::DeviceIndex index) const override {
    checked({type(), index})->device.synchronize_all();
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
}
void shutdown() {
  std::shared_ptr<Context> old;
  {
    std::lock_guard lock(context_mutex);
    TORCH_CHECK(context, "GaudiRT: device is not initialized");
    const uint64_t graph_reservation = context->graph_pool ?
        context->graph_pool->allocator().capacity_bytes() : 0;
    TORCH_CHECK(context->allocator.memory_allocated() == graph_reservation,
                "GaudiRT: device tensors must be destroyed before shutdown");
    TORCH_CHECK(context->pinned.memory_allocated() == 0,
                "GaudiRT: pinned tensors must be destroyed before shutdown");
    old = std::move(context);
  }
  old->device.synchronize_all();
  old.reset();
}
void shutdown_at_exit() {
  std::shared_ptr<Context> old;
  {
    std::lock_guard lock(context_mutex);
    old = std::move(context);
  }
  if (!old) return;
  std::vector<std::shared_ptr<AllocationLease>> live;
  {
    std::lock_guard lock(old->leases_mutex);
    old->closing = true;
    for (const auto& [id, lease] : old->leases) {
      (void)id;
      live.push_back(lease);
    }
  }
  old->device.synchronize_all();
  for (const auto& lease : live) lease->release();
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
void empty_cache() {
  auto ctx = current();
  std::lock_guard lock(ctx->cold_mutex);
  if (ctx->graph_pool && !ctx->graph_pool->allocator().memory_allocated()) ctx->graph_pool.reset();
  ctx->allocator.empty_cache();
}
void synchronize() { current()->device.synchronize_all(); }

// The DMA engine copies contiguous runs. Addresses are derived from tensor
// metadata on the host; device data never leaves HBM for device-to-device copies.
struct Layout {
  explicit Layout(const at::Tensor& tensor) : tensor(tensor), run(1) {
    for (int64_t i = tensor.dim() - 1; i >= 0; --i) {
      const auto size = tensor.size(i);
      if (size != 1 && tensor.stride(i) != static_cast<int64_t>(run)) break;
      run *= static_cast<uint64_t>(size);
    }
  }
  uint64_t offset(uint64_t index) const {
    uint64_t result = 0;
    for (int64_t i = tensor.dim() - 1; i >= 0; --i) {
      const auto size = static_cast<uint64_t>(tensor.size(i));
      result += (index % size) * static_cast<uint64_t>(tensor.stride(i));
      index /= size;
    }
    return result;
  }
  const at::Tensor& tensor;
  uint64_t run;
};

void host_copy(uint8_t* packed, const at::Tensor& tensor, uint64_t start,
               uint64_t elements, bool pack) {
  const Layout layout(tensor);
  const uint64_t item = tensor.element_size();
  auto* base = static_cast<uint8_t*>(tensor.data_ptr());
  for (uint64_t done = 0; done < elements;) {
    const uint64_t index = start + done;
    const uint64_t count = std::min(elements - done, layout.run - index % layout.run);
    auto* slot = base + layout.offset(index) * item;
    if (pack) std::memcpy(packed + done * item, slot, count * item);
    else std::memcpy(slot, packed + done * item, count * item);
    done += count;
  }
}

void dma_copy(uint64_t dst, const Layout* dst_layout, uint64_t src,
                 const Layout* src_layout, uint64_t start, uint64_t elements,
                 uint64_t item, synStreamHandle stream, synDmaDir direction) {
  std::array<uint64_t, 128> sources{}, sizes{}, destinations{};
  size_t used = 0;
  auto flush = [&] {
    if (used && synMemCopyAsyncMultiple(stream, sources.data(), sizes.data(), destinations.data(),
                                        direction, used) != synSuccess)
      throw std::runtime_error("GaudiRT: strided DMA failed");
    used = 0;
  };
  for (uint64_t done = 0; done < elements;) {
    const uint64_t index = start + done;
    uint64_t count = elements - done;
    if (src_layout) count = std::min(count, src_layout->run - index % src_layout->run);
    if (dst_layout) count = std::min(count, dst_layout->run - index % dst_layout->run);
    sources[used] = src + (src_layout ? src_layout->offset(index) : done) * item;
    destinations[used] = dst + (dst_layout ? dst_layout->offset(index) : done) * item;
    sizes[used] = count * item;
    ++used;
    if (used == sources.size()) flush();
    done += count;
  }
  flush();
}

struct TemporaryDevice {
  TemporaryDevice(Context& ctx, uint64_t bytes)
      : ctx(ctx), address(ctx.allocator.allocate(bytes, ctx.device.compute_stream())) {}
  ~TemporaryDevice() { ctx.allocator.free(address); }
  Context& ctx;
  uint64_t address;
};
struct TemporaryHost {
  TemporaryHost(Context& ctx, uint64_t bytes) : ctx(ctx), address(ctx.pinned.allocate(bytes)) {}
  ~TemporaryHost() { ctx.pinned.free(address); }
  Context& ctx;
  void* address;
};

ColdRecipe& broadcast_recipe(Context& ctx, uint64_t count, uint64_t width) {
  const std::string key = "cold_broadcast_" + std::to_string(width) + "_" + std::to_string(count);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>();
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 2> sections{};
  std::array<synTensor, 2> tensors{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "graph creation failed");
    TORCH_CHECK(width == 1 || width == 2 || width == 4 || width == 8,
                "GaudiRT: unsupported fill width");
    const uint64_t lanes = width == 8 ? 2 : 1;
    const uint64_t output_columns = width == 8 ? count : (count * width + 3) / 4;
    for (size_t i = 0; i < tensors.size(); ++i) {
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "section creation failed");
      check(synSectionSetPersistent(sections[i], true), "section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = syn_type_uint32;
      desc.m_dims = 2;
      desc.m_sizes[0] = desc.m_minSizes[0] = lanes;
      desc.m_sizes[1] = desc.m_minSizes[1] = i == 0 ? 1 : output_columns;
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
      recipe->tensors[i].tensorSize[0] = desc.m_sizes[0];
      recipe->tensors[i].tensorSize[1] = desc.m_sizes[1];
    }
    const std::string guid = "broadcast_non_fcd_u32";
    const auto node = key + "_n0";
    ns_BroadcastNonFcd::Params params{1};
    check(synNodeCreate(graph, &tensors[0], &tensors[1], 1, 1, &params, sizeof(params),
                        guid.c_str(), node.c_str(), nullptr, nullptr), "broadcast node creation failed");
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate cold recipe");
  return *it->second;
}

struct CastType { synDataType syn; const char* name; };
CastType cast_type(at::ScalarType type) {
  switch (type) {
    case at::kBFloat16: return {syn_type_bf16, "bf16"};
    case at::kFloat: return {syn_type_float, "f32"};
    case at::kInt: return {syn_type_int32, "i32"};
    case at::kLong: return {syn_type_int64, "i64"};
    case at::kBool: return {syn_type_uint8, "u8"};
    default: TORCH_CHECK(false, "GaudiRT: unsupported cast dtype");
  }
}

bool supported_cast(at::ScalarType from, at::ScalarType to) {
  return
      (from == at::kBFloat16 && (to == at::kFloat || to == at::kInt || to == at::kLong)) ||
      (from == at::kFloat && (to == at::kBFloat16 || to == at::kInt || to == at::kLong)) ||
      (from == at::kInt && (to == at::kBFloat16 || to == at::kFloat || to == at::kLong)) ||
      (from == at::kLong && (to == at::kInt || to == at::kFloat || to == at::kBFloat16)) ||
      (from == at::kBool && (to == at::kBFloat16 || to == at::kFloat ||
                             to == at::kInt || to == at::kLong)) ||
      (to == at::kBool && (from == at::kBFloat16 || from == at::kFloat ||
                           from == at::kInt || from == at::kLong));
}
ColdRecipe& cast_recipe(Context& ctx, uint64_t count, at::ScalarType from, at::ScalarType to) {
  TORCH_CHECK(supported_cast(from, to), "GaudiRT: requested dtype conversion is not implemented");
  const auto input_type = cast_type(from);
  const auto output_type = cast_type(to);
  const std::string key = "cold_cast_" + std::string(input_type.name) + "_" +
                          output_type.name + "_" + std::to_string(count);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>();
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 2> sections{};
  std::array<synTensor, 9> tensors{};
  std::array<std::string, 7> internal_names{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "cast graph creation failed");
    for (size_t i = 0; i < recipe->tensors.size(); ++i) {
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "cast section creation failed");
      check(synSectionSetPersistent(sections[i], true), "cast section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = i == 0 ? input_type.syn : output_type.syn;
      desc.m_dims = 1;
      desc.m_sizes[0] = desc.m_minSizes[0] = count;
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "cast tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
      recipe->tensors[i].tensorSize[0] = count;
    }
    if (to == at::kBool) {
      for (size_t i = 2; i < 4; ++i) {
        internal_names[i - 2] = key + "_t" + std::to_string(i);
        synTensorDescriptor desc{};
        desc.m_name = internal_names[i - 2].c_str();
        desc.m_dataType = i == 2 ? input_type.syn : syn_type_uint8;
        desc.m_dims = 1;
        desc.m_sizes[0] = desc.m_minSizes[0] = count;
        check(synTensorCreate(&tensors[i], &desc, nullptr, 0), "comparison tensor creation failed");
      }
      ns_ConstantKernel::ParamsV3 zero{};
      const auto zero_guid = "constant_" + std::string(input_type.name);
      const auto zero_node = key + "_n0";
      check(synNodeCreate(graph, nullptr, &tensors[2], 0, 1, &zero, sizeof(zero),
                          zero_guid.c_str(), zero_node.c_str(), nullptr, nullptr), "zero node creation failed");
      const auto equal_guid = "equal_fwd_" + std::string(input_type.name);
      const auto equal_node = key + "_n1";
      synTensor inputs[] = {tensors[0], tensors[2]};
      check(synNodeCreate(graph, inputs, &tensors[3], 2, 1, nullptr, 0,
                          equal_guid.c_str(), equal_node.c_str(), nullptr, nullptr), "equal node creation failed");
      const auto not_node = key + "_n2";
      check(synNodeCreate(graph, &tensors[3], &tensors[1], 1, 1, nullptr, 0,
                          "not_fwd_u8", not_node.c_str(), nullptr, nullptr), "not node creation failed");
    } else if (to == at::kInt && (from == at::kFloat || from == at::kBFloat16)) {
      const synDataType types[] = {syn_type_int32, input_type.syn, syn_type_uint8,
                                   syn_type_uint8, syn_type_uint8, syn_type_float, syn_type_int32};
      for (size_t i = 2; i < tensors.size(); ++i) {
        internal_names[i - 2] = key + "_t" + std::to_string(i);
        synTensorDescriptor desc{};
        desc.m_name = internal_names[i - 2].c_str();
        desc.m_dataType = types[i - 2];
        desc.m_dims = 1;
        desc.m_sizes[0] = desc.m_minSizes[0] = count;
        check(synTensorCreate(&tensors[i], &desc, nullptr, 0), "safe cast tensor creation failed");
      }
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_ZERO;
      const auto cast_guid = "cast_" + std::string(input_type.name) + "_to_i32";
      const auto cast_node = key + "_n0";
      check(synNodeCreate(graph, &tensors[0], &tensors[2], 1, 1, &cast, sizeof(cast),
                          cast_guid.c_str(), cast_node.c_str(), nullptr, nullptr), "safe cast node creation failed");
      ns_ConstantKernel::ParamsV3 positive{};
      positive.constant.f = 2147483648.0f;
      const auto constant_guid = "constant_" + std::string(input_type.name);
      const auto positive_node = key + "_n1";
      check(synNodeCreate(graph, nullptr, &tensors[3], 0, 1, &positive, sizeof(positive),
                          constant_guid.c_str(), positive_node.c_str(), nullptr, nullptr), "threshold node creation failed");
      synTensor comparison_inputs[] = {tensors[0], tensors[3]};
      const auto compare_guid = "greater_equal_fwd_" + std::string(input_type.name);
      const auto compare_node = key + "_n2";
      check(synNodeCreate(graph, comparison_inputs, &tensors[4], 2, 1, nullptr, 0,
                          compare_guid.c_str(), compare_node.c_str(), nullptr, nullptr), "overflow node creation failed");
      const auto isnan_guid = "isnan_fwd_" + std::string(input_type.name);
      const auto isnan_node = key + "_n3";
      check(synNodeCreate(graph, &tensors[0], &tensors[5], 1, 1, nullptr, 0,
                          isnan_guid.c_str(), isnan_node.c_str(), nullptr, nullptr), "isnan node creation failed");
      synTensor masks[] = {tensors[4], tensors[5]};
      const auto or_node = key + "_n4";
      check(synNodeCreate(graph, masks, &tensors[6], 2, 1, nullptr, 0,
                          "or_fwd_u8", or_node.c_str(), nullptr, nullptr), "mask node creation failed");
      ns_ConstantKernel::ParamsV3 negative{};
      negative.constant.f = -2147483648.0f;
      const auto negative_node = key + "_n5";
      check(synNodeCreate(graph, nullptr, &tensors[7], 0, 1, &negative, sizeof(negative),
                          "constant_f32", negative_node.c_str(), nullptr, nullptr), "sentinel node creation failed");
      const auto sentinel_node = key + "_n6";
      check(synNodeCreate(graph, &tensors[7], &tensors[8], 1, 1, &cast, sizeof(cast),
                          "cast_f32_to_i32", sentinel_node.c_str(), nullptr, nullptr), "sentinel cast creation failed");
      synTensor choices[] = {tensors[6], tensors[8], tensors[2]};
      const auto where_node = key + "_n7";
      check(synNodeCreate(graph, choices, &tensors[1], 3, 1, nullptr, 0,
                          "where_fwd_i32", where_node.c_str(), nullptr, nullptr), "safe cast selection failed");
    } else {
      const auto guid = "cast_" + std::string(input_type.name) + "_to_" + output_type.name;
      const auto node = key + "_n0";
      ns_CastKernel::ParamsV3 params{};
      params.round_mode = (to == at::kInt || to == at::kLong) ? CAST_ROUND_ZERO : CAST_ROUND_HALF_NE;
      check(synNodeCreate(graph, &tensors[0], &tensors[1], 1, 1, &params, sizeof(params),
                          guid.c_str(), node.c_str(), nullptr, nullptr), "cast node creation failed");
    }
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "cast workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate cold cast recipe");
  return *it->second;
}

void check_factory(const std::optional<at::Layout>& layout,
                   const std::optional<at::Device>& device,
                   const std::optional<bool>& pin_memory) {
  TORCH_CHECK(!layout || *layout == at::kStrided, "GaudiRT: only strided layout is supported");
  TORCH_CHECK(!pin_memory.value_or(false), "GaudiRT: pin_memory applies to CPU allocations");
  TORCH_CHECK(!device || (device->type() == c10::kPrivateUse1 &&
              (device->index() < 0 || device->index() == 0)), "GaudiRT: invalid device");
}
at::Tensor empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                 std::optional<at::Layout> layout, std::optional<at::Device> device,
                 std::optional<bool> pin_memory, std::optional<at::MemoryFormat> format) {
  check_factory(layout, device, pin_memory);
  return at::detail::empty_generic(size, c10::GetAllocator(c10::kPrivateUse1),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), dtype.value_or(at::kFloat), format);
}
at::Tensor empty_strided(at::IntArrayRef size, at::IntArrayRef stride,
                         std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                         std::optional<at::Device> device, std::optional<bool> pin_memory) {
  check_factory(layout, device, pin_memory);
  return at::detail::empty_strided_generic(size, stride, c10::GetAllocator(c10::kPrivateUse1),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), dtype.value_or(at::kFloat));
}
at::Tensor& set_storage(at::Tensor& tensor, at::Storage storage, int64_t storage_offset,
                        at::IntArrayRef size, at::IntArrayRef stride) {
  TORCH_CHECK(storage.device().type() == c10::kPrivateUse1 && storage.device().index() == 0,
              "GaudiRT: invalid storage device");
  TORCH_CHECK(storage_offset >= 0 && (stride.empty() || stride.size() == size.size()),
              "GaudiRT: invalid storage view");
  std::vector<int64_t> contiguous;
  if (stride.empty()) {
    contiguous.resize(size.size());
    int64_t next = 1;
    for (size_t i = size.size(); i-- > 0;) {
      TORCH_CHECK(size[i] >= 0 && (!size[i] || next <= INT64_MAX / size[i]),
                  "GaudiRT: invalid storage size");
      contiguous[i] = next;
      next *= std::max<int64_t>(size[i], 1);
    }
    stride = contiguous;
  }
  uint64_t last = static_cast<uint64_t>(storage_offset);
  const bool empty = std::find(size.begin(), size.end(), 0) != size.end();
  for (size_t i = 0; i < size.size(); ++i) {
    TORCH_CHECK(size[i] >= 0 && stride[i] >= 0, "GaudiRT: invalid storage view");
    if (empty) continue;
    auto count = static_cast<uint64_t>(size[i] - 1);
    TORCH_CHECK(!count || static_cast<uint64_t>(stride[i]) <= (UINT64_MAX - last) / count,
                "GaudiRT: storage view overflow");
    last += count * static_cast<uint64_t>(stride[i]);
  }
  const auto item = tensor.element_size();
  TORCH_CHECK(last <= UINT64_MAX / item &&
              (empty ? last * item <= storage.nbytes() :
                       last < UINT64_MAX / item && (last + 1) * item <= storage.nbytes()),
              "GaudiRT: storage view is out of bounds");
  auto* impl = tensor.unsafeGetTensorImpl();
  impl->set_storage_keep_dtype(std::move(storage));
  impl->set_sizes_and_strides(size, stride, storage_offset);
  return tensor;
}

at::Tensor& fill(at::Tensor& tensor, const at::Scalar& scalar) {
  if (!tensor.numel()) return tensor;
  auto ctx = current();
  const auto address = address_of(tensor, *ctx);
  const uint64_t width = tensor.element_size();
  uint64_t pattern = 0;
  switch (tensor.scalar_type()) {
    case at::kBool: pattern = scalar.to<bool>(); break;
    case at::kBFloat16: pattern = scalar.to<c10::BFloat16>().x; break;
    case at::kFloat: pattern = std::bit_cast<uint32_t>(scalar.to<float>()); break;
    case at::kInt: pattern = std::bit_cast<uint32_t>(scalar.to<int32_t>()); break;
    case at::kLong: pattern = std::bit_cast<uint64_t>(scalar.to<int64_t>()); break;
    default: TORCH_CHECK(false, "GaudiRT: fill dtype is not implemented");
  }
  if (width == 1) pattern = (pattern & 0xff) * uint64_t{0x01010101};
  if (width == 2) pattern = (pattern & 0xffff) | ((pattern & 0xffff) << 16);
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = broadcast_recipe(*ctx, tensor.numel(), width);
  auto compute = ctx->device.compute_stream();
  auto h2d = ctx->device.h2d_stream();
  const uint64_t scalar_bytes = std::max<uint64_t>(width, 4);
  TemporaryDevice value(*ctx, scalar_bytes);
  TemporaryHost host(*ctx, scalar_bytes);
  std::memcpy(host.address, &pattern, scalar_bytes);
  ctx->allocator.record_stream(value.address, h2d);
  StreamEvent reusable(ctx->device.device_id());
  reusable.record(compute);
  reusable.wait(h2d);
  ctx->pinned.copy_h2d(value.address, host.address, scalar_bytes, h2d, true);
  StreamEvent ready(ctx->device.device_id());
  ready.record(h2d);
  ready.wait(compute);
  std::unique_ptr<TemporaryDevice> output;
  const uint64_t graph_bytes = width == 8 ? tensor.nbytes() : ((tensor.nbytes() + 3) & ~uint64_t{3});
  if (!tensor.is_contiguous() || address % BlockCachingAllocator::kAlignment ||
      graph_bytes != tensor.nbytes())
    output = std::make_unique<TemporaryDevice>(*ctx, graph_bytes);
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  auto tensors = recipe.tensors;
  tensors[0].pTensorAddress = value.address;
  tensors[1].pTensorAddress = output ? output->address : address;
  ctx->allocator.record_stream(address, compute);
  TORCH_CHECK(synLaunch(compute, tensors.data(), tensors.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: fill launch failed");
  if (output) {
    const Layout layout(tensor);
    dma_copy(address, &layout, output->address, nullptr, 0, tensor.numel(), width,
             compute, DRAM_TO_DRAM);
  }
  return tensor;
}
at::Tensor& zero(at::Tensor& tensor) { return fill(tensor, at::Scalar(0)); }

void cast_direct(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  auto ctx = current();
  auto source = address_of(src, *ctx);
  auto target = address_of(dst, *ctx);
  at::assert_no_overlap(dst, src);
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = cast_recipe(*ctx, src.numel(), src.scalar_type(), dst.scalar_type());
  auto stream = ctx->device.compute_stream();
  ctx->allocator.record_stream(source, stream);
  ctx->allocator.record_stream(target, stream);
  std::unique_ptr<TemporaryDevice> input, output, workspace;
  if (!src.is_contiguous() || source % BlockCachingAllocator::kAlignment) {
    input = std::make_unique<TemporaryDevice>(*ctx, src.nbytes());
    const Layout layout(src);
    dma_copy(input->address, nullptr, source, &layout, 0, src.numel(), src.element_size(),
             stream, DRAM_TO_DRAM);
  }
  if (!dst.is_contiguous() || target % BlockCachingAllocator::kAlignment)
    output = std::make_unique<TemporaryDevice>(*ctx, dst.nbytes());
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  auto tensors = recipe.tensors;
  tensors[0].pTensorAddress = input ? input->address : source;
  tensors[1].pTensorAddress = output ? output->address : target;
  TORCH_CHECK(synLaunch(stream, tensors.data(), tensors.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: cast launch failed");
  if (output) {
    const Layout layout(dst);
    dma_copy(target, &layout, output->address, nullptr, 0, dst.numel(), dst.element_size(),
             stream, DRAM_TO_DRAM);
  }
  if (!non_blocking) ctx->device.synchronize_compute();
}

at::Tensor compare(const at::Tensor& a, const at::Tensor& b, const char* op);
at::Tensor compare_scalar(const at::Tensor& a, const at::Scalar& scalar, const char* op);
at::Tensor pointwise(std::vector<at::Tensor> inputs, const std::vector<int64_t>& shape,
                     at::ScalarType output_type, const std::string& op,
                     const std::vector<synDataType>& types);
at::Tensor where(const at::Tensor& condition, const at::Tensor& a, const at::Tensor& b);
at::Tensor& copy(at::Tensor& dst, const at::Tensor& src, bool non_blocking);

void cast_device(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  if (src.scalar_type() == at::kLong && dst.scalar_type() == at::kBFloat16) {
    auto intermediate = at::empty(src.sizes(), src.options().dtype(at::kFloat));
    cast_device(intermediate, src, true);
    cast_direct(dst, intermediate, non_blocking);
    return;
  }
  if (src.scalar_type() == at::kInt && dst.scalar_type() == at::kBFloat16) {
    auto intermediate = at::empty(src.sizes(), src.options().dtype(at::kFloat));
    cast_direct(intermediate, src, true);
    cast_direct(dst, intermediate, non_blocking);
    return;
  }
  if (src.scalar_type() == at::kBool && dst.scalar_type() == at::kLong) {
    auto intermediate = at::empty(src.sizes(), src.options().dtype(at::kInt));
    cast_direct(intermediate, src, true);
    cast_direct(dst, intermediate, non_blocking);
    return;
  }
  if ((src.scalar_type() == at::kFloat || src.scalar_type() == at::kBFloat16) &&
      dst.scalar_type() == at::kLong) {
    const std::vector<int64_t> shape(src.sizes().begin(), src.sizes().end());
    auto scalar = [&](at::ScalarType dtype, const at::Scalar& value) {
      auto result = at::empty({}, src.options().dtype(dtype));
      fill(result, value);
      return result;
    };
    auto binary = [&](const at::Tensor& a, const at::Tensor& b,
                      at::ScalarType dtype, const char* guid) {
      const auto type = cast_type(dtype).syn;
      return pointwise({a, b}, shape, dtype, guid, {type, type, type});
    };
    auto convert = [&](const at::Tensor& value, at::ScalarType dtype) {
      auto result = at::empty(shape, src.options().dtype(dtype));
      cast_direct(result, value, true);
      return result;
    };
    auto value = src.scalar_type() == at::kFloat ? src : convert(src, at::kFloat);
    auto inverse_base = scalar(at::kFloat, at::Scalar(1.0 / 4294967296.0));
    auto base = scalar(at::kFloat, at::Scalar(4294967296.0));
    auto half = scalar(at::kFloat, at::Scalar(0.5));
    auto two = scalar(at::kFloat, at::Scalar(2.0));
    auto high = convert(binary(value, inverse_base, at::kFloat, "mult_fwd_f32"), at::kInt);
    auto high_float = convert(high, at::kFloat);
    auto remainder = binary(value, binary(high_float, base, at::kFloat, "mult_fwd_f32"),
                            at::kFloat, "sub_fwd_f32");
    auto half_remainder = convert(binary(remainder, half, at::kFloat, "mult_fwd_f32"), at::kInt);
    auto half_float = convert(half_remainder, at::kFloat);
    auto last = convert(binary(remainder, binary(half_float, two, at::kFloat, "mult_fwd_f32"),
                               at::kFloat, "sub_fwd_f32"), at::kInt);
    auto high64 = convert(high, at::kLong);
    auto half64 = convert(half_remainder, at::kLong);
    auto last64 = convert(last, at::kLong);
    auto base64 = scalar(at::kLong, at::Scalar(int64_t{4294967296}));
    auto two64 = scalar(at::kLong, at::Scalar(int64_t{2}));
    auto upper = binary(high64, base64, at::kLong, "mult_fwd_i64");
    auto lower = binary(half64, two64, at::kLong, "mult_fwd_i64");
    auto raw = binary(binary(upper, lower, at::kLong, "add_fwd_i64"), last64,
                      at::kLong, "add_fwd_i64");
    auto high_overflow = compare_scalar(value, at::Scalar(9223372036854775808.0), "greater_equal");
    auto low_overflow = compare_scalar(value, at::Scalar(-9223372036854775808.0), "less");
    auto nan = compare(value, value, "not_equal");
    auto outside = pointwise({high_overflow, low_overflow}, shape, at::kBool, "or_fwd_u8",
                             {syn_type_uint8, syn_type_uint8, syn_type_uint8});
    auto invalid = pointwise({outside, nan}, shape, at::kBool, "or_fwd_u8",
                             {syn_type_uint8, syn_type_uint8, syn_type_uint8});
    auto sentinel = scalar(at::kLong, at::Scalar(std::numeric_limits<int64_t>::min()));
    auto corrected = gaudirt::generic::where(invalid, sentinel, raw);
    copy(dst, corrected, non_blocking);
    return;
  }
  if (src.scalar_type() == at::kLong && dst.scalar_type() == at::kFloat) {
    const std::vector<int64_t> shape(src.sizes().begin(), src.sizes().end());
    auto scalar = [&](at::ScalarType dtype, const at::Scalar& value) {
      auto result = at::empty({}, src.options().dtype(dtype));
      fill(result, value);
      return result;
    };
    auto binary = [&](const at::Tensor& a, const at::Tensor& b,
                      at::ScalarType dtype, const char* guid) {
      const auto type = cast_type(dtype).syn;
      return pointwise({a, b}, shape, dtype, guid, {type, type, type});
    };
    auto boolean = [&](const at::Tensor& a, const at::Tensor& b, const char* guid) {
      return pointwise({a, b}, shape, at::kBool, guid,
                       {syn_type_uint8, syn_type_uint8, syn_type_uint8});
    };
    auto raw = at::empty(shape, src.options().dtype(at::kFloat));
    cast_direct(raw, src, true);
    auto raw_integer = at::empty(shape, src.options().dtype(at::kLong));
    cast_device(raw_integer, raw, true);
    auto bits = raw.view(at::kInt);
    auto one = scalar(at::kInt, at::Scalar(1));
    auto next_bits = binary(bits, one, at::kInt, "add_fwd_i32");
    auto next = next_bits.view(at::kFloat);
    auto delta = binary(next, raw, at::kFloat, "sub_fwd_f32");
    auto spacing_float = pointwise({delta}, shape, at::kFloat, "abs_fwd_f32",
                                   {syn_type_float, syn_type_float});
    auto spacing = at::empty(shape, src.options().dtype(at::kLong));
    cast_device(spacing, spacing_float, true);
    auto minus_one = scalar(at::kLong, at::Scalar(int64_t{-1}));
    auto negated_raw = binary(raw_integer, minus_one, at::kLong, "mult_fwd_i64");
    auto remainder = binary(src, negated_raw, at::kLong, "add_fwd_i64");
    auto negative = compare_scalar(remainder, at::Scalar(0), "less");
    auto negated = binary(remainder, minus_one, at::kLong, "mult_fwd_i64");
    auto magnitude = gaudirt::generic::where(negative, negated, remainder);
    auto two = scalar(at::kLong, at::Scalar(int64_t{2}));
    auto twice = binary(magnitude, two, at::kLong, "mult_fwd_i64");
    auto greater = compare(twice, spacing, "greater");
    auto tie = compare(twice, spacing, "equal");
    auto parity = binary(bits, one, at::kInt, "bitwise_and_fwd_i32");
    auto odd = compare_scalar(parity, at::Scalar(0), "not_equal");
    auto tie_odd = boolean(tie, odd, "and_fwd_u8");
    auto rounding = boolean(greater, tie_odd, "or_fwd_u8");
    auto nonzero = compare_scalar(remainder, at::Scalar(0), "not_equal");
    auto round_up = boolean(rounding, nonzero, "and_fwd_u8");
    auto rounded = gaudirt::generic::where(round_up, next, raw);
    auto too_high = compare_scalar(raw, at::Scalar(9223372036854775808.0), "greater_equal");
    auto too_low = compare_scalar(raw, at::Scalar(-9223372036854775808.0), "less_equal");
    auto boundary = boolean(too_high, too_low, "or_fwd_u8");
    auto corrected = gaudirt::generic::where(boundary, raw, rounded);
    copy(dst, corrected, non_blocking);
    return;
  }
  cast_direct(dst, src, non_blocking);
}

at::Tensor& copy(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  if (dst.sizes() != src.sizes()) {
    auto expanded = src.expand(dst.sizes());
    return copy(dst, expanded, non_blocking);
  }
  if (!dst.numel()) return dst;
  at::assert_no_internal_overlap(dst);
  auto ctx = current();
  const bool dst_device = dst.device().type() == c10::kPrivateUse1;
  const bool src_device = src.device().type() == c10::kPrivateUse1;
  TORCH_CHECK((dst_device || dst.device().is_cpu()) && (src_device || src.device().is_cpu()) &&
              (dst_device || src_device), "GaudiRT: unsupported copy devices");
  if (dst.scalar_type() != src.scalar_type()) {
    TORCH_CHECK(supported_cast(src.scalar_type(), dst.scalar_type()),
                "GaudiRT: requested dtype conversion is not implemented");
    if (!src_device) {
      auto temporary = at::empty(src.sizes(), dst.options().dtype(src.scalar_type()));
      copy(temporary, src, true);
      cast_device(dst, temporary, non_blocking);
    } else if (!dst_device) {
      auto temporary = at::empty(dst.sizes(), src.options().dtype(dst.scalar_type()));
      cast_device(temporary, src, true);
      copy(dst, temporary, non_blocking);
    } else {
      cast_device(dst, src, non_blocking);
    }
    return dst;
  }
  auto compute = ctx->device.compute_stream();
  const uint64_t bytes = src.nbytes();
  if (dst_device && src_device) {
    auto d = address_of(dst, *ctx), s = address_of(src, *ctx);
    if (d == s && dst.strides() == src.strides()) return dst;
    at::assert_no_overlap(dst, src);
    ctx->allocator.record_stream(d, compute);
    ctx->allocator.record_stream(s, compute);
    const Layout dl(dst), sl(src);
    dma_copy(d, &dl, s, &sl, 0, src.numel(), src.element_size(), compute, DRAM_TO_DRAM);
    if (!non_blocking) ctx->device.synchronize_compute();
    return dst;
  }
  const uint64_t device = address_of(dst_device ? dst : src, *ctx);
  const uint64_t item = src.element_size();
  const uint64_t elements = src.numel();
  auto staging_elements = [&] {
    const auto available = std::min<uint64_t>(4 * 1024 * 1024, ctx->pinned.largest_free_block());
    TORCH_CHECK(available >= item, "GaudiRT: pinned pool has no staging space");
    return available / item;
  };
  if (dst_device) {
    auto stream = ctx->device.h2d_stream();
    const Layout dl(dst);
    if (ctx->pinned.contains_live_range(src.data_ptr(), span_bytes(src))) {
      const Layout sl(src);
      ctx->pinned.record_stream(src.data_ptr(), stream);
      ctx->allocator.record_stream(device, stream);
      StreamEvent ready(ctx->device.device_id()), done(ctx->device.device_id());
      ready.record(compute);
      ready.wait(stream);
      dma_copy(device, &dl, reinterpret_cast<uint64_t>(src.data_ptr()), &sl,
               0, elements, item, stream, HOST_TO_DRAM);
      done.record(stream);
      done.wait(compute);
      if (!non_blocking) done.synchronize();
      return dst;
    }
    const uint64_t chunk_elements = staging_elements();
    const uint64_t chunk_size = chunk_elements * item;
    std::unique_ptr<TemporaryDevice> temporary;
    if (!dst.is_contiguous()) {
      temporary = std::make_unique<TemporaryDevice>(*ctx, std::min(chunk_size, bytes));
      ctx->allocator.record_stream(temporary->address, stream);
    } else {
      ctx->allocator.record_stream(device, stream);
    }
    StreamEvent loaded(ctx->device.device_id()), consumed(ctx->device.device_id());
    consumed.record(compute);
    consumed.wait(stream);
    for (uint64_t start = 0; start < elements; start += chunk_elements) {
      const auto count = std::min(chunk_elements, elements - start);
      TemporaryHost host(*ctx, count * item);
      host_copy(static_cast<uint8_t*>(host.address), src, start, count, true);
      if (temporary && start) consumed.wait(stream);
      const auto target = temporary ? temporary->address : device + start * item;
      ctx->pinned.copy_h2d(target, host.address, count * item, stream, true);
      if (temporary) {
        loaded.record(stream);
        loaded.wait(compute);
        dma_copy(device, &dl, temporary->address, nullptr, start, count, item, compute, DRAM_TO_DRAM);
        consumed.record(compute);
      }
    }
    if (!temporary) {
      loaded.record(stream);
      loaded.wait(compute);
    }
    if (!non_blocking) ctx->device.synchronize_compute();
  } else {
    auto stream = ctx->device.d2h_stream();
    StreamEvent ready(ctx->device.device_id());
    const Layout sl(src);
    if (ctx->pinned.contains_live_range(dst.data_ptr(), span_bytes(dst))) {
      const Layout dl(dst);
      ctx->pinned.record_stream(dst.data_ptr(), stream);
      ctx->allocator.record_stream(device, stream);
      ready.record(compute);
      ready.wait(stream);
      dma_copy(reinterpret_cast<uint64_t>(dst.data_ptr()), &dl, device, &sl,
               0, elements, item, stream, DRAM_TO_HOST);
      StreamEvent done(ctx->device.device_id());
      done.record(stream);
      done.wait(compute);
      if (!non_blocking) done.synchronize();
      return dst;
    }
    TORCH_CHECK(!non_blocking, "GaudiRT: nonblocking D2H requires pinned destination");
    const uint64_t chunk_elements = staging_elements();
    const uint64_t chunk_size = chunk_elements * item;
    std::unique_ptr<TemporaryDevice> temporary;
    if (!src.is_contiguous()) {
      temporary = std::make_unique<TemporaryDevice>(*ctx, std::min(chunk_size, bytes));
      ctx->allocator.record_stream(temporary->address, stream);
    } else {
      ctx->allocator.record_stream(device, stream);
    }
    for (uint64_t start = 0; start < elements; start += chunk_elements) {
      const auto count = std::min(chunk_elements, elements - start);
      TemporaryHost host(*ctx, count * item);
      if (temporary)
        dma_copy(temporary->address, nullptr, device, &sl, start, count, item, compute, DRAM_TO_DRAM);
      ready.record(compute);
      ready.wait(stream);
      const auto source = temporary ? temporary->address : device + start * item;
      ctx->pinned.copy_d2h(host.address, source, count * item, stream, false);
      host_copy(static_cast<uint8_t*>(host.address), dst, start, count, false);
    }
  }
  return dst;
}

ColdRecipe& cat_recipe(Context& ctx, const std::vector<at::Tensor>& inputs,
                       const at::Tensor& output, int64_t axis) {
  std::string signature = std::to_string(static_cast<int>(output.scalar_type())) + ":" +
                          std::to_string(axis);
  for (const auto& tensor : inputs) {
    signature += ";";
    for (auto size : tensor.sizes()) signature += std::to_string(size) + ",";
  }
  const auto digest = common::sha256_bytes(reinterpret_cast<const uint8_t*>(signature.data()),
                                            signature.size());
  const std::string key = "cold_cat_" + digest;
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(inputs.size() + 1);
  synGraphHandle graph = nullptr;
  std::vector<synSectionHandle> sections(recipe->tensors.size(), nullptr);
  std::vector<synTensor> tensors(recipe->tensors.size(), nullptr);
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "cat graph creation failed");
    for (size_t i = 0; i < tensors.size(); ++i) {
      const auto& tensor = i < inputs.size() ? inputs[i] : output;
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "cat section creation failed");
      check(synSectionSetPersistent(sections[i], true), "cat section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = cast_type(tensor.scalar_type()).syn;
      desc.m_dims = tensor.dim();
      for (int64_t dim = 0; dim < tensor.dim(); ++dim) {
        auto size = tensor.size(tensor.dim() - dim - 1);
        desc.m_sizes[dim] = desc.m_minSizes[dim] = size;
        recipe->tensors[i].tensorSize[dim] = size;
      }
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "cat tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
    }
    synConcatenateParams params{static_cast<unsigned>(output.dim() - axis - 1)};
    const auto node = key + "_n0";
    check(synNodeCreate(graph, tensors.data(), &tensors.back(), inputs.size(), 1,
                        &params, sizeof(params), "concat", node.c_str(), nullptr, nullptr),
          "cat node creation failed");
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "cat workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate cold cat recipe");
  return *it->second;
}

at::Tensor cat(const at::ITensorListRef& tensors, int64_t axis) {
  TORCH_CHECK(!tensors.empty(), "GaudiRT: cat requires at least one tensor");
  std::vector<at::Tensor> all;
  for (const auto& tensor : tensors) all.push_back(tensor);
  const at::Tensor* reference = &all.front();
  for (const auto& tensor : all)
    if (!(tensor.dim() == 1 && tensor.size(0) == 0)) { reference = &tensor; break; }
  TORCH_CHECK(reference->dim() > 0 && reference->dim() <= 5,
              "GaudiRT: cat requires tensors of rank 1 to 5");
  axis = at::maybe_wrap_dim(axis, reference->dim());
  auto dtype = reference->scalar_type();
  for (const auto& tensor : all) dtype = c10::promoteTypes(dtype, tensor.scalar_type());
  cast_type(dtype);
  std::vector<int64_t> sizes(reference->sizes().begin(), reference->sizes().end());
  sizes[axis] = 0;
  std::vector<at::Tensor> active;
  for (const auto& tensor : all) {
    TORCH_CHECK(tensor.device() == reference->device() &&
                tensor.device().type() == c10::kPrivateUse1,
                "GaudiRT: cat requires same-device tensors");
    if (tensor.dim() == 1 && tensor.size(0) == 0 && reference->dim() != 1) continue;
    TORCH_CHECK(tensor.dim() == reference->dim(), "GaudiRT: cat rank mismatch");
    for (int64_t dim = 0; dim < tensor.dim(); ++dim)
      TORCH_CHECK(dim == axis || tensor.size(dim) == sizes[dim], "GaudiRT: cat size mismatch");
    TORCH_CHECK(tensor.size(axis) <= std::numeric_limits<int64_t>::max() - sizes[axis],
                "GaudiRT: cat size overflow");
    sizes[axis] += tensor.size(axis);
    if (tensor.numel()) active.push_back(tensor.scalar_type() == dtype ? tensor : tensor.to(dtype));
  }
  auto result = at::empty(sizes, reference->options().dtype(dtype));
  if (!result.numel()) return result;
  TORCH_CHECK(!active.empty(), "GaudiRT: cat has no nonempty inputs");
  if (active.size() == 1) {
    copy(result, active.front(), true);
    return result;
  }
  auto ctx = current();
  std::vector<std::pair<uint64_t, uint64_t>> regions;
  for (auto& tensor : active) {
    auto address = address_of(tensor, *ctx);
    const bool overlaps = std::any_of(regions.begin(), regions.end(), [&](const auto& region) {
      return address < region.first + region.second && region.first < address + tensor.nbytes();
    });
    if (!tensor.is_contiguous() || address % BlockCachingAllocator::kAlignment || overlaps) {
      auto staged = at::empty(tensor.sizes(), tensor.options());
      copy(staged, tensor, true);
      tensor = std::move(staged);
      address = address_of(tensor, *ctx);
    }
    regions.emplace_back(address, tensor.nbytes());
  }
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = cat_recipe(*ctx, active, result, axis);
  auto stream = ctx->device.compute_stream();
  auto bindings = recipe.tensors;
  for (size_t i = 0; i < active.size(); ++i) {
    bindings[i].pTensorAddress = address_of(active[i], *ctx);
    ctx->allocator.record_stream(bindings[i].pTensorAddress, stream);
  }
  bindings.back().pTensorAddress = address_of(result, *ctx);
  ctx->allocator.record_stream(bindings.back().pTensorAddress, stream);
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: cat launch failed");
  return result;
}

at::Tensor& cat_out(const at::ITensorListRef& tensors, int64_t axis, at::Tensor& out) {
  auto value = gaudirt::generic::cat(tensors, axis);
  TORCH_CHECK(out.sizes() == value.sizes() && out.scalar_type() == value.scalar_type(),
              "GaudiRT: cat out must already have the expected shape and dtype");
  return copy(out, value, true);
}

ColdRecipe& range_recipe(Context& ctx, uint32_t count, at::ScalarType dtype, bool wide) {
  const auto type = cast_type(dtype);
  const std::string key = "cold_range_v3_" + std::string(type.name) + "_" + std::to_string(count) +
                          (wide ? "_wide" : "_narrow");
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(3);
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 3> sections{};
  std::array<synTensor, 9> tensors{};
  std::array<std::string, 6> internal_names{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "range graph creation failed");
    for (size_t i = 0; i < 3; ++i) {
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "range section creation failed");
      check(synSectionSetPersistent(sections[i], true), "range section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = i < 2 ? syn_type_int32 : type.syn;
      desc.m_dims = 1;
      desc.m_sizes[0] = desc.m_minSizes[0] = i < 2 ? 1 : count;
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "range tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
      recipe->tensors[i].tensorSize[0] = i < 2 ? 1 : count;
    }
    const size_t internal_count = wide ? (dtype == at::kInt ? 6 : 5) :
                                         (dtype == at::kLong ? 3 : 2);
    for (size_t i = 0; i < internal_count; ++i) {
      internal_names[i] = key + "_t" + std::to_string(i + 3);
      synTensorDescriptor desc{};
      desc.m_name = internal_names[i].c_str();
      desc.m_dataType = wide && i > 0 ? syn_type_int64 : syn_type_int32;
      desc.m_dims = 1;
      desc.m_sizes[0] = desc.m_minSizes[0] = wide && (i == 2 || i == 3) ? 1 : count;
      check(synTensorCreate(&tensors[i + 3], &desc, nullptr, 0), "range intermediate creation failed");
    }
    ns_RangeKernel::Params params{};
    params.start.i = 0;
    params.limit.i = count;
    params.delta.i = 1;
    const auto node = key + "_n0";
    check(synNodeCreate(graph, nullptr, &tensors[3], 0, 1, &params, sizeof(params),
                        "range_i32", node.c_str(), nullptr, nullptr), "range node creation failed");
    if (wide) {
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_ZERO;
      for (size_t i = 0; i < 3; ++i) {
        const size_t input = i == 0 ? 3 : i - 1;
        const size_t output = i + 4;
        const auto cast_node = key + "_n" + std::to_string(i + 1);
        check(synNodeCreate(graph, &tensors[input], &tensors[output], 1, 1, &cast, sizeof(cast),
                            "cast_i32_to_i64", cast_node.c_str(), nullptr, nullptr), "wide range cast failed");
      }
      synTensor multiply[] = {tensors[4], tensors[6]};
      const auto multiply_node = key + "_n4";
      check(synNodeCreate(graph, multiply, &tensors[7], 2, 1, nullptr, 0,
                          "mult_fwd_i64", multiply_node.c_str(), nullptr, nullptr), "wide range multiply failed");
      synTensor add[] = {tensors[7], tensors[5]};
      auto* sum = dtype == at::kInt ? &tensors[8] : &tensors[2];
      const auto add_node = key + "_n5";
      check(synNodeCreate(graph, add, sum, 2, 1, nullptr, 0,
                          "add_fwd_i64", add_node.c_str(), nullptr, nullptr), "wide range add failed");
      if (dtype == at::kInt) {
        const auto cast_node = key + "_n6";
        check(synNodeCreate(graph, &tensors[8], &tensors[2], 1, 1, &cast, sizeof(cast),
                            "cast_i64_to_i32", cast_node.c_str(), nullptr, nullptr), "wide range output cast failed");
      }
    } else {
      synTensor multiply[] = {tensors[3], tensors[1]};
      const auto multiply_node = key + "_n1";
      check(synNodeCreate(graph, multiply, &tensors[4], 2, 1, nullptr, 0,
                          "mult_fwd_i32", multiply_node.c_str(), nullptr, nullptr), "range multiply creation failed");
      synTensor add[] = {tensors[4], tensors[0]};
      auto* sum = dtype == at::kLong ? &tensors[5] : &tensors[2];
      const auto add_node = key + "_n2";
      check(synNodeCreate(graph, add, sum, 2, 1, nullptr, 0,
                          "add_fwd_i32", add_node.c_str(), nullptr, nullptr), "range add creation failed");
      if (dtype == at::kLong) {
        ns_CastKernel::ParamsV3 cast{};
        cast.round_mode = CAST_ROUND_ZERO;
        const auto cast_node = key + "_n3";
        check(synNodeCreate(graph, &tensors[5], &tensors[2], 1, 1, &cast, sizeof(cast),
                            "cast_i32_to_i64", cast_node.c_str(), nullptr, nullptr), "range cast creation failed");
      }
    }
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "range workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate cold range recipe");
  return *it->second;
}

ColdRecipe& range64_recipe(Context& ctx, uint32_t count, at::ScalarType dtype) {
  const auto type = cast_type(dtype);
  const std::string key = "cold_range64_v1_" + std::string(type.name) + "_" + std::to_string(count);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(3);
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 3> sections{};
  std::array<synTensor, 7> tensors{};
  std::array<std::string, 4> internal_names{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "wide-parameter range graph creation failed");
    for (size_t i = 0; i < 3; ++i) {
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "wide-parameter range section creation failed");
      check(synSectionSetPersistent(sections[i], true), "wide-parameter range section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = i < 2 ? syn_type_int64 : type.syn;
      desc.m_dims = 1;
      desc.m_sizes[0] = desc.m_minSizes[0] = i < 2 ? 1 : count;
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "wide-parameter range tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
      recipe->tensors[i].tensorSize[0] = i < 2 ? 1 : count;
    }
    const size_t internal_count = dtype == at::kInt ? 4 : 3;
    for (size_t i = 0; i < internal_count; ++i) {
      internal_names[i] = key + "_t" + std::to_string(i + 3);
      synTensorDescriptor desc{};
      desc.m_name = internal_names[i].c_str();
      desc.m_dataType = i == 0 ? syn_type_int32 : syn_type_int64;
      desc.m_dims = 1;
      desc.m_sizes[0] = desc.m_minSizes[0] = count;
      check(synTensorCreate(&tensors[i + 3], &desc, nullptr, 0), "wide-parameter range intermediate failed");
    }
    ns_RangeKernel::Params params{};
    params.start.i = 0;
    params.limit.i = count;
    params.delta.i = 1;
    const auto range_node = key + "_n0";
    check(synNodeCreate(graph, nullptr, &tensors[3], 0, 1, &params, sizeof(params),
                        "range_i32", range_node.c_str(), nullptr, nullptr), "range node creation failed");
    ns_CastKernel::ParamsV3 cast{};
    cast.round_mode = CAST_ROUND_ZERO;
    const auto cast_node = key + "_n1";
    check(synNodeCreate(graph, &tensors[3], &tensors[4], 1, 1, &cast, sizeof(cast),
                        "cast_i32_to_i64", cast_node.c_str(), nullptr, nullptr), "range index cast failed");
    synTensor multiply[] = {tensors[4], tensors[1]};
    const auto multiply_node = key + "_n2";
    check(synNodeCreate(graph, multiply, &tensors[5], 2, 1, nullptr, 0,
                        "mult_fwd_i64", multiply_node.c_str(), nullptr, nullptr), "range multiply failed");
    synTensor add[] = {tensors[5], tensors[0]};
    auto* sum = dtype == at::kInt ? &tensors[6] : &tensors[2];
    const auto add_node = key + "_n3";
    check(synNodeCreate(graph, add, sum, 2, 1, nullptr, 0,
                        "add_fwd_i64", add_node.c_str(), nullptr, nullptr), "range add failed");
    if (dtype == at::kInt) {
      const auto output_node = key + "_n4";
      check(synNodeCreate(graph, &tensors[6], &tensors[2], 1, 1, &cast, sizeof(cast),
                          "cast_i64_to_i32", output_node.c_str(), nullptr, nullptr), "range output cast failed");
    }
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "range workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate wide-parameter range recipe");
  return *it->second;
}

at::Tensor pointwise(std::vector<at::Tensor> inputs, const std::vector<int64_t>& shape,
                     at::ScalarType output_type, const std::string& op,
                     const std::vector<synDataType>& types);
at::Tensor& arange_out(const at::Scalar& start_value, const at::Scalar& end_value,
                       const at::Scalar& step_value, at::Tensor& out);

at::Tensor& arange_dyadic_out(const at::Scalar& start_value, const at::Scalar& end_value,
                              const at::Scalar& step_value, at::Tensor& out) {
  TORCH_CHECK(out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16,
              "GaudiRT: noninteger arange parameters require a floating output");
  const double start = start_value.toDouble(), end = end_value.toDouble(), step = step_value.toDouble();
  TORCH_CHECK(std::isfinite(start) && std::isfinite(end) && std::isfinite(step) && step != 0 &&
              !(start == 0.0 && std::signbit(start)), "GaudiRT: unsupported floating arange bounds");
  TORCH_CHECK((step > 0 && start <= end) || (step < 0 && start >= end),
              "GaudiRT: arange bounds and step are inconsistent");
  const double count_double = std::ceil((end - start) / step);
  TORCH_CHECK(std::isfinite(count_double) && count_double >= 0 && count_double <= (1 << 24),
              "GaudiRT: arange is too large");
  const int64_t count = static_cast<int64_t>(count_double);
  const std::array<int64_t, 1> shape{count};
  const int64_t exact_limit = out.scalar_type() == at::kBFloat16 ? 1 << 8 : 1 << 24;
  int scale_shift = -1;
  int64_t scaled_start = 0, scaled_step = 0;
  for (int shift = 0; shift <= 16; ++shift) {
    const double a = std::ldexp(start, shift), s = std::ldexp(step, shift);
    if (!std::isfinite(a) || !std::isfinite(s) || std::trunc(a) != a || std::trunc(s) != s ||
        std::abs(a) > exact_limit || std::abs(s) > exact_limit) continue;
    const auto ai = static_cast<int64_t>(a), si = static_cast<int64_t>(s);
    const __int128 offset = static_cast<__int128>(count - 1) * si;
    const __int128 last = static_cast<__int128>(ai) + offset;
    if (offset < -exact_limit || offset > exact_limit || last < -exact_limit || last > exact_limit) continue;
    scale_shift = shift;
    scaled_start = ai;
    scaled_step = si;
    break;
  }
  TORCH_CHECK(!count || scale_shift >= 0, "GaudiRT: floating arange exceeds the verified dyadic range");
  if (out.sizes() != at::IntArrayRef(shape)) {
    TORCH_CHECK(out.numel() == 0, "GaudiRT: nonempty arange out has the wrong shape");
    auto replacement = at::empty(shape, out.options());
    set_storage(out, replacement.storage(), 0, shape, {});
  }
  if (!count) return out;
  auto integral = at::empty(shape, out.options().dtype(at::kInt));
  const int64_t scaled_end = scaled_start + count * scaled_step;
  arange_out(at::Scalar(scaled_start), at::Scalar(scaled_end), at::Scalar(scaled_step), integral);
  auto values = integral.to(at::kFloat);
  auto scale = at::empty({}, out.options().dtype(at::kFloat));
  fill(scale, at::Scalar(std::ldexp(1.0, -scale_shift)));
  const std::vector<int64_t> result_shape{count};
  auto result = pointwise({values, scale}, result_shape, at::kFloat, "mult_fwd_f32",
                          {syn_type_float, syn_type_float, syn_type_float});
  return copy(out, result, true);
}

at::Tensor& arange_out(const at::Scalar& start_value, const at::Scalar& end_value,
                       const at::Scalar& step_value, at::Tensor& out) {
  if (!start_value.isIntegral(false) || !end_value.isIntegral(false) || !step_value.isIntegral(false))
    return arange_dyadic_out(start_value, end_value, step_value, out);
  TORCH_CHECK(out.scalar_type() == at::kInt || out.scalar_type() == at::kLong ||
              out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16,
              "GaudiRT: arange dtype is not implemented");
  const int64_t start = start_value.toLong(), end = end_value.toLong(), step = step_value.toLong();
  TORCH_CHECK(step != 0, "GaudiRT: arange step must be nonzero");
  TORCH_CHECK((step > 0 && start <= end) || (step < 0 && start >= end),
              "GaudiRT: arange bounds and step are inconsistent");
  const __int128 distance = step > 0 ? static_cast<__int128>(end) - start :
                                       static_cast<__int128>(start) - end;
  const __int128 magnitude = step > 0 ? static_cast<__int128>(step) : -static_cast<__int128>(step);
  __int128 count128 = distance / magnitude + (distance % magnitude != 0);
  if (out.scalar_type() == at::kLong) {
    const __int128 difference = static_cast<__int128>(end) - start;
    const __int128 sum = difference + step;
    const __int128 adjusted = sum - (step > 0 ? 1 : -1);
    TORCH_CHECK(difference >= INT64_MIN && difference <= INT64_MAX &&
                sum >= INT64_MIN && sum <= INT64_MAX &&
                adjusted >= INT64_MIN && adjusted <= INT64_MAX,
                "GaudiRT: arange int64 size calculation exceeds the supported range");
  }
  if (out.scalar_type() == at::kInt) {
    // PyTorch computes the INT32 output size from FP64 scalar bounds.
    const double count_double = std::ceil((static_cast<double>(end) - static_cast<double>(start)) /
                                          static_cast<double>(step));
    TORCH_CHECK(count_double >= 0.0 && count_double <= (1 << 24), "GaudiRT: arange is too large");
    count128 = static_cast<int64_t>(count_double);
  }
  TORCH_CHECK(count128 <= (1 << 24), "GaudiRT: arange is too large");
  const int64_t count = static_cast<int64_t>(count128);
  const __int128 last_offset = count ? static_cast<__int128>(count - 1) * step : 0;
  const __int128 last_value = static_cast<__int128>(start) + last_offset;
  const bool wide = last_offset < INT32_MIN || last_offset > INT32_MAX ||
                    last_value < INT32_MIN || last_value > INT32_MAX;
  const bool wide_parameters = start < INT32_MIN || start > INT32_MAX ||
                               step < INT32_MIN || step > INT32_MAX;
  const bool floating = out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16;
  if (floating && count) {
    const int64_t exact_limit = out.scalar_type() == at::kBFloat16 ? 1 << 8 : 1 << 24;
    const auto in_exact_range = [exact_limit](__int128 value) {
      return value >= -exact_limit && value <= exact_limit;
    };
    TORCH_CHECK(in_exact_range(start) && in_exact_range(step) &&
                in_exact_range(last_offset) && in_exact_range(last_value),
                "GaudiRT: floating arange exceeds the verified exact integer range");
  }
  const std::array<int64_t, 1> shape{count};
  if (out.sizes() != at::IntArrayRef(shape)) {
    TORCH_CHECK(out.numel() == 0, "GaudiRT: nonempty arange out has the wrong shape");
    auto replacement = at::empty(shape, out.options());
    set_storage(out, replacement.storage(), 0, shape, {});
  }
  if (!count) return out;
  if (floating) {
    auto integral = at::empty(shape, out.options().dtype(at::kInt));
    arange_out(start_value, end_value, step_value, integral);
    return copy(out, integral, true);
  }
  at::assert_no_internal_overlap(out);
  auto ctx = current();
  const auto target = address_of(out, *ctx);
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = wide_parameters ? range64_recipe(*ctx, count, out.scalar_type()) :
                                   range_recipe(*ctx, count, out.scalar_type(), wide);
  auto stream = ctx->device.compute_stream();
  ctx->allocator.record_stream(target, stream);
  TemporaryDevice scalars(*ctx, 512);
  TemporaryHost host(*ctx, 512);
  if (wide_parameters) {
    std::memcpy(host.address, &start, sizeof(start));
    std::memcpy(static_cast<uint8_t*>(host.address) + 256, &step, sizeof(step));
  } else {
    const int32_t start32 = start, step32 = step;
    std::memcpy(host.address, &start32, sizeof(start32));
    std::memcpy(static_cast<uint8_t*>(host.address) + 256, &step32, sizeof(step32));
  }
  auto h2d = ctx->device.h2d_stream();
  ctx->allocator.record_stream(scalars.address, h2d);
  StreamEvent reusable(ctx->device.device_id());
  reusable.record(stream);
  reusable.wait(h2d);
  ctx->pinned.copy_h2d(scalars.address, host.address, 512, h2d, true);
  StreamEvent ready(ctx->device.device_id());
  ready.record(h2d);
  ready.wait(stream);
  std::unique_ptr<TemporaryDevice> output, workspace;
  if (!out.is_contiguous() || target % BlockCachingAllocator::kAlignment)
    output = std::make_unique<TemporaryDevice>(*ctx, out.nbytes());
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  auto bindings = recipe.tensors;
  bindings[0].pTensorAddress = scalars.address;
  bindings[1].pTensorAddress = scalars.address + 256;
  bindings[2].pTensorAddress = output ? output->address : target;
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: arange launch failed");
  if (output) {
    const Layout layout(out);
    dma_copy(target, &layout, output->address, nullptr, 0, count, out.element_size(),
             stream, DRAM_TO_DRAM);
  }
  return out;
}

at::Tensor& arange_end_out(const at::Scalar& end, at::Tensor& out) {
  return arange_out(at::Scalar(0), end, at::Scalar(1), out);
}

ColdRecipe& pointwise_recipe(Context& ctx, const std::string& op,
                             const std::vector<synDataType>& types, uint32_t count) {
  std::string key = "cold_pointwise_" + op + "_" + std::to_string(count);
  for (auto type : types) key += "_" + std::to_string(static_cast<int>(type));
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(types.size());
  synGraphHandle graph = nullptr;
  const bool negate_equal = op.rfind("not_equal_fwd_", 0) == 0;
  std::vector<synSectionHandle> sections(types.size(), nullptr);
  std::vector<synTensor> tensors(types.size() + (negate_equal ? 1 : 0), nullptr);
  const std::string intermediate_name = key + "_t" + std::to_string(types.size());
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "pointwise graph creation failed");
    for (size_t i = 0; i < types.size(); ++i) {
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "pointwise section creation failed");
      check(synSectionSetPersistent(sections[i], true), "pointwise section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = types[i];
      desc.m_dims = 1;
      desc.m_sizes[0] = desc.m_minSizes[0] = count;
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "pointwise tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
      recipe->tensors[i].tensorSize[0] = count;
    }
    const auto node = key + "_n0";
    if (negate_equal) {
      synTensorDescriptor desc{};
      desc.m_name = intermediate_name.c_str();
      desc.m_dataType = syn_type_uint8;
      desc.m_dims = 1;
      desc.m_sizes[0] = desc.m_minSizes[0] = count;
      check(synTensorCreate(&tensors.back(), &desc, nullptr, 0), "comparison intermediate creation failed");
      const auto equal_op = "equal_fwd_" + op.substr(std::strlen("not_equal_fwd_"));
      check(synNodeCreate(graph, tensors.data(), &tensors.back(), types.size() - 1, 1,
                          nullptr, 0, equal_op.c_str(), node.c_str(), nullptr, nullptr),
            "equality node creation failed");
      const auto not_node = key + "_n1";
      check(synNodeCreate(graph, &tensors.back(), &tensors[types.size() - 1], 1, 1,
                          nullptr, 0, "not_fwd_u8", not_node.c_str(), nullptr, nullptr),
            "inequality node creation failed");
    } else {
      check(synNodeCreate(graph, tensors.data(), &tensors.back(), types.size() - 1, 1,
                          nullptr, 0, op.c_str(), node.c_str(), nullptr, nullptr),
            "pointwise node creation failed");
    }
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "pointwise workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate cold pointwise recipe");
  return *it->second;
}

at::Tensor pointwise(std::vector<at::Tensor> inputs, const std::vector<int64_t>& shape,
                     at::ScalarType output_type, const std::string& op,
                     const std::vector<synDataType>& types) {
  auto result = at::empty(shape, inputs[0].options().dtype(output_type));
  if (!result.numel()) return result;
  TORCH_CHECK(static_cast<uint64_t>(result.numel()) <= UINT32_MAX,
              "GaudiRT: pointwise tensor is too large");
  auto ctx = current();
  std::vector<std::pair<uint64_t, uint64_t>> regions;
  for (auto& input : inputs) {
    auto expanded = input.expand(shape);
    auto address = address_of(expanded, *ctx);
    const bool overlaps = std::any_of(regions.begin(), regions.end(), [&](const auto& region) {
      return address < region.first + region.second && region.first < address + expanded.nbytes();
    });
    if (!expanded.is_contiguous() || address % BlockCachingAllocator::kAlignment || overlaps) {
      auto staged = at::empty(shape, input.options());
      copy(staged, expanded, true);
      input = std::move(staged);
    } else {
      input = std::move(expanded);
    }
    regions.emplace_back(address_of(input, *ctx), input.nbytes());
  }
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = pointwise_recipe(*ctx, op, types, result.numel());
  auto stream = ctx->device.compute_stream();
  auto bindings = recipe.tensors;
  for (size_t i = 0; i < inputs.size(); ++i) {
    bindings[i].pTensorAddress = address_of(inputs[i], *ctx);
    ctx->allocator.record_stream(bindings[i].pTensorAddress, stream);
  }
  bindings.back().pTensorAddress = address_of(result, *ctx);
  ctx->allocator.record_stream(bindings.back().pTensorAddress, stream);
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: pointwise launch failed");
  return result;
}

at::Tensor compare(const at::Tensor& a, const at::Tensor& b, const char* op) {
  TORCH_CHECK(a.device() == b.device() && a.device().type() == c10::kPrivateUse1,
              "GaudiRT: comparison requires same-device tensors");
  const auto dtype = at::result_type(a, b);
  auto left = a.scalar_type() == dtype ? a : a.to(dtype);
  auto right = b.scalar_type() == dtype ? b : b.to(dtype);
  if (dtype == at::kLong && std::strcmp(op, "greater") == 0)
    return compare(right, left, "less");
  if (dtype == at::kLong && std::strcmp(op, "less_equal") == 0)
    return compare(right, left, "greater_equal");
  const auto type = cast_type(dtype);
  const auto inferred = at::infer_size(left.sizes(), right.sizes());
  const std::vector<int64_t> shape(inferred.begin(), inferred.end());
  return pointwise({left, right}, shape, at::kBool, std::string(op) + "_fwd_" + type.name,
                   {type.syn, type.syn, syn_type_uint8});
}

at::Tensor compare_scalar(const at::Tensor& a, const at::Scalar& scalar, const char* op) {
  const auto dtype = at::result_type(a, scalar);
  auto input = a.scalar_type() == dtype ? a : a.to(dtype);
  auto rhs = at::empty({}, input.options());
  if (dtype == at::kInt && scalar.isIntegral(false)) {
    const auto bits = static_cast<uint32_t>(scalar.toLong());
    fill(rhs, at::Scalar(std::bit_cast<int32_t>(bits)));
  } else {
    fill(rhs, scalar);
  }
  return compare(input, rhs, op);
}

#define GAUDIRT_COMPARE(name, guid) \
  at::Tensor name##_tensor(const at::Tensor& a, const at::Tensor& b) { return compare(a, b, guid); } \
  at::Tensor name##_scalar(const at::Tensor& a, const at::Scalar& b) { return compare_scalar(a, b, guid); }
GAUDIRT_COMPARE(eq, "equal")
GAUDIRT_COMPARE(ne, "not_equal")
GAUDIRT_COMPARE(lt, "less")
GAUDIRT_COMPARE(le, "less_equal")
GAUDIRT_COMPARE(gt, "greater")
GAUDIRT_COMPARE(ge, "greater_equal")
#undef GAUDIRT_COMPARE

at::Tensor where(const at::Tensor& condition, const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(condition.scalar_type() == at::kBool && condition.device() == a.device() &&
              a.device() == b.device() && a.device().type() == c10::kPrivateUse1,
              "GaudiRT: where requires a boolean condition and same-device values");
  const auto dtype = at::result_type(a, b);
  if (a.scalar_type() != dtype || b.scalar_type() != dtype) {
    auto left = a.scalar_type() == dtype ? a : a.to(dtype);
    auto right = b.scalar_type() == dtype ? b : b.to(dtype);
    return gaudirt::generic::where(condition, left, right);
  }
  cast_type(a.scalar_type());
  const auto first = at::infer_size(a.sizes(), b.sizes());
  const auto inferred = at::infer_size(first, condition.sizes());
  const std::vector<int64_t> shape(inferred.begin(), inferred.end());
  if (a.scalar_type() == at::kBool)
    return gaudirt::generic::where(condition, a.to(at::kInt), b.to(at::kInt)).to(at::kBool);
  if (a.scalar_type() == at::kLong) {
    auto left = at::empty(shape, a.options());
    auto right = at::empty(shape, b.options());
    if (!left.numel()) return left;
    copy(left, a.expand(shape), true);
    copy(right, b.expand(shape), true);
    std::vector<int64_t> lanes = shape;
    lanes.push_back(2);
    auto left_lanes = left.view({-1}).view(at::kInt).view(lanes);
    auto right_lanes = right.view({-1}).view(at::kInt).view(lanes);
    auto mask_lanes = condition.expand(shape).unsqueeze(-1).expand(lanes);
    auto selected = pointwise({mask_lanes, left_lanes, right_lanes}, lanes, at::kInt,
                              "where_fwd_i32", {syn_type_uint8, syn_type_int32,
                                                syn_type_int32, syn_type_int32});
    return selected.view({-1}).view(at::kLong).view(shape);
  }
  const bool bf16 = a.scalar_type() == at::kBFloat16;
  const auto type = bf16 ? syn_type_bf16 : syn_type_int32;
  return pointwise({condition, a, b}, shape, a.scalar_type(), bf16 ? "where_fwd_bf16" : "where_fwd_i32",
                   {syn_type_uint8, type, type, type});
}

ColdRecipe& invalid_reduce_recipe(Context& ctx, uint32_t count) {
  const std::string key = "cold_index_check_" + std::to_string(count);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>();
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 2> sections{};
  std::array<synTensor, 2> tensors{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "index check graph creation failed");
    for (size_t i = 0; i < 2; ++i) {
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "index check section creation failed");
      check(synSectionSetPersistent(sections[i], true), "index check section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = syn_type_float;
      desc.m_dims = 1;
      desc.m_sizes[0] = desc.m_minSizes[0] = i == 0 ? count : 1;
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "index check tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
      recipe->tensors[i].tensorSize[0] = i == 0 ? count : 1;
    }
    ns_Reduction::Params params{};
    params.reductionDimension = 0;
    const auto node = key + "_n0";
    check(synNodeCreate(graph, &tensors[0], &tensors[1], 1, 1, &params, sizeof(params),
                        "reduce_sum_f32", node.c_str(), nullptr, nullptr), "index check reduction failed");
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "index check workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate index check recipe");
  return *it->second;
}

void validate_indices(const at::Tensor& index, int64_t upper) {
  if (!index.numel()) return;
  TORCH_CHECK(index.numel() <= UINT32_MAX && upper >= 0 && upper <= INT32_MAX,
              "GaudiRT: index dimensions exceed the supported range");
  auto low = compare_scalar(index, at::Scalar(0), "less");
  auto high = compare_scalar(index, at::Scalar(upper), "greater_equal");
  const std::vector<int64_t> shape(index.sizes().begin(), index.sizes().end());
  auto invalid = pointwise({low, high}, shape, at::kBool, "or_fwd_u8",
                           {syn_type_uint8, syn_type_uint8, syn_type_uint8});
  auto values = invalid.to(at::kFloat);
  auto ctx = current();
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = invalid_reduce_recipe(*ctx, index.numel());
  auto stream = ctx->device.compute_stream();
  const auto input = address_of(values, *ctx);
  ctx->allocator.record_stream(input, stream);
  TemporaryDevice result(*ctx, sizeof(float));
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  auto bindings = recipe.tensors;
  bindings[0].pTensorAddress = input;
  bindings[1].pTensorAddress = result.address;
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: index check launch failed");
  auto d2h = ctx->device.d2h_stream();
  StreamEvent ready(ctx->device.device_id());
  ready.record(stream);
  ready.wait(d2h);
  ctx->allocator.record_stream(result.address, d2h);
  TemporaryHost host(*ctx, sizeof(float));
  ctx->pinned.copy_d2h(host.address, result.address, sizeof(float), d2h, false);
  float failures;
  std::memcpy(&failures, host.address, sizeof(failures));
  TORCH_CHECK(std::isfinite(failures) && failures == 0.0f, "GaudiRT: index out of bounds");
}

ColdRecipe& gather_recipe(Context& ctx, const at::Tensor& input, const at::Tensor& output,
                          uint32_t indices, int64_t axis) {
  std::string signature = std::to_string(static_cast<int>(input.scalar_type())) + ":" +
                          std::to_string(axis) + ":" + std::to_string(indices);
  for (auto size : input.sizes()) signature += ":" + std::to_string(size);
  const auto hash = common::sha256_bytes(reinterpret_cast<const uint8_t*>(signature.data()),
                                          signature.size());
  const std::string key = "cold_gather_" + hash;
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>(3);
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 3> sections{};
  std::array<synTensor, 3> tensors{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "gather graph creation failed");
    for (size_t i = 0; i < 3; ++i) {
      const auto& shaped = i == 0 ? input : output;
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "gather section creation failed");
      check(synSectionSetPersistent(sections[i], true), "gather section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = i == 1 ? syn_type_int32 : cast_type(input.scalar_type()).syn;
      desc.m_dims = i == 1 ? 1 : shaped.dim();
      for (unsigned dim = 0; dim < desc.m_dims; ++dim) {
        const auto size = i == 1 ? indices : shaped.size(shaped.dim() - dim - 1);
        TORCH_CHECK(size <= UINT32_MAX, "GaudiRT: gather dimension too large");
        desc.m_sizes[dim] = desc.m_minSizes[dim] = size;
        recipe->tensors[i].tensorSize[dim] = size;
      }
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "gather tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
    }
    ns_GatherKernel::ParamsV2 params{};
    params.axis = input.dim() - axis - 1;
    params.batchDims = 0;
    const auto guid = "gather_fwd_" + std::string(cast_type(input.scalar_type()).name);
    const auto node = key + "_n0";
    check(synNodeCreate(graph, tensors.data(), &tensors[2], 2, 1, &params, sizeof(params),
                        guid.c_str(), node.c_str(), nullptr, nullptr), "gather node creation failed");
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "gather workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate gather recipe");
  return *it->second;
}

at::Tensor index_select(const at::Tensor& input, int64_t axis, const at::Tensor& index) {
  TORCH_CHECK(input.device() == index.device() && input.device().type() == c10::kPrivateUse1 &&
              (index.scalar_type() == at::kInt || index.scalar_type() == at::kLong),
              "GaudiRT: index_select requires device int32 or int64 indices");
  TORCH_CHECK(input.dim() > 0 && input.dim() <= 5 && index.dim() <= 1,
              "GaudiRT: index_select rank is not supported");
  cast_type(input.scalar_type());
  axis = at::maybe_wrap_dim(axis, input.dim());
  const auto count = index.numel();
  validate_indices(index, input.size(axis));
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  shape[axis] = count;
  auto result = at::empty(shape, input.options());
  if (!result.numel()) return result;
  auto ctx = current();
  auto source = input;
  auto source_address = address_of(source, *ctx);
  if (!source.is_contiguous() || source_address % BlockCachingAllocator::kAlignment) {
    source = at::empty(input.sizes(), input.options());
    copy(source, input, true);
  }
  auto ids = index.scalar_type() == at::kInt ? index : index.to(at::kInt);
  source_address = address_of(source, *ctx);
  const auto ids_address = address_of(ids, *ctx);
  const bool overlaps = ids_address < source_address + source.nbytes() &&
                        source_address < ids_address + ids.nbytes();
  if (!ids.is_contiguous() || ids_address % BlockCachingAllocator::kAlignment || overlaps) {
    auto staged = at::empty(ids.sizes(), ids.options());
    copy(staged, ids, true);
    ids = std::move(staged);
  }
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = gather_recipe(*ctx, source, result, count, axis);
  auto stream = ctx->device.compute_stream();
  auto bindings = recipe.tensors;
  bindings[0].pTensorAddress = address_of(source, *ctx);
  bindings[1].pTensorAddress = address_of(ids, *ctx);
  bindings[2].pTensorAddress = address_of(result, *ctx);
  for (const auto& binding : bindings) ctx->allocator.record_stream(binding.pTensorAddress, stream);
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: gather launch failed");
  return result;
}

at::Tensor embedding(const at::Tensor& weight, const at::Tensor& index, int64_t padding_idx,
                     bool scale_grad_by_freq, bool sparse) {
  (void)padding_idx;
  TORCH_CHECK(weight.dim() == 2 && !scale_grad_by_freq && !sparse,
              "GaudiRT: unsupported embedding configuration");
  auto flat = index.reshape({-1});
  auto selected = gaudirt::generic::index_select(weight, 0, flat);
  std::vector<int64_t> shape(index.sizes().begin(), index.sizes().end());
  shape.push_back(weight.size(1));
  return selected.view(shape);
}

ColdRecipe& argmax_recipe(Context& ctx, const at::Tensor& input, const at::Tensor& reduced,
                          int64_t axis, bool values_only = false) {
  std::string signature = std::to_string(static_cast<int>(input.scalar_type())) + ":" +
                          std::to_string(axis);
  for (auto size : input.sizes()) signature += ":" + std::to_string(size);
  const auto hash = common::sha256_bytes(reinterpret_cast<const uint8_t*>(signature.data()),
                                          signature.size());
  const std::string key = (values_only ? "cold_max_i32_" : "cold_argmax_") + hash;
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto recipe = std::make_unique<ColdRecipe>();
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 2> sections{};
  std::array<synTensor, 16> tensors{};
  std::array<std::string, 14> internal_names{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "argmax graph creation failed");
    for (size_t i = 0; i < 2; ++i) {
      const auto& shaped = i == 0 ? input : reduced;
      recipe->names[i] = key + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "argmax section creation failed");
      check(synSectionSetPersistent(sections[i], true), "argmax section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = i == 0 ? cast_type(input.scalar_type()).syn : syn_type_int32;
      desc.m_dims = shaped.dim();
      for (unsigned dim = 0; dim < desc.m_dims; ++dim) {
        const auto size = shaped.size(shaped.dim() - dim - 1);
        TORCH_CHECK(size <= UINT32_MAX, "GaudiRT: argmax dimension too large");
        desc.m_sizes[dim] = desc.m_minSizes[dim] = size;
        recipe->tensors[i].tensorSize[dim] = size;
      }
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "argmax tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
    }
    ns_Reduction::Params params{};
    params.reductionDimension = input.dim() - axis - 1;
    const bool floating = !values_only &&
                          (input.scalar_type() == at::kFloat || input.scalar_type() == at::kBFloat16);
    if (floating) {
      const synDataType dtypes[] = {syn_type_uint8, syn_type_float, syn_type_float,
                                   syn_type_float, syn_type_uint8, syn_type_int32, syn_type_int32,
                                   cast_type(input.scalar_type()).syn, syn_type_uint8, syn_type_float,
                                   syn_type_float, syn_type_uint8, syn_type_int32, syn_type_int32};
      for (size_t i = 0; i < internal_names.size(); ++i) {
        const auto& shaped = (i < 2 || (i >= 7 && i <= 9)) ? input : reduced;
        internal_names[i] = key + "_t" + std::to_string(i + 2);
        synTensorDescriptor desc{};
        desc.m_name = internal_names[i].c_str();
        desc.m_dataType = dtypes[i];
        desc.m_dims = shaped.dim();
        for (unsigned dim = 0; dim < desc.m_dims; ++dim)
          desc.m_sizes[dim] = desc.m_minSizes[dim] = shaped.size(shaped.dim() - dim - 1);
        check(synTensorCreate(&tensors[i + 2], &desc, nullptr, 0), "argmax intermediate creation failed");
      }
    }
    const auto guid = values_only ? std::string("reduce_max_i32") :
                                    "argmax_fwd_" + std::string(cast_type(input.scalar_type()).name);
    const auto node = key + "_n0";
    auto* ordinary = floating ? &tensors[8] : &tensors[1];
    check(synNodeCreate(graph, &tensors[0], ordinary, 1, 1, &params, sizeof(params),
                        guid.c_str(), node.c_str(), nullptr, nullptr), "argmax node creation failed");
    if (floating) {
      const auto isnan_guid = "isnan_fwd_" + std::string(cast_type(input.scalar_type()).name);
      const auto isnan_node = key + "_n1";
      check(synNodeCreate(graph, &tensors[0], &tensors[2], 1, 1, nullptr, 0,
                          isnan_guid.c_str(), isnan_node.c_str(), nullptr, nullptr), "argmax isnan failed");
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_HALF_NE;
      const auto cast_node = key + "_n2";
      check(synNodeCreate(graph, &tensors[2], &tensors[3], 1, 1, &cast, sizeof(cast),
                          "cast_u8_to_f32", cast_node.c_str(), nullptr, nullptr), "argmax mask cast failed");
      const auto nan_argmax_node = key + "_n3";
      check(synNodeCreate(graph, &tensors[3], &tensors[7], 1, 1, &params, sizeof(params),
                          "argmax_fwd_f32", nan_argmax_node.c_str(), nullptr, nullptr), "argmax NaN index failed");
      const auto nan_reduce_node = key + "_n4";
      check(synNodeCreate(graph, &tensors[3], &tensors[4], 1, 1, &params, sizeof(params),
                          "reduce_max_f32", nan_reduce_node.c_str(), nullptr, nullptr), "argmax NaN reduction failed");
      ns_ConstantKernel::ParamsV3 zero{};
      const auto zero_node = key + "_n5";
      check(synNodeCreate(graph, nullptr, &tensors[5], 0, 1, &zero, sizeof(zero),
                          "constant_f32", zero_node.c_str(), nullptr, nullptr), "argmax zero creation failed");
      synTensor compare[] = {tensors[4], tensors[5]};
      const auto compare_node = key + "_n6";
      check(synNodeCreate(graph, compare, &tensors[6], 2, 1, nullptr, 0,
                          "greater_fwd_f32", compare_node.c_str(), nullptr, nullptr), "argmax NaN comparison failed");
      ns_ConstantKernel::ParamsV3 negative_inf{};
      negative_inf.constant.f = -std::numeric_limits<float>::infinity();
      const auto constant_guid = "constant_" + std::string(cast_type(input.scalar_type()).name);
      const auto inf_node = key + "_n7";
      check(synNodeCreate(graph, nullptr, &tensors[9], 0, 1, &negative_inf, sizeof(negative_inf),
                          constant_guid.c_str(), inf_node.c_str(), nullptr, nullptr), "argmax infinity creation failed");
      synTensor equal_inputs[] = {tensors[0], tensors[9]};
      const auto equal_guid = "equal_fwd_" + std::string(cast_type(input.scalar_type()).name);
      const auto equal_node = key + "_n8";
      check(synNodeCreate(graph, equal_inputs, &tensors[10], 2, 1, nullptr, 0,
                          equal_guid.c_str(), equal_node.c_str(), nullptr, nullptr), "argmax infinity check failed");
      const auto inf_cast_node = key + "_n9";
      check(synNodeCreate(graph, &tensors[10], &tensors[11], 1, 1, &cast, sizeof(cast),
                          "cast_u8_to_f32", inf_cast_node.c_str(), nullptr, nullptr), "argmax infinity cast failed");
      const auto inf_reduce_node = key + "_n10";
      check(synNodeCreate(graph, &tensors[11], &tensors[12], 1, 1, &params, sizeof(params),
                          "reduce_min_f32", inf_reduce_node.c_str(), nullptr, nullptr), "argmax infinity reduction failed");
      synTensor all_inf_inputs[] = {tensors[12], tensors[5]};
      const auto all_inf_node = key + "_n11";
      check(synNodeCreate(graph, all_inf_inputs, &tensors[13], 2, 1, nullptr, 0,
                          "greater_fwd_f32", all_inf_node.c_str(), nullptr, nullptr), "argmax infinity comparison failed");
      const auto zero_index_node = key + "_n12";
      check(synNodeCreate(graph, &tensors[5], &tensors[14], 1, 1, &cast, sizeof(cast),
                          "cast_f32_to_i32", zero_index_node.c_str(), nullptr, nullptr), "argmax zero index failed");
      synTensor safe_choices[] = {tensors[13], tensors[14], tensors[8]};
      const auto safe_node = key + "_n13";
      check(synNodeCreate(graph, safe_choices, &tensors[15], 3, 1, nullptr, 0,
                          "where_fwd_i32", safe_node.c_str(), nullptr, nullptr), "argmax infinity selection failed");
      synTensor choices[] = {tensors[6], tensors[7], tensors[15]};
      const auto select_node = key + "_n14";
      check(synNodeCreate(graph, choices, &tensors[1], 3, 1, nullptr, 0,
                          "where_fwd_i32", select_node.c_str(), nullptr, nullptr), "argmax NaN selection failed");
    }
    recipe->recipe = compile_cached(graph, key, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "argmax workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate argmax recipe");
  return *it->second;
}

at::Tensor reduce_max_i32(const at::Tensor& input, int64_t axis) {
  auto ctx = current();
  auto source = input;
  auto address = address_of(source, *ctx);
  if (!source.is_contiguous() || address % BlockCachingAllocator::kAlignment) {
    source = at::empty(input.sizes(), input.options());
    copy(source, input, true);
  }
  std::vector<int64_t> shape(source.sizes().begin(), source.sizes().end());
  shape[axis] = 1;
  auto result = at::empty(shape, source.options());
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = argmax_recipe(*ctx, source, result, axis, true);
  auto stream = ctx->device.compute_stream();
  auto bindings = recipe.tensors;
  bindings[0].pTensorAddress = address_of(source, *ctx);
  bindings[1].pTensorAddress = address_of(result, *ctx);
  for (const auto& binding : bindings) ctx->allocator.record_stream(binding.pTensorAddress, stream);
  std::unique_ptr<TemporaryDevice> workspace;
  if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                        workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
              "GaudiRT: integer maximum launch failed");
  return result;
}

at::Tensor argmax(const at::Tensor& input, std::optional<int64_t> dimension, bool keepdim) {
  TORCH_CHECK(input.device().type() == c10::kPrivateUse1 && input.numel(),
              "GaudiRT: argmax requires a nonempty device tensor");
  TORCH_CHECK(input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kFloat ||
              input.scalar_type() == at::kInt || input.scalar_type() == at::kLong,
              "GaudiRT: argmax dtype is not implemented");
  TORCH_CHECK(input.dim() <= 5, "GaudiRT: argmax rank is not supported");
  std::vector<int64_t> shape(input.sizes().begin(), input.sizes().end());
  auto source = input;
  if (!input.is_contiguous()) {
    source = at::empty(input.sizes(), input.options());
    copy(source, input, true);
  }
  if (source.dim() == 0) source = source.view({1});
  int64_t axis = 0;
  if (!dimension) {
    source = source.view({-1});
    if (keepdim) std::fill(shape.begin(), shape.end(), 1);
    else shape.clear();
  } else {
    axis = at::maybe_wrap_dim(*dimension, source.dim());
    if (input.dim() > 0) {
      if (keepdim) shape[axis] = 1;
      else shape.erase(shape.begin() + axis);
    }
  }
  TORCH_CHECK(source.size(axis) <= INT32_MAX, "GaudiRT: argmax dimension too large");
  if (source.scalar_type() == at::kLong) {
    static_assert(std::endian::native == std::endian::little);
    std::vector<int64_t> source_shape(source.sizes().begin(), source.sizes().end());
    auto lane_shape = source_shape;
    lane_shape.push_back(2);
    auto words = source.view({-1}).view(at::kInt).view(lane_shape);
    auto low = words.select(-1, 0);
    auto high = words.select(-1, 1);
    auto high_max = reduce_max_i32(high, axis);
    auto high_matches = compare(high, high_max, "equal");
    auto sign_bit = at::empty({}, source.options().dtype(at::kInt));
    fill(sign_bit, at::Scalar(INT32_MIN));
    auto ordered_low = pointwise({low, sign_bit}, source_shape, at::kInt, "bitwise_xor_fwd_i32",
                                 {syn_type_int32, syn_type_int32, syn_type_int32});
    auto eligible = gaudirt::generic::where(high_matches, ordered_low, sign_bit);
    auto low_max = reduce_max_i32(eligible, axis);
    auto low_matches = compare(ordered_low, low_max, "equal");
    auto matches = pointwise({high_matches, low_matches}, source_shape, at::kBool, "and_fwd_u8",
                             {syn_type_uint8, syn_type_uint8, syn_type_uint8});
    auto scores = matches.to(at::kFloat);
    return gaudirt::generic::argmax(scores, axis, true).view(shape);
  }
  std::vector<int64_t> reduced_shape(source.sizes().begin(), source.sizes().end());
  reduced_shape[axis] = 1;
  auto reduced = at::empty(reduced_shape, source.options().dtype(at::kInt));
  auto ctx = current();
  auto source_address = address_of(source, *ctx);
  if (source_address % BlockCachingAllocator::kAlignment) {
    auto staged = at::empty(source.sizes(), source.options());
    copy(staged, source, true);
    source = std::move(staged);
  }
  {
    std::lock_guard lock(ctx->cold_mutex);
    auto& recipe = argmax_recipe(*ctx, source, reduced, axis);
    auto stream = ctx->device.compute_stream();
    auto bindings = recipe.tensors;
    bindings[0].pTensorAddress = address_of(source, *ctx);
    bindings[1].pTensorAddress = address_of(reduced, *ctx);
    for (const auto& binding : bindings) ctx->allocator.record_stream(binding.pTensorAddress, stream);
    std::unique_ptr<TemporaryDevice> workspace;
    if (recipe.workspace) workspace = std::make_unique<TemporaryDevice>(*ctx, recipe.workspace);
    TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(),
                          workspace ? workspace->address : 0, recipe.recipe, 0) == synSuccess,
                "GaudiRT: argmax launch failed");
  }
  return reduced.to(at::kLong).view(shape);
}

// A first executable FX lowering: a weight-transposed GEMM. The hash contains
// only operator structure and static geometry, never the layer or tensor address.
ColdRecipe& bucket_linear_recipe(Context& ctx, at::ScalarType dtype, uint64_t k,
                                 uint64_t n, uint32_t bucket, bool has_bias) {
  const auto type = cast_type(dtype);
  const std::string structure = "fx_mm_transpose_weight_v1_" + std::string(type.name) +
                                "_" + std::to_string(k) + "_" + std::to_string(n) +
                                (has_bias ? "_bias" : "");
  const auto digest = gaudirt::common::sha256_bytes(
      reinterpret_cast<const uint8_t*>(structure.data()), structure.size());
  const auto prefix = "g" + digest.substr(0, 12) + "_b" + std::to_string(bucket);
  const auto key = "fx_" + digest + "_b" + std::to_string(bucket);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  const size_t external_count = has_bias ? 4 : 3;
  auto recipe = std::make_unique<ColdRecipe>(external_count);
  synGraphHandle graph = nullptr;
  std::array<synSectionHandle, 4> sections{};
  std::array<synTensor, 7> tensors{};
  std::array<std::string, 3> intermediate_names{};
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : tensors) if (tensor) synTensorDestroy(tensor);
    for (auto section : sections) if (section) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "bucket graph creation failed");
    const std::array<std::array<uint64_t, 2>, 4> shapes = has_bias ?
        std::array<std::array<uint64_t, 2>, 4>{{{k, bucket}, {k, n}, {n, 1}, {n, bucket}}} :
        std::array<std::array<uint64_t, 2>, 4>{{{k, bucket}, {k, n}, {n, bucket}, {0, 0}}};
    for (size_t i = 0; i < external_count; ++i) {
      recipe->names[i] = prefix + "_t" + std::to_string(i);
      check(synSectionCreate(&sections[i], 0, graph), "bucket section creation failed");
      check(synSectionSetPersistent(sections[i], true), "bucket section setup failed");
      synTensorDescriptor desc{};
      desc.m_name = recipe->names[i].c_str();
      desc.m_dataType = type.syn;
      desc.m_dims = 2;
      for (size_t dim = 0; dim < 2; ++dim)
        desc.m_sizes[dim] = desc.m_minSizes[dim] = shapes[i][dim];
      check(synTensorCreate(&tensors[i], &desc, sections[i], 0), "bucket tensor creation failed");
      recipe->tensors[i].tensorName = recipe->names[i].c_str();
      recipe->tensors[i].tensorType = DATA_TENSOR;
      recipe->tensors[i].tensorSize[0] = shapes[i][0];
      recipe->tensors[i].tensorSize[1] = shapes[i][1];
    }
    synTensor inputs[] = {tensors[0], tensors[1]};
    synGEMMParams params{false, true};
    const auto node = prefix + "_n0";
    if (has_bias) {
      const bool bf16 = dtype == at::kBFloat16;
      const size_t intermediate_count = bf16 ? 3 : 1;
      for (size_t i = 0; i < intermediate_count; ++i) {
        intermediate_names[i] = prefix + "_t" + std::to_string(i + external_count);
        synTensorDescriptor desc{};
        desc.m_name = intermediate_names[i].c_str();
        desc.m_dataType = syn_type_float;
        desc.m_dims = 2;
        desc.m_sizes[0] = desc.m_minSizes[0] = n;
        desc.m_sizes[1] = desc.m_minSizes[1] = bf16 && i == 1 ? 1 : bucket;
        check(synTensorCreate(&tensors[external_count + i], &desc, nullptr, 0),
              "bucket intermediate creation failed");
      }
    }
    auto* gemm_output = has_bias ? &tensors[external_count] : &tensors[2];
    check(synNodeCreate(graph, inputs, gemm_output, 2, 1, &params, sizeof(params),
                        "gemm", node.c_str(), nullptr, nullptr), "bucket GEMM node creation failed");
    if (has_bias) {
      const bool bf16 = dtype == at::kBFloat16;
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_HALF_NE;
      if (bf16) {
        const auto cast_node = prefix + "_n1";
        check(synNodeCreate(graph, &tensors[2], &tensors[5], 1, 1, &cast, sizeof(cast),
                            "cast_bf16_to_f32", cast_node.c_str(), nullptr, nullptr),
              "bucket bias cast failed");
      }
      synTensor add_inputs[] = {tensors[4], bf16 ? tensors[5] : tensors[2]};
      auto* add_output = bf16 ? &tensors[6] : &tensors[3];
      const auto add_node = prefix + "_n2";
      check(synNodeCreate(graph, add_inputs, add_output, 2, 1, nullptr, 0,
                          "add_fwd_f32", add_node.c_str(), nullptr, nullptr), "bucket bias add failed");
      if (bf16) {
        const auto output_node = prefix + "_n3";
        check(synNodeCreate(graph, &tensors[6], &tensors[3], 1, 1, &cast, sizeof(cast),
                            "cast_f32_to_bf16", output_node.c_str(), nullptr, nullptr),
              "bucket output cast failed");
      }
    }
    recipe->recipe = compile_cached(graph, prefix, key, &ctx.cache);
    check(synWorkspaceGetSize(&recipe->workspace, recipe->recipe), "bucket workspace query failed");
    initialize_launch_tensor_ids(recipe->recipe, recipe->tensors.data(), recipe->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(recipe));
  TORCH_CHECK(inserted, "GaudiRT: duplicate bucket recipe");
  return *it->second;
}

struct GraphTemporary {
  GraphTemporary(BlockCachingAllocator& pool, synStreamHandle stream, uint64_t bytes)
      : pool(pool), address(pool.allocate(bytes, stream)) {}
  ~GraphTemporary() { pool.free(address); }
  BlockCachingAllocator& pool;
  uint64_t address;
};

at::Tensor bucket_output(const at::Tensor& reference, at::ScalarType dtype,
                         uint32_t bucket, int64_t rows, int64_t width) {
  const uint64_t item = c10::elementSize(dtype);
  TORCH_CHECK(width > 0 && static_cast<uint64_t>(width) <=
              (static_cast<uint64_t>(INT64_MAX) - BlockCachingAllocator::kAlignment) / bucket / item,
              "GaudiRT: bucket output is too large");
  const uint64_t elements = static_cast<uint64_t>(bucket) * width;
  const uint64_t bytes = (elements * item + BlockCachingAllocator::kAlignment - 1) &
                         ~(BlockCachingAllocator::kAlignment - 1);
  auto storage = at::empty({static_cast<int64_t>(bytes / item)}, reference.options().dtype(dtype));
  return storage.narrow(0, 0, elements).view({static_cast<int64_t>(bucket), width}).narrow(0, 0, rows);
}

at::Tensor bucket_linear(const at::Tensor& input, const at::Tensor& weight,
                         const std::optional<at::Tensor>& bias) {
  TORCH_CHECK(input.device() == weight.device() && input.device().type() == c10::kPrivateUse1 &&
              input.dim() == 2 && weight.dim() == 2 && input.size(1) == weight.size(1),
              "GaudiRT: bucket GEMM expects same-device [M,K] and [N,K]");
  TORCH_CHECK(input.scalar_type() == weight.scalar_type() &&
              (input.scalar_type() == at::kBFloat16 || input.scalar_type() == at::kFloat),
              "GaudiRT: bucket GEMM supports matching BF16 or FP32 tensors");
  const auto m = input.size(0), k = input.size(1), n = weight.size(0);
  TORCH_CHECK(m > 0 && m <= 4096 && k > 0 && n > 0, "GaudiRT: invalid bucket GEMM shape");
  if (bias) {
    TORCH_CHECK(bias->device() == input.device() && bias->scalar_type() == input.scalar_type() &&
                bias->dim() == 1 && bias->size(0) == n,
                "GaudiRT: bucket GEMM bias must be a same-device [N] tensor of matching dtype");
  }
  const uint32_t bucket = std::bit_ceil(static_cast<uint32_t>(m));
  const uint64_t item = input.element_size();
  TORCH_CHECK(static_cast<uint64_t>(k) <= UINT64_MAX / bucket / item,
              "GaudiRT: bucket GEMM input is too large");
  const uint64_t bucket_bytes = static_cast<uint64_t>(bucket) * k * item;
  auto ctx = current();
  auto stream = ctx->device.compute_stream();
  const auto input_address = address_of(input, *ctx);
  auto weight_value = weight;
  auto weight_address = address_of(weight_value, *ctx);
  const uint64_t offset = static_cast<uint64_t>(input.storage_offset()) * item;
  const uint64_t storage_bytes = input.storage().nbytes();
  const bool direct = input.is_contiguous() && input_address % BlockCachingAllocator::kAlignment == 0 &&
                      offset <= storage_bytes && bucket_bytes <= storage_bytes - offset;
  const bool overlaps = direct && input_address < weight_address + weight.nbytes() &&
                        weight_address < input_address + bucket_bytes;
  if (!weight_value.is_contiguous() || weight_address % BlockCachingAllocator::kAlignment || overlaps) {
    weight_value = at::empty(weight.sizes(), weight.options());
    copy(weight_value, weight, true);
    weight_address = address_of(weight_value, *ctx);
  }
  at::Tensor bias_value;
  uint64_t bias_address = 0;
  if (bias) {
    bias_value = *bias;
    bias_address = address_of(bias_value, *ctx);
    const auto bytes = bias_value.nbytes();
    const bool input_overlap = direct && input_address < bias_address + bytes &&
                               bias_address < input_address + bucket_bytes;
    const bool weight_overlap = weight_address < bias_address + bytes &&
                                bias_address < weight_address + weight_value.nbytes();
    if (!bias_value.is_contiguous() || bias_address % BlockCachingAllocator::kAlignment ||
        input_overlap || weight_overlap) {
      bias_value = at::empty(bias->sizes(), bias->options());
      copy(bias_value, *bias, true);
      bias_address = address_of(bias_value, *ctx);
    }
  }
  auto result = bucket_output(input, input.scalar_type(), bucket, m, n);
  const auto output_address = address_of(result, *ctx);
  std::lock_guard lock(ctx->cold_mutex);
  auto& recipe = bucket_linear_recipe(*ctx, input.scalar_type(), k, n, bucket, bias.has_value());
  if ((!direct || recipe.workspace) && !ctx->graph_pool) {
    const uint64_t budget = std::min<uint64_t>(ctx->allocator.capacity_bytes() / 8, uint64_t{1} << 30);
    TORCH_CHECK(budget >= 256, "GaudiRT: HBM budget is too small for graph temporaries");
    ctx->graph_pool = std::make_unique<GraphPool>(ctx->allocator, budget, stream);
  }
  std::unique_ptr<GraphTemporary> staged, workspace;
  ctx->allocator.record_stream(input_address, stream);
  ctx->allocator.record_stream(weight_address, stream);
  if (bias) ctx->allocator.record_stream(bias_address, stream);
  ctx->allocator.record_stream(output_address, stream);
  if (!direct) {
    staged = std::make_unique<GraphTemporary>(ctx->graph_pool->allocator(), stream, bucket_bytes);
    const Layout layout(input);
    dma_copy(staged->address, nullptr, input_address, &layout, 0, input.numel(), item, stream, DRAM_TO_DRAM);
  }
  if (recipe.workspace)
    workspace = std::make_unique<GraphTemporary>(ctx->graph_pool->allocator(), stream, recipe.workspace);
  auto bindings = recipe.tensors;
  bindings[0].pTensorAddress = staged ? staged->address : input_address;
  bindings[1].pTensorAddress = weight_address;
  if (bias) bindings[2].pTensorAddress = bias_address;
  bindings.back().pTensorAddress = output_address;
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(), workspace ? workspace->address : 0,
                        recipe.recipe, 0) == synSuccess, "GaudiRT: bucket GEMM launch failed");
  return result;
}

struct FxValueSpec {
  at::ScalarType dtype;
  std::vector<int64_t> sizes;
  bool dynamic;
};

// An FX program is described by external input roles and normalized nodes.
// Node 0 is linear(x, weight, optional_bias); node 1 is add(x, residual).
// A node's output ID is input_count + its index in the program.
class BucketProgram {
 public:
  BucketProgram(std::vector<int64_t> roles, std::vector<std::vector<int64_t>> operations)
      : roles_(std::move(roles)), operations_(std::move(operations)) {
    TORCH_CHECK(!roles_.empty() && !operations_.empty(), "GaudiRT FX: empty program");
    bool has_dynamic = false;
    for (auto role : roles_) {
      TORCH_CHECK(role == 0 || role == 1, "GaudiRT FX: invalid input role");
      has_dynamic |= role == 1;
    }
    TORCH_CHECK(has_dynamic, "GaudiRT FX: no dynamic token input");
    for (size_t i = 0; i < operations_.size(); ++i) {
      const auto& op = operations_[i];
      const int64_t upper = roles_.size() + i;
      TORCH_CHECK(op.size() == 4 && (op[0] == 0 || op[0] == 1) &&
                  op[1] >= 0 && op[1] < upper && op[2] >= 0 && op[2] < upper &&
                  ((op[0] == 0 && (op[3] == -1 || (op[3] >= 0 && op[3] < upper))) ||
                   (op[0] == 1 && op[3] == -1)), "GaudiRT FX: invalid normalized node");
    }
  }

  at::Tensor run(const std::vector<at::Tensor>& inputs);

 private:
  void initialize_geometry(const std::vector<at::Tensor>& inputs);
  ColdRecipe& recipe(Context& ctx, uint32_t bucket);

  std::vector<int64_t> roles_;
  std::vector<std::vector<int64_t>> operations_;
  std::vector<FxValueSpec> specs_;
  std::string digest_;
  std::mutex mutex_;
};

void BucketProgram::initialize_geometry(const std::vector<at::Tensor>& inputs) {
  TORCH_CHECK(inputs.size() == roles_.size(), "GaudiRT FX: wrong input count");
  std::vector<FxValueSpec> specs;
  specs.reserve(inputs.size() + operations_.size());
  std::ostringstream normalized;
  normalized << "fx_program_v3;";
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& tensor = inputs[i];
    TORCH_CHECK(tensor.device().type() == c10::kPrivateUse1 && tensor.device().index() == 0 &&
                (tensor.scalar_type() == at::kBFloat16 || tensor.scalar_type() == at::kFloat),
                "GaudiRT FX: unsupported input device or dtype");
    TORCH_CHECK((roles_[i] && tensor.dim() == 2) ||
                (!roles_[i] && (tensor.dim() == 1 || tensor.dim() == 2)),
                "GaudiRT FX: unsupported input rank");
    std::vector<int64_t> sizes(tensor.sizes().begin(), tensor.sizes().end());
    for (auto size : sizes) TORCH_CHECK(size > 0, "GaudiRT FX: empty input dimension");
    if (roles_[i]) sizes[0] = -1;
    normalized << "i" << roles_[i] << ":" << static_cast<int>(tensor.scalar_type());
    for (auto size : sizes) normalized << ":" << size;
    normalized << ";";
    specs.push_back({tensor.scalar_type(), std::move(sizes), roles_[i] == 1});
  }
  for (const auto& op : operations_) {
    const auto& a = specs[op[1]];
    const auto& b = specs[op[2]];
    TORCH_CHECK(a.dtype == b.dtype, "GaudiRT FX: input dtype mismatch");
    if (op[0] == 0) {
      TORCH_CHECK(a.dynamic && a.sizes.size() == 2 && !b.dynamic && b.sizes.size() == 2 &&
                  a.sizes[1] == b.sizes[1], "GaudiRT FX: invalid linear geometry");
      if (op[3] >= 0) {
        const auto& bias = specs[op[3]];
        TORCH_CHECK(!bias.dynamic && bias.sizes.size() == 1 && bias.sizes[0] == b.sizes[0] &&
                    bias.dtype == a.dtype, "GaudiRT FX: invalid linear bias");
      }
      specs.push_back({a.dtype, {-1, b.sizes[0]}, true});
    } else {
      TORCH_CHECK(a.dynamic && b.dynamic && a.sizes == b.sizes,
                  "GaudiRT FX: add requires equal dynamic shapes");
      specs.push_back(a);
    }
    normalized << "n" << op[0] << ":" << op[1] << ":" << op[2] << ":" << op[3] << ";";
  }
  const auto text = normalized.str();
  const auto digest = gaudirt::common::sha256_bytes(
      reinterpret_cast<const uint8_t*>(text.data()), text.size());
  specs_ = std::move(specs);
  digest_ = digest;
}

ColdRecipe& BucketProgram::recipe(Context& ctx, uint32_t bucket) {
  const auto prefix = "g" + digest_.substr(0, 12) + "_b" + std::to_string(bucket);
  const auto key = "fx_program_" + digest_ + "_b" + std::to_string(bucket);
  auto found = ctx.cold_recipes.find(key);
  if (found != ctx.cold_recipes.end()) return *found->second;
  auto compiled = std::make_unique<ColdRecipe>(roles_.size() + 1);
  synGraphHandle graph = nullptr;
  std::vector<synSectionHandle> sections;
  std::vector<synTensor> created;
  std::vector<synTensor> values(specs_.size(), nullptr);
  std::deque<std::string> tensor_names;
  size_t next_tensor = 0, next_node = 0;
  const auto check = [](synStatus status, const char* message) {
    TORCH_CHECK(status == synSuccess, "GaudiRT FX: ", message, " status=", static_cast<int>(status));
  };
  auto cleanup = [&] {
    if (graph) synGraphDestroy(graph);
    for (auto tensor : created) synTensorDestroy(tensor);
    for (auto section : sections) synSectionDestroy(section);
  };
  try {
    check(synGraphCreate(&graph, synDeviceGaudi2), "program graph creation failed");
    auto tensor = [&](synDataType type, uint64_t fcd, uint64_t rows, int slot) {
      tensor_names.push_back(prefix + "_t" + std::to_string(next_tensor++));
      synSectionHandle section = nullptr;
      if (slot >= 0) {
        check(synSectionCreate(&section, 0, graph), "program section creation failed");
        sections.push_back(section);
        check(synSectionSetPersistent(section, true), "program section setup failed");
        compiled->names[slot] = tensor_names.back();
      }
      synTensorDescriptor desc{};
      desc.m_name = slot >= 0 ? compiled->names[slot].c_str() : tensor_names.back().c_str();
      desc.m_dataType = type;
      desc.m_dims = 2;
      desc.m_sizes[0] = desc.m_minSizes[0] = fcd;
      desc.m_sizes[1] = desc.m_minSizes[1] = rows;
      synTensor handle = nullptr;
      check(synTensorCreate(&handle, &desc, section, 0), "program tensor creation failed");
      created.push_back(handle);
      if (slot >= 0) {
        compiled->tensors[slot].tensorName = compiled->names[slot].c_str();
        compiled->tensors[slot].tensorType = DATA_TENSOR;
        compiled->tensors[slot].tensorSize[0] = fcd;
        compiled->tensors[slot].tensorSize[1] = rows;
      }
      return handle;
    };
    auto node = [&](const std::vector<synTensor>& in, synTensor out, const void* params,
                    size_t params_size, const char* guid) {
      const auto name = prefix + "_n" + std::to_string(next_node++);
      check(synNodeCreate(graph, in.data(), &out, in.size(), 1, params, params_size,
                          guid, name.c_str(), nullptr, nullptr), "program node creation failed");
    };
    for (size_t i = 0; i < roles_.size(); ++i) {
      const auto& spec = specs_[i];
      const uint64_t fcd = spec.sizes.back();
      const uint64_t rows = spec.dynamic ? bucket : (spec.sizes.size() == 1 ? 1 : spec.sizes[0]);
      values[i] = tensor(cast_type(spec.dtype).syn, fcd, rows, i);
    }
    for (size_t i = 0; i < operations_.size(); ++i) {
      const auto& op = operations_[i];
      const auto id = roles_.size() + i;
      const auto& spec = specs_[id];
      const auto type = cast_type(spec.dtype);
      const uint64_t width = spec.sizes[1];
      const int slot = i + 1 == operations_.size() ? static_cast<int>(roles_.size()) : -1;
      values[id] = tensor(type.syn, width, bucket, slot);
      if (op[0] == 1) {
        const auto guid = spec.dtype == at::kBFloat16 ? std::string("gk_add_bf16_v1") :
                          "add_fwd_" + std::string(type.name);
        node({values[op[1]], values[op[2]]}, values[id], nullptr, 0, guid.c_str());
        continue;
      }
      synGEMMParams gemm{false, true};
      if (op[3] < 0) {
        node({values[op[1]], values[op[2]]}, values[id], &gemm, sizeof(gemm), "gemm");
        continue;
      }
      auto product = tensor(syn_type_float, width, bucket, -1);
      node({values[op[1]], values[op[2]]}, product, &gemm, sizeof(gemm), "gemm");
      const bool bf16 = spec.dtype == at::kBFloat16;
      auto bias = values[op[3]];
      ns_CastKernel::ParamsV3 cast{};
      cast.round_mode = CAST_ROUND_HALF_NE;
      if (bf16) {
        auto wide_bias = tensor(syn_type_float, width, 1, -1);
        node({bias}, wide_bias, &cast, sizeof(cast), "cast_bf16_to_f32");
        bias = wide_bias;
      }
      auto sum = bf16 ? tensor(syn_type_float, width, bucket, -1) : values[id];
      node({product, bias}, sum, nullptr, 0, "add_fwd_f32");
      if (bf16) node({sum}, values[id], &cast, sizeof(cast), "cast_f32_to_bf16");
    }
    compiled->recipe = compile_cached(graph, prefix, key, &ctx.cache);
    check(synWorkspaceGetSize(&compiled->workspace, compiled->recipe), "program workspace query failed");
    initialize_launch_tensor_ids(compiled->recipe, compiled->tensors.data(), compiled->tensors.size());
    cleanup();
  } catch (...) {
    cleanup();
    throw;
  }
  auto [it, inserted] = ctx.cold_recipes.emplace(key, std::move(compiled));
  TORCH_CHECK(inserted, "GaudiRT FX: duplicate program recipe");
  return *it->second;
}

at::Tensor BucketProgram::run(const std::vector<at::Tensor>& inputs) {
  std::lock_guard program_lock(mutex_);
  TORCH_CHECK(inputs.size() == roles_.size(), "GaudiRT FX: wrong input count");
  int64_t m = -1;
  size_t dynamic_input = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!roles_[i]) continue;
    TORCH_CHECK(inputs[i].dim() == 2, "GaudiRT FX: dynamic input must have rank two");
    if (m < 0) {
      m = inputs[i].size(0);
      dynamic_input = i;
    } else {
      TORCH_CHECK(inputs[i].size(0) == m, "GaudiRT FX: inconsistent token dimensions");
    }
  }
  TORCH_CHECK(m > 0 && m <= 4096, "GaudiRT FX: token count outside 1..4096");
  if (specs_.empty()) initialize_geometry(inputs);
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];
    const auto& spec = specs_[i];
    TORCH_CHECK(input.device().type() == c10::kPrivateUse1 && input.device().index() == 0 &&
                input.scalar_type() == spec.dtype && input.dim() == static_cast<int64_t>(spec.sizes.size()),
                "GaudiRT FX: input device, dtype or rank changed");
    for (int64_t dim = 0; dim < input.dim(); ++dim) {
      const auto expected = spec.dynamic && dim == 0 ? m : spec.sizes[dim];
      TORCH_CHECK(input.size(dim) == expected, "GaudiRT FX: input geometry changed");
    }
  }
  const uint32_t bucket = std::bit_ceil(static_cast<uint32_t>(m));
  const auto& output_spec = specs_.back();
  auto ctx = current();
  auto stream = ctx->device.compute_stream();
  auto result = bucket_output(inputs[dynamic_input], output_spec.dtype, bucket, m, output_spec.sizes[1]);
  const auto output_address = address_of(result, *ctx);
  std::lock_guard recipe_lock(ctx->cold_mutex);
  auto& compiled = recipe(*ctx, bucket);
  auto graph_pool = [&]() -> BlockCachingAllocator& {
    if (!ctx->graph_pool) {
      const uint64_t budget = std::min<uint64_t>(ctx->allocator.capacity_bytes() / 8, uint64_t{1} << 30);
      TORCH_CHECK(budget >= 256, "GaudiRT FX: HBM budget is too small for graph temporaries");
      ctx->graph_pool = std::make_unique<GraphPool>(ctx->allocator, budget, stream);
    }
    return ctx->graph_pool->allocator();
  };
  std::vector<std::unique_ptr<GraphTemporary>> staged;
  std::vector<std::pair<uint64_t, uint64_t>> regions;
  auto bindings = compiled.tensors;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& input = inputs[i];
    const auto& spec = specs_[i];
    const uint64_t item = input.element_size();
    TORCH_CHECK(!spec.dynamic || static_cast<uint64_t>(spec.sizes[1]) <= UINT64_MAX / bucket / item,
                "GaudiRT FX: input allocation size overflow");
    const uint64_t bytes = spec.dynamic ? static_cast<uint64_t>(bucket) * spec.sizes[1] * item : input.nbytes();
    const uint64_t address = address_of(input, *ctx);
    ctx->allocator.record_stream(address, stream);
    const uint64_t offset = static_cast<uint64_t>(input.storage_offset()) * item;
    const uint64_t storage = input.storage().nbytes();
    bool direct = input.is_contiguous() && address % BlockCachingAllocator::kAlignment == 0 &&
                  offset <= storage && bytes <= storage - offset;
    for (const auto& [other, length] : regions)
      if (address < other + length && other < address + bytes) direct = false;
    uint64_t binding = address;
    if (!direct) {
      auto temp = std::make_unique<GraphTemporary>(graph_pool(), stream, bytes);
      const Layout layout(input);
      dma_copy(temp->address, nullptr, address, &layout, 0, input.numel(), item, stream, DRAM_TO_DRAM);
      binding = temp->address;
      staged.push_back(std::move(temp));
    }
    regions.emplace_back(binding, bytes);
    bindings[i].pTensorAddress = binding;
  }
  ctx->allocator.record_stream(output_address, stream);
  bindings[roles_.size()].pTensorAddress = output_address;
  std::unique_ptr<GraphTemporary> workspace;
  if (compiled.workspace) workspace = std::make_unique<GraphTemporary>(graph_pool(), stream, compiled.workspace);
  TORCH_CHECK(synLaunch(stream, bindings.data(), bindings.size(), workspace ? workspace->address : 0,
                        compiled.recipe, 0) == synSuccess, "GaudiRT FX: program launch failed");
  return result;
}
}  // namespace gaudirt::generic

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("copy_", TORCH_FN(gaudirt::generic::copy));
  m.impl("empty.memory_format", TORCH_FN(gaudirt::generic::empty));
  m.impl("empty_strided", TORCH_FN(gaudirt::generic::empty_strided));
  m.impl("as_strided", TORCH_FN(at::native::as_strided_tensorimpl));
  m.impl("view", TORCH_FN(at::native::view));
  m.impl("_reshape_alias", TORCH_FN(at::native::_reshape_alias));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(gaudirt::generic::set_storage));
  m.impl("fill_.Scalar", TORCH_FN(gaudirt::generic::fill));
  m.impl("zero_", TORCH_FN(gaudirt::generic::zero));
  m.impl("cat", TORCH_FN(gaudirt::generic::cat));
  m.impl("cat.out", TORCH_FN(gaudirt::generic::cat_out));
  m.impl("arange.start_out", TORCH_FN(gaudirt::generic::arange_out));
  m.impl("arange.out", TORCH_FN(gaudirt::generic::arange_end_out));
  m.impl("eq.Tensor", TORCH_FN(gaudirt::generic::eq_tensor));
  m.impl("eq.Scalar", TORCH_FN(gaudirt::generic::eq_scalar));
  m.impl("ne.Tensor", TORCH_FN(gaudirt::generic::ne_tensor));
  m.impl("ne.Scalar", TORCH_FN(gaudirt::generic::ne_scalar));
  m.impl("lt.Tensor", TORCH_FN(gaudirt::generic::lt_tensor));
  m.impl("lt.Scalar", TORCH_FN(gaudirt::generic::lt_scalar));
  m.impl("le.Tensor", TORCH_FN(gaudirt::generic::le_tensor));
  m.impl("le.Scalar", TORCH_FN(gaudirt::generic::le_scalar));
  m.impl("gt.Tensor", TORCH_FN(gaudirt::generic::gt_tensor));
  m.impl("gt.Scalar", TORCH_FN(gaudirt::generic::gt_scalar));
  m.impl("ge.Tensor", TORCH_FN(gaudirt::generic::ge_tensor));
  m.impl("ge.Scalar", TORCH_FN(gaudirt::generic::ge_scalar));
  m.impl("where.self", TORCH_FN(gaudirt::generic::where));
  m.impl("index_select", TORCH_FN(gaudirt::generic::index_select));
  m.impl("embedding", TORCH_FN(gaudirt::generic::embedding));
  m.impl("argmax", TORCH_FN(gaudirt::generic::argmax));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  gaudirt::generic::register_backend();
  m.def("initialize", &gaudirt::generic::initialize);
  m.def("shutdown", &gaudirt::generic::shutdown);
  m.def("_shutdown_at_exit", &gaudirt::generic::shutdown_at_exit);
  m.def("synchronize", &gaudirt::generic::synchronize);
  m.def("empty_cache", &gaudirt::generic::empty_cache);
  m.def("memory_allocated", &gaudirt::generic::memory_allocated);
  m.def("max_memory_allocated", &gaudirt::generic::max_memory_allocated);
  m.def("recipe_cache_stats", &gaudirt::generic::recipe_cache_stats);
  m.def("bucket_linear", &gaudirt::generic::bucket_linear,
        pybind11::arg("input"), pybind11::arg("weight"), pybind11::arg("bias") = pybind11::none());
  pybind11::class_<gaudirt::generic::BucketProgram>(m, "BucketProgram")
      .def(pybind11::init<std::vector<int64_t>, std::vector<std::vector<int64_t>>>())
      .def("run", &gaudirt::generic::BucketProgram::run);
}
