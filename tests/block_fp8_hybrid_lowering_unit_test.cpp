// SPDX-License-Identifier: Apache-2.0
// Mock-Synapse unit test for FxOp::block_fp8_linear 1K BF16 + 2K FP8_143 hybrid prefill lowering.
#include "bridge_internal.hpp"
#include "bucket_program.hpp"
#include "context.hpp"
#include "fx_geometry.hpp"
#include "graph_builder.hpp"
#include "layout_copy.hpp"

#include <ATen/EmptyTensor.h>
#include <ATen/Functions.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/view_native.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gaudirt::generic {
class AttentionStagingPool {};
class SegmentOutputPool {};
}  // namespace gaudirt::generic

namespace {

struct TensorRecord {
  std::string name;
  synDataType dtype = syn_type_na;
  synTensorType tensor_type = DATA_TENSOR;
  std::vector<uint64_t> sizes;
  synSectionHandle section = nullptr;
  bool has_fp8_metadata = false;
  double fp8_scale = 0.0;
  int fp8_exp_bias = 0;
};

struct SectionRecord {
  bool persistent = false;
  bool rmw = false;
};

struct NodeRecord {
  std::string guid;
  std::string name;
  std::vector<synTensor> inputs;
  std::vector<synTensor> outputs;
  std::vector<uint8_t> params;
};

struct GraphState {
  uint64_t next_id = 1;
  std::unordered_map<synSectionHandle, SectionRecord> sections;
  std::unordered_map<synTensor, TensorRecord> tensors;
  std::vector<NodeRecord> nodes;
  std::vector<std::string> compiled_keys;
  uint64_t launches = 0;

  void clear_graph_records() {
    sections.clear();
    tensors.clear();
    nodes.clear();
  }
};

GraphState g_state;

synStreamHandle mock_stream(uintptr_t id) {
  return reinterpret_cast<synStreamHandle>(id);
}

struct MockEvent {
  bool complete = true;
};

uint64_t g_next_device_addr = 0x10000000ULL;

}  // namespace

extern "C" {

synStatus synDeviceGetMemoryInfo(synDeviceId, uint64_t* free_mem, uint64_t* total_mem) {
  *free_mem = *total_mem = 64ULL << 30;
  return synSuccess;
}

synStatus synDeviceMalloc(synDeviceId, uint64_t size, uint64_t, uint32_t, uint64_t* addr) {
  *addr = g_next_device_addr;
  g_next_device_addr += (size + 4095ULL) & ~4095ULL;
  return synSuccess;
}

synStatus synDeviceFree(synDeviceId, uint64_t, uint32_t) { return synSuccess; }

synStatus synEventCreate(synEventHandle* handle, synDeviceId, uint32_t) {
  *handle = new MockEvent{};
  return synSuccess;
}

synStatus synEventDestroy(synEventHandle handle) {
  delete static_cast<MockEvent*>(handle);
  return synSuccess;
}

synStatus synEventRecord(synEventHandle, synStreamHandle) { return synSuccess; }
synStatus synEventQuery(synEventHandle) { return synSuccess; }
synStatus synEventSynchronize(synEventHandle) { return synSuccess; }
synStatus synStreamWaitEvent(synStreamHandle, synEventHandle, uint32_t) { return synSuccess; }
synStatus synStreamSynchronize(synStreamHandle) { return synSuccess; }

synStatus synDeviceGetAttribute(uint64_t* value, const synDeviceAttribute* attr,
                                unsigned count, synDeviceId) {
  assert(count == 1 && *attr == DEVICE_ATTRIBUTE_MAX_RMW_SIZE);
  *value = 16ULL << 20;  // 16 MiB Gaudi2 SRAM RMW limit
  return synSuccess;
}

synStatus synGraphCreate(synGraphHandle* graph, synDeviceType device_type) {
  assert(device_type == synDeviceGaudi2);
  g_state.clear_graph_records();
  *graph = reinterpret_cast<synGraphHandle>(g_state.next_id++);
  return synSuccess;
}

synStatus synGraphDestroy(synGraphHandle) { return synSuccess; }

synStatus synSectionCreate(synSectionHandle* section, uint64_t, synGraphHandle) {
  auto handle = reinterpret_cast<synSectionHandle>(g_state.next_id++);
  g_state.sections.emplace(handle, SectionRecord{});
  *section = handle;
  return synSuccess;
}

synStatus synSectionDestroy(synSectionHandle) { return synSuccess; }

synStatus synSectionSetPersistent(synSectionHandle section, bool persistent) {
  g_state.sections.at(section).persistent = persistent;
  return synSuccess;
}

synStatus synSectionSetRMW(synSectionHandle section, bool rmw) {
  g_state.sections.at(section).rmw = rmw;
  return synSuccess;
}

synStatus synTensorCreate(synTensor* tensor, const synTensorDescriptor* desc,
                          const synSectionHandle section, uint64_t) {
  auto handle = reinterpret_cast<synTensor>(g_state.next_id++);
  TensorRecord rec;
  rec.name = desc->m_name ? desc->m_name : "";
  rec.dtype = desc->m_dataType;
  rec.tensor_type = desc->m_tensorType;
  rec.sizes.assign(desc->m_sizes, desc->m_sizes + desc->m_dims);
  rec.section = section;
  g_state.tensors.emplace(handle, std::move(rec));
  *tensor = handle;
  return synSuccess;
}

synStatus synTensorDestroy(synTensor) { return synSuccess; }

synStatus synTensorSetQuantizationData(synTensor tensor, synQuantMetadataType prop, void* data,
                                       uint64_t size) {
  assert(prop == SYN_FP_QUANT_METADATA && size == sizeof(synFpQuantMetadata));
  const auto* meta = static_cast<const synFpQuantMetadata*>(data);
  assert(meta->numFpQuantParams == 1 && meta->fpQuantParams != nullptr);
  auto& rec = g_state.tensors.at(tensor);
  rec.has_fp8_metadata = true;
  rec.fp8_scale = meta->fpQuantParams[0].scale;
  rec.fp8_exp_bias = meta->fpQuantParams[0].expBias;
  return synSuccess;
}

synStatus synTensorSetPermutation(synTensor, const synTensorPermutation*) { return synSuccess; }

synStatus synNodeCreate(synGraphHandle, const synTensor* inputs, const synTensor* outputs,
                        uint32_t num_inputs, uint32_t num_outputs, const void* params,
                        uint32_t param_size, const char* guid, const char* name,
                        const char**, const char**) {
  NodeRecord node;
  node.guid = guid ? guid : "";
  node.name = name ? name : "";
  node.inputs.assign(inputs, inputs + num_inputs);
  node.outputs.assign(outputs, outputs + num_outputs);
  if (params && param_size > 0) {
    const auto* bytes = static_cast<const uint8_t*>(params);
    node.params.assign(bytes, bytes + param_size);
  }
  g_state.nodes.push_back(std::move(node));
  return synSuccess;
}

synStatus synWorkspaceGetSize(uint64_t* size, const synRecipeHandle) {
  *size = 256;
  return synSuccess;
}

synStatus synRecipeDestroy(synRecipeHandle) { return synSuccess; }

synStatus synLaunch(synStreamHandle, const synLaunchTensorInfo*, uint32_t, uint64_t,
                    const synRecipeHandle, uint32_t) {
  ++g_state.launches;
  return synSuccess;
}

}  // extern "C"

namespace gaudirt::runtime {

DeviceRuntime::DeviceRuntime(const Config& cfg)
    : cfg_(cfg),
      device_id_(0),
      compute_stream_(mock_stream(1)),
      h2d_stream_(mock_stream(2)),
      d2h_stream_(mock_stream(3)),
      owns_device_(false),
      owns_compute_stream_(false) {}

DeviceRuntime::~DeviceRuntime() = default;

StreamEvent::StreamEvent(synDeviceId) : event_(reinterpret_cast<synEventHandle>(uintptr_t{1})) {}
StreamEvent::~StreamEvent() = default;

PinnedHostPool::PinnedHostPool(synDeviceId device, uint64_t bytes)
    : device_(device), capacity_(bytes), events_(device) {}
PinnedHostPool::~PinnedHostPool() = default;

synRecipeHandle compile_cached(synGraphHandle, const std::string&, const std::string& key,
                               RecipeCache*) {
  g_state.compiled_keys.push_back(key);
  return reinterpret_cast<synRecipeHandle>(g_state.next_id++);
}

void initialize_launch_tensor_ids(synRecipeHandle, synLaunchTensorInfo* infos, size_t count) {
  for (size_t i = 0; i < count; ++i) infos[i].tensorId = i + 1;
}

}  // namespace gaudirt::runtime

namespace gaudirt::generic {

std::mutex event_mutex;
thread_local c10::StreamId active_stream_id = 0;

StreamRegistry::StreamState::StreamState(synDeviceId device, synStreamHandle s)
    : handle(s),
      metadata_reusable(device),
      metadata_ready(device),
      validation_ready(device),
      validation_completed(device),
      h2d_reusable(device),
      h2d_ready(device),
      d2h_reusable(device),
      d2h_ready(device) {}

StreamRegistry::StreamRegistry(synDeviceId device, synStreamHandle primary) : device_(device) {
  streams_.push_back(std::make_unique<StreamState>(device_, primary));
  primary_ = streams_.front().get();
}

StreamRegistry::~StreamRegistry() = default;

std::vector<synStreamHandle> StreamRegistry::handles() { return {primary_->handle}; }

Context::Context(uint32_t module, uint64_t hbm_bytes, uint64_t pinned_bytes)
    : generation(1),
      device(runtime::DeviceRuntime::Config{module, 0, 1, 1024, 1024, -1, 0}),
      streams(device.device_id(), device.compute_stream()),
      allocator(device.device_id(), hbm_bytes),
      pinned(device.device_id(), pinned_bytes) {
  allocator.prepare_stream(device.compute_stream());
}

Context::~Context() = default;

synStreamHandle Context::stream() { return device.compute_stream(); }

runtime::GraphPool& Context::ensure_graph_pool(uint64_t budget, synStreamHandle s) {
  if (!graph_pool) {
    graph_pool = std::make_unique<runtime::GraphPool>(allocator, budget, s);
    graph_pool->prepare_stream(s);
  }
  return *graph_pool;
}

uint64_t Context::row_count_address(uint32_t) { return 0x1000; }

namespace {

std::shared_ptr<Context> g_ctx;

void release_mock_allocation(void* ptr) {
  std::unique_ptr<AllocationContext> alloc(static_cast<AllocationContext*>(ptr));
  if (auto owner = alloc->owner.lock()) {
    if (alloc->kind == AllocationKind::graph) owner->graph_pool->free(alloc->address);
    else owner->allocator.free(alloc->address);
  }
}

class MockHooks final : public at::PrivateUse1HooksInterface {
 public:
  bool isBuilt() const override { return true; }
  bool isAvailable() const override { return true; }
};

class MockAllocator final : public c10::Allocator {
 public:
  c10::DataPtr allocate(size_t bytes) override {
    if (!bytes) return {nullptr, c10::Device(c10::kPrivateUse1, 0)};
    auto s = g_ctx->stream();
    const uint64_t addr = g_ctx->allocator.allocate(bytes, s);
    auto* meta = new AllocationContext{g_ctx, addr, bytes, s, AllocationKind::device};
    return {reinterpret_cast<void*>(addr), meta, release_mock_allocation,
            c10::Device(c10::kPrivateUse1, 0)};
  }
  void copy_data(void*, const void*, size_t) const override {}
};

class MockGuard final : public c10::impl::DeviceGuardImplInterface {
 public:
  c10::DeviceType type() const override { return c10::kPrivateUse1; }
  c10::Device exchangeDevice(c10::Device d) const override { return d; }
  c10::Device getDevice() const override { return {c10::kPrivateUse1, 0}; }
  void setDevice(c10::Device) const override {}
  void uncheckedSetDevice(c10::Device) const noexcept override {}
  c10::Stream getStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }
  c10::Stream exchangeStream(c10::Stream s) const override { return s; }
  c10::DeviceIndex deviceCount() const noexcept override { return 1; }
};

void init_mock_backend() {
  static std::once_flag once;
  static MockHooks hooks;
  static MockAllocator alloc;
  static MockGuard guard;
  std::call_once(once, [] {
    at::RegisterPrivateUse1HooksInterface(&hooks);
    c10::SetAllocator(c10::kPrivateUse1, &alloc, 1);
    c10::impl::DeviceGuardImplRegistrar reg(c10::kPrivateUse1, &guard);
  });
}

}  // namespace

std::shared_ptr<Context> current() { return g_ctx; }

const AllocationContext& validated_program_input(const at::Tensor& tensor, const Context& ctx) {
  const auto* alloc =
      tensor.storage().data_ptr().cast_context<AllocationContext>(release_mock_allocation);
  assert(alloc && alloc->owner.lock().get() == &ctx);
  return *alloc;
}

CastType cast_type(at::ScalarType type) {
  switch (type) {
    case at::kBFloat16: return {syn_type_bf16, "bf16"};
    case at::kFloat: return {syn_type_float, "f32"};
    case at::kInt: return {syn_type_int32, "i32"};
    default: assert(false && "unsupported mock cast_type"); return {syn_type_na, ""};
  }
}

at::Tensor bucket_output(const at::Tensor& reference, at::ScalarType dtype, uint32_t bucket,
                         int64_t rows, int64_t width) {
  auto full = at::empty({static_cast<int64_t>(bucket), width}, reference.options().dtype(dtype));
  return rows == static_cast<int64_t>(bucket) ? full : full.narrow(0, 0, rows);
}

Layout::Layout(const at::Tensor& t) : tensor(t), run(1) {}
uint64_t Layout::offset(uint64_t index) const { return index; }
void dma_copy(uint64_t, const Layout*, uint64_t, const Layout*, uint64_t, uint64_t, uint64_t,
              synStreamHandle, synDmaDir) {}

namespace {
at::Tensor mock_empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                      std::optional<at::Layout>, std::optional<at::Device>,
                      std::optional<bool>, std::optional<at::MemoryFormat> format) {
  return at::detail::empty_generic(size, c10::GetAllocator(c10::kPrivateUse1),
                                   c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                   dtype.value_or(at::kFloat), format);
}
}  // namespace

}  // namespace gaudirt::generic

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(gaudirt::generic::mock_empty));
  m.impl("as_strided", TORCH_FN(at::native::as_strided_tensorimpl));
  m.impl("view", TORCH_FN(at::native::view));
  m.impl("_reshape_alias", TORCH_FN(at::native::_reshape_alias));
}

namespace {

using namespace gaudirt::generic;

at::Tensor make_dev_tensor(std::initializer_list<int64_t> sizes, at::ScalarType dtype) {
  return at::empty(sizes, at::TensorOptions().device(c10::Device(c10::kPrivateUse1, 0)).dtype(dtype));
}

size_t count_guid(std::string_view guid) {
  return static_cast<size_t>(std::count_if(
      g_state.nodes.begin(), g_state.nodes.end(),
      [&](const NodeRecord& n) { return n.guid == guid; }));
}

std::vector<const NodeRecord*> find_nodes(std::string_view guid) {
  std::vector<const NodeRecord*> matches;
  for (const auto& n : g_state.nodes) {
    if (n.guid == guid) matches.push_back(&n);
  }
  return matches;
}

const NodeRecord& find_single_node(std::string_view guid) {
  auto matches = find_nodes(guid);
  assert(matches.size() == 1 && "expected exactly one node with GUID");
  return *matches[0];
}

void assert_gemm_nt(const NodeRecord& n) {
  assert(n.params.size() == sizeof(synGEMMParams));
  synGEMMParams params{};
  std::memcpy(&params, n.params.data(), sizeof(params));
  assert(!params.transpose_a && params.transpose_b);
}

void assert_fp8_143_metadata(const TensorRecord& rec) {
  assert(rec.dtype == syn_type_fp8_143);
  assert(rec.has_fp8_metadata && rec.fp8_scale == 1.0 && rec.fp8_exp_bias == 7);
}

void test_small_buckets_keep_batch_gemm() {
  BucketProgram prog({1, 0, 0}, {FxNode{FxOp::block_fp8_linear, {0, 1, 2}, std::monostate{}}});
  auto w = make_dev_tensor({2, 256, 128}, at::kFloat8_e4m3fn);
  auto s = make_dev_tensor({2, 2}, at::kFloat);

  for (int64_t m : {1, 64, 128, 256}) {
    g_state.clear_graph_records();
    auto x = make_dev_tensor({m, 256}, at::kBFloat16);
    auto out = prog.run({x, w, s});
    assert(out.size() == 1 && out[0].sizes().equals({m, 256}));
    assert(count_guid("gk_fp8_block_quant_bf16_v1") == 1);
    assert(count_guid("batch_gemm") == 1);
    assert(count_guid("gk_fp8_block_reduce_bf16_v1") == 1);
    assert(count_guid("gk_fp8_quant_act_bf16_1k_v1") == 0);
    assert(count_guid("gk_fp8_epilogue_k32_bf16_v1") == 0);
  }
}

void test_prefill_k32_unbiased_and_sram_row_tiling() {
  // K = 7168 (56 groups), N = 9216:
  // SRAM activation budget (~22 MiB) selects tile_rows = min(M, 1024):
  // - M in {512, 1024} -> row_tiles = 1, column_tiles = 1 (7 nodes, 0 splits/concats)
  // - M = 4096 -> row_tiles = 4, column_tiles = 1 (1 row split + 4 * 7 bundle nodes + 1 row concat)
  BucketProgram prog({1, 0, 0}, {FxNode{FxOp::block_fp8_linear, {0, 1, 2}, std::monostate{}}});
  auto w = make_dev_tensor({56, 9216, 128}, at::kFloat8_e4m3fn);
  auto s = make_dev_tensor({72, 56}, at::kFloat);

  for (uint64_t m : {512ULL, 1024ULL}) {
    g_state.clear_graph_records();
    auto x = make_dev_tensor({static_cast<int64_t>(m), 7168}, at::kBFloat16);
    auto out = prog.run({x, w, s});
    assert(out.size() == 1 && out[0].sizes().equals({static_cast<int64_t>(m), 9216}));
    assert(g_state.nodes.size() == 7);
    assert(count_guid("batch_gemm") == 0);
    assert(count_guid("split") == 0);
    assert(count_guid("concat") == 0);
    assert(count_guid("gk_fp8_epilogue_k32_bias_bf16_v1") == 0);

    const auto& qa0 = find_single_node("gk_fp8_quant_act_bf16_1k_v1");
    const auto& qap = find_single_node("gk_fp8_quant_act_bf16_k32_v1");
    const auto& dw0 = find_single_node("gk_fp8_dequant_weight_bf16_1k_v1");
    const auto& dwp = find_single_node("gk_fp8_dequant_weight_bf16_k32_v1");
    assert(qa0.inputs == qap.inputs && qa0.outputs[0] != qap.outputs[0]);
    assert(dw0.inputs == dwp.inputs && dw0.outputs[0] != dwp.outputs[0]);

    const auto& act0 = g_state.tensors.at(qa0.outputs[0]);
    const auto& actp = g_state.tensors.at(qap.outputs[0]);
    const auto& w0 = g_state.tensors.at(dw0.outputs[0]);
    const auto& wp = g_state.tensors.at(dwp.outputs[0]);
    assert(act0.dtype == syn_type_bf16 && (act0.sizes == std::vector<uint64_t>{7168, m}) && act0.section == nullptr);
    assert(actp.dtype == syn_type_bf16 && (actp.sizes == std::vector<uint64_t>{7168, m}) && actp.section == nullptr);
    assert(w0.dtype == syn_type_bf16 && (w0.sizes == std::vector<uint64_t>{7168, 9216}) && w0.section == nullptr);
    assert(wp.dtype == syn_type_bf16 && (wp.sizes == std::vector<uint64_t>{7168, 9216}) && wp.section == nullptr);
    assert_fp8_143_metadata(g_state.tensors.at(dw0.inputs[0]));

    const auto gemms = find_nodes("gemm");
    assert(gemms.size() == 2);
    assert_gemm_nt(*gemms[0]);
    assert_gemm_nt(*gemms[1]);
    assert(gemms[0]->inputs[0] == qa0.outputs[0] && gemms[0]->inputs[1] == dw0.outputs[0]);
    assert(gemms[1]->inputs[0] == qap.outputs[0] && gemms[1]->inputs[1] == dwp.outputs[0]);

    const auto& epi_node = find_single_node("gk_fp8_epilogue_k32_bf16_v1");
    assert(epi_node.inputs.size() == 2 && epi_node.outputs.size() == 1);
    assert(epi_node.inputs[0] == gemms[0]->outputs[0] &&
           epi_node.inputs[1] == gemms[1]->outputs[0]);
    const auto& epi_out = g_state.tensors.at(epi_node.outputs[0]);
    assert(epi_out.dtype == syn_type_bf16 &&
           (epi_out.sizes == std::vector<uint64_t>{9216, m}) &&
           epi_out.section != nullptr && g_state.sections.at(epi_out.section).persistent);
  }

  g_state.clear_graph_records();
  auto out4096 = prog.run({make_dev_tensor({4096, 7168}, at::kBFloat16), w, s});
  assert(out4096.size() == 1 && out4096[0].sizes().equals({4096, 9216}));
  assert(count_guid("split") == 1);
  assert(count_guid("gk_fp8_quant_act_bf16_1k_v1") == 4);
  assert(count_guid("gk_fp8_quant_act_bf16_k32_v1") == 4);
  assert(count_guid("gk_fp8_dequant_weight_bf16_1k_v1") == 4);
  assert(count_guid("gk_fp8_dequant_weight_bf16_k32_v1") == 4);
  assert(count_guid("gemm") == 8);
  assert(count_guid("gk_fp8_epilogue_k32_bf16_v1") == 4);
  assert(count_guid("concat") == 1);
  for (const auto* qa : find_nodes("gk_fp8_quant_act_bf16_1k_v1")) {
    const auto& t = g_state.tensors.at(qa->outputs[0]);
    assert((t.sizes == std::vector<uint64_t>{7168, 1024}) && t.section == nullptr);
  }

  // K = 22016 (172 groups, Down): SRAM budget selects tile_rows = 512 -> 2 tiles at M=1024, 8 tiles at M=4096.
  BucketProgram prog_down({1, 0, 0}, {FxNode{FxOp::block_fp8_linear, {0, 1, 2}, std::monostate{}}});
  auto w_down = make_dev_tensor({172, 7168, 128}, at::kFloat8_e4m3fn);
  auto s_down = make_dev_tensor({56, 172}, at::kFloat);
  g_state.clear_graph_records();
  auto down1024 = prog_down.run({make_dev_tensor({1024, 22016}, at::kBFloat16), w_down, s_down});
  assert(down1024[0].sizes().equals({1024, 7168}));
  assert(count_guid("gk_fp8_quant_act_bf16_1k_v1") == 2);
  assert(count_guid("gemm") == 4);
  for (const auto* qa : find_nodes("gk_fp8_quant_act_bf16_1k_v1")) {
    const auto& t = g_state.tensors.at(qa->outputs[0]);
    assert((t.sizes == std::vector<uint64_t>{22016, 512}) && t.section == nullptr);
  }

  assert(!g_state.compiled_keys.empty());
  auto geom = infer_fx_geometry(
      {make_dev_tensor({128, 7168}, at::kBFloat16), w, s}, {1, 0, 0},
      {FxNode{FxOp::block_fp8_linear, {0, 1, 2}, std::monostate{}}}, {});
  assert(!geom.digest.empty());
}

void test_prefill_k32_biased_bf16_and_fp32() {
  for (at::ScalarType bias_dtype : {at::kBFloat16, at::kFloat}) {
    BucketProgram prog({1, 0, 0, 0},
                       {FxNode{FxOp::block_fp8_linear, {0, 1, 2, 3}, std::monostate{}}});
    auto w = make_dev_tensor({4, 512, 128}, at::kFloat8_e4m3fn);
    auto s = make_dev_tensor({4, 4}, at::kFloat);
    auto b = make_dev_tensor({512}, bias_dtype);

    for (uint64_t m : {512ULL, 1024ULL, 4096ULL}) {
      g_state.clear_graph_records();
      auto x = make_dev_tensor({static_cast<int64_t>(m), 512}, at::kBFloat16);
      auto out = prog.run({x, w, s, b});
      assert(out.size() == 1 && out[0].sizes().equals({static_cast<int64_t>(m), 512}));
      assert(count_guid("cast_bf16_to_f32") == (bias_dtype == at::kBFloat16 ? 1 : 0));
      assert(count_guid("gk_fp8_quant_act_bf16_1k_v1") == 1);
      assert(count_guid("gk_fp8_quant_act_bf16_k32_v1") == 1);
      assert(count_guid("gk_fp8_dequant_weight_bf16_1k_v1") == 1);
      assert(count_guid("gk_fp8_dequant_weight_bf16_k32_v1") == 1);
      assert(count_guid("gemm") == 2);
      assert(count_guid("gk_fp8_epilogue_k32_bf16_v1") == 0);

      const auto& epi_node = find_single_node("gk_fp8_epilogue_k32_bias_bf16_v1");
      assert(epi_node.inputs.size() == 3 && epi_node.outputs.size() == 1);
      const auto& bias_in = g_state.tensors.at(epi_node.inputs[2]);
      assert(bias_in.dtype == syn_type_float && (bias_in.sizes == std::vector<uint64_t>{512}));
      const auto& final_out = g_state.tensors.at(epi_node.outputs[0]);
      assert(final_out.dtype == syn_type_bf16 &&
             (final_out.sizes == std::vector<uint64_t>{512, m}) &&
             final_out.section != nullptr && g_state.sections.at(final_out.section).persistent);
    }
  }
}

void test_3d_activation_threshold_and_reshapes() {
  BucketProgram prog({1, 0, 0}, {FxNode{FxOp::block_fp8_linear, {0, 1, 2}, std::monostate{}}});
  auto w = make_dev_tensor({2, 256, 128}, at::kFloat8_e4m3fn);
  auto s = make_dev_tensor({2, 2}, at::kFloat);

  // [128, 2, 256] -> bucket=128, token_rows=256 -> stays on batch_gemm
  g_state.clear_graph_records();
  auto out256 = prog.run({make_dev_tensor({128, 2, 256}, at::kBFloat16), w, s});
  assert(out256[0].sizes().equals({128, 2, 256}));
  assert(count_guid("batch_gemm") == 1);
  assert(count_guid("gk_fp8_quant_act_bf16_1k_v1") == 0);

  // [256, 2, 256] -> bucket=256, token_rows=512 -> takes Karatsuba-32 2K with 2 reshapes
  g_state.clear_graph_records();
  auto out512 = prog.run({make_dev_tensor({256, 2, 256}, at::kBFloat16), w, s});
  assert(out512[0].sizes().equals({256, 2, 256}));
  assert(count_guid("reshape") == 2);
  assert(count_guid("gk_fp8_quant_act_bf16_1k_v1") == 1);
  assert(count_guid("gk_fp8_quant_act_bf16_k32_v1") == 1);
  assert(count_guid("gk_fp8_dequant_weight_bf16_1k_v1") == 1);
  assert(count_guid("gk_fp8_dequant_weight_bf16_k32_v1") == 1);
  assert(count_guid("gemm") == 2);
  assert(count_guid("gk_fp8_epilogue_k32_bf16_v1") == 1);
}

void test_large_n_accum_budget_column_tiling() {
  // K = 7168 (56 groups), N = 44032 (gate_up):
  // - At M <= 1024: row_tiles = 1, column_tiles = 1.
  // - At M = 4096: row_tiles = 4 (tile_rows = 1024), column_tiles = 2 (tile_columns = 22016),
  //   with dedicated 1-output TPC producers per (row_tile, col_tile) GEMM.
  BucketProgram prog({1, 0, 0, 0},
                     {FxNode{FxOp::block_fp8_linear, {0, 1, 2, 3}, std::monostate{}}});
  auto w = make_dev_tensor({56, 44032, 128}, at::kFloat8_e4m3fn);
  auto s = make_dev_tensor({344, 56}, at::kFloat);
  auto b = make_dev_tensor({44032}, at::kBFloat16);

  for (uint64_t m : {512ULL, 1024ULL}) {
    g_state.clear_graph_records();
    auto x = make_dev_tensor({static_cast<int64_t>(m), 7168}, at::kBFloat16);
    auto out = prog.run({x, w, s, b});
    assert(out[0].sizes().equals({static_cast<int64_t>(m), 44032}));
    assert(count_guid("split") == 0);
    assert(count_guid("gk_fp8_quant_act_bf16_1k_v1") == 1);
    assert(count_guid("gk_fp8_quant_act_bf16_k32_v1") == 1);
    assert(count_guid("gk_fp8_dequant_weight_bf16_1k_v1") == 1);
    assert(count_guid("gk_fp8_dequant_weight_bf16_k32_v1") == 1);
    assert(count_guid("gemm") == 2);
    assert(count_guid("gk_fp8_epilogue_k32_bias_bf16_v1") == 1);
    assert(count_guid("concat") == 0);
  }

  g_state.clear_graph_records();
  auto x4096 = make_dev_tensor({4096, 7168}, at::kBFloat16);
  auto out4096 = prog.run({x4096, w, s, b});
  assert(out4096[0].sizes().equals({4096, 44032}));
  assert(count_guid("split") == 4);
  assert(count_guid("gk_fp8_quant_act_bf16_1k_v1") == 8);
  assert(count_guid("gk_fp8_quant_act_bf16_k32_v1") == 8);
  assert(count_guid("gk_fp8_dequant_weight_bf16_1k_v1") == 8);
  assert(count_guid("gk_fp8_dequant_weight_bf16_k32_v1") == 8);
  assert(count_guid("gemm") == 16);
  assert(count_guid("gk_fp8_epilogue_k32_bias_bf16_v1") == 8);
  assert(count_guid("concat") == 5);
  for (const auto* dw : find_nodes("gk_fp8_dequant_weight_bf16_1k_v1")) {
    const auto& t = g_state.tensors.at(dw->outputs[0]);
    assert((t.sizes == std::vector<uint64_t>{7168, 22016}) && t.section == nullptr);
  }
}

}  // namespace

int main() {
  gaudirt::generic::init_mock_backend();
  gaudirt::generic::g_ctx =
      std::make_shared<gaudirt::generic::Context>(0, 32ULL << 30, 4ULL << 20);

  test_small_buckets_keep_batch_gemm();
  test_prefill_k32_unbiased_and_sram_row_tiling();
  test_prefill_k32_biased_bf16_and_fp32();
  test_3d_activation_threshold_and_reshapes();
  test_large_n_accum_budget_column_tiling();

  gaudirt::generic::g_ctx.reset();
  std::cout << "block_fp8_hybrid_lowering_unit_test PASSED (" << g_state.launches
            << " mock launches)\n";
  return 0;
}
