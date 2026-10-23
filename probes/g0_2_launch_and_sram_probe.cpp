#include "gaudirt/common/abba_harness.hpp"
#include "gaudirt/common/host_policy.hpp"
#include "gaudirt/common/synapse_raii.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace gaudirt::common;

namespace {

struct CompiledRecipe {
  synDeviceId dev = 0;
  synRecipeHandle handle = nullptr;
  std::vector<synLaunchTensorInfoExt> bindings;
  std::vector<std::string> names;
  uint64_t workspace = 0;
  uint64_t workspace_bytes = 0;

  void destroy() {
    for (auto& b : bindings) {
      if (b.pTensorAddress) synDeviceFree(dev, b.pTensorAddress, 0);
    }
    if (workspace) synDeviceFree(dev, workspace, 0);
    if (handle) synRecipeDestroy(handle);
  }
};

// Build a 122ms BF16 GEMM gate recipe to hold the compute stream while measuring pre-block launch cost
CompiledRecipe make_gate_recipe(synDeviceId dev, unsigned count = 48, unsigned dim = 8192) {
  CompiledRecipe r;
  r.dev = dev;
  synGraphHandle graph = nullptr;
  GAUDIRT_SYN_CHECK(synGraphCreate(&graph, synDeviceGaudi2));
  r.names = {"gate_in", "gate_w", "gate_out"};
  std::vector<synSectionHandle> sections;

  auto mk = [&](const char* name, bool persistent) {
    synSectionHandle s = nullptr;
    if (persistent) {
      GAUDIRT_SYN_CHECK(synSectionCreate(&s, 0, graph));
      GAUDIRT_SYN_CHECK(synSectionSetPersistent(s, true));
      sections.push_back(s);
    }
    synTensorDescriptor d{};
    d.m_name = name;
    d.m_dataType = syn_type_bf16;
    d.m_dims = 2;
    d.m_sizes[0] = d.m_minSizes[0] = dim;
    d.m_sizes[1] = d.m_minSizes[1] = dim;
    synTensor t = nullptr;
    GAUDIRT_SYN_CHECK(synTensorCreate(&t, &d, s, 0));
    return t;
  };

  synTensor x = mk("gate_in", true), w = mk("gate_w", true), out = mk("gate_out", true);
  synTensor cur = x;
  std::vector<std::string> mid_names;
  for (unsigned i = 0; i < count; ++i) mid_names.push_back("gate_mid_" + std::to_string(i));
  for (unsigned i = 0; i < count; ++i) {
    synTensor next = (i + 1 == count) ? out : mk(mid_names[i].c_str(), false);
    synTensor in[2] = {cur, w};
    synGEMMParams p{false, false};
    GAUDIRT_SYN_CHECK(synNodeCreate(graph, in, &next, 2, 1, &p, sizeof(p), "gemm",
                                    ("gate_gemm_" + std::to_string(i)).c_str(), nullptr, nullptr));
    cur = next;
  }

  GAUDIRT_SYN_CHECK(synGraphCompile(&r.handle, graph, "g0_2_gate", nullptr));
  const char* cn[3] = {"gate_in", "gate_w", "gate_out"};
  uint64_t ids[3]{};
  GAUDIRT_SYN_CHECK(synTensorRetrieveIds(r.handle, cn, ids, 3));
  r.bindings.resize(3);
  for (int i = 0; i < 3; ++i) {
    GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, uint64_t(dim) * dim * 2, 0, 0, &r.bindings[i].pTensorAddress));
    r.bindings[i].tensorId = ids[i];
    r.bindings[i].tensorName = r.names[i].c_str();
    r.bindings[i].tensorType = DATA_TENSOR;
    r.bindings[i].tensorSize[0] = dim;
    r.bindings[i].tensorSize[1] = dim;
  }
  GAUDIRT_SYN_CHECK(synWorkspaceGetSize(&r.workspace_bytes, r.handle));
  if (r.workspace_bytes) {
    GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, r.workspace_bytes, 0, 0, &r.workspace));
  }
  GAUDIRT_SYN_CHECK(synGraphDestroy(graph));
  for (auto s : sections) GAUDIRT_SYN_CHECK(synSectionDestroy(s));
  return r;
}

// Build a chain of `num_nodes` add_fwd_f32 nodes with exact `target_bindings` persistent tensors.
// For num_nodes nodes, there are:
// - 1 initial input `x0` (always persistent)
// - `num_nodes` RHS inputs `b_0 .. b_{num_nodes-1}` (either shared `b0` or distinct persistent)
// - `num_nodes - 1` intermediate outputs `m_0 .. m_{num_nodes-2}` (either non-persistent or persistent)
// - 1 final output `out` (always persistent)
// Minimum persistent bindings = 3 (`x0`, `b_shared`, `out`).
// Maximum persistent bindings = 2 * num_nodes + 1 (if all `b_k` and `m_k` are persistent) or 3 * num_nodes (independent nodes).
CompiledRecipe make_decoupled_recipe(synDeviceId dev, unsigned num_nodes, unsigned target_bindings, unsigned elem_cnt = 4096) {
  CompiledRecipe r;
  r.dev = dev;
  synGraphHandle graph = nullptr;
  GAUDIRT_SYN_CHECK(synGraphCreate(&graph, synDeviceGaudi2));
  std::vector<synSectionHandle> sections;
  std::vector<std::string> all_names;
  all_names.reserve(num_nodes * 3 + 4);

  auto make_tensor = [&](const std::string& name, bool persistent) {
    all_names.push_back(name);
    synSectionHandle s = nullptr;
    if (persistent) {
      GAUDIRT_SYN_CHECK(synSectionCreate(&s, 0, graph));
      GAUDIRT_SYN_CHECK(synSectionSetPersistent(s, true));
      sections.push_back(s);
      r.names.push_back(name);
    }
    synTensorDescriptor d{};
    d.m_name = all_names.back().c_str();
    d.m_dataType = syn_type_single;
    d.m_dims = 1;
    d.m_sizes[0] = d.m_minSizes[0] = elem_cnt;
    synTensor t = nullptr;
    GAUDIRT_SYN_CHECK(synTensorCreate(&t, &d, s, 0));
    return t;
  };

  // Start with 3 mandatory persistent tensors: x0, b0, out
  synTensor x0 = make_tensor("x0", true);
  synTensor b0 = make_tensor("b0", true);
  synTensor out = make_tensor("out", true);
  unsigned extra_needed = (target_bindings > 3) ? (target_bindings - 3) : 0;

  synTensor cur = x0;
  for (unsigned k = 0; k < num_nodes; ++k) {
    synTensor rhs = b0;
    if (k > 0 && extra_needed > 0) {
      rhs = make_tensor("b_" + std::to_string(k), true);
      --extra_needed;
    }
    synTensor dst = out;
    if (k + 1 < num_nodes) {
      bool persist_mid = (extra_needed > 0);
      if (persist_mid) --extra_needed;
      dst = make_tensor("mid_" + std::to_string(k), persist_mid);
    }
    synTensor in[2] = {cur, rhs};
    GAUDIRT_SYN_CHECK(synNodeCreate(graph, in, &dst, 2, 1, nullptr, 0, "add_fwd_f32",
                                    ("add_" + std::to_string(k)).c_str(), nullptr, nullptr));
    cur = dst;
  }

  std::string rname = "g0_2_n" + std::to_string(num_nodes) + "_b" + std::to_string(r.names.size());
  GAUDIRT_SYN_CHECK(synGraphCompile(&r.handle, graph, rname.c_str(), nullptr));

  std::vector<const char*> cnames;
  for (const auto& s : r.names) cnames.push_back(s.c_str());
  std::vector<uint64_t> ids(cnames.size());
  GAUDIRT_SYN_CHECK(synTensorRetrieveIds(r.handle, cnames.data(), ids.data(), cnames.size()));
  r.bindings.resize(cnames.size());
  for (size_t i = 0; i < cnames.size(); ++i) {
    GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, uint64_t(elem_cnt) * 4, 0, 0, &r.bindings[i].pTensorAddress));
    r.bindings[i].tensorId = ids[i];
    r.bindings[i].tensorName = r.names[i].c_str();
    r.bindings[i].tensorType = DATA_TENSOR;
    r.bindings[i].tensorSize[0] = elem_cnt;
  }
  GAUDIRT_SYN_CHECK(synWorkspaceGetSize(&r.workspace_bytes, r.handle));
  if (r.workspace_bytes) {
    GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, r.workspace_bytes, 0, 0, &r.workspace));
  }
  GAUDIRT_SYN_CHECK(synGraphDestroy(graph));
  for (auto s : sections) GAUDIRT_SYN_CHECK(synSectionDestroy(s));
  return r;
}

SummaryStats measure_prequeue_launch_us(synStreamHandle stream, CompiledRecipe& gate, CompiledRecipe& target, unsigned count = 400) {
  for (int i = 0; i < 32; ++i) {
    GAUDIRT_SYN_CHECK(synLaunchWithExternalEventsExt(stream, target.bindings.data(), target.bindings.size(),
                                                     target.workspace, target.handle, nullptr, 0, 0));
  }
  GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));

  std::vector<double> samples;
  samples.reserve(count);
  GAUDIRT_SYN_CHECK(synLaunchWithExternalEventsExt(stream, gate.bindings.data(), gate.bindings.size(),
                                                   gate.workspace, gate.handle, nullptr, 0, 0));
  for (unsigned i = 0; i < count; ++i) {
    double t0 = now_us();
    GAUDIRT_SYN_CHECK(synLaunchWithExternalEventsExt(stream, target.bindings.data(), target.bindings.size(),
                                                     target.workspace, target.handle, nullptr, 0, 0));
    samples.push_back(now_us() - t0);
  }
  GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));
  return summarize(samples);
}

struct SramProbeResult {
  uint64_t mid_bytes = 0;
  bool rmw = false;
  std::string allocation = "UNKNOWN";
  uint64_t workspace_bytes = 0;
  double device_p50_us = 0.0;
};

SramProbeResult run_sram_case(synDeviceId dev, synStreamHandle stream, unsigned k, unsigned m, unsigned n, bool rmw) {
  SramProbeResult res{};
  res.mid_bytes = 2ull * k * m;
  res.rmw = rmw;

  setenv("HABANA_LOGS", "/tmp/g0_2_logs", 1);
  std::string tag = "sram_k" + std::to_string(k) + "_m" + std::to_string(m) + (rmw ? "_rmw1" : "_rmw0");
  std::string rm_cmd = "rm -f .graph_dumps/*" + tag + "*.json 2>/dev/null";
  int rc = std::system(rm_cmd.c_str());
  (void)rc;

  const unsigned sizes[5][2] = {{k, m}, {k, m}, {n, k}, {n, m}, {k, m}};
  const char* names[5] = {"x", "b", "weight", "y", "sram_sum"};
  synGraphHandle graph = nullptr;
  GAUDIRT_SYN_CHECK(synGraphCreate(&graph, synDeviceGaudi2));
  synSectionHandle sections[5]{};
  synTensor tensors[5]{};
  for (unsigned i = 0; i < 5; ++i) {
    GAUDIRT_SYN_CHECK(synSectionCreate(&sections[i], 0, graph));
    GAUDIRT_SYN_CHECK(synSectionSetPersistent(sections[i], i < 4));
    if (i == 4) {
      GAUDIRT_SYN_CHECK(synSectionSetRMW(sections[i], rmw));
    }
    synTensorDescriptor d{};
    d.m_name = names[i];
    d.m_dataType = syn_type_bf16;
    d.m_dims = 2;
    for (unsigned j = 0; j < 2; ++j) d.m_sizes[j] = d.m_minSizes[j] = sizes[i][j];
    synStatus tst = synTensorCreate(&tensors[i], &d, sections[i], 0);
    if (tst != synSuccess) {
      res.allocation = "REJECTED_TENSOR_CREATE_" + std::to_string(int(tst));
      synGraphDestroy(graph);
      for (unsigned k_idx = 0; k_idx <= i; ++k_idx) synSectionDestroy(sections[k_idx]);
      return res;
    }
  }
  synStatus n1_st = synNodeCreate(graph, tensors, &tensors[4], 2, 1, nullptr, 0, "add_fwd_bf16", "producer_add", nullptr, nullptr);
  if (n1_st != synSuccess) {
    res.allocation = "REJECTED_OVER_RMW_LIMIT_STATUS_" + std::to_string(int(n1_st));
    synGraphDestroy(graph);
    for (auto s : sections) synSectionDestroy(s);
    return res;
  }
  synTensor gemm_in[2] = {tensors[4], tensors[2]};
  synGEMMParams params{false, false};
  GAUDIRT_SYN_CHECK(synNodeCreate(graph, gemm_in, &tensors[3], 2, 1, &params, sizeof(params), "gemm", "consumer_mme", nullptr, nullptr));

  synRecipeHandle recipe = nullptr;
  synStatus compile_st = synGraphCompile(&recipe, graph, tag.c_str(), nullptr);

  if (compile_st != synSuccess) {
    res.allocation = "COMPILE_FAILED_" + std::to_string(int(compile_st));
    synGraphDestroy(graph);
    for (auto s : sections) synSectionDestroy(s);
    return res;
  }

  uint64_t ids[4]{};
  GAUDIRT_SYN_CHECK(synTensorRetrieveIds(recipe, names, ids, 4));
  synLaunchTensorInfoExt bindings[4]{};
  for (unsigned i = 0; i < 4; ++i) {
    uint64_t bytes = 2ull * sizes[i][0] * sizes[i][1];
    GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, bytes, 0, 0, &bindings[i].pTensorAddress));
    bindings[i].tensorId = ids[i];
    bindings[i].tensorType = DATA_TENSOR;
    bindings[i].tensorName = names[i];
    for (unsigned j = 0; j < 2; ++j) bindings[i].tensorSize[j] = sizes[i][j];
  }
  uint64_t workspace = 0;
  GAUDIRT_SYN_CHECK(synWorkspaceGetSize(&res.workspace_bytes, recipe));
  if (res.workspace_bytes) {
    GAUDIRT_SYN_CHECK(synDeviceMalloc(dev, res.workspace_bytes, 0, 0, &workspace));
  }
  res.allocation = (res.workspace_bytes < res.mid_bytes) ? "SRAM" : "DRAM_WORKSPACE";

  for (int i = 0; i < 10; ++i) {
    GAUDIRT_SYN_CHECK(synLaunchWithExternalEventsExt(stream, bindings, 4, workspace, recipe, nullptr, 0, 0));
  }
  GAUDIRT_SYN_CHECK(synStreamSynchronize(stream));

  TimedEventPair timer(dev);
  std::vector<double> dev_us;
  for (int i = 0; i < 30; ++i) {
    timer.record_start(stream);
    GAUDIRT_SYN_CHECK(synLaunchWithExternalEventsExt(stream, bindings, 4, workspace, recipe, nullptr, 0, 0));
    timer.record_end(stream);
    dev_us.push_back(timer.elapsed_us());
  }
  res.device_p50_us = summarize(dev_us).p50;

  for (unsigned i = 0; i < 4; ++i) synDeviceFree(dev, bindings[i].pTensorAddress, 0);
  if (workspace) synDeviceFree(dev, workspace, 0);
  synRecipeDestroy(recipe);
  synGraphDestroy(graph);
  for (auto s : sections) synSectionDestroy(s);
  return res;
}

} // namespace

int main(int argc, char** argv) {
  const int module_id = (argc > 1) ? std::atoi(argv[1]) : 7;
  auto host_st = lock_host_policy(-1, 1024);
  if (has_unwanted_instrumentation()) {
    std::fprintf(stderr, "[G0-2] ERROR: Unwanted instrumentation/profiler detected in environment!\n");
    return 1;
  }

  SynapseContext ctx(module_id);
  synStreamHandle stream = nullptr;
  GAUDIRT_SYN_CHECK(synStreamCreateGeneric(&stream, ctx.device(), 0));

  char rwm_cfg[128] = {0};
  synConfigurationGet("GCFG_RMW_SECTION_MAX_SIZE_BYTES", rwm_cfg, sizeof(rwm_cfg));

  CompiledRecipe gate = make_gate_recipe(ctx.device(), 48, 8192);

  std::printf("{\n");
  std::printf("  \"probe\": \"G0-2_launch_and_sram\",\n");
  std::printf("  \"module_id\": %d,\n", module_id);
  std::printf("  \"uclamp_applied\": %s,\n", host_st.uclamp_applied ? "true" : "false");
  std::printf("  \"pinned_cpu\": %d,\n", host_st.pinned_cpu);
  std::printf("  \"GCFG_RMW_SECTION_MAX_SIZE_BYTES\": \"%s\",\n", rwm_cfg);

  // Series A1: Fixed bindings = 3, Variable nodes = 1, 4, 8, 16, 32
  std::printf("  \"fixed_bindings_3_variable_nodes\": [\n");
  const unsigned node_list[] = {1, 4, 8, 16, 32};
  for (size_t i = 0; i < sizeof(node_list) / sizeof(node_list[0]); ++i) {
    unsigned n_nodes = node_list[i];
    CompiledRecipe r = make_decoupled_recipe(ctx.device(), n_nodes, 3);
    SummaryStats st = measure_prequeue_launch_us(stream, gate, r, 400);
    std::printf("    {\"nodes\": %u, \"bindings\": %zu, \"p50_us\": %.3f, \"p90_us\": %.3f, \"mean_us\": %.3f}%s\n",
                n_nodes, r.bindings.size(), st.p50, st.p90, st.mean,
                (i + 1 < sizeof(node_list) / sizeof(node_list[0])) ? "," : "");
    r.destroy();
  }
  std::printf("  ],\n");

  // Series A2: Fixed nodes = 16, Variable bindings = 3, 8, 16, 24, 32
  std::printf("  \"fixed_nodes_16_variable_bindings\": [\n");
  const unsigned bind_list[] = {3, 8, 16, 24, 32};
  for (size_t i = 0; i < sizeof(bind_list) / sizeof(bind_list[0]); ++i) {
    unsigned n_bind = bind_list[i];
    CompiledRecipe r = make_decoupled_recipe(ctx.device(), 16, n_bind);
    SummaryStats st = measure_prequeue_launch_us(stream, gate, r, 400);
    std::printf("    {\"nodes\": 16, \"bindings\": %zu, \"p50_us\": %.3f, \"p90_us\": %.3f, \"mean_us\": %.3f}%s\n",
                r.bindings.size(), st.p50, st.p90, st.mean,
                (i + 1 < sizeof(bind_list) / sizeof(bind_list[0])) ? "," : "");
    r.destroy();
  }
  std::printf("  ],\n");

  gate.destroy();

  // Part B: SRAM RMW=false vs RMW=true across sizes
  struct ShapeCase { unsigned k; unsigned m; unsigned n; };
  const ShapeCase shapes[] = {
      {64, 16, 64},       // 2 KB
      {512, 64, 512},     // 64 KB
      {1024, 256, 512},   // 512 KB
      {2048, 512, 512},   // 2 MB
      {4096, 1024, 512},  // 8 MB
      {4096, 1536, 512},  // 12 MB
      {4096, 2048, 512},  // 16 MB
      {4096, 2560, 512},  // 20 MB
      {4096, 3072, 512},  // 24 MB
  };
  std::printf("  \"sram_rmw_matrix\": [\n");
  size_t total_cases = (sizeof(shapes) / sizeof(shapes[0])) * 2;
  size_t idx = 0;
  for (const auto& sc : shapes) {
    for (bool rmw : {false, true}) {
      SramProbeResult sr = run_sram_case(ctx.device(), stream, sc.k, sc.m, sc.n, rmw);
      ++idx;
      std::printf("    {\"mid_bytes\": %llu, \"mid_kb\": %.1f, \"rmw\": %s, \"allocation\": \"%s\", \"workspace_bytes\": %llu, \"device_p50_us\": %.2f}%s\n",
                  (unsigned long long)sr.mid_bytes, double(sr.mid_bytes) / 1024.0,
                  sr.rmw ? "true" : "false", sr.allocation.c_str(),
                  (unsigned long long)sr.workspace_bytes, sr.device_p50_us,
                  (idx < total_cases) ? "," : "");
    }
  }
  std::printf("  ]\n");
  std::printf("}\n");

  synStreamDestroy(stream);
  return 0;
}
