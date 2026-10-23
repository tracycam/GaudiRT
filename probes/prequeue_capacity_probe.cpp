// Pure C++ Synapse (SDK 1.24.2-481) in-flight capacity probe.
// A long MME "gate" recipe occupies the compute stream; then small recipes are
// enqueued back-to-back until synLaunchWithExternalEventsExt blocks.
// Varies the small recipe size (binding count / node count) and recipe identity
// to discriminate count-limited vs byte-limited in-flight resources.
// Optional: on the first long block, raise SIGUSR1 so a parent gdb can capture
// all-thread backtraces (ptrace_scope=1 => gdb must be the parent).
#include <synapse_api.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <string>
#include <thread>
#include <vector>

#define CK(call) do { auto s_ = (call); if (s_ != synSuccess) { \
    std::fprintf(stderr, "%s failed: %d at line %d\n", #call, int(s_), __LINE__); std::exit(2); } } while (0)

using Clock = std::chrono::steady_clock;
static double now_us() {
    return std::chrono::duration<double, std::micro>(Clock::now().time_since_epoch()).count();
}

struct Recipe {
    synRecipeHandle handle{};
    std::vector<synLaunchTensorInfoExt> bindings;
    std::vector<std::string> names;
    uint64_t workspace = 0, workspace_bytes = 0;
};

static synDeviceId g_device;

// Chain of `nodes` FP32 add nodes; every node reads two persistent inputs and
// writes one persistent output (so bindings = 3*nodes). Each element count n.
static Recipe make_small(unsigned nodes, unsigned n, const char* tag) {
    Recipe r;
    synGraphHandle graph;
    CK(synGraphCreate(&graph, synDeviceGaudi2));
    std::vector<synSectionHandle> sections;
    std::vector<synTensor> tensors;
    for (unsigned i = 0; i < nodes * 3; ++i) r.names.push_back(std::string(tag) + "_t" + std::to_string(i));
    for (unsigned i = 0; i < nodes * 3; ++i) {
        synSectionHandle s; CK(synSectionCreate(&s, 0, graph)); CK(synSectionSetPersistent(s, true));
        sections.push_back(s);
        synTensorDescriptor d{}; d.m_name = r.names[i].c_str(); d.m_dataType = syn_type_single; d.m_dims = 1;
        d.m_sizes[0] = d.m_minSizes[0] = n;
        synTensor t; CK(synTensorCreate(&t, &d, s, 0)); tensors.push_back(t);
    }
    for (unsigned k = 0; k < nodes; ++k) {
        synTensor in[2] = {tensors[3*k], tensors[3*k+1]};
        CK(synNodeCreate(graph, in, &tensors[3*k+2], 2, 1, nullptr, 0, "add_fwd_f32",
                         (std::string(tag) + "_add" + std::to_string(k)).c_str(), nullptr, nullptr));
    }
    CK(synGraphCompile(&r.handle, graph, tag, nullptr));
    std::vector<const char*> cn; for (auto& s : r.names) cn.push_back(s.c_str());
    std::vector<uint64_t> ids(cn.size());
    CK(synTensorRetrieveIds(r.handle, cn.data(), ids.data(), cn.size()));
    r.bindings.resize(cn.size());
    for (size_t i = 0; i < cn.size(); ++i) {
        CK(synDeviceMalloc(g_device, uint64_t(n) * 4, 0, 0, &r.bindings[i].pTensorAddress));
        r.bindings[i].tensorId = ids[i]; r.bindings[i].tensorName = r.names[i].c_str();
        r.bindings[i].tensorType = DATA_TENSOR; r.bindings[i].tensorSize[0] = n;
    }
    CK(synWorkspaceGetSize(&r.workspace_bytes, r.handle));
    if (r.workspace_bytes) CK(synDeviceMalloc(g_device, r.workspace_bytes, 0, 0, &r.workspace));
    CK(synGraphDestroy(graph));
    for (auto s : sections) CK(synSectionDestroy(s));
    return r;
}

// Chain of `count` BF16 GEMMs [dim x dim] with non-persistent intermediates.
static Recipe make_gate(unsigned count, unsigned dim) {
    Recipe r;
    synGraphHandle graph;
    CK(synGraphCreate(&graph, synDeviceGaudi2));
    r.names = {"gate_in", "gate_w", "gate_out"};
    std::vector<synSectionHandle> sections;
    auto mk = [&](const char* name, bool persistent) {
        synSectionHandle s = nullptr;
        if (persistent) { CK(synSectionCreate(&s, 0, graph)); CK(synSectionSetPersistent(s, true)); sections.push_back(s); }
        synTensorDescriptor d{}; d.m_name = name; d.m_dataType = syn_type_bf16; d.m_dims = 2;
        d.m_sizes[0] = d.m_minSizes[0] = dim; d.m_sizes[1] = d.m_minSizes[1] = dim;
        synTensor t; CK(synTensorCreate(&t, &d, s, 0)); return t;
    };
    synTensor x = mk("gate_in", true), w = mk("gate_w", true), out = mk("gate_out", true);
    synTensor cur = x;
    std::vector<std::string> inames;
    for (unsigned i = 0; i < count; ++i) inames.push_back("gate_mid" + std::to_string(i));
    for (unsigned i = 0; i < count; ++i) {
        synTensor next = (i + 1 == count) ? out : mk(inames[i].c_str(), false);
        synTensor in[2] = {cur, w}; synGEMMParams p{false, false};
        CK(synNodeCreate(graph, in, &next, 2, 1, &p, sizeof(p), "gemm",
                         ("gate_gemm" + std::to_string(i)).c_str(), nullptr, nullptr));
        cur = next;
    }
    CK(synGraphCompile(&r.handle, graph, "gate_chain", nullptr));
    const char* cn[3] = {"gate_in", "gate_w", "gate_out"};
    uint64_t ids[3]; CK(synTensorRetrieveIds(r.handle, cn, ids, 3));
    r.bindings.resize(3);
    for (int i = 0; i < 3; ++i) {
        CK(synDeviceMalloc(g_device, uint64_t(dim) * dim * 2, 0, 0, &r.bindings[i].pTensorAddress));
        r.bindings[i].tensorId = ids[i]; r.bindings[i].tensorName = cn[i];
        r.bindings[i].tensorType = DATA_TENSOR; r.bindings[i].tensorSize[0] = dim; r.bindings[i].tensorSize[1] = dim;
    }
    CK(synWorkspaceGetSize(&r.workspace_bytes, r.handle));
    if (r.workspace_bytes) CK(synDeviceMalloc(g_device, r.workspace_bytes, 0, 0, &r.workspace));
    CK(synGraphDestroy(graph));
    for (auto s : sections) CK(synSectionDestroy(s));
    return r;
}

static void launch(synStreamHandle st, Recipe& r) {
    CK(synLaunchWithExternalEventsExt(st, r.bindings.data(), r.bindings.size(), r.workspace, r.handle, nullptr, 0, 0));
}

static std::atomic<double> g_inflight_start{0};
static std::atomic<bool> g_armed{false};
static pthread_t g_main;

int main(int argc, char** argv) {
    const int module = argc > 1 ? std::atoi(argv[1]) : 7;
    const bool want_trap = argc > 2 && std::strcmp(argv[2], "trap") == 0;
    const unsigned total = 4000;
    CK(synInitialize());
    CK(synDeviceAcquireByModuleId(&g_device, module));
    synStreamHandle stream; CK(synStreamCreateGeneric(&stream, g_device, 0));
    g_main = pthread_self();
    std::thread watcher([&] {
        while (true) {
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            double s = g_inflight_start.load();
            if (s < 0) break;
            if (want_trap && g_armed.load() && s > 0 && now_us() - s > 5000) {
                g_armed = false;
                std::printf("TRAP raising SIGUSR1 during blocked launch\n"); std::fflush(stdout);
                pthread_kill(g_main, SIGUSR1);
            }
        }
    });
    std::signal(SIGUSR1, SIG_IGN);  // gdb intercepts first; without gdb it is ignored

    Recipe gate = make_gate(48, 8192);
    std::printf("gate workspace=%llu\n", (unsigned long long)gate.workspace_bytes);
    // Gate-only duration.
    for (int i = 0; i < 2; ++i) { launch(stream, gate); CK(synStreamSynchronize(stream)); }
    double g0 = now_us(); launch(stream, gate); CK(synStreamSynchronize(stream));
    std::printf("gate_only_us=%.1f\n", now_us() - g0);

    struct Cfg { const char* label; unsigned nodes; unsigned n; bool alternate; };
    const Cfg cfgs[] = {
        {"small_1node_3bind", 1, 4096, false},
        {"small_1node_3bind_alt2recipes", 1, 4096, true},
        {"small_8node_24bind", 8, 4096, false},
        {"small_32node_96bind", 32, 4096, false},
    };
    for (const auto& cfg : cfgs) {
        Recipe a = make_small(cfg.nodes, cfg.n, (std::string(cfg.label) + "_a").c_str());
        Recipe b = cfg.alternate ? make_small(cfg.nodes, cfg.n, (std::string(cfg.label) + "_b").c_str()) : Recipe{};
        for (int i = 0; i < 64; ++i) { launch(stream, a); if (cfg.alternate) launch(stream, b); }
        CK(synStreamSynchronize(stream));
        for (int rep = 0; rep < 2; ++rep) {
            std::vector<double> dur(total);
            double t0 = now_us();
            launch(stream, gate);
            double tg = now_us() - t0;
            if (rep == 0 && cfg.nodes == 1 && !cfg.alternate) g_armed = true;
            for (unsigned i = 0; i < total; ++i) {
                double s = now_us();
                g_inflight_start = s;
                launch(stream, (cfg.alternate && (i & 1)) ? b : a);
                dur[i] = now_us() - s;
                g_inflight_start = 0;
            }
            double submit = now_us() - t0;
            CK(synStreamSynchronize(stream));
            double done = now_us() - t0;
            int first_block = -1; double maxd = 0; int maxi = -1; unsigned over1ms = 0;
            for (unsigned i = 0; i < total; ++i) {
                if (dur[i] > 1000 && first_block < 0) first_block = int(i);
                if (dur[i] > 1000) ++over1ms;
                if (dur[i] > maxd) { maxd = dur[i]; maxi = int(i); }
            }
            // Median of the pre-block launches.
            std::vector<double> pre(dur.begin(), dur.begin() + (first_block > 0 ? first_block : total));
            std::sort(pre.begin(), pre.end());
            std::printf("cfg=%s rep=%d bindings=%zu ws=%llu gate_launch_us=%.1f first_block_index=%d "
                        "max_us=%.1f at=%d launches_over_1ms=%u pre_block_p50_us=%.2f submit_us=%.1f complete_us=%.1f\n",
                        cfg.label, rep, a.bindings.size(), (unsigned long long)a.workspace_bytes, tg, first_block,
                        maxd, maxi, over1ms, pre.empty() ? 0.0 : pre[pre.size()/2], submit, done);
            // Post-block pattern: indices of the next few long launches.
            std::printf("  long_launch_indices:");
            unsigned shown = 0;
            for (unsigned i = 0; i < total && shown < 12; ++i) if (dur[i] > 200) { std::printf(" %u(%.0fus)", i, dur[i]); ++shown; }
            std::printf("\n"); std::fflush(stdout);
        }
    }
    g_inflight_start = -1; watcher.join();
    CK(synStreamDestroy(stream));
    CK(synDeviceRelease(g_device));
    CK(synDestroy());
    return 0;
}
