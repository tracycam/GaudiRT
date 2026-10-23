#include <synapse_api.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

#define CK(call) do { auto status = (call); if (status != synSuccess) { \
    std::fprintf(stderr, "%s failed: %d at line %d\n", #call, int(status), __LINE__); \
    std::exit(2); }} while (0)

using Clock = std::chrono::steady_clock;
double us(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b-a).count();
}

int main(int argc, char** argv) {
    const bool rmw = argc > 1 && std::strcmp(argv[1], "rmw") == 0;
    constexpr unsigned n = 4096;
    constexpr uint64_t bytes = n * sizeof(float);
    CK(synInitialize());
    synDeviceId device;
    CK(synDeviceAcquireByModuleId(&device, 0));
    synDeviceInfoV2 info{};
    CK(synDeviceGetInfoV2(device, &info));
    std::printf("device_sram_base=0x%llx device_sram_bytes=%u rmw=%d\n",
                (unsigned long long)info.sramBaseAddress, info.sramSize, rmw);
    synStreamHandle stream;
    CK(synStreamCreateGeneric(&stream, device, 0));
    synGraphHandle graph;
    CK(synGraphCreate(&graph, synDeviceGaudi2));
    const char* names[] = {"input_x", "input_b", "output_y", "intermediate"};
    synSectionHandle sections[4]{};
    synTensor tensors[4]{};
    for (unsigned i = 0; i < 4; ++i) {
        CK(synSectionCreate(&sections[i], 0, graph));
        CK(synSectionSetPersistent(sections[i], i < 3));
        if (i == 3 && rmw) CK(synSectionSetRMW(sections[i], true));
        synTensorDescriptor desc{};
        desc.m_name = names[i];
        desc.m_dataType = syn_type_single;
        desc.m_dims = 1;
        desc.m_sizes[0] = desc.m_minSizes[0] = n;
        CK(synTensorCreate(&tensors[i], &desc, sections[i], 0));
    }
    CK(synNodeCreate(graph, tensors, &tensors[3], 2, 1, nullptr, 0,
                     "add_fwd_f32", "add", nullptr, nullptr));
    CK(synNodeCreate(graph, &tensors[3], &tensors[2], 1, 1, nullptr, 0,
                     "relu_fwd_f32", "relu", nullptr, nullptr));
    synRecipeHandle recipe;
    CK(synGraphCompile(&recipe, graph, rmw ? "review_add_relu_rmw" : "review_add_relu", nullptr));
    uint64_t ids[3];
    CK(synTensorRetrieveIds(recipe, names, ids, 3));
    synLaunchTensorInfoExt bindings[3]{};
    for (unsigned i = 0; i < 3; ++i) {
        CK(synDeviceMalloc(device, bytes, 0, 0, &bindings[i].pTensorAddress));
        bindings[i].tensorId = ids[i];
        bindings[i].tensorName = names[i];
        bindings[i].tensorType = DATA_TENSOR;
        bindings[i].tensorSize[0] = n;
    }
    uint64_t workspace_bytes = 0, workspace = 0;
    CK(synWorkspaceGetSize(&workspace_bytes, recipe));
    if (workspace_bytes) CK(synDeviceMalloc(device, workspace_bytes, 0, 0, &workspace));
    std::printf("graph_nodes=2 recipes=1 workspace_bytes=%llu\n", (unsigned long long)workspace_bytes);
    float* host = nullptr;
    CK(synHostMalloc(device, 3*bytes, 0, reinterpret_cast<void**>(&host)));
    auto launch = [&] {
        CK(synLaunchWithExternalEventsExt(stream, bindings, 3, workspace, recipe, nullptr, 0, 0));
    };
    auto refresh = [&](int seed) {
        for (unsigned i=0; i<n; ++i) {
            host[i] = float(int(i % 37) - 18 + seed);
            host[n+i] = float(int(i % 11) - 5 - 2*seed);
            host[2*n+i] = -99999.f;
        }
        for (unsigned i=0; i<3; ++i)
            CK(synMemCopyAsync(stream, reinterpret_cast<uint64_t>(host+i*n), bytes,
                              bindings[i].pTensorAddress, HOST_TO_DRAM));
        CK(synStreamSynchronize(stream));
    };
    auto check = [&] {
        CK(synMemCopyAsync(stream, bindings[2].pTensorAddress, bytes,
                          reinterpret_cast<uint64_t>(host+2*n), DRAM_TO_HOST));
        CK(synStreamSynchronize(stream));
        unsigned mismatches=0;
        for (unsigned i=0; i<n; ++i) {
            float expected = std::max(0.f, host[i]+host[n+i]);
            if (std::memcmp(&expected, host+2*n+i, sizeof(float))) ++mismatches;
        }
        std::printf("bitwise_mismatches=%u words=%u\n", mismatches, n);
        if (mismatches) std::exit(3);
    };
    refresh(0);
    for (int i=0; i<256; ++i) launch();
    CK(synStreamSynchronize(stream));
    check();
    auto reset_stats = reinterpret_cast<void(*)()>(dlsym(RTLD_DEFAULT, "reset_synapse_launch_stats"));
    if (reset_stats) reset_stats();
    for (unsigned batch : {1u, 16u, 256u}) {
        std::vector<double> enqueue, total;
        enqueue.reserve(32); total.reserve(32);
        for (unsigned trial=0; trial<32; ++trial) {
            auto begin = Clock::now();
            for (unsigned j=0; j<batch; ++j) launch();
            auto submitted = Clock::now();
            CK(synStreamSynchronize(stream));
            auto completed = Clock::now();
            enqueue.push_back(us(begin, submitted)/batch);
            total.push_back(us(begin, completed)/batch);
        }
        double se=0, st=0;
        for (double v: enqueue) se+=v;
        for (double v: total) st+=v;
        std::sort(enqueue.begin(), enqueue.end());
        std::sort(total.begin(), total.end());
        std::printf("batch=%u trials=32 enqueue_mean_us=%.6f enqueue_p50_us=%.6f "
                    "enqueue_p95_us=%.6f completion_mean_us=%.6f completion_p50_us=%.6f\n",
                    batch, se/32, enqueue[16], enqueue[30], st/32, total[16]);
    }
    check();
    refresh(7); launch(); check();
    CK(synHostFree(device, host, 0));
    for (auto& binding : bindings) CK(synDeviceFree(device, binding.pTensorAddress, 0));
    if (workspace) CK(synDeviceFree(device, workspace, 0));
    CK(synRecipeDestroy(recipe));
    CK(synGraphDestroy(graph));
    for (auto section : sections) CK(synSectionDestroy(section));
    CK(synStreamDestroy(stream));
    CK(synDeviceRelease(device));
    CK(synDestroy());
    return 0;
}
