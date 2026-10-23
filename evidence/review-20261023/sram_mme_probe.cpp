#include <synapse_api.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CK(call) do { auto status = (call); if (status != synSuccess) { \
    std::fprintf(stderr, "%s failed: %d at line %d\n", #call, int(status), __LINE__); \
    std::exit(2); }} while (0)

static uint16_t bf16(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return uint16_t(bits >> 16);
}

int main() {
    constexpr unsigned k=64, m=16, n=64;
    const unsigned sizes[5][2]={{k,m},{k,m},{n,k},{n,m},{k,m}};
    const char* names[5]={"x","b","weight","y","sram_sum"};
    CK(synInitialize());
    synDeviceId device;
    CK(synDeviceAcquireByModuleId(&device,0));
    synStreamHandle stream;
    CK(synStreamCreateGeneric(&stream,device,0));
    synGraphHandle graph;
    CK(synGraphCreate(&graph,synDeviceGaudi2));
    synSectionHandle sections[5]{};
    synTensor tensors[5]{};
    for (unsigned i=0;i<5;++i) {
        CK(synSectionCreate(&sections[i],0,graph));
        CK(synSectionSetPersistent(sections[i],i<4));
        if (i==4) CK(synSectionSetRMW(sections[i],true));
        synTensorDescriptor d{};
        d.m_name=names[i]; d.m_dataType=syn_type_bf16; d.m_dims=2;
        for (unsigned j=0;j<2;++j) d.m_sizes[j]=d.m_minSizes[j]=sizes[i][j];
        CK(synTensorCreate(&tensors[i],&d,sections[i],0));
    }
    CK(synNodeCreate(graph,tensors,&tensors[4],2,1,nullptr,0,"add_fwd_bf16","producer_add",nullptr,nullptr));
    synTensor gemm_in[2]={tensors[4],tensors[2]};
    synGEMMParams params{false,false};
    CK(synNodeCreate(graph,gemm_in,&tensors[3],2,1,&params,sizeof(params),"gemm","consumer_mme",nullptr,nullptr));
    synRecipeHandle recipe;
    CK(synGraphCompile(&recipe,graph,"review_tpc_sram_mme",nullptr));
    uint64_t ids[4];
    CK(synTensorRetrieveIds(recipe,names,ids,4));
    synLaunchTensorInfoExt bindings[4]{};
    uint16_t* host[4]{};
    for (unsigned i=0;i<4;++i) {
        uint64_t bytes=2ull*sizes[i][0]*sizes[i][1];
        CK(synDeviceMalloc(device,bytes,0,0,&bindings[i].pTensorAddress));
        CK(synHostMalloc(device,bytes,0,reinterpret_cast<void**>(&host[i])));
        bindings[i].tensorId=ids[i]; bindings[i].tensorType=DATA_TENSOR;
        bindings[i].tensorName=names[i];
        for (unsigned j=0;j<2;++j) bindings[i].tensorSize[j]=sizes[i][j];
    }
    uint64_t workspace_size=0,workspace=0;
    CK(synWorkspaceGetSize(&workspace_size,recipe));
    if(workspace_size) CK(synDeviceMalloc(device,workspace_size,0,0,&workspace));
    for (int seed: {0,7}) {
        for(unsigned i=0;i<k*m;++i) {
            host[0][i]=bf16(float(int(i%17)-8+seed));
            host[1][i]=bf16(float(int(i%7)-3));
            host[3][i]=uint16_t(0x7fc1);
        }
        for(unsigned row=0;row<k;++row)
            for(unsigned col=0;col<n;++col) host[2][row*n+col]=bf16(row==col ? 1.f : 0.f);
        for(unsigned i=0;i<4;++i)
            CK(synMemCopyAsync(stream,reinterpret_cast<uint64_t>(host[i]),2ull*sizes[i][0]*sizes[i][1],
                               bindings[i].pTensorAddress,HOST_TO_DRAM));
        CK(synLaunchWithExternalEventsExt(stream,bindings,4,workspace,recipe,nullptr,0,0));
        CK(synMemCopyAsync(stream,bindings[3].pTensorAddress,2ull*n*m,reinterpret_cast<uint64_t>(host[3]),DRAM_TO_HOST));
        CK(synStreamSynchronize(stream));
        unsigned bad=0;
        for(unsigned i=0;i<n*m;++i)
            if(host[3][i]!=bf16(float(int(i%17)-8+seed+int(i%7)-3))) ++bad;
        std::printf("seed=%d bf16_words=%u bitwise_mismatches=%u\n",seed,n*m,bad);
        if(bad) return 3;
    }
    std::printf("requested_nodes=TPC_add,MME_gemm recipes=1 workspace_bytes=%llu\n",
                (unsigned long long)workspace_size);
    for(unsigned i=0;i<4;++i) {
        CK(synHostFree(device,host[i],0));
        CK(synDeviceFree(device,bindings[i].pTensorAddress,0));
    }
    if(workspace_size) CK(synDeviceFree(device,workspace,0));
    CK(synRecipeDestroy(recipe)); CK(synGraphDestroy(graph));
    for(auto section:sections) CK(synSectionDestroy(section));
    CK(synStreamDestroy(stream)); CK(synDeviceRelease(device)); CK(synDestroy());
}
