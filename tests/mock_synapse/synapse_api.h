// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "synapse_common_types.hpp"

#define SYN_API_CALL

extern "C" {
synStatus synInitialize();
synStatus synDestroy();
synStatus synDeviceGetCount(uint32_t* pCount);
synStatus synDeviceGetModuleIDs(uint32_t* pDeviceModuleIds, uint32_t* size);
synStatus synDeviceAcquireByModuleId(synDeviceId* pDeviceId, synModuleId moduleId);
synStatus synDeviceRelease(synDeviceId devId);
synStatus synDeviceSynchronize(synDeviceId devId);
synStatus synDeviceGetMemoryInfo(synDeviceId, uint64_t*, uint64_t*);
synStatus synDeviceGetAttribute(uint64_t*, const synDeviceAttribute*, unsigned, synDeviceId);
synStatus synDeviceMalloc(synDeviceId, uint64_t, uint64_t, uint32_t, uint64_t*);
synStatus synDeviceFree(synDeviceId, uint64_t, uint32_t);
synStatus synHostMalloc(synDeviceId, uint64_t, uint32_t, void**);
synStatus synHostFree(synDeviceId, const void*, uint32_t);

synStatus synStreamCreateGeneric(synStreamHandle* pStreamHandle, synDeviceId devId, uint32_t flags);
synStatus synStreamDestroy(synStreamHandle streamHandle);
synStatus synStreamQuery(synStreamHandle streamHandle);
synStatus synStreamWaitEvent(synStreamHandle, synEventHandle, uint32_t);
synStatus synStreamSynchronize(synStreamHandle);

synStatus synEventCreate(synEventHandle*, synDeviceId, uint32_t);
synStatus synEventDestroy(synEventHandle);
synStatus synEventRecord(synEventHandle, synStreamHandle);
synStatus synEventQuery(synEventHandle);
synStatus synEventSynchronize(synEventHandle);
synStatus synEventElapsedTime(uint64_t* pNanoSeconds, synEventHandle eventHandleStart, synEventHandle eventHandleEnd);

synStatus synMemCopyAsync(synStreamHandle streamHandle, uint64_t src, uint64_t size, uint64_t dst, synDmaDir direction);
synStatus synMemCopyAsyncMultiple(synStreamHandle streamHandle, const uint64_t* src, const uint64_t* size,
                                  const uint64_t* dst, synDmaDir direction, uint64_t numCopies);
synStatus synMemsetD32Async(uint64_t pDeviceMem, uint32_t value, size_t numOfElements, synStreamHandle streamHandle);

synStatus synGraphCreate(synGraphHandle* pGraphHandle, synDeviceType deviceType);
synStatus synGraphDestroy(synGraphHandle graphHandle);
synStatus synSectionCreate(synSectionHandle* sectionHandle, uint64_t sectionDescriptor, synGraphHandle graph);
synStatus synSectionDestroy(synSectionHandle sectionHandle);
synStatus synSectionSetPersistent(synSectionHandle sectionHandle, bool sectionIsPersistent);
synStatus synSectionSetRMW(synSectionHandle sectionHandle, bool sectionIsRMW);

synStatus synTensorCreate(synTensor* pTensor, const synTensorDescriptor* descriptor,
                          const synSectionHandle pSectionHandle, uint64_t sectionOffset);
synStatus synTensorDestroy(synTensor tensor);
synStatus synTensorSetPermutation(synTensor tensor, const synTensorPermutation* permutation);
synStatus synTensorSetQuantizationData(synTensor tensor, synQuantMetadataType prop, void* propVal, uint64_t propSize);

synStatus synNodeCreate(synGraphHandle graphHandle, const synTensor* pInputsTensorList,
                        const synTensor* pOutputsTensorList, uint32_t numberInputs, uint32_t numberOutputs,
                        const void* pUserParams, unsigned paramsSize, const char* pGuid, const char* pName,
                        const char** inputLayouts, const char** outputLayouts);
synStatus synNodeCreateWithId(synGraphHandle graphHandle, const synTensor* pInputsTensorList,
                              const synTensor* pOutputsTensorList, uint32_t numberInputs, uint32_t numberOutputs,
                              const void* pUserParams, unsigned paramsSize, const char* pGuid, const char* pName,
                              synNodeId* nodeUniqueId, const char** inputLayouts, const char** outputLayouts);

synStatus synGraphCompile(synRecipeHandle* pRecipeHandle, const synGraphHandle graphHandle,
                          const char* pFileName, const char* pBuildLog);
synStatus synRecipeDestroy(synRecipeHandle recipeHandle);
synStatus synRecipeSerialize(const synRecipeHandle recipeHandle, const char* recipeFileName);
synStatus synRecipeDeSerialize(synRecipeHandle* pRecipeHandle, const char* recipeFileName);
synStatus synWorkspaceGetSize(uint64_t* pWorkspaceSize, const synRecipeHandle recipeHandle);

synStatus synTensorRetrieveLaunchAmount(const synRecipeHandle pRecipeHandle, uint32_t* numOfTensors);
synStatus synTensorRetrieveLaunchIds(const synRecipeHandle pRecipeHandle, uint64_t* tensorsIds, uint32_t numOfTensors);
synStatus synTensorRetrieveLaunchInfoByIdExt(const synRecipeHandle pRecipeHandle, uint32_t numOfTensors,
                                             synRetrievedLaunchTensorInfoExt* tensorsLaunchInfo);

synStatus synLaunch(synStreamHandle streamHandle, const synLaunchTensorInfo* launchTensorsInfo,
                    uint32_t numberTensors, uint64_t pWorkspace, const synRecipeHandle pRecipeHandle, uint32_t flags);
synStatus synLaunchWithExternalEvents(synStreamHandle streamHandle, const synLaunchTensorInfo* launchTensorsInfo,
                                      uint32_t numberTensors, uint64_t pWorkspace, const synRecipeHandle pRecipeHandle,
                                      synEventHandle* eventHandleList, uint32_t numberOfEvents, uint32_t flags);
}
