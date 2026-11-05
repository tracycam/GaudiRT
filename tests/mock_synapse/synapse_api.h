#pragma once

#include <cstdint>

using synDeviceId = unsigned;
using synEventHandle = void*;
using synStreamHandle = void*;
enum synStatus { synSuccess = 0 };

extern "C" {
synStatus synDeviceGetMemoryInfo(synDeviceId, uint64_t*, uint64_t*);
synStatus synDeviceMalloc(synDeviceId, uint64_t, uint64_t, uint32_t, uint64_t*);
synStatus synDeviceFree(synDeviceId, uint64_t, uint32_t);
synStatus synEventCreate(synEventHandle*, synDeviceId, uint32_t);
synStatus synEventDestroy(synEventHandle);
synStatus synEventRecord(synEventHandle, synStreamHandle);
synStatus synEventSynchronize(synEventHandle);
synStatus synStreamWaitEvent(synStreamHandle, synEventHandle, uint32_t);
synStatus synStreamSynchronize(synStreamHandle);
}
