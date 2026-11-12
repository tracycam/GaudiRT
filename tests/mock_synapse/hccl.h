// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

using hcclComm_t = void*;

struct hcclUniqueId {
  uint8_t internal[1024]{};
  size_t length = 0;
};

enum hcclRedOp_t {
  hcclSum = 0,
  hcclProd = 1,
  hcclMin = 2,
  hcclMax = 3,
  hcclAvg = 4,
  hcclOpNone = 5,
};

enum hcclResult_t {
  hcclUninitialized = -1,
  hcclSuccess = 0,
  hcclInvalidArgument = 7,
};

enum hcclDataType_t {
  hcclInt8 = 0,
  hcclUint8 = 1,
  hcclInt32 = 2,
  hcclUint32 = 3,
  hcclInt64 = 4,
  hcclUint64 = 5,
  hcclFloat16 = 6,
  hcclFloat32 = 7,
  hcclFloat64 = 8,
  hcclBfloat16 = 9,
};

extern "C" {
hcclResult_t hcclGetUniqueId(hcclUniqueId* uniqueId);
hcclResult_t hcclCommInitRank(hcclComm_t* comm, int nranks, hcclUniqueId commId, int rank);
hcclResult_t hcclCommFinalize(hcclComm_t comm);
hcclResult_t hcclCommDestroy(hcclComm_t comm);
hcclResult_t hcclCommAbort(hcclComm_t comm);
const char* hcclGetErrorString(hcclResult_t result);
hcclResult_t hcclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
                           hcclDataType_t datatype, hcclRedOp_t reduceOp,
                           hcclComm_t comm, void* stream_handle);
hcclResult_t hcclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount,
                               hcclDataType_t datatype, hcclRedOp_t reduceOp,
                               hcclComm_t comm, void* stream_handle);
hcclResult_t hcclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount,
                           hcclDataType_t datatype, hcclComm_t comm, void* stream_handle);
hcclResult_t hcclAlltoAll(const void* sendbuff, void* recvbuff, size_t count,
                          hcclDataType_t datatype, hcclComm_t comm, void* stream_handle);
hcclResult_t hcclBroadcast(const void* sendbuff, void* recvbuff, size_t count,
                           hcclDataType_t datatype, int root, hcclComm_t comm,
                           void* stream_handle);
hcclResult_t hcclReduce(const void* sendbuff, void* recvbuff, size_t count,
                        hcclDataType_t datatype, hcclRedOp_t reduceOp, int root,
                        hcclComm_t comm, void* stream_handle);
hcclResult_t hcclSend(const void* sendbuff, size_t count, hcclDataType_t datatype,
                      int peer, hcclComm_t comm, void* stream);
hcclResult_t hcclRecv(void* recvbuff, size_t count, hcclDataType_t datatype,
                      int peer, hcclComm_t comm, void* stream);
hcclResult_t hcclBarrier(hcclComm_t comm, void* stream_handle);
hcclResult_t hcclGroupStart();
hcclResult_t hcclGroupEnd();
}
