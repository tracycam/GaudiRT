// SPDX-License-Identifier: Apache-2.0
#pragma once

using hcclComm_t = void*;

struct hcclUniqueId {
  char internal[128]{};
};

enum hcclResult_t {
  hcclSuccess = 0,
};

extern "C" {
hcclResult_t hcclCommInitRank(hcclComm_t* comm, int nranks, hcclUniqueId commId, int rank);
hcclResult_t hcclCommDestroy(hcclComm_t comm);
}
