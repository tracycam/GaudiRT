// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

using synDeviceId = unsigned;
using synModuleId = uint32_t;
using synEventHandle = void*;
using synStreamHandle = void*;
using synGraphHandle = void*;
using synRecipeHandle = void*;
using synSectionHandle = void*;
using synTensor = void*;
using synNodeId = uint64_t;
using TSize = uint64_t;

inline constexpr synDeviceId SYN_INVALID_DEVICE_ID = static_cast<synDeviceId>(-1);
inline constexpr unsigned SYN_MAX_TENSOR_DIM = 5;
inline constexpr int HABANA_DIM_MAX = 5;
inline constexpr uint32_t EVENT_COLLECT_TIME = 1;

enum synStatus {
  synSuccess = 0,
  synBusy = 1,
  synFail = 2,
};

enum synDeviceType {
  synDeviceGaudi2 = 0,
};

enum synDmaDir {
  HOST_TO_DRAM = 0,
  DRAM_TO_HOST = 1,
  DRAM_TO_DRAM = 2,
};

enum synDataType {
  syn_type_na = 0,
  syn_type_int4,
  syn_type_uint4,
  syn_type_packed_nf4,
  syn_type_packed_mxfp4,
  syn_type_int8,
  syn_type_uint8,
  syn_type_fp8_152,
  syn_type_fp8_143,
  syn_type_bf16,
  syn_type_fp16,
  syn_type_int16,
  syn_type_uint16,
  syn_type_ufp16,
  syn_type_float,
  syn_type_int32,
  syn_type_uint32,
  syn_type_tf32,
  syn_type_hb_float,
  syn_type_int64,
  syn_type_uint64,
};

enum synTensorType {
  DATA_TENSOR = 0,
  DATA_TENSOR_DYNAMIC = 1,
};

enum synQuantMetadataType {
  SYN_FP_QUANT_METADATA = 0,
};

enum TransposePermutationDim {
  TPD_Channel = 0,
  TPD_Width = 1,
  TPD_Height = 2,
  TPD_4Dim_Batch = 3,
  TPD_Batch = 4,
};

struct synTensorDescriptor {
  synDataType m_dataType = syn_type_na;
  unsigned m_dims = 0;
  TSize m_sizes[SYN_MAX_TENSOR_DIM]{};
  TSize m_strides[SYN_MAX_TENSOR_DIM]{};
  void* m_ptr = nullptr;
  const char* m_name = nullptr;
  TSize m_minSizes[SYN_MAX_TENSOR_DIM]{};
  synTensorType m_tensorType = DATA_TENSOR;
};

struct synTensorPermutation {
  uint8_t permutation[HABANA_DIM_MAX]{};
  uint8_t dims = 0;
};

struct synFpQuantParam {
  double scale = 1.0;
  double expBias = 7.0;
};

struct synFpQuantMetadata {
  synDataType dataType = syn_type_na;
  synFpQuantParam* fpQuantParams = nullptr;
  uint32_t numFpQuantParams = 0;
};

struct synGEMMParams {
  bool transpose_a = false;
  bool transpose_b = false;
};

struct synTransposeParams {
  TransposePermutationDim permutation[HABANA_DIM_MAX]{};
  unsigned int tensorDim = 0;
};

struct synAxisParams {
  unsigned axis = 0;
};

struct synLaunchTensorInfo {
  const char* tensorName = nullptr;
  uint64_t pTensorAddress = 0;
  synTensorType tensorType = DATA_TENSOR;
  TSize tensorSize[HABANA_DIM_MAX]{};
  uint64_t tensorId = 0;
};

using synLaunchTensorInfoExt = synLaunchTensorInfo;

struct synRetrievedLaunchTensorInfoExt {
  char tensorName[256];
  uint64_t tensorId;
  synTensorType tensorType;
  synDataType tensorDataType;
  uint32_t tensorDims;
  TSize tensorMaxSize[HABANA_DIM_MAX];
  TSize tensorMinSize[HABANA_DIM_MAX];
  uint8_t isInput;
};
