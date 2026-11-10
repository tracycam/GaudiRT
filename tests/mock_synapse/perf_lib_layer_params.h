// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

union fint_t {
  float f;
  int32_t i;
  uint32_t u;
};

enum CastF32RoundMode_t {
  CAST_ROUND_HALF_NE = 0,
  CAST_ROUND_ZERO = 1,
};

namespace ns_GatherKernel {
struct Params {
  int axis = 0;
};
struct ParamsV2 : public Params {
  int batchDims = 0;
};
}  // namespace ns_GatherKernel

namespace ns_CastKernel {
struct Params {
  CastF32RoundMode_t round_mode = CAST_ROUND_HALF_NE;
};
struct ParamsV3 : public Params {
  unsigned int seed = 0;
};
}  // namespace ns_CastKernel

namespace ns_ConstantKernel {
struct Params {
  fint_t constant{};
};
struct ParamsV3 : public Params {
  uint32_t constant_high = 0;
};
}  // namespace ns_ConstantKernel

namespace ns_BroadcastNonFcd {
struct Params {
  int axis = 0;
};
}  // namespace ns_BroadcastNonFcd
