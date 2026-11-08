// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "attention_plan.hpp"

namespace gaudirt::generic {

struct TiledAttentionInputs {
  uint64_t query, key, value, locations, pool, output;
};

void launch_tiled_attention(Context& ctx, const AttentionPlan& plan, const AttentionBlock& block,
                            const TiledAttentionInputs& input, float scale);

}  // namespace gaudirt::generic
