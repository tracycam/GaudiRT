// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "bridge_internal.hpp"
#include "cold_recipe.hpp"

namespace gaudirt::generic {
// Caller holds the cold-recipe mutex and has checked the contiguous matrix.
ColdRecipe& vocabulary_argmax_recipe(Context& context, int64_t width, int64_t rows, bool bf16);
}
