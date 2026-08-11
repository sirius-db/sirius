/*
 * Copyright 2025, Sirius Contributors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuvs/distance/distance.hpp>

#include <string_view>

namespace sirius::vss {

/**
 * @brief Map a user metric string to the cuVS DistanceType for enn. L2 uses the
 * Unexpanded form to avoid catastrophic cancellation on large-magnitude vectors.
 */
cuvs::distance::DistanceType enn_distance_type_from_metric(std::string_view metric);

/**
 * @brief Map a user metric string to the cuVS DistanceType for ann.
 */
cuvs::distance::DistanceType ann_distance_type_from_metric(std::string_view metric);

/**
 * @brief Map a user metric string to the cuVS DistanceType for the exact vector
 * join's selection pass.
 *
 * Uses the Expanded forms so the pairwise distances ride the GEMM/tensor-core
 * path (`x·y` via matrix multiply + norms). This is the fast first pass; where an
 * accurate value near zero is needed (any distance output), the refine pass
 * recomputes it Unexpanded on the selected survivors. Expanded is safe for
 * *ranking* because a true near-neighbor stays in the top-k despite the
 * near-zero cancellation.
 */
cuvs::distance::DistanceType join_selection_distance_type_from_metric(std::string_view metric);

}  // namespace sirius::vss
