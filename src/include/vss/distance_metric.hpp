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
 * @brief Map a user metric string to the cuVS DistanceType for the vector join's select pass.
 *
 * L2 with @p exact_unexpanded true uses the Unexpanded form (no GEMM); false uses the Expanded
 * form so the pairwise distances ride the GEMM/tensor-core path. Cosine always uses
 * CosineExpanded, so @p exact_unexpanded has no effect.
 */
cuvs::distance::DistanceType join_selection_distance_type_from_metric(std::string_view metric,
                                                                      bool exact_unexpanded);

}  // namespace sirius::vss
