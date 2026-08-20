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

#include "vss/distance_metric.hpp"

#include <stdexcept>
#include <string>

namespace sirius::vss {

cuvs::distance::DistanceType enn_distance_type_from_metric(std::string_view metric)
{
  if (metric == "l2") { return cuvs::distance::DistanceType::L2SqrtUnexpanded; }
  if (metric == "cosine") { return cuvs::distance::DistanceType::CosineExpanded; }
  throw std::invalid_argument("enn_distance_type_from_metric: unsupported metric '" +
                              std::string(metric) + "'");
}

}  // namespace sirius::vss
