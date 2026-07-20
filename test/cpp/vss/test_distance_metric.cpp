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

// test
#include <catch.hpp>

// sirius
#include <vss/distance_metric.hpp>

#include <stdexcept>

using Metric = cuvs::distance::DistanceType;

TEST_CASE("enn_distance_type_from_metric maps l2 to the unexpanded form", "[vss]")
{
  REQUIRE(sirius::vss::enn_distance_type_from_metric("l2") == Metric::L2SqrtUnexpanded);
  REQUIRE(sirius::vss::enn_distance_type_from_metric("cosine") == Metric::CosineExpanded);
  REQUIRE_THROWS_AS(sirius::vss::enn_distance_type_from_metric("dot"), std::invalid_argument);
  REQUIRE_THROWS_AS(sirius::vss::enn_distance_type_from_metric(""), std::invalid_argument);
}
