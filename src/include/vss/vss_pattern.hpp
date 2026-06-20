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

#include <cudf/types.hpp>

#include <cuvs/distance/distance.hpp>

#include <cstdint>
#include <optional>
#include <vector>

namespace duckdb {
class LogicalTopN;
}  // namespace duckdb

namespace sirius::vss {

/**
 * @brief A single output column of the fused VSS operator.
 */
struct vss_output_column {
  enum class kind {
    gather_input,
    distance,
  };
  kind which;
  cudf::size_type input_index;
};

/**
 * @brief A recognized "vector search top-k" shape.
 */
struct vss_top_k_pattern {
  cudf::size_type vector_column_index;
  std::vector<float> query;
  int64_t dim;
  cuvs::distance::DistanceType metric;
  std::vector<vss_output_column> output_columns;
  cudf::size_type distance_output_index;
};

/**
 * @brief Try to recognize op as a brute-force vector-search top-k.
 *
 * @return The extracted pattern, or std::nullopt if op does not match,
 *         in which case the caller falls back to ordinary TopN.
 */
std::optional<vss_top_k_pattern> match_vss_top_n(duckdb::LogicalTopN const& op);

}  // namespace sirius::vss
