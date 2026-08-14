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

#include "duckdb/function/function.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace sirius::vss {

enum class vector_join_mode : std::uint8_t {
  global_top_k,
  per_row_top_k,
  threshold,
};

/// - exact:      brute force, L2 computed Unexpanded, no GEMM.
/// - exact_gemm: brute force, L2 computed Expanded (with GEMM). Exact for
///               normalized / moderate-magnitude vectors.
/// - approx:     not implemented yet
enum class vector_join_search_mode : std::uint8_t {
  exact,
  exact_gemm,
  approx,
};

/// The type of score to emit for each result pair, and which value space the join
/// selects/thresholds in.
/// - similarity: higher is closer. For cosine this is the inner product
///               (1 - distance). Threshold semantics are score >= eps.
/// - distance:   lower is closer. The natural output for L2. Threshold semantics
///               are score <= eps.
enum class vector_join_output_type : std::uint8_t {
  similarity,
  distance,
};

struct vector_join_side {
  std::string catalog;                      ///< resolved catalog of the pinned table
  std::string schema;                       ///< resolved schema
  std::string table;                        ///< base table
  std::string column;                       ///< vector column
  std::vector<std::string> output_columns;  ///< base-table columns to emit in order
  bool is_fast_path{false};
};

struct vector_join_request {
  vector_join_side left;
  vector_join_side right;
  vector_join_mode mode;
  std::string metric;  ///< distance metric
  vector_join_search_mode search_mode{vector_join_search_mode::exact_gemm};
  std::int64_t k{0};           ///< top-k
  std::int64_t n_clusters{0};  ///< number of clusters of k-means cluster
  std::int64_t n_probes{1};    ///< nearest clusters each point is assigned to
  std::int64_t dim{0};         ///< vector dimensionality
  double eps{0.0};             ///< distance/similarity threshold
  vector_join_output_type output_type{vector_join_output_type::distance};
};

struct SiriusVectorJoinBindData : public duckdb::TableFunctionData {
  vector_join_request req;
};

}  // namespace sirius::vss
