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

#include "helper/logical_type.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace cucascade {
class host_data_representation;
}  // namespace cucascade

namespace sirius::vss {

struct vector_search_request {
  std::string catalog;                      ///< Resolved catalog of the pinned table.
  std::string schema;                       ///< Resolved schema of the pinned table.
  std::string table_name;                   ///< GPU-pinned base table to search.
  std::string column_name;                  ///< FLOAT[dim] vector column.
  std::string metric;                       ///< Distance metric.
  std::string index_type{"ivf_flat"};       ///< ann index type.
  std::vector<float> query;                 ///< Query vector, length == dim.
  std::int64_t dim{0};                      ///< Vector dimensionality.
  std::int64_t k{10};                       ///< Top-k neighbors to return.
  std::vector<std::string> output_columns;  ///< Base-table columns to return (in order).
  std::vector<sirius::logical_type> output_column_types;  ///< Catalog type of each output column
  bool use_index{true};                                   ///< true => ann; false => enn.
  std::int64_t n_probes{0};                               ///< IVF lists to probe for ann;
};

/// Run a single-query k-NN search over a GPU-pinned table and return the result
/// materialized on the HOST tier: one column per @c output_columns entry in
/// order followed by a trailing FLOAT32 @c distance column, up to @c k rows
/// nearest-first. The table must be pinned on the GPU tier; when @c use_index
/// is true a matching cuVS index must exist.
std::unique_ptr<cucascade::host_data_representation> run_vector_search(
  duckdb::SiriusContext& ctx, const vector_search_request& req);

}  // namespace sirius::vss
