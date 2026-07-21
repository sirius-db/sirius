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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace duckdb {
class SiriusContext;
}  // namespace duckdb

namespace cucascade {
class host_data_representation;
}  // namespace cucascade

namespace sirius::vss {

/// k-NN (top-k) similarity join: for every row of @c probe_table, find its
/// @c k nearest rows in @c corpus_table under @c metric on the vector columns.
struct vector_join_request {
  std::string probe_catalog;         ///< Resolved catalog of the probe table.
  std::string probe_schema;          ///< Resolved schema of the probe table.
  std::string probe_table;           ///< Table being iterated against corpus.
  std::string probe_vector_column;   ///< FLOAT[dim] vector column on the probe.
  std::string corpus_catalog;        ///< Resolved catalog of the corpus table.
  std::string corpus_schema;         ///< Resolved schema of the corpus table.
  std::string corpus_table;          ///< Table being searched for neighbors.
  std::string corpus_vector_column;  ///< FLOAT[dim] vector column on the corpus.
  std::string metric;                ///< Distance metric.
  std::int64_t dim{0};               ///< vector dimensionality.
  std::int64_t k{10};                ///< Neighbors per probe row, or global cap.
  bool global{false};                ///< true = global top-k pairs; false = per probe row.
  std::optional<float> threshold;    ///< If set, drop pairs where distance > threshold.
  std::vector<std::string> probe_output_columns;   ///< Probe columns to emit (in order).
  std::vector<std::string> corpus_output_columns;  ///< Corpus columns to emit (in order).
};

/// Run a top-k similarity join over two tables and return the result materialized on
/// the HOST tier: the probe output columns, then the corpus output columns, then the
/// FLOAT32 @c distance column. Each probe row contributes @c k consecutive rows of its
/// nearing neighbors.
std::unique_ptr<cucascade::host_data_representation> run_vector_join(
  duckdb::SiriusContext& ctx, const vector_join_request& req);

}  // namespace sirius::vss
