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

// This header is intentionally included ONLY at DuckDB boundaries:
//   - src/planner/  (entry: duckdb::LogicalType → sirius::logical_type)
//   - src/op/result/ (exit: sirius::logical_type → duckdb::LogicalType)
// Do NOT include from operator headers or GPU kernel headers.

#include "helper/logical_type.hpp"

#include <duckdb/common/identifier.hpp>
#include <duckdb/common/projection_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

#include <cstddef>
#include <string>

namespace sirius {

/**
 * @brief Convert a DuckDB logical type to a Sirius logical type.
 *
 * Used at the plan-generator entry boundary to convert types from the DuckDB
 * logical plan into Sirius operator types. DECIMAL precision and scale are
 * fully preserved.
 *
 * @throws duckdb::InvalidInputException if the type is unsupported.
 */
logical_type from_duckdb(const duckdb::LogicalType& t);

/**
 * @brief Convert a Sirius logical type back to a DuckDB logical type.
 *
 * Used at the result-collector exit boundary to reconstruct DuckDB types for
 * DataChunk initialization and ColumnDataCollection. DECIMAL precision and
 * scale are fully preserved.
 *
 * @throws duckdb::InvalidInputException if the type is unsupported.
 */
duckdb::LogicalType to_duckdb(const logical_type& t);

/**
 * @brief Batch-convert a DuckDB type vector to a Sirius type vector.
 *
 * Convenience wrapper over from_duckdb() for operator constructor call sites
 * in the planner. Returns duckdb::vector to preserve bounds-checking on
 * subsequent operator[] accesses.
 */
duckdb::vector<logical_type> from_duckdb_vec(const duckdb::vector<duckdb::LogicalType>& types);

/**
 * @brief Batch-convert a Sirius type vector to a DuckDB type vector.
 *
 * Convenience wrapper over to_duckdb() for result-collector call sites that
 * need a duckdb::vector<duckdb::LogicalType> (e.g. DataChunk::Initialize,
 * ColumnDataCollection). Accepts duckdb::vector via base-class reference.
 */
duckdb::vector<duckdb::LogicalType> to_duckdb_vec(const duckdb::vector<logical_type>& types);

/**
 * @brief Unwrap DuckDB Identifier column names to plain strings.
 *
 * Convenience wrapper for operator constructor call sites: DuckDB plan nodes
 * carry names as case-insensitive Identifier, while sirius operators store
 * plain strings.
 */
duckdb::vector<std::string> from_duckdb_names(const duckdb::vector<duckdb::Identifier>& names);

/**
 * @brief Unwrap DuckDB ProjectionIndex vectors to plain positions.
 *
 * Convenience wrapper for operator constructor call sites: DuckDB plan nodes
 * carry projection maps as ProjectionIndex, while sirius operators store
 * plain positions.
 */
duckdb::vector<std::size_t> from_duckdb_projection_ids(
  const duckdb::vector<duckdb::ProjectionIndex>& ids);

}  // namespace sirius
