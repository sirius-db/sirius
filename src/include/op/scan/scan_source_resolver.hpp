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

// sirius
#include <op/scan/parquet_scan_operator_data.hpp>  // post_convert_fn_t

// duckdb
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

// standard library
#include <cstddef>
#include <string>
#include <vector>

namespace sirius::op::scan {

/**
 * @brief Polymorphic description of a scan source (file list + read projection + per-batch
 *        transform).
 *
 * The metadata scan operator is scan-kind-agnostic: plain parquet, Iceberg, and other future
 * sources differ only in how they resolve their file list, widen their read projection, and
 * what (if any) per-batch transform they install on each materialized GPU batch. Each source
 * kind provides a concrete resolver; the metadata operator holds a unique_ptr to the base
 * class and calls resolve() once at construction.
 *
 * Lifetime: the resolver is consumed by the metadata scan operator; resolve() is expected to
 * be invoked exactly once.
 */
class scan_source_resolver {
 public:
  /**
   * @brief Output of resolve(). Everything the metadata scan operator needs to describe its
   *        scan, beyond the filter/planner bits that stay on the operator constructor.
   */
  struct resolved {
    /// Data-file paths the metadata scan operator will parse footers for.
    std::vector<std::string> file_paths;

    /// Parquet column indices to read. For plain parquet, this is the planner projection
    /// expanded via make_selected_column_indices. For Iceberg it is additionally widened
    /// with equality-delete key columns.
    std::vector<std::size_t> selected_column_indices;

    /// Column names in selected_column_indices order. Empty when no projection should be
    /// applied to the parquet reader (i.e., read all columns). When non-empty, the metadata
    /// operator calls reader_options::set_column_names with these names.
    std::vector<std::string> projected_column_names;

    /// Optional GPU transform invoked by the GPU scan operator after a batch has been
    /// materialized. Null for plain parquet. Used by Iceberg to apply V2 positional and
    /// equality deletes without a dedicated pipeline operator.
    post_convert_fn_t per_batch_transform;

    /// Number of trailing columns the GPU scan operator strips from each batch after the
    /// per_batch_transform runs. 0 for plain parquet. Used by Iceberg to drop delete-key
    /// columns that were widened into selected_column_indices but are not part of the user's
    /// output schema.
    int trailing_columns_to_strip = 0;
  };

  virtual ~scan_source_resolver() = default;

  /**
   * @brief Compute the resolved scan description.
   *
   * Invoked exactly once by the metadata scan operator at construction time.
   */
  virtual resolved resolve() = 0;
};

/**
 * @brief Resolver for plain parquet scans.
 *
 * Produces file_paths verbatim and computes selected_column_indices / projected_column_names
 * from DuckDB's column_ids + projection_ids + names, using the same rules as the legacy
 * parquet scan task (detail::make_selected_column_indices). Sets per_batch_transform to null
 * and trailing_columns_to_strip to 0.
 *
 * @throws from resolve() if projection_ids is non-empty but names is empty (projection
 *         needs column names).
 */
class parquet_scan_source_resolver : public scan_source_resolver {
 public:
  parquet_scan_source_resolver(std::vector<std::string> file_paths,
                               duckdb::vector<duckdb::ColumnIndex> const& column_ids,
                               duckdb::vector<duckdb::idx_t> const& projection_ids,
                               duckdb::vector<std::string> const& names);

  resolved resolve() override;

 private:
  std::vector<std::string> _file_paths;
  duckdb::vector<duckdb::ColumnIndex> _column_ids;
  duckdb::vector<duckdb::idx_t> _projection_ids;
  duckdb::vector<std::string> _names;
};

}  // namespace sirius::op::scan
