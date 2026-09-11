/*
 * Copyright 2026, Sirius Contributors.
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

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <duckdb/common/types/data_chunk.hpp>
#include <duckdb/common/vector.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace sirius {

/**
 * @brief Host-side accumulation of DuckDB rows, uploaded once as a cudf::table.
 *
 * Every GPU consumer of CPU-side DuckDB data faces the same three problems -- a DataChunk's
 * vectors may be dictionary/constant encoded, its invalid rows hold uninitialized bytes, and a
 * per-chunk H2D copy would be one tiny transfer per column per 2048 rows. Staging into host
 * vectors and uploading once solves all three, so it lives here rather than being written again
 * per call site (`sirius_physical_gpu_values` for VALUES/materialized data, the `.hpln` COPY
 * writer for an outbound file).
 *
 * Rows accumulate until @ref build, which is the only GPU step. @ref reset then starts a fresh
 * table with the same schema, which is what lets a writer cut a stream into fixed-size chunks
 * without re-deriving the staging layout each time.
 *
 * Fixed-width and VARCHAR columns only: a nested type has no flat host staging here, and the
 * caller is expected to have refused one before constructing this.
 */
class duckdb_chunk_staging {
 public:
  /**
   * @param types       One per column, in the order rows are appended.
   * @param reserve_rows Hint for the host reservation; exceeding it costs a reallocation, not
   *                     correctness.
   */
  explicit duckdb_chunk_staging(duckdb::vector<sirius::logical_type> types,
                                cudf::size_type reserve_rows = 0);

  /// Append every row of @p chunk.
  void append(duckdb::DataChunk& chunk);

  /// Append rows [@p offset, @p offset + @p count) of @p chunk.
  ///
  /// The row range is what lets a writer close a chunk at an exact row count rather than at
  /// whatever boundary DuckDB's vector size happens to fall on.
  void append(duckdb::DataChunk& chunk, duckdb::idx_t offset, duckdb::idx_t count);

  /// Append one row that is NULL in every column (what a DUMMY_SCAN produces).
  void append_null_row();

  [[nodiscard]] cudf::size_type num_rows() const noexcept { return _num_rows; }
  [[nodiscard]] std::size_t num_columns() const noexcept { return _columns.size(); }

  /// Nulls staged so far in @p column.
  [[nodiscard]] cudf::size_type null_count(std::size_t column) const;

  /// Index of the first column that has staged a null, or nullopt when none has.
  ///
  /// For callers whose sink cannot represent nullability and must refuse rather than write a
  /// file that silently loses the information.
  [[nodiscard]] std::optional<std::size_t> first_column_with_nulls() const;

  /// Upload the staged rows as one table. The staging is left intact; call @ref reset to reuse it.
  [[nodiscard]] std::unique_ptr<cudf::table> build(rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr);

  /// Drop all staged rows, keeping the schema.
  void reset();

 private:
  /// Host-side staging for one column. Pageable host memory + cudaMemcpyAsync is deliberate:
  /// the sources feeding this are either small (VALUES) or bounded by one chunk, so neither a
  /// pinned allocation nor the IO machinery is warranted.
  struct column_staging {
    std::vector<std::uint8_t> fixed_data;        // fixed-width payload
    std::vector<std::int32_t> offsets;           // varchar: num_rows + 1 entries
    std::vector<char> chars;                     // varchar payload
    std::vector<cudf::bitmask_type> mask_words;  // cudf validity bitmask (1 = valid)
    cudf::size_type null_count = 0;
    bool is_varchar            = false;
  };

  void stage_column(column_staging& s,
                    duckdb::Vector& vec,
                    duckdb::idx_t offset,
                    duckdb::idx_t count,
                    const sirius::logical_type& type);

  duckdb::vector<sirius::logical_type> _types;
  std::vector<column_staging> _columns;
  cudf::size_type _num_rows = 0;
};

}  // namespace sirius
