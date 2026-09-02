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

// cudf
#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>

// standard library
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sirius::op::scan::detail {

/**
 * @brief Byte offset where row group @p index starts, per the StarRocks reader convention.
 *
 * A distributed byte-range scan is correct only if every reader of the same file derives the
 * same start offset per row group — the FE load-balances splits assuming its readers follow
 * the StarRocks BE rule (be/src/formats/parquet/utils.cpp): the minimum of the first column
 * chunk's data/index/dictionary page offsets and the row group's own file_offset, where each
 * candidate counts only when present. cudf's thrift structs carry no __isset, so a page
 * offset of 0 is treated as absent (real offsets start after the 4-byte magic).
 *
 * @throws sirius::invalid_input_exception if no candidate offset is present — ownership
 *         would be undefined, and guessing risks reading rows twice or not at all.
 */
[[nodiscard]] std::int64_t row_group_start_offset(cudf::io::parquet::FileMetaData const& metadata,
                                                  std::size_t index);

/**
 * @brief Row groups owned by the byte range [start, start+length).
 *
 * Ownership is start-offset containment: row group i belongs to the range iff
 * `start <= row_group_start_offset(i) < start + length`. A row group straddling the range end
 * is still owned in full by this range (the byte range bounds ownership, not I/O), and a range
 * that contains no start offset owns nothing — the caller must treat that as a valid empty
 * scan. Together these make any exact tiling of the file read every row group exactly once.
 *
 * @return Indices into @p metadata.row_groups, ascending.
 */
[[nodiscard]] std::vector<cudf::size_type> row_groups_in_byte_range(
  cudf::io::parquet::FileMetaData const& metadata, std::uint64_t start, std::uint64_t length);

}  // namespace sirius::op::scan::detail
