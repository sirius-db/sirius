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

/**
 * NOTE:
 * This file provides utilities for building cucascade::memory::column_metadata
 * used to describe column layouts within a host_table_allocation.
 */

// sirius
#include <cudf_utils.hpp>

// duckdb
#include <duckdb/common/types.hpp>

// cucascade
#include <cucascade/memory/host_table.hpp>

// cudf
#include <cudf/types.hpp>

// standard library
#include <cstdint>
#include <vector>

namespace sirius {

using column_metadata = cucascade::memory::column_metadata;

/**
 * @brief Create column metadata for a fixed-width type.
 *
 * @param type The DuckDB logical type of the column.
 * @param num_rows The number of rows in the column.
 * @param null_count The number of NULL values in the column.
 * @param data_offset The byte offset of the column data within the allocation.
 * @param null_mask_offset The byte offset of the null mask within the allocation.
 * @return column_metadata The constructed column metadata.
 */
inline column_metadata make_flat_column_metadata(duckdb::LogicalType type,
                                                 cudf::size_type num_rows,
                                                 cudf::size_type null_count,
                                                 std::size_t data_offset,
                                                 std::size_t null_mask_offset)
{
  auto cudf_type = duckdb::GetCudfType(type);
  column_metadata cm;
  cm.type_id          = cudf_type.id();
  cm.num_rows         = num_rows;
  cm.null_count       = null_count;
  cm.scale            = cudf_type.scale();
  cm.has_data         = true;
  cm.data_offset      = data_offset;
  cm.data_size        = 0;
  cm.has_null_mask    = (null_count > 0);
  cm.null_mask_offset = cm.has_null_mask ? null_mask_offset : 0;
  cm.null_mask_size   = cm.has_null_mask ? static_cast<std::size_t>((num_rows + 7) / 8) : 0;
  return cm;
}

/**
 * @brief Create column metadata for a VARCHAR (STRING) type.
 *
 * @param num_rows The number of rows in the column.
 * @param null_count The number of NULL values in the column.
 * @param data_offset The byte offset of the string char data within the allocation.
 * @param null_mask_offset The byte offset of the null mask within the allocation.
 * @param offsets_offset The byte offset of the offsets array within the allocation.
 * @return column_metadata The constructed column metadata.
 */
inline column_metadata make_string_column_metadata(cudf::size_type num_rows,
                                                   cudf::size_type null_count,
                                                   std::size_t data_offset,
                                                   std::size_t null_mask_offset,
                                                   std::size_t offsets_offset)
{
  column_metadata cm;
  cm.type_id          = cudf::type_id::STRING;
  cm.num_rows         = num_rows;
  cm.null_count       = null_count;
  cm.scale            = 0;
  cm.has_data         = true;
  cm.data_offset      = data_offset;
  cm.data_size        = 0;
  cm.has_null_mask    = (null_count > 0);
  cm.null_mask_offset = cm.has_null_mask ? null_mask_offset : 0;
  cm.null_mask_size   = cm.has_null_mask ? static_cast<std::size_t>((num_rows + 7) / 8) : 0;

  // Offsets child column (INT32 offsets, num_rows + 1 entries, no nulls)
  column_metadata offsets_child;
  offsets_child.type_id          = cudf::type_id::INT32;
  offsets_child.num_rows         = num_rows + 1;
  offsets_child.null_count       = 0;
  offsets_child.scale            = 0;
  offsets_child.has_data         = true;
  offsets_child.data_offset      = offsets_offset;
  offsets_child.data_size        = 0;
  offsets_child.has_null_mask    = false;
  offsets_child.null_mask_offset = 0;
  offsets_child.null_mask_size   = 0;
  cm.children.push_back(std::move(offsets_child));

  return cm;
}

}  // namespace sirius
