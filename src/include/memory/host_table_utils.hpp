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
#include <cudf_utils.hpp>

// duckdb
#include <duckdb/common/types.hpp>

// cudf
#include <cudf/detail/contiguous_split.hpp>
#include <cudf/types.hpp>

// standard library
#include <cstdint>
#include <vector>

namespace sirius {

/**
 * @brief Metadata node structure for serializing cudf column metadata.
 *
 * This structure is used to create host_table metadata, similar to cudf::pack(), but without a
 * cudf::table_view. This is important in contexts where data has not been migrated to CPU memory
 * from GPU memory, e.g., after table scans, into a `host_table_allocation`.
 *
 */
struct metadata_node {
  cudf::data_type type;
  cudf::size_type size;
  cudf::size_type null_count;
  int64_t data_offset;
  int64_t null_mask_offset;
  std::vector<metadata_node> children;
};

/**
 * @brief Create a flat metadata node for fixed-width types.
 *
 * @param type The DuckDB logical type of the column.
 * @param size The number of rows in the column.
 * @param null_count The number of NULL values in the column.
 * @param data_offset The byte offset of the column data within a compact allocation.
 * @param null_mask_offset The byte offset of the null mask within a compact allocation.
 * @return metadata_node The constructed metadata node.
 */
inline metadata_node make_flat_metadata_node(duckdb::LogicalType type,
                                             cudf::size_type size,
                                             cudf::size_type null_count,
                                             int64_t data_offset,
                                             int64_t null_mask_offset)
{
  return metadata_node{
    duckdb::GetCudfType(type), size, null_count, data_offset, null_mask_offset, {}};
}

/**
 * @brief Create a metadata node for VARCHAR type.
 *
 * @param size The number of rows in the column.
 * @param null_count The number of NULL values in the column.
 * @param data_offset The byte offset of the column data within a compact allocation.
 * @param null_mask_offset The byte offset of the null mask within a compact allocation.
 * @param offsets_offset The byte offset of the offsets array within a compact allocation.
 * @return metadata_node The constructed metadata node.
 */
inline metadata_node make_string_metadata_node(cudf::size_type size,
                                               cudf::size_type null_count,
                                               int64_t data_offset,
                                               int64_t null_mask_offset,
                                               int64_t offsets_offset)
{
  return metadata_node{
    cudf::data_type(cudf::type_id::STRING),
    size,
    null_count,
    data_offset,
    null_mask_offset,
    {metadata_node{cudf::data_type(cudf::type_id::INT64), size + 1, 0, offsets_offset, -1, {}}}};
}

/**
 * @brief Recursive helper to emit metadata nodes into a cudf::metadata_builder.
 *
 * @param mb The metadata builder to which to add the nodes.
 * @param n The current metadata node to process.
 */
static void emit(cudf::detail::metadata_builder& mb, metadata_node const& n)
{
  mb.add_column_info_to_meta(n.type,
                             n.size,
                             n.null_count,
                             n.data_offset,
                             n.null_mask_offset,
                             static_cast<cudf::size_type>(n.children.size()));
  for (auto const& child : n.children) {
    emit(mb, child);
  }
}

/**
 * @brief Pack metadata nodes into a byte vector.
 *
 * This function serializes a vector of metadata_node structures into a contiguous byte vector for
 * use with `host_table_allocation`s
 *
 * @param nodes The vector of metadata_node structures to pack.
 * @return std::vector<uint8_t> The packed byte vector representing the metadata.
 */
inline std::vector<uint8_t> pack_metadata_from_nodes(std::vector<metadata_node> const& nodes)
{
  if (nodes.empty()) { return {}; }
  cudf::detail::metadata_builder mb(static_cast<cudf::size_type>(nodes.size()));
  for (auto const& n : nodes) {
    emit(mb, n);
  }
  return mb.build();
};

}  // namespace sirius
