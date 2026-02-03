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
 * This file provides utilities for working with cudf metadata.
 * See cudf::pack() and cudf::detail::unpack() for reference.
 * See https://github.com/rapidsai/cudf/issues/20966 for a feature request to make similar
 * functionality available via cudf itself.
 */

// sirius
#include <cudf_utils.hpp>

// duckdb
#include <duckdb/common/types.hpp>

// cudf
#include <cudf/detail/contiguous_split.hpp>
#include <cudf/types.hpp>

// standard library
#include <cstdint>
#include <cstring>
#include <stdexcept>
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

namespace detail {
/**
 * @brief Recursive helper to emit metadata nodes into a cudf::metadata_builder.
 *
 * @param mb The metadata builder to which to add the nodes.
 * @param n The current metadata node to process.
 */
inline void emit(cudf::detail::metadata_builder& mb, metadata_node const& n)
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
}  // namespace detail

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
    detail::emit(mb, n);
  }
  return mb.build();
};

namespace detail {

/**
 * @brief Local copy of cudf::detail::serialized_column for metadata parsing.
 *
 * This struct is a copy of cudf::used by cudf::pack(). See
 * https://github.com/rapidsai/cudf/blob/d63d978bf949e278a650a829b85c0744e52d60b0/cpp/src/copying/pack.cpp#L31
 */
struct serialized_column {
  cudf::data_type type;
  cudf::size_type size;
  cudf::size_type null_count;
  int64_t data_offset;       // offset into contiguous data buffer, or -1 if column data is null
  int64_t null_mask_offset;  // offset into contiguous data buffer, or -1 if column data is null
  cudf::size_type num_children;
  // Explicitly pad to avoid uninitialized padding bits, allowing `serialized_column` to be bit-wise
  // comparable
  int pad{};
};

/**
 * @brief Recursive helper to unpack metadata nodes from serialized columns.
 *
 * @param columns Pointer to the array of serialized_column structures.
 * @param column_count The total number of serialized columns.
 * @param current_index Reference to the current index in the columns array.
 * @param num_columns The number of columns to unpack at this level.
 * @return std::vector<metadata_node> The unpacked metadata nodes.
 */
inline std::vector<metadata_node> unpack_nodes(serialized_column const* columns,
                                               size_t column_count,
                                               size_t& current_index,
                                               size_t num_columns)
{
  std::vector<metadata_node> nodes;
  nodes.reserve(num_columns);
  for (size_t i = 0; i < num_columns; ++i) {
    if (current_index >= column_count) {
      throw std::runtime_error("metadata underflow while parsing host table metadata");
    }

    auto const& col = columns[current_index++];
    metadata_node node{
      col.type, col.size, col.null_count, col.data_offset, col.null_mask_offset, {}};
    if (col.num_children > 0) {
      node.children =
        unpack_nodes(columns, column_count, current_index, static_cast<size_t>(col.num_children));
    }
    nodes.push_back(std::move(node));
  }
  return nodes;
}

}  // namespace detail

/**
 * @brief Unpack host table metadata into a vector of metadata_node objects.
 *
 * This mirrors cudf::detail::unpack(), but returns the tree of metadata_node objects for use
 * in host-side accessors without constructing cudf::column_view.
 *
 * @param metadata Packed metadata buffer from cudf::pack().
 * @return std::vector<metadata_node> Parsed metadata nodes.
 */
inline std::vector<metadata_node> unpack_metadata_to_nodes(
  std::unique_ptr<std::vector<uint8_t>> const& metadata)
{
  // Sanity checks
  if (metadata->empty()) { return {}; }
  if (metadata->size() % sizeof(detail::serialized_column) != 0) {
    throw std::runtime_error("Invalid metadata size for host table unpack");
  }

  auto const column_count = metadata->size() / sizeof(detail::serialized_column);
  auto const* raw_ptr     = metadata->data();

  // Ensure proper alignment
  detail::serialized_column const* columns = nullptr;
  if (reinterpret_cast<uintptr_t>(raw_ptr) % alignof(detail::serialized_column) != 0) {
    std::vector<detail::serialized_column> aligned(column_count);
    std::memcpy(aligned.data(), raw_ptr, metadata->size());
    columns = aligned.data();
  } else {
    columns = reinterpret_cast<detail::serialized_column const*>(raw_ptr);
  }

  // Invoke recursive unpack helper
  auto const num_columns = static_cast<size_t>(columns[0].size);
  size_t current_index   = 1;
  auto nodes             = detail::unpack_nodes(columns, column_count, current_index, num_columns);

  if (current_index != column_count) {
    throw std::runtime_error("Metadata size mismatch while parsing host table metadata");
  }

  return nodes;
}

}  // namespace sirius
