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

#include "io/types.hpp"
#include "op/scan/table_scan/parquet_physical_profile.hpp"

#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_schema.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {

/// Parquet-flavored @c io_object_metadata stored in the ioctx metadata
/// store alongside an io_object.  Holds the parsed @c FileMetaData so future
/// scans of the same file can skip the footer fetch + parse and construct a
/// @c hybrid_scan_reader directly from the cached struct.  @c footer_byte_len
/// is the body size returned by @c fetch_footer_to_host (excludes the 8-byte
/// trailer) — kept here so callers reassembling the footer byte range for
/// prefetch don't have to round-trip through the datasource.
///
/// Lives with the parquet ingestible (its only producer/consumer): the bind
/// path (@c sirius_scan_manager::describe_parquet) parses and parks it, and the
/// metadata scan (@c parquet_gpu_ingestible::build_file_scan_info) reuses it.
class parquet_metadata final : public sirius::io::io_object_metadata,
                               public std::enable_shared_from_this<parquet_metadata> {
 public:
  parquet_metadata(std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata,
                   std::size_t footer_byte_len,
                   parquet_encryption_evidence encryption            = {},
                   std::vector<uint8_t> original_schema              = {},
                   std::string arrow_schema                          = {},
                   std::vector<uint8_t> original_logical_annotations = {})
    : encryption_evidence(encryption),
      original_schema(std::move(original_schema)),
      arrow_schema(std::move(arrow_schema)),
      original_logical_annotations(std::move(original_logical_annotations)),
      _file_metadata(std::move(file_metadata)),
      _footer_byte_len(footer_byte_len)
  {
    account_retention();
  }

  [[nodiscard]] std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata() const
  {
    // Export an alias that owns the entire evidence record, including accounting.
    // Removing the cache entry cannot free evidence still used by a split.
    return {shared_from_this(), _file_metadata.get()};
  }

  /// Conservative retained allocation estimate, computed outside cache locks.
  /// Includes vector capacities and nested statistics, not compressed footer bytes.
  [[nodiscard]] std::size_t retained_bytes() const noexcept override
  {
    size_t bytes = sizeof(*this) + sizeof(cudf::io::parquet::FileMetaData) + 64;
    auto add     = [&](size_t n) {
      bytes = n > std::numeric_limits<size_t>::max() - bytes ? std::numeric_limits<size_t>::max()
                                                                 : bytes + n;
    };
    auto vector = [&](auto const& v) {
      add(v.capacity() * sizeof(typename std::decay_t<decltype(v)>::value_type));
    };
    auto optional_vector = [&](auto const& v) {
      if (v) vector(*v);
    };
    auto string = [&](auto const& v) { add(v.capacity() + 1); };
    vector(original_schema);
    vector(original_logical_annotations);
    string(arrow_schema);
    if (!_file_metadata) return bytes;
    auto const& f = *_file_metadata;
    vector(f.schema);
    for (auto const& s : f.schema) {
      string(s.name);
      vector(s.children_idx);
    }
    vector(f.key_value_metadata);
    for (auto const& kv : f.key_value_metadata) {
      string(kv.key);
      string(kv.value);
    }
    string(f.created_by);
    optional_vector(f.column_orders);
    vector(f.row_groups);
    for (auto const& rg : f.row_groups) {
      vector(rg.columns);
      optional_vector(rg.sorting_columns);
      for (auto const& c : rg.columns) {
        string(c.file_path);
        auto const& m = c.meta_data;
        vector(m.encodings);
        vector(m.path_in_schema);
        for (auto const& p : m.path_in_schema)
          string(p);
        optional_vector(m.statistics.min);
        optional_vector(m.statistics.max);
        optional_vector(m.statistics.min_value);
        optional_vector(m.statistics.max_value);
        optional_vector(m.encoding_stats);
        if (m.size_statistics) {
          optional_vector(m.size_statistics->repetition_level_histogram);
          optional_vector(m.size_statistics->definition_level_histogram);
        }
        if (c.offset_index) {
          vector(c.offset_index->page_locations);
          optional_vector(c.offset_index->unencoded_byte_array_data_bytes);
        }
        if (c.column_index) {
          auto const& ci = *c.column_index;
          add(ci.null_pages.capacity());
          vector(ci.min_values);
          vector(ci.max_values);
          for (auto const& v : ci.min_values)
            vector(v);
          for (auto const& v : ci.max_values)
            vector(v);
          optional_vector(ci.null_counts);
          optional_vector(ci.repetition_level_histogram);
          optional_vector(ci.definition_level_histogram);
        }
      }
    }
    return bytes;
  }

  [[nodiscard]] std::size_t footer_byte_len() const noexcept { return _footer_byte_len; }

  parquet_encryption_evidence const encryption_evidence;
  // Retained before hybrid_scan_reader normalizes REQUIRED fields to OPTIONAL.
  std::vector<uint8_t> const original_schema;
  std::string const arrow_schema;
  // One byte per raw SchemaElement; distinguishes a real logicalType from a
  // converted_type annotation synthesized by cuDF during sanitize_schema.
  std::vector<uint8_t> const original_logical_annotations;

 private:
  std::shared_ptr<cudf::io::parquet::FileMetaData const> _file_metadata;
  std::size_t _footer_byte_len{0};
};

// The three footer producers must publish the same complete evidence. Cache hits
// return the very same record, including the original serialized schema.
std::shared_ptr<parquet_metadata> resolve_parquet_metadata(
  io::sirius_datasource& source,
  scan_contract_id contract,
  std::string const& identity,
  cudf::io::parquet_reader_options const& options,
  bool* cache_hit = nullptr);

}  // namespace sirius::op::scan
