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

#include <cudf/io/parquet_schema.hpp>

#include <cstddef>
#include <memory>
#include <utility>

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
class parquet_metadata final : public sirius::io::io_object_metadata {
 public:
  parquet_metadata(std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata,
                   std::size_t footer_byte_len)
    : _file_metadata(std::move(file_metadata)), _footer_byte_len(footer_byte_len)
  {
  }

  [[nodiscard]] std::shared_ptr<cudf::io::parquet::FileMetaData const> const& file_metadata()
    const noexcept
  {
    return _file_metadata;
  }

  [[nodiscard]] std::size_t footer_byte_len() const noexcept { return _footer_byte_len; }

 private:
  std::shared_ptr<cudf::io::parquet::FileMetaData const> _file_metadata;
  std::size_t _footer_byte_len{0};
};

}  // namespace sirius::op::scan
