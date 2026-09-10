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

// Ingesting a .hpln file as a pinned compressed chunk.
//
// A pin today reads parquet, materializes it on the GPU, and compresses it on the way into the
// cache -- which is why pinning dominates the SF1000 wall clock (~151 s) while the queries it
// serves take ~9 s. A file that is ALREADY in the pinned representation needs none of that: the
// payload lands in pinned host blocks byte-for-byte as stored, so ingest is an I/O copy rather
// than a decode plus a re-compress. Nothing is decoded here and no GPU is touched.
//
// The served-from side then needs no new code at all: a compressed_host_representation built this
// way goes through the same converter as a pinned one, including the range-skipped fetch
// (CHUNK_SKIPPING_PLAN.md 6.5).

#include "compressed_representation.hpp"
#include "scan_manager/pinned_chunk_stats.hpp"

#include <cudf/table/table_view.hpp>

#include <api/compressed_table_io.hpp>
#include <cucascade/memory/memory_space.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace sirius {

/// A .hpln file staged into pinned host memory, plus what its header says is in it.
struct ingested_hpln {
  std::shared_ptr<pinned_compressed_blob> blob;
  simpatico::hpln_schema schema;
  /// Per-group min/max read straight out of the file's `zone_maps` segment. Empty when the file
  /// carries none, or when the segment did not decode — both mean "serve unpruned", never
  /// "prune wrongly". This is what lets an ingested table prune without decoding anything.
  scan_manager::group_bounds_arena group_bounds;
  /// The engine's logical type per column, from the file's `logical_types` segment. Empty when the
  /// file carries none — a caller then has only the cuDF physical types, which cannot express
  /// DECIMAL precision, nullability or a timestamp's zone.
  duckdb::vector<duckdb::LogicalType> column_types;
};

/// The schema a reader binds against: column names and the engine's logical types, obtained
/// without touching the payload and without a GPU.
struct hpln_bind_schema {
  std::vector<std::string> names;
  duckdb::vector<duckdb::LogicalType> types;
  /// The cuDF type each column DECODES to, which is not always the physical layout of @ref types:
  /// a DECIMAL(12,2) is INT64 to the engine but may be DECIMAL32 in the file. A reader sizing the
  /// decoded table has to size it by these.
  std::vector<cudf::data_type> physical_types;
  std::int64_t num_rows = 0;
};

/// Open @p path far enough to answer "what columns does this file have".
///
/// This is what a `read_simpatico()` bind needs, and what an ingestible's table_info reports. The
/// logical types come from the file's `logical_types` segment when it has one; otherwise they are
/// derived from the cuDF physical types in the header, which is lossy in exactly the ways
/// CHUNK_SKIPPING_PLAN.md 7.9 lists (DECIMAL precision, nullability, time zone) — so a file
/// written without that segment binds to approximate types rather than failing.
///
/// Throws std::runtime_error if the file cannot be read or parsed.
[[nodiscard]] hpln_bind_schema read_hpln_schema(std::string const& path);

/// Pack @p types into the bytes a `logical_types` segment carries, and back.
///
/// Positional with the header's columns. Only the types Sirius can pin are representable; an
/// unrepresentable one packs as SQLNULL, which a reader treats as "this column has no declared
/// type" rather than silently substituting a wrong one.
[[nodiscard]] std::vector<std::uint8_t> pack_logical_types(
  duckdb::vector<duckdb::LogicalType> const& types);
[[nodiscard]] duckdb::vector<duckdb::LogicalType> unpack_logical_types(
  std::span<const std::uint8_t> bytes, std::string* error = nullptr);

/// Read @p path into pinned host memory belonging to @p host_space.
///
/// The header is located by reading a speculative prefix and growing it if the parse reports
/// truncation -- .hpln carries no length prefix or footer, so its extent cannot be known without
/// parsing (see CHUNK_SKIPPING_PLAN.md 7.5). Throws std::runtime_error on a missing, truncated or
/// malformed file, since a partially ingested table must never become a pinned entry.
[[nodiscard]] ingested_hpln read_hpln_into_pinned(std::string const& path,
                                                  cucascade::memory::memory_space& host_space);

/// Compress @p table with @p plan_dsl and write it to @p path, carrying per-group zone maps.
///
/// The counterpart of read_hpln_into_pinned: without a writer that emits statistics there is
/// nothing for an ingesting reader to prune with, since the bounds cannot be recovered from
/// compressed bytes without decoding them. @p group_rows of 0 writes no zone-map segment.
///
/// Returns an empty string on success.
[[nodiscard]] std::string write_table_to_hpln(
  cudf::table_view const& table,
  duckdb::vector<duckdb::LogicalType> const& column_types,
  std::vector<std::string> const& column_names,
  std::string const& plan_dsl,
  std::size_t group_rows,
  std::string const& path,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace sirius
