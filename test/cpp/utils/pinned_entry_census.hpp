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

// Storage census of a pinned entry, shared by the tests that assert what a pin actually stored.
// A pin can look correct from its query results while having silently failed to compress or to
// narrow, so any test whose subject is the pin itself censuses the entry rather than trusting a
// counter.

#include "sirius_context.hpp"

#include <catch.hpp>
#include <compression/compressed_representation.hpp>
#include <cudf/types.hpp>
#include <duckdb.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <utils/transparent_execution_test_utils.hpp>

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace sirius::test {

/// What a pinned entry holds: chunk count, how many chunks are compressed representations, total
/// stored bytes, the recorded storage metadata's narrowing shape, and the recorded carrier of
/// every cell of the first chunk.
struct pinned_entry_census {
  std::size_t chunks{0};
  std::size_t compressed_chunks{0};
  std::size_t stored_bytes{0};
  bool any_marker_true{false};
  bool all_columns_narrowed{false};
  std::vector<cudf::data_type> first_chunk_carriers;
};

/// Census the pinned entry named @p name on @p con. Reads host_chunks or device_chunks per the
/// entry's form. Fails the calling assertion when no entry of that name holds any chunk.
inline pinned_entry_census census_entry(duckdb::Connection& con, const std::string& name)
{
  pinned_entry_census out;
  auto ctx = sirius::test::get_registered_sirius_context(con);
  REQUIRE(ctx);
  ctx->get_scan_manager().visit_pinned_entries(
    [&](std::string_view entry_name, sirius::scan_manager::pinned_entry const& entry) {
      if (entry_name != name) { return true; }
      out.all_columns_narrowed = !entry.column_storage.empty();
      for (auto const& row : entry.column_storage) {
        for (auto const& column : row) {
          out.any_marker_true      = out.any_marker_true || column.narrowed;
          out.all_columns_narrowed = out.all_columns_narrowed && column.narrowed;
        }
      }
      if (!entry.column_storage.empty()) {
        for (auto const& column : entry.column_storage.front()) {
          out.first_chunk_carriers.push_back(column.carrier);
        }
      }
      if (!entry.device_chunks.empty()) {
        out.chunks = entry.device_chunks.size();
        for (auto const& chunk : entry.device_chunks) {
          if (chunk.compressed) {
            ++out.compressed_chunks;
            out.stored_bytes += chunk.compressed->get_size_in_bytes();
          } else {
            for (auto const& column : chunk.columns) {
              REQUIRE(column);
              out.stored_bytes += column->alloc_size();
            }
          }
        }
        return false;
      }
      out.chunks = entry.host_chunks.size();
      for (auto const& chunk : entry.host_chunks) {
        REQUIRE(chunk);
        out.stored_bytes += chunk->get_size_in_bytes();
        if (dynamic_cast<sirius::compressed_host_representation const*>(chunk.get())) {
          ++out.compressed_chunks;
        }
      }
      return false;
    });
  REQUIRE(out.chunks > 0);
  return out;
}

}  // namespace sirius::test
