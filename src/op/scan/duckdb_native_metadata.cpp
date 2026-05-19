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

#include "op/scan/duckdb_native_metadata.hpp"

#include "log/logging.hpp"

#include <duckdb/function/partition_stats.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/storage/table_storage_info.hpp>

#include <algorithm>
#include <cassert>
#include <charconv>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

// Keys sourced from `duckdb/common/enums/compression_type.cpp` at v1.5.2.
// "Empty Validity" (not "Empty") is the canonical spelling for COMPRESSION_EMPTY.
const std::unordered_map<std::string, duckdb::CompressionType>& compression_string_to_enum()
{
  static const std::unordered_map<std::string, duckdb::CompressionType> map = {
    {"Auto", duckdb::CompressionType::COMPRESSION_AUTO},
    {"Uncompressed", duckdb::CompressionType::COMPRESSION_UNCOMPRESSED},
    {"Constant", duckdb::CompressionType::COMPRESSION_CONSTANT},
    {"RLE", duckdb::CompressionType::COMPRESSION_RLE},
    {"Dictionary", duckdb::CompressionType::COMPRESSION_DICTIONARY},
    {"PFOR", duckdb::CompressionType::COMPRESSION_PFOR_DELTA},
    {"BitPacking", duckdb::CompressionType::COMPRESSION_BITPACKING},
    {"FSST", duckdb::CompressionType::COMPRESSION_FSST},
    {"Chimp", duckdb::CompressionType::COMPRESSION_CHIMP},
    {"Patas", duckdb::CompressionType::COMPRESSION_PATAS},
    {"ZSTD", duckdb::CompressionType::COMPRESSION_ZSTD},
    {"ALP", duckdb::CompressionType::COMPRESSION_ALP},
    {"ALPRD", duckdb::CompressionType::COMPRESSION_ALPRD},
    {"Roaring", duckdb::CompressionType::COMPRESSION_ROARING},
    {"DICT_FSST", duckdb::CompressionType::COMPRESSION_DICT_FSST},
    {"Empty Validity", duckdb::CompressionType::COMPRESSION_EMPTY},
  };
  return map;
}

// Unrecognized strings map to COMPRESSION_COUNT, DuckDB's trailing sentinel
// — `is_supported_*` rejects it via the default arm.
duckdb::CompressionType parse_compression_string(const std::string& s)
{
  const auto& map = compression_string_to_enum();
  auto it         = map.find(s);
  if (it == map.end()) { return duckdb::CompressionType::COMPRESSION_COUNT; }
  return it->second;
}

// `StandardColumnData::GetColumnSegmentInfo` recurses into the validity child
// with `col_path.push_back(0)` — "[col, 0]" for validity, "[col]" for data.
bool is_validity_path(const std::string& column_path)
{
  return column_path.find(',') != std::string::npos;
}

// Exhaustive switch: a new `sirius::type_id` enumerator should compile-fail
// here rather than be silently accepted.
bool is_supported_logical_type(const sirius::logical_type& type, std::string& reason_out)
{
  switch (type.id()) {
    case sirius::type_id::HUGEINT:
    case sirius::type_id::UHUGEINT:
      reason_out = "type " + type.to_string() + " has 128-bit storage; sirius decode lacks it";
      return false;
    case sirius::type_id::STRUCT:
    case sirius::type_id::LIST:
      reason_out =
        "type " + type.to_string() + " is a nested type; sirius decode does not support it";
      return false;
    case sirius::type_id::INVALID:
    case sirius::type_id::SQLNULL:
      reason_out = "type " + type.to_string() + " is a sentinel; not a valid scan column type";
      return false;
    case sirius::type_id::DECIMAL:
      // <=18 → DECIMAL64 (supported); >18 → DECIMAL128 (no decode path).
      if (type.decimal_precision() > sirius::logical_type::decimal_max_precision_int64) {
        reason_out = "type " + type.to_string() + " has DECIMAL128 storage; sirius decode lacks it";
        return false;
      }
      return true;
    case sirius::type_id::BOOLEAN:
    case sirius::type_id::TINYINT:
    case sirius::type_id::UTINYINT:
    case sirius::type_id::SMALLINT:
    case sirius::type_id::USMALLINT:
    case sirius::type_id::INTEGER:
    case sirius::type_id::UINTEGER:
    case sirius::type_id::BIGINT:
    case sirius::type_id::UBIGINT:
    case sirius::type_id::FLOAT:
    case sirius::type_id::DOUBLE:
    case sirius::type_id::DATE:
    case sirius::type_id::TIMESTAMP_SEC:
    case sirius::type_id::TIMESTAMP_MS:
    case sirius::type_id::TIMESTAMP:
    case sirius::type_id::TIMESTAMP_NS:
    case sirius::type_id::VARCHAR: return true;
  }
  reason_out = "type " + type.to_string() + " is not enumerated by the walker viability switch";
  return false;
}

// Parse the per-segment Max String Length out of DuckDB's stats blob
// (`ColumnSegmentInfo::segment_stats`). Format:
//   "[Min: ..., Max: ..., Has Unicode: ..., Max String Length: N][Has Null: ...]"
// The "Max String Length: " token only appears in StringStats, so a forward
// strstr is safe (no risk of matching a `Max:` value containing the word).
//
// Returns nullopt iff the field is absent from the blob — meaning the segment
// was written without string stats (pathological on modern DuckDB; every
// VARCHAR storage path in duckdb/src/storage/compression/*.cpp updates these
// stats unconditionally).
std::optional<std::uint32_t> parse_segment_max_string_length(std::string_view blob)
{
  constexpr std::string_view kNeedle = "Max String Length: ";
  auto pos                           = blob.find(kNeedle);
  if (pos == std::string_view::npos) { return std::nullopt; }
  pos += kNeedle.size();
  std::uint32_t value = 0;
  auto [ptr, ec]      = std::from_chars(blob.data() + pos, blob.data() + blob.size(), value);
  if (ec != std::errc{}) { return std::nullopt; }
  return value;
}

duckdb_segment_descriptor build_segment_descriptor(const duckdb::ColumnSegmentInfo& seg,
                                                   duckdb::CompressionType compression)
{
  duckdb_segment_descriptor desc{};
  desc.block_id          = seg.block_id;
  desc.additional_blocks = seg.additional_blocks;
  desc.block_offset      = seg.block_offset;
  desc.segment_start     = seg.segment_start;
  desc.segment_count     = seg.segment_count;
  desc.compression       = compression;
  return desc;
}

using row_group_handles =
  std::vector<std::pair<duckdb::idx_t, duckdb::shared_ptr<duckdb::PartitionRowGroup>>>;

// Use PartitionStatistics::count as the source of truth so rowid-only
// projections (no data segments to sum) still get a real row_count.
void compute_row_counts(duckdb_native_metadata& md,
                        const std::vector<duckdb::PartitionStatistics>& partition_stats)
{
  for (auto& rg_md : md.row_groups) {
    if (rg_md.row_group_index < partition_stats.size()) {
      rg_md.row_count = partition_stats[rg_md.row_group_index].count;
      continue;
    }
    duckdb::idx_t row_count = 0;
    for (const auto& col_md : rg_md.columns) {
      if (col_md.is_rowid) { continue; }
      duckdb::idx_t col_count = 0;
      for (const auto& d : col_md.data_segments) {
        col_count += d.segment_count;
      }
      if (col_count > 0) {
        row_count = col_count;
        break;
      }
    }
    rg_md.row_count = row_count;
  }
}

// Per-segment chars exact: budget = Σ(seg.count × *seg.max_string_length) + offsets.
// The walker refuses on absent stat, so every VARCHAR data segment is guaranteed
// Some(...) when this runs.
void compute_decoded_byte_budgets(duckdb_native_metadata& md,
                                  const std::vector<sirius::logical_type>& projected_types)
{
  for (auto& rg_md : md.row_groups) {
    std::size_t budget = 0;
    for (std::size_t ci = 0; ci < rg_md.columns.size(); ++ci) {
      const auto& col_md = rg_md.columns[ci];
      if (col_md.is_rowid) {
        budget += static_cast<std::size_t>(rg_md.row_count) * sizeof(std::int64_t);
        continue;
      }
      if (projected_types[ci].is_varchar()) {
        std::size_t chars_total = 0;
        for (const auto& seg : col_md.data_segments) {
          chars_total +=
            static_cast<std::size_t>(seg.segment_count) * static_cast<std::size_t>(*seg.max_string_length);
        }
        budget += chars_total + static_cast<std::size_t>(rg_md.row_count) * sizeof(std::uint32_t);
      } else {
        budget +=
          static_cast<std::size_t>(rg_md.row_count) * projected_types[ci].fixed_width_byte_size();
      }
    }
    rg_md.decoded_bytes_budget = budget;
  }
}

// The walker over-allocates to `max_row_group_index + 1`; row groups
// whose every projected segment was filtered out arrive here with
// row_count == 0. Rowid-only entries are kept.
void drop_empty_trailing_row_groups(duckdb_native_metadata& md)
{
  while (!md.row_groups.empty() && md.row_groups.back().row_count == 0) {
    bool has_rowid = false;
    for (const auto& col_md : md.row_groups.back().columns) {
      if (col_md.is_rowid) {
        has_rowid = true;
        break;
      }
    }
    if (has_rowid) { break; }
    md.row_groups.pop_back();
  }
}

}  // namespace

bool is_supported_data_compression(duckdb::CompressionType c)
{
  switch (c) {
    case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
    case duckdb::CompressionType::COMPRESSION_CONSTANT:
    case duckdb::CompressionType::COMPRESSION_RLE:
    case duckdb::CompressionType::COMPRESSION_DICTIONARY:
    case duckdb::CompressionType::COMPRESSION_BITPACKING:
    case duckdb::CompressionType::COMPRESSION_FSST:
    case duckdb::CompressionType::COMPRESSION_DICT_FSST:
    case duckdb::CompressionType::COMPRESSION_ALP:
    case duckdb::CompressionType::COMPRESSION_ALPRD: return true;
    default: return false;
  }
}

bool is_supported_validity_compression(duckdb::CompressionType c)
{
  switch (c) {
    // CONSTANT is the all-valid case (all-null columns land in EMPTY).
    // ROARING is host-decoded to a plain bitmap before the GPU sees it.
    case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
    case duckdb::CompressionType::COMPRESSION_EMPTY:
    case duckdb::CompressionType::COMPRESSION_CONSTANT:
    case duckdb::CompressionType::COMPRESSION_ROARING: return true;
    default: return false;
  }
}

duckdb_native_metadata walk_duckdb_native_metadata(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  const std::vector<projected_column>& projected_cols,
  const std::vector<sirius::logical_type>& projected_types)
{
  duckdb_native_metadata md;
  md.viable = false;

  auto refuse = [&md](std::string reason) {
    md.viability_failure_reason = std::move(reason);
    SIRIUS_LOG_DEBUG("[duckdb_native_metadata] refused: {}", md.viability_failure_reason);
  };

  if (projected_cols.empty()) {
    refuse("no projected columns");
    return md;
  }
  if (projected_cols.size() != projected_types.size()) {
    refuse("projected_cols and projected_types size mismatch");
    return md;
  }

  for (std::size_t ci = 0; ci < projected_types.size(); ++ci) {
    if (projected_cols[ci].is_rowid) { continue; }
    std::string reason;
    if (!is_supported_logical_type(projected_types[ci], reason)) {
      refuse("column " + std::to_string(projected_cols[ci].storage_idx.GetPrimaryIndex()) + ": " +
             reason);
      return md;
    }
  }

  // PartitionStatistics order matches `RowGroupCollection::SegmentNodes()`
  // iteration order at v1.5.2 — relied on for indexing by row_group_index.
  auto partition_stats = storage.GetPartitionStats(context);
  row_group_handles handles;
  handles.reserve(partition_stats.size());
  for (std::size_t i = 0; i < partition_stats.size(); ++i) {
    auto& ps                = partition_stats[i];
    duckdb::idx_t row_start = 0;
    if (ps.row_start.IsValid()) {
      row_start = ps.row_start.GetIndex();
    } else {
      // Not expected on a materialized row group at v1.5.2; rowid synthesis
      // for this group would emit incorrect ids.
      SIRIUS_LOG_WARN(
        "[duckdb_native_metadata] partition_stats[{}].row_start is not valid; defaulting to 0", i);
    }
    handles.emplace_back(row_start, ps.partition_row_group);
  }

  // O(1) skip for non-projected columns in the GetColumnSegmentInfo loop.
  std::unordered_map<duckdb::idx_t, std::size_t> projected_lookup;
  projected_lookup.reserve(projected_cols.size());
  for (std::size_t ci = 0; ci < projected_cols.size(); ++ci) {
    if (projected_cols[ci].is_rowid) { continue; }
    projected_lookup.emplace(projected_cols[ci].storage_idx.GetPrimaryIndex(), ci);
  }

  duckdb::QueryContext qc{context};
  auto column_segments = storage.GetColumnSegmentInfo(qc);

  // Size to max_row_group_index + 1 so rowid-only and all-filtered row
  // groups still have an entry for rowid synthesis and trailing-empty
  // pruning to inspect.
  duckdb::idx_t max_rg_idx = 0;
  for (const auto& seg : column_segments) {
    max_rg_idx = std::max(max_rg_idx, seg.row_group_index);
  }
  const std::size_t num_row_groups =
    column_segments.empty() ? 0 : static_cast<std::size_t>(max_rg_idx) + 1;
  md.row_groups.resize(num_row_groups);
  for (std::size_t rg = 0; rg < num_row_groups; ++rg) {
    md.row_groups[rg].row_group_index = rg;
    md.row_groups[rg].columns.resize(projected_cols.size());
    for (std::size_t ci = 0; ci < projected_cols.size(); ++ci) {
      md.row_groups[rg].columns[ci].column_id =
        projected_cols[ci].is_rowid ? std::numeric_limits<duckdb::idx_t>::max()
                                    : projected_cols[ci].storage_idx.GetPrimaryIndex();
      md.row_groups[rg].columns[ci].is_rowid = projected_cols[ci].is_rowid;
    }
    if (rg < handles.size()) { md.row_groups[rg].row_group_start = handles[rg].first; }
  }

  for (const auto& seg : column_segments) {
    auto pl = projected_lookup.find(seg.column_id);
    if (pl == projected_lookup.end()) { continue; }  // not projected
    const std::size_t ci    = pl->second;
    const auto rg_idx       = seg.row_group_index;
    const bool validity_seg = is_validity_path(seg.column_path);
    const auto compression  = parse_compression_string(seg.compression_type);

    const bool compression_ok = validity_seg ? is_supported_validity_compression(compression)
                                             : is_supported_data_compression(compression);
    if (!compression_ok) {
      refuse(std::string{validity_seg ? "validity" : "data"} + " segment on column " +
             std::to_string(seg.column_id) + " row group " + std::to_string(rg_idx) +
             ": unsupported compression \"" + seg.compression_type + "\"");
      return md;
    }

    auto desc = build_segment_descriptor(seg, compression);

    if (!validity_seg && projected_types[ci].is_varchar()) {
      // Per-segment Max String Length from the stats blob. Every VARCHAR
      // storage path in DuckDB writes this (UpdateStringStats in
      // duckdb/src/storage/compression/*.cpp and string_uncompressed.hpp), so
      // a missing field is pathological — refuse the row group rather than
      // silently fall back to a guessed bound. Some(0) is legal data (every
      // value in this segment is "") — accept; the kernel/partitioner size
      // a zero-byte chars region for it.
      desc.max_string_length = parse_segment_max_string_length(seg.segment_stats);
      if (!desc.max_string_length.has_value()) {
        refuse("varchar segment on column " + std::to_string(seg.column_id) + " row group " +
               std::to_string(rg_idx) + ": Max String Length stat absent from segment_stats");
        return md;
      }
    }

    auto& col_md = md.row_groups[rg_idx].columns[ci];
    if (validity_seg) {
      col_md.validity_segments.push_back(std::move(desc));
    } else {
      col_md.data_segments.push_back(std::move(desc));
    }
  }

#ifndef NDEBUG
  // Invariant check: DuckDB's typed `PartitionRowGroup::GetColumnStatistics`
  // returns max(per-segment max) via StringStats::Merge (see
  // duckdb/src/storage/statistics/string_stats.cpp:181). If the assert ever
  // fires, DuckDB has changed its merge semantics or we've parsed the blob
  // wrong — either way the per-segment data we ship to the kernel is
  // suspect.
  for (std::size_t rg_idx = 0; rg_idx < handles.size(); ++rg_idx) {
    auto& prg = handles[rg_idx].second;
    if (!prg) { continue; }
    for (std::size_t ci = 0; ci < projected_cols.size(); ++ci) {
      if (projected_cols[ci].is_rowid || !projected_types[ci].is_varchar()) { continue; }
      auto stats = prg->GetColumnStatistics(projected_cols[ci].storage_idx);
      if (!stats || !duckdb::StringStats::HasMaxStringLength(*stats)) { continue; }
      const auto rg_typed_max  = duckdb::StringStats::MaxStringLength(*stats);
      const auto& data_segs    = md.row_groups[rg_idx].columns[ci].data_segments;
      std::uint32_t per_seg_max = 0;
      for (const auto& seg : data_segs) {
        if (seg.max_string_length.has_value()) {
          per_seg_max = std::max(per_seg_max, *seg.max_string_length);
        }
      }
      assert(rg_typed_max == per_seg_max &&
             "DuckDB rg-level MaxStringLength != max(per-segment) — Merge semantics drifted?");
    }
  }
#endif

  // GetColumnSegmentInfo at v1.5.2 already yields segments in segment_start
  // order per (column, row group); this sort is a guard against future
  // upstream changes to that order.
  for (auto& rg_md : md.row_groups) {
    auto seg_less = [](const duckdb_segment_descriptor& a, const duckdb_segment_descriptor& b) {
      return a.segment_start < b.segment_start;
    };
    for (auto& col_md : rg_md.columns) {
      std::sort(col_md.data_segments.begin(), col_md.data_segments.end(), seg_less);
      std::sort(col_md.validity_segments.begin(), col_md.validity_segments.end(), seg_less);
    }
  }

  compute_row_counts(md, partition_stats);
  compute_decoded_byte_budgets(md, projected_types);
  drop_empty_trailing_row_groups(md);

  md.viable = true;
  SIRIUS_LOG_DEBUG(
    "[duckdb_native_metadata] walked {} row groups across {} projected columns; viable=true",
    md.row_groups.size(),
    projected_cols.size());
  return md;
}

}  // namespace sirius::op::scan
