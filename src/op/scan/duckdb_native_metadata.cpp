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
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

// Strings sourced from `duckdb/src/common/enums/compression_type.cpp` at v1.5.2.
// Note: "Empty Validity" (NOT "Empty") is the canonical spelling.
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

duckdb::CompressionType parse_compression_string(const std::string& s)
{
  const auto& map = compression_string_to_enum();
  auto it         = map.find(s);
  if (it == map.end()) { return duckdb::CompressionType::COMPRESSION_AUTO; }
  return it->second;
}

// `column_path` formatter:
// `StandardColumnData::GetColumnSegmentInfo` recurses into the validity child
// with `col_path.push_back(0)`, producing "[col, 0]" vs "[col]" for data.
bool is_validity_path(const std::string& column_path)
{
  return column_path.find(',') != std::string::npos;
}

// Complete switch — every `sirius::type_id` must appear so a future
// enumerator becomes a compile error rather than a silent accept.
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
    // CONSTANT here is in practice the all-valid case (genuinely all-null
    // columns get EMPTY); the decoder's all-valid pre-fill covers it.
    // ROARING is host-decoded to a regular bitmap before the GPU sees it.
    case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
    case duckdb::CompressionType::COMPRESSION_EMPTY:
    case duckdb::CompressionType::COMPRESSION_CONSTANT:
    case duckdb::CompressionType::COMPRESSION_ROARING: return true;
    default: return false;
  }
}

std::uint32_t lookup_max_string_length(duckdb::PartitionRowGroup& prg,
                                       const duckdb::StorageIndex& storage_idx)
{
  auto stats = prg.GetColumnStatistics(storage_idx);
  if (!stats) { return 0; }
  if (!duckdb::StringStats::HasMaxStringLength(*stats)) { return 0; }
  return duckdb::StringStats::MaxStringLength(*stats);
}

using row_group_handles =
  std::vector<std::pair<duckdb::idx_t, duckdb::shared_ptr<duckdb::PartitionRowGroup>>>;

class varchar_max_length_resolver {
 public:
  varchar_max_length_resolver(const row_group_handles& handles,
                              const std::vector<projected_column>& projected_cols)
    : _handles(handles), _projected_cols(projected_cols)
  {
  }

  /// Returns 0 when the row group did not advertise the stat.
  std::uint32_t get(duckdb::idx_t rg_idx, duckdb::idx_t col_id, std::size_t projected_ci)
  {
    const auto k = key(rg_idx, col_id);
    auto it      = _cache.find(k);
    if (it != _cache.end()) { return it->second; }
    std::uint32_t max_len = 0;
    if (rg_idx < _handles.size() && _handles[rg_idx].second) {
      max_len = lookup_max_string_length(*_handles[rg_idx].second,
                                         _projected_cols[projected_ci].storage_idx);
    }
    _cache.emplace(k, max_len);
    return max_len;
  }

 private:
  // col_id is narrowed to uint32 — a schema with >2^32 columns would alias
  // in the cache. Realistic schemas are far below that.
  static std::uint64_t key(duckdb::idx_t rg, duckdb::idx_t col)
  {
    return (static_cast<std::uint64_t>(rg) << 32) | static_cast<std::uint32_t>(col);
  }

  const row_group_handles& _handles;
  const std::vector<projected_column>& _projected_cols;
  std::unordered_map<std::uint64_t, std::uint32_t> _cache;
};

duckdb_segment_descriptor build_segment_descriptor(const duckdb::ColumnSegmentInfo& seg,
                                                   duckdb::CompressionType compression,
                                                   std::uint32_t max_string_length)
{
  duckdb_segment_descriptor desc{};
  desc.block_id          = seg.block_id;
  desc.additional_blocks = seg.additional_blocks;
  desc.block_offset      = seg.block_offset;
  desc.segment_start     = seg.segment_start;
  desc.segment_count     = seg.segment_count;
  desc.compression       = compression;
  desc.max_string_length = max_string_length;
  return desc;
}

// PartitionStatistics::count is the source of truth — it's correct even when
// the projection is rowid-only (no data segments to sum) or when every
// projected non-rowid column happens to have zero data segments. Falls back
// to summing data segments only when partition stats don't cover this row
// group (shouldn't happen at v1.5.2 but defensive).
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

// VARCHAR with no advertised max-string-length falls back to
// VARCHAR_UNKNOWN_LENGTH_FALLBACK_BYTES per row and flips the row group's
// `decoded_bytes_budget_is_lower_bound`. Without the flag, downstream
// split-batch sizing would silently under-count when the actual strings
// exceed the fallback.
void compute_decoded_byte_budgets(duckdb_native_metadata& md,
                                  const std::vector<sirius::logical_type>& projected_types)
{
  for (auto& rg_md : md.row_groups) {
    std::size_t budget = 0;
    bool any_unknown   = false;
    for (std::size_t ci = 0; ci < rg_md.columns.size(); ++ci) {
      const auto& col_md = rg_md.columns[ci];
      if (col_md.is_rowid) {
        budget += static_cast<std::size_t>(rg_md.row_count) * sizeof(std::int64_t);
        continue;
      }
      if (projected_types[ci].is_varchar()) {
        std::uint32_t max_len = 0;
        for (const auto& d : col_md.data_segments) {
          max_len = std::max(max_len, d.max_string_length);
        }
        if (max_len == 0) {
          max_len     = VARCHAR_UNKNOWN_LENGTH_FALLBACK_BYTES;
          any_unknown = true;
        }
        budget += static_cast<std::size_t>(rg_md.row_count) * (sizeof(std::uint32_t) + max_len);
      } else {
        budget +=
          static_cast<std::size_t>(rg_md.row_count) * projected_types[ci].fixed_width_byte_size();
      }
    }
    rg_md.decoded_bytes_budget                = budget;
    rg_md.decoded_bytes_budget_is_lower_bound = any_unknown;
  }
}

// The walker over-allocates entries up to `max_row_group_index + 1`, so
// row groups with all projected segments filtered out arrive here at 0.
// Rowid-only entries are legitimately 0-data and are kept.
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

duckdb_native_metadata walk_duckdb_native_metadata(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  const std::vector<projected_column>& projected_cols,
  const std::vector<sirius::logical_type>& projected_types)
{
  duckdb_native_metadata md;
  md.viable = false;

  if (projected_cols.empty()) {
    md.viability_failure_reason = "no projected columns";
    return md;
  }
  if (projected_cols.size() != projected_types.size()) {
    md.viability_failure_reason = "projected_cols and projected_types size mismatch";
    return md;
  }

  for (std::size_t ci = 0; ci < projected_types.size(); ++ci) {
    if (projected_cols[ci].is_rowid) { continue; }
    std::string reason;
    if (!is_supported_logical_type(projected_types[ci], reason)) {
      md.viability_failure_reason =
        "column " + std::to_string(projected_cols[ci].storage_idx.GetPrimaryIndex()) + ": " +
        reason;
      return md;
    }
  }

  // PartitionStatistics order matches `RowGroupCollection::SegmentNodes()`
  // iteration order at v1.5.2 — relied on for indexing by row_group_index.
  auto partition_stats = storage.GetPartitionStats(context);
  row_group_handles handles;
  handles.reserve(partition_stats.size());
  for (auto& ps : partition_stats) {
    duckdb::idx_t row_start = ps.row_start.IsValid() ? ps.row_start.GetIndex() : 0;
    handles.emplace_back(row_start, ps.partition_row_group);
  }

  // GetColumnSegmentInfo yields all table columns; this lookup skips
  // non-projected ones in O(1).
  std::unordered_map<duckdb::idx_t, std::size_t> projected_lookup;
  projected_lookup.reserve(projected_cols.size());
  for (std::size_t ci = 0; ci < projected_cols.size(); ++ci) {
    if (projected_cols[ci].is_rowid) { continue; }
    projected_lookup.emplace(projected_cols[ci].storage_idx.GetPrimaryIndex(), ci);
  }

  duckdb::QueryContext qc{context};
  auto column_segments = storage.GetColumnSegmentInfo(qc);

  // Pre-allocate up to max_row_group_index + 1: a row group whose every
  // projected segment was filtered (or is rowid-only) still needs an entry
  // so rowid synthesis and trailing-empty pruning have something to look at.
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

  varchar_max_length_resolver max_len_resolver(handles, projected_cols);

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
      md.viability_failure_reason = std::string{validity_seg ? "validity" : "data"} +
                                    " segment on column " + std::to_string(seg.column_id) +
                                    " row group " + std::to_string(rg_idx) +
                                    ": unsupported compression \"" + seg.compression_type + "\"";
      return md;
    }

    std::uint32_t max_string_length = 0;
    if (!validity_seg && projected_types[ci].is_varchar()) {
      // Dictionary-family codecs need the stat to size the host-side
      // pre-decode buffer; without it pass-2 would OOB.
      const bool needs_max_len = compression == duckdb::CompressionType::COMPRESSION_DICTIONARY ||
                                 compression == duckdb::CompressionType::COMPRESSION_FSST ||
                                 compression == duckdb::CompressionType::COMPRESSION_DICT_FSST;
      if (needs_max_len) {
        max_string_length = max_len_resolver.get(rg_idx, seg.column_id, ci);
        if (max_string_length == 0) {
          md.viability_failure_reason =
            "varchar segment on column " + std::to_string(seg.column_id) + " row group " +
            std::to_string(rg_idx) + ": codec \"" + seg.compression_type +
            "\" needs max_string_length stat but row group did not advertise one";
          return md;
        }
      }
    }

    auto desc    = build_segment_descriptor(seg, compression, max_string_length);
    auto& col_md = md.row_groups[rg_idx].columns[ci];
    if (validity_seg) {
      col_md.validity_segments.push_back(std::move(desc));
    } else {
      col_md.data_segments.push_back(std::move(desc));
    }
  }

  // DuckDB's iteration is already monotonic at v1.5.2 — the explicit sort
  // guards against future ordering drift.
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
