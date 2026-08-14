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
#include "op/scan/duckdb_native_metadata_cache.hpp"
#include "op/scan/metadata_walk_parallel.hpp"

#include <nvtx3/nvtx3.hpp>

#include <duckdb/common/column_index.hpp>
#include <duckdb/common/enums/compression_type.hpp>
#include <duckdb/common/enums/filter_propagate_result.hpp>
#include <duckdb/function/compression_function.hpp>
#include <duckdb/function/partition_stats.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/block_manager.hpp>
#include <duckdb/storage/segment/uncompressed.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/storage/storage_manager.hpp>
#include <duckdb/storage/table/array_column_data.hpp>
#include <duckdb/storage/table/column_data.hpp>
#include <duckdb/storage/table/column_segment.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/row_group_collection.hpp>
#include <duckdb/storage/table/segment_tree.hpp>
#include <duckdb/storage/table/standard_column_data.hpp>
#include <duckdb/storage/table_storage_info.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {

std::size_t metadata_parse_chunk()
{
  // This is an internal scheduling granularity, not a user tuning surface.
  constexpr std::size_t kParseChunk = 8;
  return kParseChunk;
}

namespace {

// Collects a segment's compression-function "additional" block ids (overflow blocks for FSST tables
// etc.) into a vector. It visits ONLY the compression function's extra blocks, NOT the segment's
// main block.
struct collect_block_ids : public duckdb::BlockIdVisitor {
  explicit collect_block_ids(std::vector<duckdb::block_id_t>& out) : out(out) {}
  void Visit(duckdb::block_id_t block_id) override { out.push_back(block_id); }
  std::vector<duckdb::block_id_t>& out;
};

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
    case sirius::type_id::ARRAY: {
      if (!type.has_child()) {
        reason_out = "type " + type.to_string() + " is an ARRAY without child type metadata";
        return false;
      }
      auto const& child = type.array_child();
      // Only fixed-width child for now (no VARCHAR or nested ARRAY/LIST/STRUCT element)
      if (!child.is_fixed_width()) {
        reason_out = "type " + type.to_string() + " has a non-fixed-width ARRAY element";
        return false;
      }
      std::string child_reason;
      if (!is_supported_logical_type(child, child_reason)) {
        reason_out = "type " + type.to_string() + " has unsupported ARRAY element: " + child_reason;
        return false;
      }
      return true;
    }
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

// Build a descriptor from a persistent/transient segment, reading typed fields
// directly (block id/offset, compression enum, row counts, additional blocks).
// bytes_size and max_string_length are filled by the caller.
duckdb_segment_descriptor fill_segment_descriptor(duckdb::ColumnSegment& segment,
                                                  duckdb::idx_t segment_start)
{
  duckdb_segment_descriptor desc{};
  desc.compression   = segment.GetCompressionFunction().type;
  desc.segment_start = segment_start;
  desc.segment_count = segment.count;
  if (segment.segment_type == duckdb::ColumnSegmentType::PERSISTENT) {
    desc.block_id     = segment.GetBlockId();
    desc.block_offset = segment.GetBlockOffset();
  } else {
    desc.block_id     = INVALID_BLOCK;
    desc.block_offset = 0;
  }
  // additional_blocks: compression-function extra blocks only (guarded by the
  // segment's compressed state)
  auto const& cf = segment.GetCompressionFunction();
  auto seg_state = segment.GetSegmentState();
  if (seg_state && cf.visit_block_ids) {
    collect_block_ids visitor(desc.additional_blocks);
    cf.visit_block_ids(segment, visitor);
  }
  if (desc.compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
    // Snapshot the segment's own stats: the constant value lives here, and
    // row-group-level stats drift as later appends merge into them.
    desc.segment_stats = std::make_shared<duckdb::BaseStatistics>(segment.stats.statistics.Copy());
  }
  return desc;
}

// A CONSTANT validity segment is uniform, with the direction in its stats:
// all-valid decodes to nothing, all-NULL carries the marker so the decoder
// synthesizes zero validity bits.
bool constant_validity_is_all_null(duckdb::ColumnSegment& segment)
{
  return segment.GetCompressionFunction().type == duckdb::CompressionType::COMPRESSION_CONSTANT &&
         segment.stats.statistics.CanHaveNull();
}

// Grants access to ArrayColumnData's protected child/validity members. C++
// permits a derived class to reach a protected base member through a reference
// of its own type; static_cast'ing the existing ArrayColumnData object to this
// (layout-identical, member-less) subclass is well-defined and avoids the
// fragile offset/padding assumptions a reinterpret_cast would require.
struct array_column_access : duckdb::ArrayColumnData {
  static duckdb::ColumnData* get_child(duckdb::ArrayColumnData& a)
  {
    return static_cast<array_column_access&>(a).child_column.get();
  }
  static duckdb::ValidityColumnData* get_validity(duckdb::ArrayColumnData& a)
  {
    return static_cast<array_column_access&>(a).validity.get();
  }
};

// Walks an ARRAY column's segment trees. DuckDB lays out a fixed-size ARRAY as:
// array-level validity (path [col,0]) + a child column of count * array_size
// contiguous values (child data [col,1], child validity [col,1,0]). The decoder
// reads array-level validity from data_segments, so it lands there. Returns a
// viability-failure reason, or nullopt on success.
std::optional<std::string> walk_array_column(duckdb::ColumnData& col_data,
                                             duckdb::idx_t column_id,
                                             std::size_t rg_idx,
                                             duckdb_column_metadata& col_md)
{
  // Walk a fixed-width data segment tree into out
  auto walk_data = [&](duckdb::ColumnSegmentTree& tree,
                       std::vector<duckdb_segment_descriptor>& out,
                       const char* label) -> std::optional<std::string> {
    // Tree order is row-start order; the caller re-sorts anyway
    for (auto& node : tree.SegmentNodes()) {
      auto& segment          = node.GetNode();
      auto const compression = segment.GetCompressionFunction().type;
      if (!is_supported_data_compression(compression)) {
        return std::string(label) + " segment on column " + std::to_string(column_id) +
               " row group " + std::to_string(rg_idx) + ": unsupported compression " +
               duckdb::CompressionTypeToString(compression);
      }
      out.push_back(fill_segment_descriptor(segment, node.GetRowStart()));
    }
    return std::nullopt;
  };

  // Walk a validity segment tree into out
  auto walk_validity = [&](duckdb::ColumnSegmentTree& tree,
                           std::vector<duckdb_segment_descriptor>& out,
                           const char* label) -> std::optional<std::string> {
    for (auto& node : tree.SegmentNodes()) {
      auto& segment          = node.GetNode();
      auto const compression = segment.GetCompressionFunction().type;
      if (!is_supported_validity_compression(compression)) {
        return std::string(label) + " segment on column " + std::to_string(column_id) +
               " row group " + std::to_string(rg_idx) + ": unsupported compression " +
               duckdb::CompressionTypeToString(compression);
      }
      auto desc     = fill_segment_descriptor(segment, node.GetRowStart());
      desc.all_null = constant_validity_is_all_null(segment);
      out.push_back(std::move(desc));
    }
    return std::nullopt;
  };

  col_md.is_array = true;
  auto* array_col = dynamic_cast<duckdb::ArrayColumnData*>(&col_data);
  if (!array_col) {
    return "ARRAY column " + std::to_string(column_id) + " row group " + std::to_string(rg_idx) +
           ": expected ArrayColumnData but got " + col_data.GetType().ToString();
  }

  // Array-level validity to data_segments
  auto* array_validity = array_column_access::get_validity(*array_col);
  if (!array_validity) {
    return "ARRAY column " + std::to_string(column_id) + " row group " + std::to_string(rg_idx) +
           ": no array validity column";
  }
  if (auto reason =
        walk_validity(array_validity->GetSegmentTree(), col_md.data_segments, "array validity")) {
    return reason;
  }

  // Child column. The decode path supports a fixed-width child only; its
  // storage is StandardColumnData.
  auto* child = array_column_access::get_child(*array_col);
  if (!child) {
    return "ARRAY column " + std::to_string(column_id) + " row group " + std::to_string(rg_idx) +
           ": no child column found";
  }
  auto* child_std = dynamic_cast<duckdb::StandardColumnData*>(child);
  if (!child_std) {
    return "ARRAY child on column " + std::to_string(column_id) + " row group " +
           std::to_string(rg_idx) + ": child storage is not StandardColumnData";
  }
  if (auto reason = walk_data(
        child_std->GetSegmentTree(), col_md.array_child_data_segments, "ARRAY child data")) {
    return reason;
  }
  if (auto reason = walk_validity(child_std->GetValidityData().GetSegmentTree(),
                                  col_md.array_child_validity_segments,
                                  "ARRAY child validity")) {
    return reason;
  }
  return std::nullopt;
}

// Walks the segment trees of a projected column, collecting metadata about its data and validity
// segments. Returns a viability-failure reason, or nullopt on success.
std::optional<std::string> walk_standard_column(duckdb::ColumnData& col_data,
                                                bool is_varchar,
                                                duckdb::idx_t column_id,
                                                std::size_t rg_idx,
                                                duckdb_column_metadata& col_md)
{
  auto* std_col = dynamic_cast<duckdb::StandardColumnData*>(&col_data);
  if (!std_col) {
    return "column " + std::to_string(column_id) + " row group " + std::to_string(rg_idx) +
           ": column storage for type " + col_data.GetType().ToString() +
           " is not StandardColumnData (nested/unsupported)";
  }

  // Data segments (tree order is row-start order; the caller re-sorts anyway).
  for (auto& node : std_col->GetSegmentTree().SegmentNodes()) {
    auto& segment          = node.GetNode();
    auto const compression = segment.GetCompressionFunction().type;
    if (!is_supported_data_compression(compression)) {
      return "data segment on column " + std::to_string(column_id) + " row group " +
             std::to_string(rg_idx) + ": unsupported compression " +
             duckdb::CompressionTypeToString(compression);
    }
    auto desc = fill_segment_descriptor(segment, node.GetRowStart());
    if (is_varchar) {
      // The varchar decoder cannot read CONSTANT-compressed segments.
      if (compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
        return "varchar segment on column " + std::to_string(column_id) + " row group " +
               std::to_string(rg_idx) + ": CONSTANT compression is unsupported for varchar";
      }
      // Read the per-segment Max String Length stat TYPED (exact)
      // Absent stat -> refuse so consumers deref unchecked.
      if (!duckdb::StringStats::HasMaxStringLength(segment.stats.statistics)) {
        return "varchar segment on column " + std::to_string(column_id) + " row group " +
               std::to_string(rg_idx) + ": Max String Length stat absent from segment stats";
      }
      desc.max_string_length = duckdb::StringStats::MaxStringLength(segment.stats.statistics);
      // Stats-drift guard: a marker-bearing segment must never reach the GPU string
      // decoder. Mirrors the refusal in prepare_duckdb_native_walk (see rationale there).
      if (*desc.max_string_length >=
          duckdb::StringUncompressed::GetStringBlockLimit(segment.GetBlockSize())) {
        return "varchar segment on column " + std::to_string(column_id) + " row group " +
               std::to_string(rg_idx) +
               ": max string length reaches the overflow-block limit; overflow strings are not "
               "GPU-decodable";
      }
    }
    col_md.data_segments.push_back(std::move(desc));
  }

  // Validity child segments (StandardColumnData always has a validity child).
  for (auto& node : std_col->GetValidityData().GetSegmentTree().SegmentNodes()) {
    auto& segment          = node.GetNode();
    auto const compression = segment.GetCompressionFunction().type;
    if (!is_supported_validity_compression(compression)) {
      return "validity segment on column " + std::to_string(column_id) + " row group " +
             std::to_string(rg_idx) + ": unsupported compression " +
             duckdb::CompressionTypeToString(compression);
    }
    auto desc     = fill_segment_descriptor(segment, node.GetRowStart());
    desc.all_null = constant_validity_is_all_null(segment);
    col_md.validity_segments.push_back(std::move(desc));
  }
  return std::nullopt;
}

// ColumnSegmentInfo lacks segment_size. Derive via sorted-by-(block_id,
// block_offset) delta to the next walked segment; last-in-block falls back
// to `block_size - block_offset`. Upper bound only: trailing free space and
// cross-table partial-block neighbors inflate it. Codec headers self-bound
// reads, so correctness-safe; only H2D and staging bytes pay the overshoot.
void compute_segment_bytes_size(std::vector<duckdb_row_group_metadata>& row_groups,
                                std::size_t block_size)
{
  std::vector<duckdb_segment_descriptor*> refs;
  for (auto& rg : row_groups) {
    for (auto& col : rg.columns) {
      for (auto& s : col.data_segments)
        if (s.block_id >= 0) refs.push_back(&s);
      for (auto& s : col.validity_segments)
        if (s.block_id >= 0) refs.push_back(&s);
      for (auto& s : col.array_child_data_segments)
        if (s.block_id >= 0) refs.push_back(&s);
      for (auto& s : col.array_child_validity_segments)
        if (s.block_id >= 0) refs.push_back(&s);
    }
  }
  std::sort(refs.begin(), refs.end(), [](const auto* a, const auto* b) {
    if (a->block_id != b->block_id) return a->block_id < b->block_id;
    return a->block_offset < b->block_offset;
  });
  for (std::size_t i = 0; i < refs.size(); ++i) {
    auto& seg                = *refs[i];
    auto const last_in_block = i + 1 == refs.size() || refs[i + 1]->block_id != seg.block_id;
    auto const end =
      last_in_block ? block_size : static_cast<std::size_t>(refs[i + 1]->block_offset);
    seg.bytes_size = end - static_cast<std::size_t>(seg.block_offset);
  }
}

/// @brief Check if the filter can be applied to row-group pruning.
///
/// This DuckDB-native statistics walker only consumes the static payloads represented directly by
/// DuckDB @c TableFilter nodes. @c DYNAMIC_FILTER is a routing placeholder, while Sirius runtime
/// join filters use their own publication channel and scan-consumer paths, so it is not translated
/// by this walker.
bool filter_is_prunable(duckdb::TableFilterType t)
{
  return t != duckdb::TableFilterType::DYNAMIC_FILTER;
}

std::size_t estimate_decoded_bytes_budget(duckdb::idx_t row_count,
                                          const std::vector<projected_column>& projected_cols,
                                          const std::vector<sirius::logical_type>& projected_types)
{
  std::size_t budget = 0;
  for (std::size_t ci = 0; ci < projected_cols.size(); ++ci) {
    if (projected_cols[ci].is_rowid) {
      budget += static_cast<std::size_t>(row_count) * sizeof(std::int64_t);
    } else if (projected_types[ci].is_varchar()) {
      // String payload bytes require segment-level max-string stats. At prepare
      // time we can only account for offsets; this counter is diagnostic.
      budget += static_cast<std::size_t>(row_count) * sizeof(std::uint32_t);
    } else if (projected_types[ci].is_array()) {
      // ARRAY: offsets (int32) + child values (array_size × child_width × row_count).
      auto const array_size  = projected_types[ci].array_size();
      auto const child_width = projected_types[ci].array_child().fixed_width_byte_size();
      budget +=
        static_cast<std::size_t>(row_count) * (sizeof(std::int32_t) + array_size * child_width);
    } else {
      budget += static_cast<std::size_t>(row_count) * projected_types[ci].fixed_width_byte_size();
    }
  }
  return budget;
}

bool column_index_can_have_storage_stats(const duckdb::ColumnIndex& column_id)
{
  return column_id.HasPrimaryIndex() && !column_id.IsRowIdColumn() && !column_id.IsEmptyColumn() &&
         !column_id.IsVirtualColumn();
}

//===----------Fused per-row-group statistics pass----------===//
// The prepare walk previously ran one serial statistics pass over all row
// groups per prunable filter column (pruning) and one per projected varchar
// column (overflow refusal). Both consume the same per-(row group, column)
// statistics reads, so they are fused into ONE pass per row group and
// parallelized across row groups (see parallel_over_row_groups). Only
// GetPartitionStats itself must stay serial: it touches ClientContext /
// LocalStorage. The statistics reads here go through RowGroup::GetStatistics,
// which locks internally (per-row-group row_group_lock for lazy column loads,
// per-column stats_lock for the copy-out) and returns a self-contained copy;
// TableFilter::CheckStatistics implementations are const and read-only
// (audited: constant/zonemap/conjunction/bloom/expression paths), so a shared
// filter object is safe to probe from multiple workers.

/// A pushed-down filter resolved to its storage primary index, restricted to
/// the prunable, stats-bearing subset (mirrors the old pass-1 guards).
struct resolved_prunable_filter {
  duckdb::idx_t primary             = 0;
  const duckdb::TableFilter* filter = nullptr;
};

/// A projected varchar column resolved for the overflow refusal (old pass 2).
struct resolved_varchar_probe {
  std::size_t ci        = 0;  ///< Position in projected_cols — the refusal order key.
  duckdb::idx_t primary = 0;
  const duckdb::StorageIndex* storage_idx = nullptr;  ///< Full index for the uncached read.
};

std::vector<resolved_prunable_filter> resolve_prunable_filters(
  const duckdb::TableFilterSet* table_filters,
  const duckdb::vector<duckdb::ColumnIndex>* column_ids)
{
  std::vector<resolved_prunable_filter> out;
  if (table_filters == nullptr || column_ids == nullptr || column_ids->empty()) { return out; }
  for (auto const& [col_idx, filter] : table_filters->filters) {
    if (!filter_is_prunable(filter->filter_type)) { continue; }
    if (col_idx >= column_ids->size()) { continue; }  // defensive
    auto const& column_id = (*column_ids)[col_idx];
    if (!column_index_can_have_storage_stats(column_id)) { continue; }
    out.push_back({column_id.GetPrimaryIndex(), filter.get()});
  }
  // Canonical order for the product-cache key. TableFilterSet::filters is an
  // ordered map over col_idx, but the product key is expressed in primary
  // indexes; sorting makes equal filter SETS compare equal regardless of the
  // scan's column_ids layout. Pruning is an any-of, so order never changes it.
  std::stable_sort(
    out.begin(), out.end(), [](auto const& a, auto const& b) { return a.primary < b.primary; });
  return out;
}

std::vector<resolved_varchar_probe> resolve_varchar_probes(
  const std::vector<projected_column>& projected_cols,
  const std::vector<sirius::logical_type>& projected_types)
{
  std::vector<resolved_varchar_probe> out;
  for (std::size_t ci = 0; ci < projected_cols.size(); ++ci) {
    if (projected_cols[ci].is_rowid || !projected_types[ci].is_varchar()) { continue; }
    out.push_back(
      {ci, projected_cols[ci].storage_idx.GetPrimaryIndex(), &projected_cols[ci].storage_idx});
  }
  return out;
}

inline const duckdb::BaseStatistics* fused_stats_ptr(
  const duckdb::unique_ptr<duckdb::BaseStatistics>& p)
{
  return p.get();
}
inline const duckdb::BaseStatistics* fused_stats_ptr(const duckdb::BaseStatistics* p) { return p; }

struct fused_pass_result {
  /// Per row group: proven empty by a pushed-down filter's statistics.
  std::vector<std::uint8_t> pruned;
  /// Overflow refusal, if any: the lexicographically (ci, rg) smallest — the
  /// exact refusal the old serial column-outer/row-group-inner pass reported.
  bool refused = false;
  std::string refusal_reason;
};

/// @brief The fused pruning + varchar-overflow statistics pass.
///
/// @p skip_rg: row groups with no statistics source (uncached path: row groups
/// past the PartitionStatistics range) — never pruned, never overflow-checked,
/// exactly like the old passes. @p prune_stats / @p varchar_stats return either
/// a `duckdb::unique_ptr<BaseStatistics>` (fresh copy, uncached path) or a
/// `const BaseStatistics*` (cached snapshot); null means "no stats".
///
/// Refusal-order equivalence with the old serial passes: per row group the
/// FIRST refusing probe (min ci) is recorded; the global pick minimizes
/// (ci, rg) lexicographically. The old pass returned the min-rg refusal of the
/// min refusing ci; since the global-min ci is by definition <= every other
/// refusing ci in its row group, the per-row-group min-ci records always
/// contain that pair, and the lexicographic reduction selects exactly it.
template <typename SkipFn, typename PruneStatsFn, typename VarcharStatsFn>
fused_pass_result run_fused_stats_pass(std::size_t n_row_groups,
                                       const std::vector<resolved_prunable_filter>& filters,
                                       const std::vector<resolved_varchar_probe>& varchar_probes,
                                       std::size_t overflow_limit,
                                       SkipFn&& skip_rg,
                                       PruneStatsFn&& prune_stats,
                                       VarcharStatsFn&& varchar_stats)
{
  fused_pass_result res;
  res.pruned.assign(n_row_groups, 0);
  if (filters.empty() && varchar_probes.empty()) { return res; }

  constexpr std::size_t kNoRefusal = std::numeric_limits<std::size_t>::max();
  // Workers write only their own row groups' slots — deterministic under any
  // worker count.
  std::vector<std::size_t> refusal_ci(varchar_probes.empty() ? 0 : n_row_groups, kNoRefusal);
  std::vector<std::string> refusal_reason(varchar_probes.empty() ? 0 : n_row_groups);

  parallel_over_row_groups(n_row_groups, [&](std::size_t begin, std::size_t end) {
    for (std::size_t rg = begin; rg < end; ++rg) {
      if (skip_rg(rg)) { continue; }
      bool pruned = false;
      for (auto const& f : filters) {
        auto holder       = prune_stats(rg, f);
        auto const* stats = fused_stats_ptr(holder);
        if (stats == nullptr) { continue; }  // no stats -> cannot prune
        // CheckStatistics takes a non-const ref but every implementation is
        // read-only (see the audit note above), so probing a shared snapshot
        // statistics object is safe.
        if (f.filter->CheckStatistics(const_cast<duckdb::BaseStatistics&>(*stats)) ==
            duckdb::FilterPropagateResult::FILTER_ALWAYS_FALSE) {
          pruned = true;
          break;
        }
      }
      if (pruned) {
        res.pruned[rg] = 1;
        continue;  // pruned -> never decoded -> no overflow check
      }
      for (auto const& probe : varchar_probes) {
        auto holder       = varchar_stats(rg, probe);
        auto const* stats = fused_stats_ptr(holder);
        if (stats == nullptr || !duckdb::StringStats::HasMaxStringLength(*stats)) {
          refusal_ci[rg]     = probe.ci;
          refusal_reason[rg] = "row group " + std::to_string(rg) + " varchar column " +
                               std::to_string(probe.primary) +
                               ": max-string-length stat absent; cannot rule out overflow strings";
          break;
        }
        auto const max_len = duckdb::StringStats::MaxStringLength(*stats);
        if (max_len >= overflow_limit) {
          refusal_ci[rg]     = probe.ci;
          refusal_reason[rg] = "row group " + std::to_string(rg) + " varchar column " +
                               std::to_string(probe.primary) + ": max string length " +
                               std::to_string(max_len) + " reaches the overflow-block limit (" +
                               std::to_string(overflow_limit) +
                               "); overflow strings are not GPU-decodable";
          break;
        }
      }
    }
  });

  // Serial reduction: pick the (ci, rg)-lexicographic minimum refusal.
  std::size_t best_ci = kNoRefusal;
  std::size_t best_rg = kNoRefusal;
  for (std::size_t rg = 0; rg < refusal_ci.size(); ++rg) {
    if (refusal_ci[rg] < best_ci) {
      best_ci = refusal_ci[rg];
      best_rg = rg;
    }
  }
  if (best_ci != kNoRefusal) {
    res.refused        = true;
    res.refusal_reason = std::move(refusal_reason[best_rg]);
  }
  return res;
}

/// Fold a fused pass into the plan's pruning bookkeeping (serial, so the
/// pruned-byte sums are deterministic).
void fold_pruning_into_plan(duckdb_native_walk_plan& plan, const fused_pass_result& pass)
{
  auto const& projected_cols  = *plan.projected_cols;
  auto const& projected_types = *plan.projected_types;
  for (std::size_t rg = 0; rg < plan.n_row_groups; ++rg) {
    if (!pass.pruned[rg]) { continue; }
    plan.row_group_pruned_by_stats[rg] = true;
    auto const pruned_bytes =
      estimate_decoded_bytes_budget(plan.row_count[rg], projected_cols, projected_types);
    plan.pruned_decoded_bytes_by_row_group[rg] = pruned_bytes;
    ++plan.pruned_row_groups;
    plan.pruned_decoded_bytes += pruned_bytes;
  }
}

//===----------Cached prepare (see duckdb_native_metadata_cache.hpp)----------===//

/// Storage primary indexes whose row-group statistics the cached prepare
/// consumes: prunable filter columns and projected varchar columns (overflow
/// refusal). nullopt when a projected varchar carries child indexes — a shape
/// the per-primary-index stats cache does not model, so the caller bypasses.
std::optional<std::vector<duckdb::idx_t>> stats_columns_for_cached_prepare(
  const std::vector<resolved_prunable_filter>& prunable_filters,
  const std::vector<projected_column>& projected_cols,
  const std::vector<sirius::logical_type>& projected_types)
{
  std::vector<duckdb::idx_t> cols;
  auto add = [&cols](duckdb::idx_t c) {
    if (std::find(cols.begin(), cols.end(), c) == cols.end()) { cols.push_back(c); }
  };
  for (auto const& f : prunable_filters) {
    add(f.primary);
  }
  for (std::size_t ci = 0; ci < projected_cols.size(); ++ci) {
    if (projected_cols[ci].is_rowid || !projected_types[ci].is_varchar()) { continue; }
    auto const& storage_idx = projected_cols[ci].storage_idx;
    if (!storage_idx.GetChildIndexes().empty()) { return std::nullopt; }
    add(storage_idx.GetPrimaryIndex());
  }
  return cols;
}

void append_storage_index_signature(const duckdb::StorageIndex& idx, std::string& out)
{
  out += std::to_string(idx.GetPrimaryIndex());
  auto const& children = idx.GetChildIndexes();
  if (!children.empty()) {
    out += '[';
    for (auto const& child : children) {
      append_storage_index_signature(child, out);
      out += ',';
    }
    out += ']';
  }
}

/// Canonical string for the projected column set: identity (storage index,
/// including child indexes) and type of every projected column, in emission
/// order. Everything the walk product derives from the projection — pruned
/// decoded-byte estimates and the varchar overflow probes — is a function of
/// this signature.
std::string projection_signature_for_product_key(
  const std::vector<projected_column>& projected_cols,
  const std::vector<sirius::logical_type>& projected_types)
{
  std::string sig;
  sig.reserve(projected_cols.size() * 12);
  for (std::size_t ci = 0; ci < projected_cols.size(); ++ci) {
    if (projected_cols[ci].is_rowid) {
      sig += "r;";
      continue;
    }
    append_storage_index_signature(projected_cols[ci].storage_idx, sig);
    sig += ':';
    sig += projected_types[ci].to_string();
    sig += ';';
  }
  return sig;
}

/// Assemble the query-dependent walk product from a validated cache snapshot
/// via the fused statistics pass, over the snapshot's cached statistics.
/// Mirrors the uncached prepare exactly: same pruning decisions and same
/// refusal reasons.
std::shared_ptr<const walk_plan_product> assemble_product_from_snapshot(
  const duckdb_native_metadata_cache::acquired_snapshot& snap,
  const std::vector<resolved_prunable_filter>& prunable_filters,
  const std::vector<resolved_varchar_probe>& varchar_probes,
  const std::vector<projected_column>& projected_cols,
  const std::vector<sirius::logical_type>& projected_types)
{
  auto const& core = *snap.core;
  auto product     = std::make_shared<walk_plan_product>();
  product->row_group_pruned_by_stats.assign(core.n_row_groups, false);
  product->pruned_decoded_bytes_by_row_group.assign(core.n_row_groups, 0);

  // Defensive: stats_columns_for_cached_prepare requested every probed column,
  // so a missing snapshot column cannot happen; refuse rather than skip a
  // safety check.
  for (auto const& probe : varchar_probes) {
    if (snap.column_stats.find(probe.primary) == snap.column_stats.end()) {
      product->viable                   = false;
      product->viability_failure_reason = "varchar column " + std::to_string(probe.primary) +
                                          ": statistics missing from the metadata cache snapshot";
      return product;
    }
  }

  // Hoist the per-column stats snapshots out of the parallel loop (read-only
  // map lookups are thread-safe, but pay per row group otherwise).
  std::vector<const column_stats_snapshot*> filter_stats(prunable_filters.size(), nullptr);
  for (std::size_t i = 0; i < prunable_filters.size(); ++i) {
    auto it = snap.column_stats.find(prunable_filters[i].primary);
    if (it != snap.column_stats.end()) { filter_stats[i] = it->second.get(); }
  }
  std::vector<const column_stats_snapshot*> varchar_stats(varchar_probes.size(), nullptr);
  for (std::size_t i = 0; i < varchar_probes.size(); ++i) {
    varchar_stats[i] = snap.column_stats.at(varchar_probes[i].primary).get();
  }
  // Probe index maps for the accessor callbacks (identity lookups by element
  // address keep the shared run_fused_stats_pass signature simple).
  auto filter_index = [&prunable_filters](const resolved_prunable_filter& f) {
    return static_cast<std::size_t>(&f - prunable_filters.data());
  };
  auto probe_index = [&varchar_probes](const resolved_varchar_probe& p) {
    return static_cast<std::size_t>(&p - varchar_probes.data());
  };

  auto const overflow_limit = duckdb::StringUncompressed::GetStringBlockLimit(core.block_size);
  auto pass                 = run_fused_stats_pass(
    core.n_row_groups,
    prunable_filters,
    varchar_probes,
    overflow_limit,
    [](std::size_t) { return false; },  // snapshot stats cover every row group
    [&](std::size_t rg, const resolved_prunable_filter& f) -> const duckdb::BaseStatistics* {
      auto const* col = filter_stats[filter_index(f)];
      if (col == nullptr) { return nullptr; }  // defensive: no stats -> cannot prune
      return col->per_row_group[rg].get();
    },
    [&](std::size_t rg, const resolved_varchar_probe& p) -> const duckdb::BaseStatistics* {
      return varchar_stats[probe_index(p)]->per_row_group[rg].get();
    });

  for (std::size_t rg = 0; rg < core.n_row_groups; ++rg) {
    if (!pass.pruned[rg]) { continue; }
    product->row_group_pruned_by_stats[rg] = true;
    auto const pruned_bytes =
      estimate_decoded_bytes_budget(core.row_count[rg], projected_cols, projected_types);
    product->pruned_decoded_bytes_by_row_group[rg] = pruned_bytes;
    ++product->pruned_row_groups;
    product->pruned_decoded_bytes += pruned_bytes;
  }

  if (pass.refused) {
    product->viable                   = false;
    product->viability_failure_reason = std::move(pass.refusal_reason);
  } else {
    product->viable = true;
  }
  return product;
}

/// Copy a snapshot's geometry and a walk product into @p plan.
/// partition_row_groups stays empty: it exists to feed the uncached prepare's
/// statistics reads, which the snapshot's cached statistics replaced.
void apply_snapshot_and_product(duckdb_native_walk_plan& plan,
                                const table_walk_snapshot& core,
                                const walk_plan_product& product)
{
  plan.n_row_groups    = core.n_row_groups;
  plan.block_size      = core.block_size;
  plan.row_group_start = core.row_group_start;
  plan.row_count       = core.row_count;

  plan.row_group_pruned_by_stats         = product.row_group_pruned_by_stats;
  plan.pruned_decoded_bytes_by_row_group = product.pruned_decoded_bytes_by_row_group;
  plan.pruned_row_groups                 = product.pruned_row_groups;
  plan.pruned_decoded_bytes              = product.pruned_decoded_bytes;

  if (plan.pruned_row_groups > 0) {
    SIRIUS_LOG_DEBUG(
      "[duckdb_native_metadata] prepare stats-pruned {} row groups (~{} decoded bytes)",
      plan.pruned_row_groups,
      plan.pruned_decoded_bytes);
  }
  if (plan.n_row_groups > 0 && plan.pruned_row_groups == plan.n_row_groups) {
    SIRIUS_LOG_DEBUG(
      "[duckdb_native_metadata] all {} row groups stats-pruned; scan yields an "
      "empty result via the coalescer fallback",
      plan.n_row_groups);
  }

  plan.viable = product.viable;
  if (!product.viable) {
    plan.viability_failure_reason = product.viability_failure_reason;
    SIRIUS_LOG_DEBUG("[duckdb_native_metadata] refused (prepare): {}",
                     plan.viability_failure_reason);
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
    // CONSTANT validity is uniform — all-valid or all-NULL, disambiguated by
    // the segment stats (see constant_validity_is_all_null). EMPTY means the
    // base data codec covers validity itself.
    // ROARING is host-decoded to a plain bitmap before the GPU sees it.
    case duckdb::CompressionType::COMPRESSION_UNCOMPRESSED:
    case duckdb::CompressionType::COMPRESSION_EMPTY:
    case duckdb::CompressionType::COMPRESSION_CONSTANT:
    case duckdb::CompressionType::COMPRESSION_ROARING: return true;
    default: return false;
  }
}

//===----------prepare_duckdb_native_walk----------===//
duckdb_native_walk_plan prepare_duckdb_native_walk(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  const std::vector<projected_column>& projected_cols,
  const std::vector<sirius::logical_type>& projected_types,
  const duckdb::TableFilterSet* table_filters,
  const duckdb::vector<duckdb::ColumnIndex>* column_ids)
{
  nvtx3::scoped_range nvtx_prep{"sirius::native_metadata_prepare"};

  duckdb_native_walk_plan plan;
  plan.viable          = false;
  plan.storage         = &storage;
  plan.context         = &context;
  plan.projected_cols  = &projected_cols;
  plan.projected_types = &projected_types;
  plan.table_filters   = table_filters;
  plan.column_ids      = column_ids;

  auto refuse = [&plan](std::string reason) {
    plan.viability_failure_reason = std::move(reason);
    SIRIUS_LOG_DEBUG("[duckdb_native_metadata] refused (prepare): {}",
                     plan.viability_failure_reason);
  };

  if (projected_cols.empty()) {
    refuse("no projected columns");
    return plan;
  }
  if (projected_cols.size() != projected_types.size()) {
    refuse("projected_cols and projected_types size mismatch");
    return plan;
  }

  // Type gate
  for (std::size_t ci = 0; ci < projected_types.size(); ++ci) {
    if (projected_cols[ci].is_rowid) { continue; }
    std::string reason;
    if (!is_supported_logical_type(projected_types[ci], reason)) {
      refuse("column " + std::to_string(projected_cols[ci].storage_idx.GetPrimaryIndex()) + ": " +
             reason);
      return plan;
    }
  }

  // Resolved query shape: shared by the product-cache key and both fused
  // statistics passes below.
  auto const prunable_filters = resolve_prunable_filters(table_filters, column_ids);
  auto const varchar_probes   = resolve_varchar_probes(projected_cols, projected_types);

  // Serve the serial prepare from the process-wide metadata cache. On a
  // product hit this replaces GetPartitionStats and BOTH statistics passes
  // with a structural validity probe; on a snapshot hit with a new query
  // shape, the fused pass runs over cached statistics (no storage reads).
  // A bypass (cache disabled, transaction-local appends, nested varchar
  // storage index, or a capture torn by a concurrent commit) falls through
  // to the uncached walk below.
  if (auto stats_columns =
        stats_columns_for_cached_prepare(prunable_filters, projected_cols, projected_types)) {
    auto projection_signature =
      projection_signature_for_product_key(projected_cols, projected_types);
    std::vector<std::pair<duckdb::idx_t, const duckdb::TableFilter*>> key_filters;
    key_filters.reserve(prunable_filters.size());
    for (auto const& f : prunable_filters) {
      key_filters.emplace_back(f.primary, f.filter);
    }
    walk_product_key_view key_view;
    key_view.projection_signature = &projection_signature;
    key_view.prunable_filters     = &key_filters;

    auto& cache = duckdb_native_metadata_cache::instance();
    if (auto snapshot = cache.acquire(storage, context, *stats_columns, &key_view)) {
      auto product = snapshot->product;
      if (product == nullptr) {
        product = assemble_product_from_snapshot(
          *snapshot, prunable_filters, varchar_probes, projected_cols, projected_types);
        walk_product_key key;
        key.projection_signature = std::move(projection_signature);
        key.prunable_filters.reserve(prunable_filters.size());
        for (auto const& f : prunable_filters) {
          key.prunable_filters.emplace_back(f.primary, f.filter->Copy());
        }
        cache.store_product(storage, snapshot->generation, std::move(key), product);
      }
      apply_snapshot_and_product(plan, *snapshot->core, *product);
      return plan;
    }
  }

  // GetPartitionStats touches LocalStorage/ClientContext. Runs before the
  // concurrent range walks.
  duckdb::vector<duckdb::PartitionStatistics> partition_stats;
  {
    /// @note Synchronous pread()s happen here when cold.
    nvtx3::scoped_range nvtx_ps{"sirius::native_metadata_partition_stats"};
    partition_stats = storage.GetPartitionStats(context);
  }

  auto const& row_groups = *storage.GetRowGroupCollection();
  plan.n_row_groups      = row_groups.GetRowGroupCount();
  plan.block_size = storage.GetAttached().GetStorageManager().GetBlockManager().GetBlockSize();

  plan.row_group_start.assign(plan.n_row_groups, 0);
  plan.row_count.assign(plan.n_row_groups, 0);
  plan.partition_row_groups.assign(plan.n_row_groups, nullptr);
  plan.row_group_pruned_by_stats.assign(plan.n_row_groups, false);
  plan.pruned_decoded_bytes_by_row_group.assign(plan.n_row_groups, 0);
  // PartitionStatistics is expected to carry one entry per row group; a larger
  // count means the DuckDB layout assumption below (index i == row group i) has
  // drifted and trailing entries would be silently dropped.
  assert(partition_stats.size() <= plan.n_row_groups &&
         "partition_stats count exceeds row group count — DuckDB layout drift");
  for (std::size_t i = 0; i < partition_stats.size(); ++i) {
    auto const& ps = partition_stats[i];
    if (!ps.row_start.IsValid()) {
      refuse("partition_stats[" + std::to_string(i) +
             "].row_start is not valid; cannot synthesize rowids");
      return plan;
    }
    // PartitionStatistics order matches `RowGroupCollection::SegmentNodes()`
    // iteration order at v1.5.2.
    if (i < plan.n_row_groups) {
      plan.row_group_start[i]      = ps.row_start.GetIndex();
      plan.row_count[i]            = ps.count;
      plan.partition_row_groups[i] = ps.partition_row_group;
    }
  }

  // Fused statistics pass: row-group pruning against pushed-down filter stats
  // and the varchar overflow (big-string) refusal, in ONE pass per row group,
  // parallel across row groups (previously one serial pass per filter column
  // plus one per varchar column). Overflow rationale: the UNCOMPRESSED codec
  // stores any single string at/over StringUncompressed::GetStringBlockLimit
  // in an overflow block, leaving a BIG_STRING_MARKER the GPU string decoder
  // would silently emit as string content. The stat is a per-string max, so
  // stat < limit proves a row group marker-free. Conservative for DICT_FSST,
  // which inlines strings up to 16 KiB (DictFSSTCompression::STRING_SIZE_LIMIT)
  // without markers — codecs are invisible in row-group stats, so its
  // limit..16 KiB row groups are refused unnecessarily (rare in practice).
  fused_pass_result pass;
  {
    nvtx3::scoped_range nvtx_pass{"sirius::native_metadata_stats_pass"};
    auto const overflow_limit = duckdb::StringUncompressed::GetStringBlockLimit(plan.block_size);
    pass                      = run_fused_stats_pass(
      plan.n_row_groups,
      prunable_filters,
      varchar_probes,
      overflow_limit,
      [&plan](std::size_t rg) {
        // No PartitionRowGroup handle -> no stats source: never pruned, never
        // overflow-checked (same skip as the old serial passes).
        return rg >= plan.partition_row_groups.size() || !plan.partition_row_groups[rg];
      },
      [&plan](std::size_t rg, const resolved_prunable_filter& f) {
        return plan.partition_row_groups[rg]->GetColumnStatistics(duckdb::StorageIndex(f.primary));
      },
      [&plan](std::size_t rg, const resolved_varchar_probe& p) {
        return plan.partition_row_groups[rg]->GetColumnStatistics(*p.storage_idx);
      });
  }

  fold_pruning_into_plan(plan, pass);
  if (plan.pruned_row_groups > 0) {
    SIRIUS_LOG_DEBUG(
      "[duckdb_native_metadata] prepare stats-pruned {} row groups (~{} decoded bytes)",
      plan.pruned_row_groups,
      plan.pruned_decoded_bytes);
  }
  // A fully-pruned table (every row group removed by filter stats) is viable: the
  // ranges walk yields empty row-group lists, and the coalescer's empty-batch
  // fallback emits one schema-correct 0-row split so the scan still creates a task
  // and the pipeline completes (mirrors the parquet all-pruned path). Refusing here
  // instead throws "duckdb-native scan rejected query" and hangs the query.
  if (plan.n_row_groups > 0 && plan.pruned_row_groups == plan.n_row_groups) {
    SIRIUS_LOG_DEBUG(
      "[duckdb_native_metadata] all {} row groups stats-pruned; scan yields an "
      "empty result via the coalescer fallback",
      plan.n_row_groups);
  }

  if (pass.refused) {
    refuse(std::move(pass.refusal_reason));
    return plan;
  }

  plan.viable = true;
  return plan;
}

//===----------walk_duckdb_native_row_group_range----------===//
duckdb_native_row_group_range walk_duckdb_native_row_group_range(
  const duckdb_native_walk_plan& plan, std::size_t rg_begin, std::size_t rg_end)
{
  duckdb_native_row_group_range result;

  auto const& projected_cols  = *plan.projected_cols;
  auto const& projected_types = *plan.projected_types;

  rg_end = std::min(rg_end, plan.n_row_groups);
  if (rg_begin >= rg_end) { return result; }  // viable=true, empty range

  auto refuse = [&result](std::string reason) {
    result.viable                   = false;
    result.viability_failure_reason = std::move(reason);
    SIRIUS_LOG_DEBUG("[duckdb_native_metadata] refused (range): {}",
                     result.viability_failure_reason);
  };

  auto const n     = rg_end - rg_begin;
  auto const n_pos = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> local_index_by_rg(n, n_pos);

  // One entry per surviving row group in [rg_begin, rg_end). Row groups that
  // were stats-pruned during prepare are skipped before any segment metadata is
  // requested from DuckDB.
  for (std::size_t i = 0; i < n; ++i) {
    auto const rg = rg_begin + i;
    if (rg < plan.row_group_pruned_by_stats.size() && plan.row_group_pruned_by_stats[rg]) {
      ++result.pruned_row_groups;
      if (rg < plan.pruned_decoded_bytes_by_row_group.size()) {
        result.pruned_decoded_bytes += plan.pruned_decoded_bytes_by_row_group[rg];
      }
      continue;
    }

    local_index_by_rg[i]  = result.row_groups.size();
    auto& rg_md           = result.row_groups.emplace_back();
    rg_md.row_group_index = rg;
    rg_md.row_group_start = plan.row_group_start[rg];
    rg_md.row_count       = plan.row_count[rg];
    rg_md.columns.resize(projected_cols.size());
    for (std::size_t ci = 0; ci < projected_cols.size(); ++ci) {
      rg_md.columns[ci].column_id = projected_cols[ci].is_rowid
                                      ? std::numeric_limits<duckdb::idx_t>::max()
                                      : projected_cols[ci].storage_idx.GetPrimaryIndex();
      rg_md.columns[ci].is_rowid  = projected_cols[ci].is_rowid;
    }
  }

  if (result.row_groups.empty()) { return result; }

  // Walk segment metadata for surviving row groups only — reading the typed
  // segment trees directly
  {
    nvtx3::scoped_range nvtx_si{"sirius::native_metadata_segment_info"};
    auto& row_groups = *plan.storage->GetRowGroupCollection();
    for (std::size_t rg = rg_begin; rg < rg_end; ++rg) {
      auto const local_rgi = local_index_by_rg[rg - rg_begin];
      if (local_rgi == n_pos) { continue; }
      auto row_group = row_groups.GetRowGroup(static_cast<duckdb::idx_t>(rg));
      if (!row_group) { continue; }
      auto& rg_md = result.row_groups[local_rgi];
      for (std::size_t ci = 0; ci < projected_cols.size(); ++ci) {
        auto const& pc = projected_cols[ci];
        if (pc.is_rowid) { continue; }
        auto reason = projected_types[ci].is_array()
                        ? walk_array_column(row_group->GetRawColumnData(pc.storage_idx),
                                            pc.storage_idx.GetPrimaryIndex(),
                                            rg,
                                            rg_md.columns[ci])
                        : walk_standard_column(row_group->GetRawColumnData(pc.storage_idx),
                                               projected_types[ci].is_varchar(),
                                               pc.storage_idx.GetPrimaryIndex(),
                                               rg,
                                               rg_md.columns[ci]);
        if (reason) {
          refuse(std::move(*reason));
          return result;
        }
      }
    }
  }

  // Sort data_segments by segment_start ascending for codec run coalescing.
  for (auto& rg_md : result.row_groups) {
    auto seg_less = [](const duckdb_segment_descriptor& a, const duckdb_segment_descriptor& b) {
      return a.segment_start < b.segment_start;
    };
    for (auto& col_md : rg_md.columns) {
      std::sort(col_md.data_segments.begin(), col_md.data_segments.end(), seg_less);
      std::sort(col_md.validity_segments.begin(), col_md.validity_segments.end(), seg_less);
      std::sort(
        col_md.array_child_data_segments.begin(), col_md.array_child_data_segments.end(), seg_less);
      std::sort(col_md.array_child_validity_segments.begin(),
                col_md.array_child_validity_segments.end(),
                seg_less);
    }
  }

  // Compute row_count manually for any row group beyond PartitionStatistics range.
  for (auto& rg_md : result.row_groups) {
    if (rg_md.row_count != 0) { continue; }
    for (const auto& col_md : rg_md.columns) {
      if (col_md.is_rowid) { continue; }
      duckdb::idx_t col_count = 0;
      for (const auto& d : col_md.data_segments) {
        col_count += d.segment_count;
      }
      if (col_count > 0) {
        rg_md.row_count = col_count;
        break;
      }
    }
  }

  // Per-segment on-disk byte sizes, over this range's segments. A segment whose
  // DuckDB block extends past the range is sized to the block end: an upper
  // bound, since decoders self-bound reads via their segment headers.
  compute_segment_bytes_size(result.row_groups, plan.block_size);

  // Per row group, compute the decoded-byte budget and per-column varchar char
  // count. Refuse a varchar column whose char count would overflow cudf's int32
  // string offsets.
  for (auto& rg_md : result.row_groups) {
    rg_md.varchar_bytes_per_col.assign(projected_cols.size(), 0);
    std::size_t budget = 0;
    for (std::size_t ci = 0; ci < projected_cols.size(); ++ci) {
      const auto& col_md = rg_md.columns[ci];
      if (col_md.is_rowid) {
        budget += static_cast<std::size_t>(rg_md.row_count) * sizeof(std::int64_t);
        continue;
      }
      if (projected_types[ci].is_varchar()) {
        std::size_t chars = 0;
        for (const auto& seg : col_md.data_segments) {
          chars += static_cast<std::size_t>(seg.segment_count) *
                   static_cast<std::size_t>(*seg.max_string_length);
        }
        if (chars >= kCudfInt32StringsThreshold) {
          refuse("row group " + std::to_string(rg_md.row_group_index) + " column " +
                 std::to_string(col_md.column_id) + " varchar chars upper bound (" +
                 std::to_string(chars) + ") >= cudf int32 chars threshold");
          return result;
        }
        rg_md.varchar_bytes_per_col[ci] = chars;
        budget += chars + static_cast<std::size_t>(rg_md.row_count) * sizeof(std::uint32_t);
      } else if (projected_types[ci].is_array()) {
        // ARRAY: offsets (int32) + child values (array_size × child_width × row_count)
        auto const array_size  = projected_types[ci].array_size();
        auto const child_width = projected_types[ci].array_child().fixed_width_byte_size();
        budget += static_cast<std::size_t>(rg_md.row_count) *
                  (sizeof(std::int32_t) + array_size * child_width);
      } else {
        budget +=
          static_cast<std::size_t>(rg_md.row_count) * projected_types[ci].fixed_width_byte_size();
      }
    }
    rg_md.decoded_bytes_budget = budget;
  }

  return result;
}

}  // namespace sirius::op::scan
