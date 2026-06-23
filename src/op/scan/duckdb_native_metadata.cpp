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

#include <nvtx3/nvtx3.hpp>

#include <duckdb/common/column_index.hpp>
#include <duckdb/common/enums/compression_type.hpp>
#include <duckdb/common/enums/filter_propagate_result.hpp>
#include <duckdb/function/compression_function.hpp>
#include <duckdb/function/partition_stats.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/planner/table_filter.hpp>
#include <duckdb/storage/block_manager.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/storage/storage_manager.hpp>
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
  return desc;
}

// Walks the segment trees of a projected column, collecting metadata about its data and validity
// segments. Returns a viability-failure reason, or nullopt on success.
std::optional<std::string> walk_typed_column(duckdb::ColumnData& col_data,
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
    col_md.validity_segments.push_back(fill_segment_descriptor(segment, node.GetRowStart()));
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
/// The only filter type we must exclude from statistics pruning is DYNAMIC_FILTER:
/// its bounds come from a runtime source (e.g. a hash-join build) and are not
/// currently populated at metadata-walk time.
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

bool row_group_pruned_by_filter_stats(duckdb::PartitionRowGroup& prg,
                                      const duckdb::TableFilterSet& table_filters,
                                      const duckdb::vector<duckdb::ColumnIndex>& column_ids)
{
  for (auto const& [col_idx, filter] : table_filters.filters) {
    if (!filter_is_prunable(filter->filter_type)) { continue; }
    if (col_idx >= column_ids.size()) { continue; }  // defensive
    auto const& column_id = column_ids[col_idx];
    if (!column_index_can_have_storage_stats(column_id)) { continue; }

    auto stats = prg.GetColumnStatistics(duckdb::StorageIndex(column_id.GetPrimaryIndex()));
    if (!stats) { continue; }  // no stats -> cannot prune
    if (filter->CheckStatistics(*stats) == duckdb::FilterPropagateResult::FILTER_ALWAYS_FALSE) {
      return true;
    }
  }
  return false;
}

/// @brief Mark row groups a pushed-down filter proves can hold no matching rows.
///
/// Runs during prepare, before worker-thread segment walks, because statistics
/// are available from PartitionRowGroup handles and do not require segment
/// metadata. This avoids parsing metadata for row groups that will be skipped
/// and lets the all-pruned case route to DuckDB CPU before the async scan starts.
void mark_row_groups_pruned_by_filter_stats(duckdb_native_walk_plan& plan)
{
  if (plan.table_filters == nullptr || plan.table_filters->filters.empty() ||
      plan.column_ids == nullptr || plan.column_ids->empty()) {
    return;
  }

  const auto& table_filters   = *plan.table_filters;
  const auto& column_ids      = *plan.column_ids;
  const auto& projected_cols  = *plan.projected_cols;
  const auto& projected_types = *plan.projected_types;

  for (std::size_t rg = 0; rg < plan.n_row_groups; ++rg) {
    if (rg >= plan.partition_row_groups.size() || !plan.partition_row_groups[rg]) { continue; }
    auto& prg = *plan.partition_row_groups[rg];
    if (!row_group_pruned_by_filter_stats(prg, table_filters, column_ids)) { continue; }

    plan.row_group_pruned_by_stats[rg] = true;
    auto const pruned_bytes =
      estimate_decoded_bytes_budget(plan.row_count[rg], projected_cols, projected_types);
    plan.pruned_decoded_bytes_by_row_group[rg] = pruned_bytes;
    ++plan.pruned_row_groups;
    plan.pruned_decoded_bytes += pruned_bytes;
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

  mark_row_groups_pruned_by_filter_stats(plan);
  if (plan.pruned_row_groups > 0) {
    SIRIUS_LOG_DEBUG(
      "[duckdb_native_metadata] prepare stats-pruned {} row groups (~{} decoded bytes)",
      plan.pruned_row_groups,
      plan.pruned_decoded_bytes);
  }
  if (plan.n_row_groups > 0 && plan.pruned_row_groups == plan.n_row_groups) {
    refuse("no row groups in table (empty or fully pruned)");
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
        auto reason = walk_typed_column(row_group->GetRawColumnData(pc.storage_idx),
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
