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

#include "op/scan/duckdb_insert_delta.hpp"

#include <cudf/utilities/bit.hpp>

#include <duckdb/common/enums/scan_options.hpp>
#include <duckdb/common/types/selection_vector.hpp>
#include <duckdb/common/vector_size.hpp>
#include <duckdb/function/compression_function.hpp>
#include <duckdb/function/partition_stats.hpp>
#include <duckdb/main/attached_database.hpp>
#include <duckdb/storage/buffer/buffer_handle.hpp>
#include <duckdb/storage/buffer_manager.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/segment/uncompressed.hpp>
#include <duckdb/storage/statistics/base_statistics.hpp>
#include <duckdb/storage/statistics/string_stats.hpp>
#include <duckdb/storage/table/array_column_data.hpp>
#include <duckdb/storage/table/column_data.hpp>
#include <duckdb/storage/table/column_segment.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/row_group_collection.hpp>
#include <duckdb/storage/table/row_group_segment_tree.hpp>
#include <duckdb/storage/table/segment_tree.hpp>
#include <duckdb/storage/table/standard_column_data.hpp>
#include <duckdb/storage/table/validity_column_data.hpp>
#include <duckdb/transaction/duck_transaction.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace sirius::op::scan {

namespace {

constexpr char const* kTag = "[insert_delta]";

[[noreturn]] void throw_capture(std::string what)
{
  throw std::runtime_error(std::string(kTag) + " " + std::move(what));
}

// Collects a segment's extra compression-function block ids (overflow blocks).
struct collect_block_ids : public duckdb::BlockIdVisitor {
  explicit collect_block_ids(std::vector<duckdb::block_id_t>& out) : out(out) {}
  void Visit(duckdb::block_id_t block_id) override { out.push_back(block_id); }
  std::vector<duckdb::block_id_t>& out;
};

bool is_constant_or_empty_validity(duckdb::CompressionType c)
{
  return c == duckdb::CompressionType::COMPRESSION_CONSTANT ||
         c == duckdb::CompressionType::COMPRESSION_EMPTY;
}

// Grants access to ArrayColumnData's protected child/validity members.
// (similar to that defined in duckdb_native_metadata.cpp)
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

/// Walk one column tree over the delta slice [k, k + row_count), in
/// row-group-relative rows. Validity trees get their own codec rules and
/// bit-granular staging extents.
void walk_delta_tree(duckdb::ColumnSegmentTree& tree,
                     bool is_validity,
                     bool is_varchar,
                     sirius::logical_type const& type,
                     duckdb::idx_t column_id,
                     std::size_t rg_index,
                     std::size_t k,
                     std::size_t row_count,
                     std::vector<insert_delta_segment>& out)
{
  auto const delta_end = k + row_count;
  auto lock            = tree.Lock();
  for (auto& node : tree.SegmentNodes(lock)) {
    auto& segment        = node.GetNode();
    auto const seg_start = static_cast<std::size_t>(node.GetRowStart());
    auto const seg_end   = seg_start + static_cast<std::size_t>(segment.count);
    auto const lo        = std::max(seg_start, k);
    auto const hi        = std::min(seg_end, delta_end);
    if (hi <= lo) { continue; }

    auto const compression = segment.GetCompressionFunction().type;
    // CONSTANT validity is uniform, with the direction in the segment stats:
    // all-valid runs (like EMPTY, a no-op mask) decode to nothing, but an
    // all-NULL run must reach the decoder so it synthesizes zero validity
    // bits.
    bool const all_null_validity = is_validity &&
                                   compression == duckdb::CompressionType::COMPRESSION_CONSTANT &&
                                   segment.stats.statistics.CanHaveNull();
    if (is_validity && !all_null_validity && is_constant_or_empty_validity(compression)) {
      continue;
    }

    insert_delta_segment out_seg;
    out_seg.segment       = &segment;
    out_seg.is_validity   = is_validity;
    out_seg.all_null      = all_null_validity;
    out_seg.segment_start = lo - k;
    out_seg.segment_count = hi - lo;
    out_seg.compression   = compression;

    if (all_null_validity) {
      // Nothing to stage or read; the marker alone drives the decode.
    } else if (segment.segment_type == duckdb::ColumnSegmentType::TRANSIENT) {
      // DuckDB compresses transient segments only at checkpoint, and
      // checkpoints are suppressed while pinned; a compressed one here means
      // a checkpoint ran anyway.
      if (compression != duckdb::CompressionType::COMPRESSION_UNCOMPRESSED) {
        throw_capture("TRANSIENT segment on column " + std::to_string(column_id) + " row group " +
                      std::to_string(rg_index) + " is compressed (" +
                      duckdb::CompressionTypeToString(compression) +
                      ") — a CHECKPOINT ran while the table was pinned");
      }
      out_seg.is_transient = true;
      if (is_validity) {
        // Validity is bit-packed, so the copy must start on a byte boundary.
        // Transient validity starts exactly at the delta boundary, so this
        // holds; check rather than assume.
        if ((lo - seg_start) % 8 != 0 || (lo - k) % 8 != 0) {
          throw_capture("transient validity segment on column " + std::to_string(column_id) +
                        " row group " + std::to_string(rg_index) +
                        " does not start byte-aligned in the delta slice");
        }
        out_seg.copy_src_offset = (lo - seg_start) / 8;
        out_seg.bytes_size      = ((hi - lo) + 7) / 8;
      } else if (is_varchar) {
        // Stage the segment's whole used extent. The uncompressed-string
        // dictionary is self-describing, so a tail-clamped row count still
        // decodes correctly.
        if (lo != seg_start) {
          throw_capture("transient varchar segment on column " + std::to_string(column_id) +
                        " row group " + std::to_string(rg_index) +
                        " starts inside the cached prefix — positional drift");
        }
        out_seg.copy_src_offset = 0;
        out_seg.bytes_size      = segment.SegmentSize();
      } else {
        auto const type_size    = type.fixed_width_byte_size();
        out_seg.copy_src_offset = (lo - seg_start) * type_size;
        out_seg.bytes_size      = (hi - lo) * type_size;
      }
    } else {
      // Persistent segments decode straight from the file; nothing is staged.
      // Partial coverage is legal only at the n_total snapshot tail, where a
      // merge committed mid-capture.
      if (lo != seg_start || (hi != seg_end && hi != delta_end)) {
        throw_capture("persistent segment on column " + std::to_string(column_id) + " row group " +
                      std::to_string(rg_index) +
                      " straddles the delta boundary — positional drift");
      }
      bool const supported = is_validity ? is_supported_validity_compression(compression)
                                         : is_supported_data_compression(compression);
      if (!supported ||
          (is_varchar && compression == duckdb::CompressionType::COMPRESSION_CONSTANT)) {
        throw_capture("persistent delta segment on column " + std::to_string(column_id) +
                      " row group " + std::to_string(rg_index) + ": unsupported compression " +
                      duckdb::CompressionTypeToString(compression));
      }
      if (compression == duckdb::CompressionType::COMPRESSION_CONSTANT) {
        // CONSTANT segments have no backing block (GetBlockSize() would
        // null-deref); the block/byte fields stay unset. Snapshot the
        // segment's own stats for the decoder: the constant value lives
        // here, and row-group-level stats drift as later appends (e.g. into
        // an indexed table's tail row group) merge into them.
        out_seg.segment_stats =
          std::make_shared<duckdb::BaseStatistics>(segment.stats.statistics.Copy());
      } else {
        out_seg.block_id     = segment.GetBlockId();
        out_seg.block_offset = segment.GetBlockOffset();
        out_seg.bytes_size   = static_cast<std::size_t>(segment.GetBlockSize()) -
                             static_cast<std::size_t>(out_seg.block_offset);
        auto const& cf = segment.GetCompressionFunction();
        if (segment.GetSegmentState() && cf.visit_block_ids) {
          collect_block_ids visitor(out_seg.additional_blocks);
          cf.visit_block_ids(segment, visitor);
        }
      }
    }

    if (is_varchar && !is_validity) {
      // Overflow strings must never reach the GPU string decoder. The
      // plan-time table-stat probe normally declines first; this catches
      // stats drift.
      if (!duckdb::StringStats::HasMaxStringLength(segment.stats.statistics)) {
        throw_capture("varchar delta segment on column " + std::to_string(column_id) +
                      " row group " + std::to_string(rg_index) + ": Max String Length stat absent");
      }
      auto const max_len = duckdb::StringStats::MaxStringLength(segment.stats.statistics);
      if (max_len >= duckdb::StringUncompressed::GetStringBlockLimit(segment.GetBlockSize())) {
        throw_capture("varchar delta segment on column " + std::to_string(column_id) +
                      " row group " + std::to_string(rg_index) +
                      ": max string length reaches the overflow-block limit");
      }
      out_seg.max_string_length = static_cast<std::uint32_t>(max_len);
    }

    out.push_back(std::move(out_seg));
  }
}

/// Fill one skeleton row group: walk its column segment trees into columns,
/// assign staging slab offsets, and compute the byte budgets. Task-safe.
insert_delta_row_group complete_delta_row_group(
  insert_delta_row_group const& skeleton,
  std::span<duckdb::storage_t const> storage_column_indices,
  std::span<sirius::logical_type const> column_types)
{
  insert_delta_row_group rg_plan = skeleton;
  auto& rg                       = *rg_plan.row_group;
  auto const rg_index            = rg_plan.row_group_index;

  std::size_t slab   = 0;
  std::size_t budget = 0;
  rg_plan.varchar_bytes_per_col.assign(storage_column_indices.size(), 0);

  for (std::size_t ci = 0; ci < storage_column_indices.size(); ++ci) {
    auto const col   = storage_column_indices[ci];
    auto& col_data   = rg.GetRawColumnData(col);
    auto const& type = column_types[ci];

    if (type.is_array() || dynamic_cast<duckdb::ArrayColumnData*>(&col_data) != nullptr) {
      // Fixed-size ARRAY: DuckDB stores it as array-level validity plus a child column
      // of row_count * array_size contiguous elements.
      auto* array_col = dynamic_cast<duckdb::ArrayColumnData*>(&col_data);
      if (array_col == nullptr) {
        throw_capture("ARRAY column " + std::to_string(col) + " row group " +
                      std::to_string(rg_index) + ": expected ArrayColumnData");
      }
      auto const array_size  = static_cast<std::size_t>(type.array_size());
      auto const& child_type = type.array_child();

      // Only a fixed-width scalar child has a staging and decode path.
      if (!child_type.is_fixed_width()) {
        throw_capture("ARRAY column " + std::to_string(col) + " row group " +
                      std::to_string(rg_index) + ": child type " + child_type.to_string() +
                      " is not a fixed-width scalar (only fixed-width ARRAY children are supported)");
      }

      insert_delta_column col_plan;
      col_plan.column_id = col;
      col_plan.is_array  = true;

      // Array-level validity -> data_segments (one bit per array row).
      auto* array_validity = array_column_access::get_validity(*array_col);
      if (array_validity == nullptr) {
        throw_capture("ARRAY column " + std::to_string(col) + " row group " +
                      std::to_string(rg_index) + ": no array validity column");
      }
      walk_delta_tree(array_validity->GetSegmentTree(),
                      /*is_validity=*/true,
                      /*is_varchar=*/false,
                      type,
                      col,
                      rg_index,
                      rg_plan.k_offset,
                      rg_plan.row_count,
                      col_plan.data_segments);

      // The array's elements live in a fixed-width child column (StandardColumnData).
      // Elements are stored flat so here we use offset and row count to index into
      // the delta slice.
      auto* child = array_column_access::get_child(*array_col);
      auto* child_std =
        child != nullptr ? dynamic_cast<duckdb::StandardColumnData*>(child) : nullptr;
      if (child_std == nullptr) {
        throw_capture("ARRAY child on column " + std::to_string(col) + " row group " +
                      std::to_string(rg_index) + ": child storage is not StandardColumnData");
      }
      auto const k_child  = rg_plan.k_offset * array_size;
      auto const rc_child = rg_plan.row_count * array_size;
      walk_delta_tree(child_std->GetSegmentTree(),
                      /*is_validity=*/false,
                      /*is_varchar=*/false,
                      child_type,
                      col,
                      rg_index,
                      k_child,
                      rc_child,
                      col_plan.array_child_data_segments);
      walk_delta_tree(child_std->GetValidityData().GetSegmentTree(),
                      /*is_validity=*/true,
                      /*is_varchar=*/false,
                      child_type,
                      col,
                      rg_index,
                      k_child,
                      rc_child,
                      col_plan.array_child_validity_segments);

      // The child data segments must fully cover the child slice with no gaps.
      // Array-level validity can have gaps (all-valid runs are dropped), so it
      // can't be used to check coverage.
      std::size_t child_rows = 0;
      for (auto const& s : col_plan.array_child_data_segments) {
        child_rows += s.segment_count;
      }
      if (child_rows != rc_child) {
        throw_capture("ARRAY column " + std::to_string(col) + " row group " +
                      std::to_string(rg_index) + ": child data segments cover " +
                      std::to_string(child_rows) + " of " + std::to_string(rc_child) + " elements");
      }

      // Stage every transient segment across the three buckets.
      for (auto* segs : {&col_plan.data_segments,
                         &col_plan.array_child_data_segments,
                         &col_plan.array_child_validity_segments}) {
        for (auto& s : *segs) {
          if (!s.is_transient) { continue; }
          // 8-byte aligned
          slab          = (slab + 7) / 8 * 8;
          s.slab_offset = slab;
          slab += s.bytes_size;
        }
      }

      // Decoded bytes: the child element payload plus the LIST offsets buffer.
      // Array-level validity is one bit per row and not counted, matching the fixed-width path.
      budget += rc_child * child_type.fixed_width_byte_size() +
                rg_plan.row_count * sizeof(std::int32_t);

      rg_plan.columns.push_back(std::move(col_plan));
      continue;
    }
    auto* std_col = dynamic_cast<duckdb::StandardColumnData*>(&col_data);
    if (std_col == nullptr) {
      throw_capture("column " + std::to_string(col) + " storage for type " + type.to_string() +
                    " is not StandardColumnData (nested/unsupported)");
    }

    insert_delta_column col_plan;
    col_plan.column_id  = col;
    col_plan.is_varchar = type.is_varchar();
    walk_delta_tree(std_col->GetSegmentTree(),
                    /*is_validity=*/false,
                    col_plan.is_varchar,
                    type,
                    col,
                    rg_index,
                    rg_plan.k_offset,
                    rg_plan.row_count,
                    col_plan.data_segments);
    walk_delta_tree(std_col->GetValidityData().GetSegmentTree(),
                    /*is_validity=*/true,
                    /*is_varchar=*/false,
                    type,
                    col,
                    rg_index,
                    rg_plan.k_offset,
                    rg_plan.row_count,
                    col_plan.validity_segments);

    // The data segments must tile the delta slice exactly; a gap would
    // decode garbage rows the mask never drops.
    std::size_t data_rows = 0;
    for (auto const& s : col_plan.data_segments) {
      data_rows += s.segment_count;
    }
    if (data_rows != rg_plan.row_count) {
      throw_capture("column " + std::to_string(col) + " row group " + std::to_string(rg_index) +
                    ": delta segments cover " + std::to_string(data_rows) + " of " +
                    std::to_string(rg_plan.row_count) + " rows");
    }

    for (auto* segs : {&col_plan.data_segments, &col_plan.validity_segments}) {
      for (auto& s : *segs) {
        if (!s.is_transient) { continue; }
        slab          = (slab + 7) / 8 * 8;  // 8-byte align within the slab
        s.slab_offset = slab;
        slab += s.bytes_size;
      }
    }
    if (col_plan.is_varchar) {
      std::size_t varchar_bytes = 0;
      for (auto const& s : col_plan.data_segments) {
        varchar_bytes +=
          s.segment_count * (s.max_string_length.has_value() ? *s.max_string_length : 0);
      }
      rg_plan.varchar_bytes_per_col[ci] = varchar_bytes;
      budget += rg_plan.row_count * sizeof(std::uint32_t) + varchar_bytes;
    } else {
      budget += rg_plan.row_count * type.fixed_width_byte_size();
    }

    rg_plan.columns.push_back(std::move(col_plan));
  }

  rg_plan.transient_staging_bytes = slab;
  rg_plan.decoded_bytes_budget    = budget;
  return rg_plan;
}

}  // namespace

insert_delta_plan prepare_insert_delta_capture(duckdb::DataTable& storage,
                                               duckdb::ClientContext& context,
                                               std::size_t n_cache)
{
  insert_delta_plan plan;
  auto& txn           = duckdb::DuckTransaction::Get(context, storage.GetAttached());
  plan.transaction    = duckdb::TransactionData(txn);
  plan.buffer_manager = &duckdb::BufferManager::GetBufferManager(context);
  plan.n_cache        = n_cache;
  plan.n_total        = static_cast<std::size_t>(storage.GetTotalRows());
  if (plan.n_total <= n_cache) { return plan; }

  auto tree = storage.GetRowGroupCollection()->GetRowGroups();
  auto lock = tree->Lock();
  // Row groups are contiguous and tail-append-only, so nothing below the row
  // group holding the last cached row can contain delta rows: binary-search
  // that boundary instead of scanning the cached prefix. The boundary row
  // group itself stays in the walk — an indexed table's appends grow its tail
  // (the k_offset > 0 case) — and an unchanged one costs a single iteration.
  auto const boundary_rg =
    n_cache == 0 ? 0 : tree->GetSegmentIndex(lock, static_cast<duckdb::idx_t>(n_cache - 1));
  for (duckdb::idx_t rg_index = boundary_rg;; ++rg_index) {
    auto node = tree->GetSegmentByIndex(lock, static_cast<int64_t>(rg_index));
    if (!node) { break; }
    auto& rg            = node->GetNode();
    auto const rg_start = static_cast<std::size_t>(node->GetRowStart());
    if (rg_start >= plan.n_total) { break; }
    auto const live_end = rg_start + static_cast<std::size_t>(rg.count.load());
    if (live_end <= n_cache) { continue; }

    insert_delta_row_group rg_plan;
    rg_plan.row_group       = &rg;
    rg_plan.row_group_index = rg_index;
    rg_plan.k_offset        = n_cache > rg_start ? n_cache - rg_start : 0;
    auto const covered_end  = std::min(live_end, plan.n_total);
    if (covered_end <= rg_start + rg_plan.k_offset) { continue; }
    rg_plan.row_count       = covered_end - (rg_start + rg_plan.k_offset);
    rg_plan.row_group_start = rg_start + rg_plan.k_offset;
    rg_plan.has_version_state =
      duckdb::RowGroup::GetPartitionStats(*node).count_type == duckdb::CountType::COUNT_APPROXIMATE;

    plan.row_groups.push_back(std::move(rg_plan));
  }
  return plan;
}

std::vector<insert_delta_row_group> capture_insert_delta_row_group_range(
  insert_delta_plan const& plan,
  std::size_t rg_begin,
  std::size_t rg_end,
  std::span<duckdb::storage_t const> storage_column_indices,
  std::span<sirius::logical_type const> column_types)
{
  if (storage_column_indices.size() != column_types.size()) {
    throw_capture("column index / type lists are not parallel");
  }

  rg_end = std::min(rg_end, plan.row_groups.size());
  std::vector<insert_delta_row_group> out;
  if (rg_begin >= rg_end) { return out; }
  out.reserve(rg_end - rg_begin);
  for (std::size_t i = rg_begin; i < rg_end; ++i) {
    out.push_back(
      complete_delta_row_group(plan.row_groups[i], storage_column_indices, column_types));
  }
  return out;
}

insert_delta_plan capture_insert_delta_plan(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  std::size_t n_cache,
  std::span<duckdb::storage_t const> storage_column_indices,
  std::span<sirius::logical_type const> column_types)
{
  auto plan = prepare_insert_delta_capture(storage, context, n_cache);
  // Run the range walk even on an empty plan so the parallel-lists check
  // fires the same way in both call shapes.
  plan.row_groups = capture_insert_delta_row_group_range(
    plan, 0, plan.row_groups.size(), storage_column_indices, column_types);
  return plan;
}

std::size_t count_visible_delta_rows(insert_delta_row_group const& rg,
                                     duckdb::TransactionData transaction,
                                     std::span<std::uint32_t> mask_words,
                                     std::size_t mask_bit_offset)
{
  auto const k   = rg.k_offset;
  auto const end = k + rg.row_count;

  auto set_range = [&](std::size_t from, std::size_t to) {  // rg-relative [from, to)
    for (std::size_t r = from; r < to; ++r) {
      cudf::set_bit_unsafe(mask_words.data(),
                           static_cast<cudf::size_type>(mask_bit_offset + (r - k)));
    }
  };

  if (!rg.has_version_state) {
    // No version state at capture means every covered row is visible. The
    // probe can flag a clean row group but never misses a dirty one.
    set_range(k, end);
    return rg.row_count;
  }

  duckdb::SelectionVector sel(STANDARD_VECTOR_SIZE);
  duckdb::ScanOptions const options{transaction};
  std::size_t visible = 0;

  for (std::size_t vec_start = (k / STANDARD_VECTOR_SIZE) * STANDARD_VECTOR_SIZE; vec_start < end;
       vec_start += STANDARD_VECTOR_SIZE) {
    auto const vector_idx = vec_start / STANDARD_VECTOR_SIZE;
    auto const max_count  = std::min<std::size_t>(STANDARD_VECTOR_SIZE, end - vec_start);
    auto const count      = static_cast<std::size_t>(rg.row_group->GetSelVector(
      options, static_cast<duckdb::idx_t>(vector_idx), sel, static_cast<duckdb::idx_t>(max_count)));
    auto const from       = std::max(vec_start, k);
    if (count == max_count) {
      // All rows visible. GetSelVector may leave sel unfilled here; never read it.
      set_range(from, vec_start + max_count);
      visible += vec_start + max_count - from;
      continue;
    }
    for (std::size_t i = 0; i < count; ++i) {
      auto const pos = vec_start + static_cast<std::size_t>(sel.get_index(i));
      if (pos < from) { continue; }
      cudf::set_bit_unsafe(mask_words.data(),
                           static_cast<cudf::size_type>(mask_bit_offset + (pos - k)));
      ++visible;
    }
  }
  return visible;
}

void copy_delta_row_group(insert_delta_row_group const& rg,
                          duckdb::BufferManager& buffer_manager,
                          std::uint8_t* slab_base)
{
  for (auto const& col : rg.columns) {
    for (auto const* segs : {&col.data_segments,
                             &col.validity_segments,
                             &col.array_child_data_segments,
                             &col.array_child_validity_segments}) {
      for (auto const& s : *segs) {
        if (!s.is_transient) { continue; }
        // Pin just long enough to memcpy; no DuckDB handle outlives the call.
        auto handle = s.segment->block;
        auto pin    = buffer_manager.Pin(handle);
        std::memcpy(slab_base + s.slab_offset, pin.Ptr() + s.copy_src_offset, s.bytes_size);
      }
    }
  }
}

}  // namespace sirius::op::scan
