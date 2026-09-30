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

#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>

#include <duckdb/common/enums/scan_options.hpp>
#include <duckdb/common/types/selection_vector.hpp>
#include <duckdb/common/vector_size.hpp>
#include <duckdb/function/partition_stats.hpp>
#include <duckdb/storage/data_table.hpp>
#include <duckdb/storage/table/array_column_data.hpp>
#include <duckdb/storage/table/column_data.hpp>
#include <duckdb/storage/table/column_segment.hpp>
#include <duckdb/storage/table/row_group.hpp>
#include <duckdb/storage/table/row_group_collection.hpp>
#include <duckdb/storage/table/row_group_segment_tree.hpp>
#include <duckdb/storage/table/validity_column_data.hpp>
#include <duckdb/transaction/duck_transaction.hpp>
#include <op/scan/duckdb_mvcc_visibility.hpp>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace sirius::op::scan {

namespace {

constexpr std::size_t kWordBits = 32;

// The host fill writes uint32 words that mask_to_bools reads as
// cudf::bitmask_type.
static_assert(sizeof(std::uint32_t) == sizeof(cudf::bitmask_type),
              "bit-packed keep-masks assume cudf::bitmask_type is 32-bit");

/// Set or clear bits [begin, begin + count) of @p words: read-modify-write on
/// partial edge words, whole-word fills between, so any begin/count is legal.
/// Caller contract: no concurrent writer shares an edge word (the mask job
/// slices parallel tasks only on word-aligned offsets), and the words start
/// zeroed so edge reads never see indeterminate memory.
void write_bit_range(std::span<std::uint32_t> words,
                     std::size_t begin,
                     std::size_t count,
                     bool value)
{
  if (count == 0) { return; }
  auto const end       = begin + count;
  auto const first     = begin / kWordBits;
  auto const last      = (end - 1) / kWordBits;
  auto const head_bits = begin % kWordBits;
  auto const tail_bits = end % kWordBits;  // 0 => the last word is fully covered

  std::uint32_t const head_mask = ~0u << head_bits;
  std::uint32_t const tail_mask = tail_bits ? ~(~0u << tail_bits) : ~0u;

  if (first == last) {
    std::uint32_t const m = head_mask & tail_mask;
    words[first]          = value ? (words[first] | m) : (words[first] & ~m);
    return;
  }
  words[first] = value ? (words[first] | head_mask) : (words[first] & ~head_mask);
  std::fill(words.begin() + static_cast<std::ptrdiff_t>(first) + 1,
            words.begin() + static_cast<std::ptrdiff_t>(last),
            value ? 0xFFFFFFFFu : 0u);
  words[last] = value ? (words[last] | tail_mask) : (words[last] & ~tail_mask);
}

/// Non-loading per-row-group version-state probe (see header docs for why
/// this is a safe skip signal). Goes through the public GetPartitionStats
/// surface: DuckDB marks a row group's count APPROXIMATE exactly when it has
/// version state — in-memory row versions or unloaded persisted deletes
/// (RowGroup::HasUnloadedDeletes / GetVersionInfoIfLoaded are private).
bool row_group_has_version_state(duckdb::SegmentNode<duckdb::RowGroup>& node)
{
  return duckdb::RowGroup::GetPartitionStats(node).count_type ==
         duckdb::CountType::COUNT_APPROXIMATE;
}

}  // namespace

bool mvcc_visibility_plan::any_version_state() const
{
  return std::any_of(chunk_has_version_state.begin(),
                     chunk_has_version_state.end(),
                     [](bool dirty) { return dirty; });
}

mvcc_visibility_plan capture_mvcc_visibility_plan(
  duckdb::DataTable& storage,
  duckdb::ClientContext& context,
  scan_manager::duckdb_mvcc_metadata const& metadata)
{
  auto& txn = duckdb::DuckTransaction::Get(context, storage.GetAttached());
  mvcc_visibility_plan plan{duckdb::TransactionData(txn)};

  if (plan.transaction.start_time < metadata.v_base) {
    throw std::runtime_error(
      "[capture_mvcc_visibility_plan] query snapshot (start_time " +
      std::to_string(plan.transaction.start_time) + ") predates the pin snapshot (v_base " +
      std::to_string(metadata.v_base) + ") — a re-pin raced this query between plan and prepare");
  }

  auto const& counts = metadata.base_row_count_per_chunk;
  auto const n_cache = metadata.n_cache();
  plan.mvcc_row_groups.resize(counts.size());
  plan.chunk_has_version_state.assign(counts.size(), false);
  if (n_cache == 0) { return plan; }

  auto tree = storage.GetRowGroupCollection()->GetRowGroups();

  std::size_t chunk_idx      = 0;
  std::size_t chunk_start    = 0;
  std::size_t chunk_end      = counts[0];
  std::size_t expected_start = 0;

  for (duckdb::idx_t rg_index = 0; expected_start < n_cache; ++rg_index) {
    // Advance to the chunk containing expected_start (skips zero-row chunks).
    while (expected_start == chunk_end && chunk_idx + 1 < counts.size()) {
      ++chunk_idx;
      chunk_start = chunk_end;
      chunk_end += counts[chunk_idx];
    }

    auto node = tree->GetSegmentByIndex(static_cast<int64_t>(rg_index));
    if (!node) {
      throw std::runtime_error("[capture_mvcc_visibility_plan] row groups end at rowid " +
                               std::to_string(expected_start) + " but the pin covers [0, " +
                               std::to_string(n_cache) +
                               ") — live coverage below the pinned prefix");
    }
    auto const rg_start = static_cast<std::size_t>(node->GetRowStart());
    if (rg_start != expected_start) {
      throw std::runtime_error("[capture_mvcc_visibility_plan] non-contiguous row groups: index " +
                               std::to_string(rg_index) + " starts at rowid " +
                               std::to_string(rg_start) + ", expected " +
                               std::to_string(expected_start));
    }

    auto& rg              = node->GetNode();
    auto const live_count = static_cast<std::size_t>(rg.count.load());
    // Clamp to pin coverage: post-pin appends grow the LAST covered row
    // group's live count; the pinned prefix never moves.
    auto const covered = std::min(live_count, n_cache - expected_start);
    if (covered == 0) {
      throw std::runtime_error(
        "[capture_mvcc_visibility_plan] zero-row row group inside the pinned prefix at rowid " +
        std::to_string(expected_start));
    }
    if (expected_start + covered > chunk_end) {
      // Chunks are contiguous runs of WHOLE row groups (validated at pin by
      // validate_duckdb_pin_chunk); a straddle means the invariant broke.
      throw std::runtime_error(
        "[capture_mvcc_visibility_plan] chunk boundary at rowid " + std::to_string(chunk_end) +
        " does not land on a row-group boundary (row group [" + std::to_string(expected_start) +
        ", " + std::to_string(expected_start + covered) + "))");
    }

    // Checkpoint-grown tables have row groups of arbitrary sizes, so this
    // offset may be unaligned; the fill writer handles word edges and the
    // mask job serializes unaligned chunks into a single fill task.
    auto const chunk_offset = expected_start - chunk_start;

    bool const dirty = row_group_has_version_state(*node);
    plan.mvcc_row_groups[chunk_idx].push_back(
      {&rg, node->GetRowStart(), chunk_offset, covered, dirty});
    if (dirty) { plan.chunk_has_version_state[chunk_idx] = true; }

    expected_start += covered;
  }

  return plan;
}

bool fill_keep_mask_for_row_groups(std::span<mvcc_row_group_slice const> slices,
                                   duckdb::TransactionData transaction,
                                   std::span<std::uint32_t> chunk_mask_words)
{
  bool any_dropped = false;
  duckdb::SelectionVector sel(STANDARD_VECTOR_SIZE);
  duckdb::ScanOptions const options{transaction};

  for (auto const& slice : slices) {
    if (!slice.has_version_state) {
      // No version state at capture: every covered row is visible. No
      // GetVersionInfo, no GetSelVector, no version lock.
      write_bit_range(chunk_mask_words, slice.chunk_offset, slice.row_count, true);
      continue;
    }

    auto const vectors = (slice.row_count + STANDARD_VECTOR_SIZE - 1) / STANDARD_VECTOR_SIZE;
    for (std::size_t v = 0; v < vectors; ++v) {
      auto const off = slice.chunk_offset + v * STANDARD_VECTOR_SIZE;
      auto const max_count =
        std::min<std::size_t>(STANDARD_VECTOR_SIZE, slice.row_count - v * STANDARD_VECTOR_SIZE);
      auto const count = static_cast<std::size_t>(slice.row_group->GetSelVector(
        options, static_cast<duckdb::idx_t>(v), sel, static_cast<duckdb::idx_t>(max_count)));

      if (count == max_count) {
        // All visible — sel may be UNFILLED on this fast path; never read it.
        write_bit_range(chunk_mask_words, off, max_count, true);
        continue;
      }
      any_dropped = true;
      write_bit_range(chunk_mask_words, off, max_count, false);
      for (std::size_t i = 0; i < count; ++i) {
        cudf::set_bit_unsafe(chunk_mask_words.data(),
                             static_cast<cudf::size_type>(off + sel.get_index(i)));
      }
    }
  }
  return any_dropped;
}

bool any_update_chains(duckdb::DataTable& storage,
                       std::span<duckdb::storage_t const> storage_column_indices,
                       std::size_t row_prefix)
{
  if (row_prefix == 0 || storage_column_indices.empty()) { return false; }
  auto tree           = storage.GetRowGroupCollection()->GetRowGroups();
  std::size_t covered = 0;
  for (duckdb::idx_t rg_index = 0; covered < row_prefix; ++rg_index) {
    auto node = tree->GetSegmentByIndex(static_cast<int64_t>(rg_index));
    if (!node) { break; }  // fewer live rows than the prefix — nothing left to check
    auto& rg = node->GetNode();
    for (auto col : storage_column_indices) {
      if (rg.GetRawColumnData(col).HasUpdates()) { return true; }
    }
    covered =
      static_cast<std::size_t>(node->GetRowStart()) + static_cast<std::size_t>(rg.count.load());
  }
  return false;
}

namespace {

// Grants access to ArrayColumnData's protected child/validity members.
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

// True if any segment in the tree is a transient (uncheckpointed) append.
bool tree_has_transient_segment(duckdb::ColumnSegmentTree& tree)
{
  for (auto& seg_node : tree.SegmentNodes()) {
    if (seg_node.GetNode().segment_type == duckdb::ColumnSegmentType::TRANSIENT) { return true; }
  }
  return false;
}

// True if the column carries an uncheckpointed append. A fixed-size ARRAY keeps
// appended rows in its array-level validity tree and its child column tree, not
// the parent tree, so those two trees have to be checked as well.
bool column_has_transient_append(duckdb::ColumnData& col_data)
{
  if (tree_has_transient_segment(col_data.GetSegmentTree())) { return true; }
  auto* array_col = dynamic_cast<duckdb::ArrayColumnData*>(&col_data);
  if (array_col == nullptr) { return false; }
  if (auto* validity = array_column_access::get_validity(*array_col)) {
    if (tree_has_transient_segment(validity->GetSegmentTree())) { return true; }
  }
  if (auto* child = array_column_access::get_child(*array_col)) {
    if (tree_has_transient_segment(child->GetSegmentTree())) { return true; }
  }
  return false;
}

}  // namespace

bool any_uncheckpointed_appends(duckdb::DataTable& storage,
                                std::span<duckdb::storage_t const> storage_column_indices)
{
  if (storage_column_indices.empty()) { return false; }
  auto tree = storage.GetRowGroupCollection()->GetRowGroups();
  for (duckdb::idx_t rg_index = 0;; ++rg_index) {
    auto node = tree->GetSegmentByIndex(static_cast<int64_t>(rg_index));
    if (!node) { return false; }
    auto& rg = node->GetNode();
    for (auto col : storage_column_indices) {
      if (column_has_transient_append(rg.GetRawColumnData(col))) { return true; }
    }
  }
}

native_read_mvcc_state check_native_read_mvcc_state(
  duckdb::DataTable& storage,
  std::span<duckdb::storage_t const> scanned_columns,
  duckdb::TransactionData transaction)
{
  auto tree = storage.GetRowGroupCollection()->GetRowGroups();
  for (duckdb::idx_t rg_index = 0;; ++rg_index) {
    auto node = tree->GetSegmentByIndex(static_cast<int64_t>(rg_index));
    if (!node) { return native_read_mvcc_state::exact; }
    auto& rg = node->GetNode();
    for (auto col : scanned_columns) {
      if (rg.GetRawColumnData(col).HasUpdates()) {
        return native_read_mvcc_state::has_update_chains;
      }
    }
    if (!row_group_has_version_state(*node)) { continue; }
    // Read the physical count first: a concurrent append makes the two
    // counts diverge in either order, so a race can only refuse.
    auto const physical = static_cast<duckdb::idx_t>(rg.count.load());
    if (rg.GetVisibleRowCount(transaction) != physical) {
      return native_read_mvcc_state::has_invisible_rows;
    }
  }
}

}  // namespace sirius::op::scan
