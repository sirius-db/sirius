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

// Late-materialization origin tracking (env gate: SIRIUS_EXP_LATE_MAT).
//
// A column that will be late-materialized is described by WHERE it came from
// (a pinned-table entry + column position: @ref column_origin) and WHICH rows
// of that origin are live (@ref row_selection). Both are metadata-only: no
// device work happens in this header, and when the gate is off none of these
// types is ever instantiated (zero-cost-when-off contract).
//
// GLOBAL ROW ADDRESSING. A row's global id is its position in pinned-table
// order (the concatenation of the entry's chunks in emission order). Each
// served scan batch covers one pinned chunk, i.e. the contiguous global span
// [range.start, range.start + range.rows). Simpatico decode coordinates are
// BATCH-LOCAL: local = gid - range.start, chunk_id = local / 1024,
// offset = local % 1024 — exactly the fused selection_mask geometry
// (sirius::codegen::SELECTION_CHUNK_ROWS), because every pinned chunk's
// compressed table numbers its 1024-row chunks from its own row 0. There is
// deliberately NO table-global chunk_id*1024+offset encoding: batch row
// starts are not multiples of 1024.
//
// Consumed by the late materializer
// (materialize(column_origin, row_selection, stream, mr)).

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace sirius::scan_manager {
struct pinned_entry;  // scan_manager/sirius_scan_manager.hpp
}  // namespace sirius::scan_manager

namespace sirius::late_mat {

/// Env gate for late materialization. Set and != "0" means on. This is the
/// ONLY reader — every TU must use it so the on/off semantics can never fork
/// (the fused scan-filter gate ended up with two subtly different readers).
inline bool late_mat_enabled()
{
  static const bool enabled = []() {
    char const* v = std::getenv("SIRIUS_EXP_LATE_MAT");
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
  }();
  return enabled;
}

/// v2 sub-gate (SIRIUS_EXP_LATE_MAT_V2, default off): the PLANNER lifetime
/// pass decides deferrals and the v1 pipeline walk demotes to a lowering/
/// verification backend. Implies the main gate — v2 can never be on while
/// late_mat_enabled() is false, so v1 banked behavior stays re-measurable
/// byte-identically with the sub-gate unset.
inline bool late_mat_v2_enabled()
{
  static const bool enabled = []() {
    char const* v = std::getenv("SIRIUS_EXP_LATE_MAT_V2");
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
  }();
  return enabled && late_mat_enabled();
}

/// v3 sub-gate (SIRIUS_EXP_LATE_MAT_V3, default off): FD/composite-key
/// group-by-rowid proofs (determination closures over pin-unique seeds and
/// INNER-join equality transfer). Implies the v2 sub-gate, which implies the
/// main gate — the stack can never invert.
inline bool late_mat_v3_enabled()
{
  static const bool enabled = []() {
    char const* v = std::getenv("SIRIUS_EXP_LATE_MAT_V3");
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
  }();
  return enabled && late_mat_v2_enabled();
}

/// Columns selected by SIRIUS_LATE_MAT_PIN_UNIQUE_COLS for the pin-time
/// uniqueness probes, as positions into @p column_names. Two value forms:
/// a comma-separated case-sensitive name list, or a boolean-style value
/// ("1"/"all"/"*") selecting every column. Shared by the cheap per-chunk
/// probe and the exact-count fallback so the two can never track different
/// column sets. Empty when the env is unset/empty or the main gate is off.
inline std::vector<std::uint32_t> pin_unique_probe_columns(
  std::span<std::string const> column_names)
{
  std::vector<std::uint32_t> tracked;
  if (!late_mat_enabled()) { return tracked; }
  char const* env = std::getenv("SIRIUS_LATE_MAT_PIN_UNIQUE_COLS");
  if (env == nullptr || env[0] == '\0') { return tracked; }
  std::string const list(env);
  if (list == "1" || list == "all" || list == "*") {
    tracked.reserve(column_names.size());
    for (std::size_t i = 0; i < column_names.size(); ++i) {
      tracked.push_back(static_cast<std::uint32_t>(i));
    }
    return tracked;
  }
  std::size_t start = 0;
  while (start <= list.size()) {
    auto const comma = list.find(',', start);
    auto name =
      list.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    while (!name.empty() && name.front() == ' ') { name.erase(name.begin()); }
    while (!name.empty() && name.back() == ' ') { name.pop_back(); }
    if (!name.empty()) {
      auto const it = std::find(column_names.begin(), column_names.end(), name);
      if (it != column_names.end()) {
        tracked.push_back(
          static_cast<std::uint32_t>(std::distance(column_names.begin(), it)));
      }
    }
    if (comma == std::string::npos) { break; }
    start = comma + 1;
  }
  return tracked;
}

/// Pin generation. 0 is never a live generation (it is the invalidated /
/// "no origin" value), so a zero-initialized column_origin fails closed.
using pin_generation_t = std::uint64_t;

/// A contiguous span of rows in pinned-table order.
struct row_range {
  std::int64_t start{0};  ///< first global row id covered
  std::int64_t rows{0};   ///< number of rows covered

  [[nodiscard]] std::int64_t end() const noexcept { return start + rows; }
};

/**
 * @brief Stable, never-dangling handle to one pinned-table entry.
 *
 * Created by the scan manager when an entry is inserted (gate-on only) and
 * shared into every column_origin minted against the entry. The HANDLE
 * outlives the entry (shared_ptr); the ENTRY pointer it carries is validated
 * by generation:
 *   - unpin / replacing re-pin  -> invalidate(): entry=null, generation=0;
 *   - in-place column merge     -> bump_generation(): pre-merge origins fail
 *     closed (origins never legitimately span a pin call — pin/unpin is
 *     query-lifecycle-serialized).
 * resolve(expected) therefore returns nullptr for ANY origin captured against
 * a pin state that no longer exists, never a dangling pointer.
 *
 * Thread-safety: pin/unpin is serialized against query execution by the
 * engine (the same discipline visit_pinned_entries documents), so readers
 * during a query observe a stable value; the atomics keep concurrent readers
 * (multiple pipeline threads resolving origins) race-free.
 */
class pin_entry_handle {
 public:
  pin_entry_handle(std::string name, pin_generation_t generation)
    : _name(std::move(name)), _generation(generation)
  {
  }

  pin_entry_handle(const pin_entry_handle&)            = delete;
  pin_entry_handle& operator=(const pin_entry_handle&) = delete;

  /// The entry this handle points at iff @p expected matches the live
  /// generation; nullptr otherwise (fails closed).
  [[nodiscard]] scan_manager::pinned_entry const* resolve(pin_generation_t expected) const
  {
    if (expected == 0 || _generation.load(std::memory_order_acquire) != expected) {
      return nullptr;
    }
    return _entry.load(std::memory_order_acquire);
  }

  [[nodiscard]] pin_generation_t generation() const
  {
    return _generation.load(std::memory_order_acquire);
  }

  /// Pin-time name of the entry (diagnostics only — never used for lookup).
  [[nodiscard]] const std::string& name() const noexcept { return _name; }

  // ── scan-manager lifecycle ─────────────────────────────────────────────────

  /// Point the handle at its (map-node-stable) entry. Called once, right after
  /// the entry is installed in _pinned_entries.
  void set_entry(scan_manager::pinned_entry const* entry)
  {
    _entry.store(entry, std::memory_order_release);
  }

  /// Entry destroyed or replaced: all outstanding origins fail closed.
  void invalidate()
  {
    _entry.store(nullptr, std::memory_order_release);
    _generation.store(0, std::memory_order_release);
  }

  /// Entry mutated in place (append-only column merge): origins captured
  /// before the merge fail closed; the entry stays live under a new generation.
  void bump_generation() { _generation.fetch_add(1, std::memory_order_acq_rel); }

 private:
  std::string _name;
  std::atomic<pin_generation_t> _generation{0};
  std::atomic<scan_manager::pinned_entry const*> _entry{nullptr};
};

/**
 * @brief The origin of one (potentially late-materialized) column.
 *
 * { stable pin-entry handle, column position, pin generation }. column_pos is
 * the position into the pinned entry's cache_info.column_ids — which is also
 * the storage position inside device/host chunks and the key
 * cache_info.names[column_pos] for the plain GPU pin's per-column map, so a
 * consumer needs no further translation.
 */
struct column_origin {
  std::shared_ptr<const pin_entry_handle> handle;  ///< empty = no origin
  std::uint32_t column_pos{0};
  pin_generation_t generation{0};

  [[nodiscard]] bool has_origin() const noexcept { return handle != nullptr; }

  /// Generation-checked entry resolution; nullptr when the origin is stale
  /// (re-pin / merge / unpin since capture) or empty.
  [[nodiscard]] scan_manager::pinned_entry const* resolve() const
  {
    return handle ? handle->resolve(generation) : nullptr;
  }
};

/// Which form a @ref row_selection takes.
enum class row_selection_kind : std::uint8_t {
  dense   = 0,  ///< every row of `range` is live, in order
  mask    = 1,  ///< fused wave-1 geometry: packed bit-mask + chunk offsets
  id_list = 2,  ///< explicit batch-local row ids
};

/**
 * @brief The live rows of one batch, relative to its origin span.
 *
 * All device buffers are batch-local over range.rows rows:
 *  - mask: mask_words is uint32 x selection_mask::WordsFor(rows) (full 32-word
 *    strips per 1024-row chunk, tail bits zero) and chunk_offsets is
 *    uint32 x (ChunksFor(rows)+1) — EXACTLY the fused scan-filter wave-1
 *    output (include/codegen/selection/selection.hpp), so when the fused
 *    pipeline ran, survivors are captured, never recomputed.
 *  - id_list: row_ids is int32 x num_ids, ascending, batch-local
 *    (global id = range.start + id). Batch-local int32 always suffices:
 *    batches are bounded by cudf::size_type rows; 64-bit ids only exist as
 *    range.start + local at the consumer's edge.
 *
 * Buffers are stream-ordered RMM allocations shared via shared_ptr (the same
 * batch's selection may be referenced by several deferred columns); callers
 * rebinding a batch to a pipeline stream must apply the same set_stream
 * discipline used for scan_filter_result.
 */
struct row_selection {
  row_selection_kind kind{row_selection_kind::dense};
  row_range range;

  // kind == mask
  std::shared_ptr<rmm::device_buffer> mask_words;
  std::shared_ptr<rmm::device_buffer> chunk_offsets;
  std::int64_t survivor_count{-1};

  // kind == id_list
  std::shared_ptr<rmm::device_buffer> row_ids;
  std::int64_t num_ids{0};

  /// Number of live rows this selection describes.
  [[nodiscard]] std::int64_t live_rows() const noexcept
  {
    switch (kind) {
      case row_selection_kind::dense: return range.rows;
      case row_selection_kind::mask: return survivor_count;
      case row_selection_kind::id_list: return num_ids;
    }
    return 0;
  }

  [[nodiscard]] static row_selection make_dense(row_range r)
  {
    row_selection s;
    s.kind  = row_selection_kind::dense;
    s.range = r;
    return s;
  }
};

/**
 * @brief Origin annotation for one served scan batch (== one pinned chunk).
 *
 * Stamped by the cached databatch provider (gate-on, no MVCC masks, no
 * insert-delta splits — the same invariants the fused pipeline requires) and
 * carried through the split to the scan output. `columns` is shared across
 * every batch of the scan (identical content) and is in MATERIALIZED column
 * order — output columns first, in output order, then pure-filter columns
 * (the materialized-order mapping invariant), so output column j maps to
 * (*columns)[j].
 */
struct scan_batch_origin {
  std::shared_ptr<const std::vector<column_origin>> columns;
  row_range range;             ///< this chunk's global row span
  std::size_t chunk_index{0};  ///< index into the entry's chunk storage
};

}  // namespace sirius::late_mat
