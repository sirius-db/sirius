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
// A deferred column is described by WHERE it came from — a pinned entry and a
// column position (@ref column_origin) — and WHICH of its rows are still live
// (@ref row_selection). Everything here is metadata: no device work, and with
// the gate off none of these types is ever instantiated.
//
// GLOBAL ROW ADDRESSING. A row's global id is its position in pinned-table
// order, i.e. in the concatenation of the entry's chunks in emission order.
// One served scan batch is one pinned chunk, so it covers the contiguous span
// [range.start, range.start + range.rows), and the decode's coordinates are
// batch-local: local = gid - range.start, then chunk = local / 1024 and
// position = local % 1024 — exactly the geometry
// sirius::codegen::SELECTION_CHUNK_ROWS describes, because every chunk's
// compressed table numbers its own 1024-row chunks from its own row 0.
//
// There is deliberately NO table-global chunk*1024 + position encoding. Batch
// row starts are not multiples of 1024, so such an id would not decompose into
// the batch-local coordinates any decode actually indexes by, and the error
// would be a silent off-by-a-chunk rather than a failure.
//
// The id widths follow from that: a global id addresses a whole pinned entry
// (lineitem at sf1000 is 6.0e9 rows, past 32 bits), while a batch-local id is
// bounded by one batch and fits int32. So 64-bit ids exist only as
// range.start + local, at the consumer's edge —
// codegen/selection/row_id_space.hpp is where they are narrowed back.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rmm {
class device_buffer;
}

namespace sirius::scan_manager {
struct pinned_entry;  // scan_manager/sirius_scan_manager.hpp
}  // namespace sirius::scan_manager

namespace sirius::late_mat {

/// Env gate for late materialization: set and not "0" means on.
///
/// This is the ONLY reader, and every translation unit must come through it.
/// The fused scan-filter gate ended up with two readers that disagreed in a
/// corner, and a gate whose on/off semantics can fork is worse than no gate.
inline bool late_mat_enabled()
{
  static bool const enabled = []() {
    char const* v = std::getenv("SIRIUS_EXP_LATE_MAT");
    return v != nullptr && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
  }();
  return enabled;
}

/// Pin generation. 0 is never a live generation — it is the invalidated, "no
/// origin" value — so a zero-initialized column_origin fails closed.
using pin_generation_t = std::uint64_t;

/// A contiguous span of rows in pinned-table order.
struct row_range {
  std::int64_t start{0};  ///< first global row id covered
  std::int64_t rows{0};   ///< number of rows covered

  [[nodiscard]] std::int64_t end() const noexcept { return start + rows; }

  [[nodiscard]] bool contains(std::int64_t global_id) const noexcept
  {
    return global_id >= start && global_id < end();
  }
};

/**
 * @brief A generation-checked handle to a pinned entry.
 *
 * A deferred column outlives the operator that deferred it, so it holds a
 * reference to pinned data that the scan manager may meanwhile have unpinned,
 * replaced or merged. Holding the pointer would make that a use-after-free
 * discovered as wrong data; holding a name would make it a lookup that quietly
 * finds a DIFFERENT entry. So the handle carries a generation, and every
 * lifecycle event moves it:
 *
 *   - unpin, or a re-pin that replaces the entry -> invalidate(): the entry
 *     goes null and the generation goes to 0, which no origin can match;
 *   - an in-place column merge -> bump_generation(): origins captured before
 *     the merge fail closed, since an origin never legitimately spans a pin
 *     call (pin/unpin is serialized against query execution).
 *
 * resolve(expected) therefore yields nullptr for any origin captured against a
 * pin state that no longer exists — never a dangling pointer. Failing closed
 * costs a re-read; failing open costs a wrong answer.
 *
 * Thread-safety: pin/unpin is serialized against query execution by the
 * engine, so readers during a query see a stable value; the atomics are what
 * keep several pipeline threads resolving origins concurrently race-free.
 */
class pin_entry_handle {
 public:
  pin_entry_handle(std::string name, pin_generation_t generation)
    : _name(std::move(name)), _generation(generation)
  {
  }

  pin_entry_handle(pin_entry_handle const&)            = delete;
  pin_entry_handle& operator=(pin_entry_handle const&) = delete;

  /// The entry this handle points at, iff @p expected is still the live
  /// generation; nullptr otherwise.
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

  /// Pin-time name of the entry. Diagnostics only — resolution never goes
  /// through a name, which is the whole point of the handle.
  [[nodiscard]] std::string const& name() const noexcept { return _name; }

  // ── scan-manager lifecycle ────────────────────────────────────────────────

  /// Point the handle at its entry. Called once, after the entry is installed.
  void set_entry(scan_manager::pinned_entry const* entry)
  {
    _entry.store(entry, std::memory_order_release);
  }

  /// The entry was destroyed or replaced: every outstanding origin fails closed.
  void invalidate()
  {
    _entry.store(nullptr, std::memory_order_release);
    _generation.store(0, std::memory_order_release);
  }

  /// The entry changed in place: origins captured before now fail closed.
  void bump_generation(pin_generation_t next)
  {
    _generation.store(next, std::memory_order_release);
  }

 private:
  std::string _name;
  std::atomic<scan_manager::pinned_entry const*> _entry{nullptr};
  std::atomic<pin_generation_t> _generation{0};
};

/**
 * @brief Where a deferred column's values live: a pinned entry and a position.
 *
 * `column_pos` indexes the entry's cache_info.column_ids, which is also the
 * storage position within each chunk and the key into the per-column map, so a
 * consumer needs no further translation.
 */
struct column_origin {
  std::shared_ptr<pin_entry_handle const> handle;  ///< empty = no origin
  std::uint32_t column_pos{0};
  pin_generation_t generation{0};

  [[nodiscard]] bool has_origin() const noexcept { return handle != nullptr; }

  /// Generation-checked resolution; nullptr when the origin is stale or empty.
  [[nodiscard]] scan_manager::pinned_entry const* resolve() const
  {
    return handle ? handle->resolve(generation) : nullptr;
  }
};

/// Which form a @ref row_selection takes.
enum class row_selection_kind : std::uint8_t {
  dense   = 0,  ///< every row of `range` is live, in order
  mask    = 1,  ///< the fused wave-1 geometry: packed bits + chunk offsets
  id_list = 2,  ///< explicit batch-local row ids
};

/**
 * @brief The live rows of one batch, relative to its origin span.
 *
 * Three forms because the selection arrives in whichever shape its producer
 * already had, and converting costs a pass:
 *  - mask: `mask_words` is uint32 x selection_mask::WordsFor(rows) (full
 *    32-word strips per chunk, tail bits zero) and `chunk_offsets` is
 *    uint32 x (ChunksFor(rows)+1) — EXACTLY the fused scan-filter's wave-1
 *    output, so when that pipeline ran its survivors are captured rather than
 *    recomputed.
 *  - id_list: `row_ids` is int32 x num_ids, ascending and batch-local, so a
 *    global id is range.start + id.
 *
 * The buffers are shared rather than owned because one batch's selection may
 * be referenced by several deferred columns; a caller rebinding a batch to a
 * pipeline stream owes them the same set_stream discipline as any other
 * stream-ordered allocation.
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

  /// How many rows this selection describes.
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
 * @brief Origin annotation for one served scan batch (one pinned chunk).
 *
 * `columns` is shared across every batch of a scan — the content is identical
 * — and is in materialized column order: output columns first, in output
 * order, then the columns only a filter reads. So output column j is
 * (*columns)[j], with no mapping table to keep in step.
 */
struct scan_batch_origin {
  std::shared_ptr<std::vector<column_origin> const> columns;
  row_range range;             ///< this chunk's global row span
  std::size_t chunk_index{0};  ///< index into the entry's chunk storage
};

}  // namespace sirius::late_mat
