/*
 * Copyright 2025, Sirius Contributors.
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

#include "op/groupby_surrogate_deferral.hpp"

#include <cudf/types.hpp>

#include <cucascade/data/data_batch.hpp>

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <vector>

namespace sirius::op {

/// @brief Retention store shared between the deferral join and the group-by merge (see
/// `op/groupby_surrogate_deferral.hpp` for the surrogate-key group-by overview).
///
/// Thread-safe: every public operation takes the store's single mutex; entry lookups are linear
/// scans, which beat map overhead at the small entry counts involved (one entry per
/// deferral-join source batch).
///
/// Address-space invariants (relied on by the merge's finalize gather):
///  - each SOURCE BATCH occupies exactly one contiguous range [base, base + rows), assigned once
///    per batch id (`reserve` is idempotent per (side, id), so task retries and BUILD_PROBE
///    probe tasks sharing one build table reuse the same range and emit identical rowids for the
///    same row);
///  - ranges are contiguous and entries are kept in base order, so concatenating the committed
///    sources in entry order reproduces the absolute rowid address space exactly.
///
/// Retention protocol (OOM-downgrade friendly): `reserve` takes NO accessor -- a task that fails
/// after reserving leaves the source batch downgradable and only names an address range (reused
/// on retry via the id dedupe). The pinning `read_only_data_batch` accessor is attached by the
/// reservation token's `commit` only after the task's output was produced successfully.
/// `snapshot` therefore requires every reserved range to be committed: an uncommitted range can
/// only belong to a batch whose task never succeeded, in which case the query has already failed
/// before any merge finalize could run.
///
/// Ownership and lifetime: the planner pass creates one store per rewritten group-by; it is
/// co-owned by the join's `surrogate_emit_plan` and the aggregate/merge's
/// `surrogate_restore_plan` and dies with the physical plan. The transparent optimizer's probe
/// plan gets its own store, discarded with the probe plan. Destruction is the RAII backstop for
/// `release`: a failed query drops the retained accessors at plan teardown without any explicit
/// call.
class surrogate_deferral_store {
 public:
  /// One committed source batch: its address range plus the read-only accessor that OWNS a pin
  /// on the batch's representation for as long as the snapshot element lives.
  struct retained_source {
    std::int64_t base;     ///< first absolute rowid of this source
    cudf::size_type rows;  ///< rows in this source (range is [base, base + rows))
    ::cucascade::read_only_data_batch batch;  ///< pinning accessor onto the source batch
  };

  /// Observability counts returned by `release`.
  struct release_stats {
    std::size_t sources;  ///< retained accessors dropped by this call
    std::size_t bytes;    ///< bytes those accessors were pinning
  };

  /// Move-only token naming one reserved range. Dropping it without commit is legal and leaves
  /// the range burned-but-unreferenced (the OOM-retry path relies on this: the source batch
  /// stays downgradable and a retried task reuses the same range via the id dedupe).
  class reservation {
   public:
    reservation(reservation&&) noexcept            = default;
    reservation& operator=(reservation&&) noexcept = default;
    reservation(reservation const&)                = delete;
    reservation& operator=(reservation const&)     = delete;
    ~reservation()                                 = default;

    /// First absolute rowid of the reserved range.
    [[nodiscard]] std::int64_t base() const noexcept { return _base; }

    /// Attach the retaining accessor. Call only after the task's output was produced.
    /// Idempotent per batch id (first commit wins). Throws sirius::internal_exception when
    /// `batch.get_batch_id()` differs from the reserved id.
    void commit(::cucascade::read_only_data_batch batch) &&;

   private:
    friend class surrogate_deferral_store;
    reservation(surrogate_deferral_store& store,
                join_side side,
                std::uint64_t batch_id,
                std::int64_t base) noexcept
      : _store{&store}, _side{side}, _batch_id{batch_id}, _base{base}
    {
    }

    surrogate_deferral_store* _store;
    join_side _side;
    std::uint64_t _batch_id;
    std::int64_t _base;
  };

  /// Reserve (or look up) the address range for `batch_id` on one side. Idempotent per
  /// (side, id): a retried task or a BUILD_PROBE build table shared by N probe tasks reuses ONE
  /// range and emits identical rowids for the same row. Overflow of int32 row addressing is
  /// checked BEFORE any state mutates (strong guarantee). Throws sirius::internal_exception
  /// when `rows` disagrees with the existing reservation; throws std::runtime_error
  /// (user-actionable, names the setting to disable) on overflow.
  [[nodiscard]] reservation reserve(join_side side, std::uint64_t batch_id, cudf::size_type rows);

  /// One side's committed sources in base order. Throws sirius::internal_exception if any
  /// reserved range is uncommitted -- unreachable in a successfully-running query (see the
  /// retention protocol in the class comment).
  [[nodiscard]] std::vector<retained_source> snapshot(join_side side) const;

  /// Drop every retained accessor; idempotent. Early-release optimization invoked from the
  /// merge's `on_finalize_operator`; destruction is the RAII backstop (see the class comment).
  /// Not declared noexcept: the byte accounting goes through cucascade accessor interfaces
  /// (`get_data`, `get_size_in_bytes`) that make no noexcept guarantee.
  release_stats release();

 private:
  struct entry {
    std::uint64_t batch_id;
    std::int64_t base;
    cudf::size_type rows;
    std::optional<::cucascade::read_only_data_batch> batch;  ///< set by commit only
  };
  struct side_state {
    std::vector<entry> entries;  ///< base-ordered by construction
    std::int64_t next_base = 0;
  };

  /// Reservation-token backend: validates the id and attaches the accessor (first commit wins).
  void commit(join_side side, std::uint64_t batch_id, ::cucascade::read_only_data_batch batch);

  [[nodiscard]] static entry* find_entry(side_state& state, std::uint64_t batch_id);
  [[nodiscard]] side_state& state_for(join_side side) noexcept
  {
    return side == join_side::left ? _left : _right;
  }
  [[nodiscard]] side_state const& state_for(join_side side) const noexcept
  {
    return side == join_side::left ? _left : _right;
  }

  mutable std::mutex _mutex;
  side_state _left;
  side_state _right;
};

}  // namespace sirius::op
