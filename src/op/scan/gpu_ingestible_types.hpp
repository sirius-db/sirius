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

#include "io/cache/types.hpp"
#include "op/scan/owning_table_view.hpp"
#include "op/scan/table_scan/scan_contract.hpp"

#include <expression/ast/node.hpp>
#include <io/sirius_datasource.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#pragma once

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// ingestible_table_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-table bind data carrier; polymorphic factory for gpu_ingestible.
 *
 * Built from a scan binding by the plan generator or by pin_table, then passed
 * to make_ingestible. prepare_for_query reads it back to match pinned entries.
 * Implementations: parquet_ingestible_table_info,
 * duckdb_native_ingestible_table_info.
 */
class ingestible_table_info {
 public:
  scan_contract_id contract_id = 0;
  test_injections injections;
  std::shared_ptr<physical_profile_table> profiles = std::make_shared<physical_profile_table>();

  virtual ~ingestible_table_info() = default;

  ingestible_table_info(ingestible_table_info const&)            = delete;
  ingestible_table_info& operator=(ingestible_table_info const&) = delete;

  [[nodiscard]] virtual std::span<std::string const> column_names() const = 0;

  /**
   * @brief Resolved file paths captured at bind time.
   *
   * Used by sirius_scan_manager to match an incoming scan against pinned
   * entries. Returned span
   * must remain valid for the lifetime of @c *this.
   */
  [[nodiscard]] virtual std::span<std::string const> file_paths() const = 0;

  /// Human-readable identity for diagnostics (logging, error messages). Parquet
  /// tables use the first resolved file path; duckdb-native tables use the
  /// qualified catalog/schema/table name, since @ref file_paths for them is the
  /// shared .db path and would collapse every table in that database to the
  /// same string.
  [[nodiscard]] virtual std::string display_name() const = 0;

 protected:
  ingestible_table_info() = default;
};

//===----------------------------------------------------------------------===//
// scan_info
//===----------------------------------------------------------------------===//
/**
 * @brief Per-split scan descriptor. Polymorphic; each gpu_ingestible
 *        implementation defines its own subclass with the per-split
 *        information its @ref gpu_ingestible::materialize_table requires.
 *
 * Distinct from per-table bind data (@ref ingestible_table_info): one
 * ingestible produces many @c scan_info instances during its lifetime —
 * one per emitted split.
 */
class scan_info : public std::enable_shared_from_this<scan_info> {
 public:
  struct fadvise_entry {
    std::shared_ptr<sirius::io::sirius_datasource> datasource;
    std::vector<cudf::io::text::byte_range_info> ranges;
  };

  scan_info() = default;

  explicit scan_info(std::vector<fadvise_entry> hints) : _hints(std::move(hints))
  {
    _datasources.reserve(_hints.size());
    for (auto const& entry : _hints) {
      if (entry.datasource) { _datasources.push_back(entry.datasource); }
    }
  }

  virtual ~scan_info() = default;

  [[nodiscard]] std::span<const fadvise_entry> fadvise_hints() const { return _hints; }

  [[nodiscard]] std::span<const std::shared_ptr<sirius::io::sirius_datasource>> datasources() const
  {
    return _datasources;
  }

  // ---- readahead -----------------------------------------------------------

  /// How one @ref prefetch call resolved, once every datasource has settled.
  /// A backend refusing the request is not an error -- it is busy -- but it is
  /// worth counting, because a refusal while work is queued means capacity went
  /// unused.
  struct prefetch_outcome {
    std::size_t issued{0};    ///< datasources that started IO
    std::size_t declined{0};  ///< datasources that refused the request
    /// Refused because the cache pool could not attach staging buffers.  The
    /// only refusal reason that has to travel: every other one is a statement
    /// about the consumer, which the readahead manager can read off its own
    /// per-split state at the moment the attempt completes.
    std::size_t declined_memory_pressure{0};
    bool ok{true};  ///< every datasource that started IO completed it
    /// Every requested extent was covered by successful prefetch or resident data.
    bool covered{true};
  };

  /// How preparing this split's prefetch requests turned out, across its
  /// datasources.  A split spanning several files can have some prepared and the
  /// rest refused, so both are counted rather than reduced to a verdict here.
  struct prepare_outcome {
    std::size_t prepared{0};     ///< requests that now own staging buffers
    std::size_t failed{0};       ///< requests the cache pool could not satisfy
    std::size_t fell_behind{0};  ///< requests overtaken by their consumer

    /// Whether every request that needed preparation is ready for the following
    /// prefetch. A partially prepared split must retry its failed datasource;
    /// issuing only the successful subset would silently abandon the rest.
    [[nodiscard]] bool ready() const noexcept
    {
      return prepared > 0 && failed == 0 && fell_behind == 0;
    }
  };

  /// Where the consumer has got to with this split, reported by
  /// @c scan_operator_input::update as it moves through the executor.
  ///
  /// The stage belongs to the split, not to its files: they all advance with it,
  /// so polling N per-file prefetch handles to rediscover one fact only invites
  /// them to disagree.
  ///
  /// Monotone, because stages only ever advance and a report that arrives late
  /// must not walk one back.
  void set_scan_stage(io::cache::scan_stage stage) noexcept
  {
    auto cur = _scan_stage.load(std::memory_order_relaxed);
    while (stage > cur && !_scan_stage.compare_exchange_weak(
                            cur, stage, std::memory_order_release, std::memory_order_relaxed)) {}
  }

  [[nodiscard]] io::cache::scan_stage get_scan_stage() const noexcept
  {
    return _scan_stage.load(std::memory_order_acquire);
  }

  /// The consumer has reached this split -- it is preparing to read it, or
  /// already reading -- so a prefetch started now would only duplicate the IO
  /// the executor is doing for itself.
  [[nodiscard]] bool has_fallen_behind() const noexcept
  {
    return get_scan_stage() >= io::cache::scan_stage::preparing;
  }

  /// Allocate staging buffers for this split's prefetch requests.  A chunk
  /// without one cannot be claimed for loading, so a prefetch issued over it
  /// reads nothing and settles `ready` having done nothing -- and the reader
  /// then pays for the whole split.
  ///
  /// @p wait_for_eviction: see @c prefetching_cache::prepare.
  prepare_outcome prepare_for_prefetching(bool wait_for_eviction = false)
  {
    prepare_outcome out;
    for (auto const& ds : _datasources) {
      switch (ds->prepare_prefetch(wait_for_eviction)) {
        case sirius::io::prepare_result::prepared: ++out.prepared; break;
        case sirius::io::prepare_result::allocation_failed: ++out.failed; break;
        case sirius::io::prepare_result::fallen_behind: ++out.fell_behind; break;
        case sirius::io::prepare_result::nothing_to_prepare: break;
      }
    }
    return out;
  }

  /// Issue prefetch IO for every datasource this split reads.  Reports through
  /// @p on_done exactly once, when the last datasource has settled -- there is
  /// no return value, because the answer is not known when this returns.
  ///
  /// The report is what frees the split's readahead slot, so it must reach the
  /// readahead rather than be waited on here: the caller is the worker thread
  /// that would otherwise be collecting the next batch.  A split with no
  /// datasources reports inline.
  ///
  /// @p on_done may run on this thread (a datasource can complete inline) or on
  /// an IO completion thread, so it must be safe on either and must not block.
  void prefetch(sirius::exec::invocable<void(prefetch_outcome) noexcept> on_done)
  {
    if (_datasources.empty()) {
      // Nothing to read ahead of: an attempt that could never have issued.
      on_done(prefetch_outcome{});
      return;
    }

    // The +1 is a guard held across the loop: prefetch_async may invoke its
    // completion inline, so without it the countdown could reach zero -- and
    // report an outcome still missing its later datasources -- before every one
    // had even been asked.
    auto pending =
      std::make_shared<prefetch_completion>(_datasources.size() + 1, std::move(on_done));
    for (auto const& ds : _datasources) {
      switch (ds->prefetch_async(
        [pending](io::cache::prefetch_result result) noexcept { pending->arrive(result); })) {
        case sirius::io::prefetch_refusal::issued:
          pending->issued.fetch_add(1, std::memory_order_relaxed);
          continue;
        case sirius::io::prefetch_refusal::memory_pressure:
          pending->declined_memory_pressure.fetch_add(1, std::memory_order_relaxed);
          break;
        case sirius::io::prefetch_refusal::consumer_ahead:
        case sirius::io::prefetch_refusal::other:
        case sirius::io::prefetch_refusal::no_cache: break;
      }
      pending->declined.fetch_add(1, std::memory_order_relaxed);
    }
    pending->release_guard();
  }

  [[nodiscard]] virtual std::span<split_materializer_certificate const> certificates() const
  {
    return certificates_;
  }

  [[nodiscard]] virtual std::span<split_dependencies const> dependencies() const
  {
    return dependencies_;
  }

  [[nodiscard]] virtual scan_contract_id contract_id() const { return contract_id_; }

  void set_contract_payload(scan_contract_id contract_id,
                            std::vector<split_materializer_certificate> certificates,
                            std::vector<split_dependencies> dependencies)
  {
    if (certificates.size() != dependencies.size()) {
      throw std::invalid_argument("split certificates and dependencies must be parallel");
    }
    for (auto const& certificate : certificates) {
      if (certificate.contract_id != contract_id) {
        throw std::invalid_argument("split certificate contract does not match split contract");
      }
    }
    contract_id_  = contract_id;
    certificates_ = std::move(certificates);
    dependencies_ = std::move(dependencies);
  }

  /**
   * @brief Estimated decoded bytes for projected data columns before row filtering.
   *
   * Read by @c scan_operator_input::get_estimated_size_in_bytes for the
   * reservation system and execution history. A format may use decoded read
   * columns as a nonzero history basis when no data column is projected.
   * Returns 0 for splits with no a-priori size estimate.
   */
  [[nodiscard]] virtual std::size_t estimated_bytes() const noexcept { return 0; }
  // Additive decoded work retained from preparation until the consumer takes its batch.
  [[nodiscard]] virtual std::size_t preparation_bytes() const noexcept
  {
    return estimated_working_set_bytes();
  }

  /**
   * @brief Estimated decoded column buffers needed to materialize the split.
   *
   * Defaults to the projected-column estimate. Formats that decode additional
   * transient columns, such as parquet pure-filter columns, override this to
   * expose that memory separately from the execution-history basis. Decoder
   * scratch and synthesized columns are not included.
   */
  [[nodiscard]] virtual std::size_t estimated_working_set_bytes() const noexcept
  {
    return estimated_bytes();
  }

 private:
  /// Fan-in for @ref prefetch: counts the per-datasource completions down to a
  /// single report.  Shared by every callback, so it outlives the call that
  /// created it.
  struct prefetch_completion {
    prefetch_completion(std::size_t n, sirius::exec::invocable<void(prefetch_outcome) noexcept> f)
      : remaining(n), on_done(std::move(f))
    {
    }

    void arrive(io::cache::prefetch_result result) noexcept
    {
      if (!result.ok) { ok.store(false, std::memory_order_relaxed); }
      if (!result.covered) { covered.store(false, std::memory_order_relaxed); }
      settle();
    }

    /// Drops the caller's guard once every datasource has been asked.
    void release_guard() noexcept { settle(); }

    std::atomic<std::size_t> remaining;
    std::atomic<std::size_t> issued{0};
    std::atomic<std::size_t> declined{0};
    std::atomic<std::size_t> declined_memory_pressure{0};
    std::atomic<bool> ok{true};
    std::atomic<bool> covered{true};
    sirius::exec::invocable<void(prefetch_outcome) noexcept> on_done;

   private:
    void settle() noexcept
    {
      if (remaining.fetch_sub(1, std::memory_order_acq_rel) != 1) { return; }
      on_done(prefetch_outcome{
        .issued                   = issued.load(std::memory_order_relaxed),
        .declined                 = declined.load(std::memory_order_relaxed),
        .declined_memory_pressure = declined_memory_pressure.load(std::memory_order_relaxed),
        .ok                       = ok.load(std::memory_order_relaxed),
        .covered                  = covered.load(std::memory_order_relaxed)});
    }
  };

  std::vector<fadvise_entry> _hints;
  std::vector<std::shared_ptr<sirius::io::sirius_datasource>> _datasources;
  std::atomic<io::cache::scan_stage> _scan_stage{io::cache::scan_stage::none};

  scan_contract_id contract_id_ = 0;
  std::vector<split_materializer_certificate> certificates_;
  std::vector<split_dependencies> dependencies_;
};

//===----------------------------------------------------------------------===//
// filter_state / filtered_table
//===----------------------------------------------------------------------===//
/**
 * @brief How much of the per-split filter + projection work the ingestible
 *        already absorbed during @ref gpu_ingestible::materialize_table.
 *
 * Returned alongside the materialized table so the scan operator can skip
 * a redundant @ref gpu_ingestible::post_filter_and_project call when the
 * ingestible already applied both the row-level filter and projection
 * inline (the parquet reader-side pushdown path).
 */
enum class filter_state {
  UNFILTERED,                  // pinned table is an example of this
  ROWGROUP_FILTERED,           // hybrid_scan materialize is an example of this
  ROW_FILTERED,                // read_parquet is an example of this
  ROW_FILTERED_AND_PROJECTED,  // table for the particular query is cached
};

/**
 * @brief Result of @ref gpu_ingestible::materialize_table.
 *
 * Bundles the materialized cudf::table with a tag describing how much of
 * the split's filter + projection work was applied during materialization.
 */
struct filtered_table {
  owning_table_view table;
  filter_state state{filter_state::UNFILTERED};
  /// Positions in @c table delivered as a BOOL8 predicate result rather than
  /// values (see sirius::pushdown_outcome::predicate_columns). Their filter
  /// conjunct is already answered, so post_filter_and_project references the
  /// column instead of re-expressing the comparison. Empty on every path that
  /// substitutes nothing.
  std::vector<std::size_t> predicate_columns;
  /// The decode also applied those conjuncts to the rows (see
  /// sirius::pushdown_outcome::predicates_enforced), so post_filter_and_project
  /// drops them from the residual instead of referencing the answer.
  bool predicates_enforced{false};
  /// Bytes the cached batch behind a resident view stores, read through the lease @c table holds;
  /// zero on every other path.
  std::size_t stored_bytes{0};
};

/**
 * @brief Owns the original materialized-column coordinates and residual for a cached checkpoint.
 */
struct cached_view_layout {
  /// Maps each canonical endpoint ordinal to its source column; a layout that cannot map every
  /// ordinal is declined rather than produced. output_columns[endpoint_ordinal] = source_column
  std::vector<cudf::size_type> output_columns;
  /// The source columns to gather after payload elision, in output order and including duplicates.
  /// The residual references the same unmodified source rows.
  std::vector<cudf::size_type> projected_columns;
  /// The residual expression that must still be applied to the projected columns.
  std::unique_ptr<sirius::ast::node> residual;
};

}  // namespace sirius::op::scan
