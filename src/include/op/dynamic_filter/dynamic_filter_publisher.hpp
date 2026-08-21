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

#include "op/dynamic_filter/dynamic_filter_publish_plan.hpp"
#include "op/dynamic_filter/dynamic_filter_stats.hpp"

#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace sirius::op {

class sirius_dynamic_bloom_filter;

/**
 * @brief Structurally validated identity and global row geometry of a complete partitioned build
 *
 * The caller supplies the completeness and pre-scatter meaning of the owned batch IDs. The type
 * validates that the snapshot has more than one partition, a non-empty set of unique IDs, and a
 * representable global row count. Zero build rows are valid. Moving a snapshot invalidates the
 * source object; `valid()` reports whether a snapshot still carries its validated structure.
 */
class complete_build_snapshot final {
 public:
  /**
   * @brief Validates and takes ownership of a complete build snapshot
   *
   * @param[in] total_rows Global build row count
   * @param[in] batch_ids Original pre-scatter batch IDs
   * @param[in] partition_count Number of hash partitions
   * @return The validated snapshot, or `std::nullopt` when the row count is not representable as
   * `std::size_t`, the IDs are empty or non-unique, or `partition_count` is not greater than one
   */
  [[nodiscard]] static std::optional<complete_build_snapshot> try_create(
    std::uint64_t total_rows, std::vector<std::uint64_t> batch_ids, std::size_t partition_count);

  complete_build_snapshot(complete_build_snapshot const&)            = delete;
  complete_build_snapshot& operator=(complete_build_snapshot const&) = delete;
  complete_build_snapshot(complete_build_snapshot&& other) noexcept;
  complete_build_snapshot& operator=(complete_build_snapshot&& other) noexcept;
  ~complete_build_snapshot() = default;

  [[nodiscard]] bool valid() const noexcept { return _partition_count > 1 && !_batch_ids.empty(); }
  [[nodiscard]] std::size_t total_rows() const noexcept { return _total_rows; }
  [[nodiscard]] std::size_t partition_count() const noexcept { return _partition_count; }
  [[nodiscard]] std::span<std::uint64_t const> batch_ids() const noexcept { return _batch_ids; }

 private:
  complete_build_snapshot(std::size_t total_rows,
                          std::vector<std::uint64_t> batch_ids,
                          std::size_t partition_count)
    : _total_rows(total_rows), _batch_ids(std::move(batch_ids)), _partition_count(partition_count)
  {
  }

  std::size_t _total_rows;
  std::vector<std::uint64_t> _batch_ids;
  std::size_t _partition_count;
};

namespace detail {

/**
 * @brief Row contribution of one original build batch
 */
struct complete_build_batch_summary {
  std::uint64_t batch_id;  ///< Original pre-scatter batch ID
  std::uint64_t rows;      ///< Rows in the GPU table representation
};

/**
 * @brief Sums exact batch rows and creates a validated complete build snapshot
 *
 * @param[in] batches Original build batch identities and row counts
 * @param[in] partition_count Number of hash partitions
 * @return The snapshot, or `std::nullopt` when row summation overflows or validation fails
 */
[[nodiscard]] std::optional<complete_build_snapshot> try_summarize_complete_build(
  std::span<complete_build_batch_summary const> batches, std::size_t partition_count);

}  // namespace detail

struct dynamic_filter_publication_outcome {
  std::size_t keys_considered              = 0;
  std::size_t keys_with_known_domain       = 0;
  std::size_t keys_build_exceeded_domain   = 0;
  std::size_t skipped_targets_drained      = 0;
  std::size_t keys_skipped_domain_gate     = 0;
  std::size_t keys_skipped_bloom_size_gate = 0;
  std::size_t keys_skipped_type_mismatch   = 0;
  std::size_t membership_filters_built     = 0;
  std::size_t zone_map_filters_built       = 0;
  std::size_t active_targets               = 0;
  std::size_t filters_pushed               = 0;
};

/**
 * @brief Builds and publishes filters from a complete hash-join build table
 *
 * Replicas are ready before filters reach bound channels. The function retains no inputs; the
 * caller owns source readiness and one-shot arbitration. Type-mismatched keys are skipped.
 *
 * @pre @p plan is enabled
 * @throw std::runtime_error if the source GPU cannot be identified
 * @throw std::logic_error for inconsistent plan or filter metadata
 */
[[nodiscard]] dynamic_filter_publication_outcome publish_dynamic_filters(
  dynamic_filter_publish_plan const& plan,
  cudf::table_view const& build_view,
  rmm::cuda_stream_view stream);

/**
 * @brief Outcome of one exact build-batch contribution
 */
struct dynamic_filter_accumulation_result {
  enum class status : std::uint8_t {
    pending,    ///< Accepted; expected IDs remain
    duplicate,  ///< Ignored after completion or because the expected ID was already accepted
    published,  ///< This call completed publication
    aborted     ///< Publication cannot proceed for this contribution
  };

  status state = status::pending;
  dynamic_filter_publication_outcome publication;  ///< Current outcome counters
  std::size_t exact_contribution_count = 0;        ///< Completed unique IDs at publication
  std::size_t global_build_rows        = 0;        ///< Validated global build geometry
  int root_device_id = -1;  ///< Final contributor selected as reduction/replication source
};

namespace detail {
struct dynamic_filter_accumulator_test_hooks {
  std::function<void(std::uint64_t)> after_id_claim;
  std::function<void(std::uint64_t)> after_insert_sync;
  std::function<void(sirius_dynamic_bloom_filter&, std::span<dynamic_filter_replica_space const>)>
    strict_replicate;  ///< Replaces strict replication at the pre-fan-out boundary
};
}  // namespace detail

/**
 * @brief Accumulates expected pre-scatter batches into one global Bloom snapshot
 *
 * `contribute()` is thread-safe and accepts each expected ID at most once. Nothing is exposed until
 * all expected IDs finish. The retained plan reference must outlive this object.
 */
class dynamic_filter_accumulator final {
 public:
  /**
   * @brief Creates an accumulator for a complete build snapshot
   *
   * @throw std::invalid_argument if @p plan is disabled or @p snapshot is invalid
   *
   * @param[in] plan Enabled plan that outlives this accumulator
   * @param[in] snapshot Validated complete pre-scatter build identity and row geometry
   */
  dynamic_filter_accumulator(dynamic_filter_publish_plan const& plan,
                             complete_build_snapshot snapshot);

  /**
   * @copydoc dynamic_filter_accumulator(dynamic_filter_publish_plan const&,complete_build_snapshot)
   * @param[in] test_hooks Deterministic test hooks
   */
  dynamic_filter_accumulator(dynamic_filter_publish_plan const& plan,
                             complete_build_snapshot snapshot,
                             detail::dynamic_filter_accumulator_test_hooks test_hooks);
  ~dynamic_filter_accumulator();

  dynamic_filter_accumulator(dynamic_filter_accumulator const&)            = delete;
  dynamic_filter_accumulator& operator=(dynamic_filter_accumulator const&) = delete;

  /**
   * @brief Contributes one expected build batch
   *
   * A newly accepted batch is consumed on the current GPU, and @p stream is synchronized before
   * return. Invalid IDs, devices, or columns produce an `aborted` result.
   *
   * @param[in] batch_id Original pre-scatter batch ID
   * @param[in] build_view Batch containing the admitted build-key ordinals
   * @param[in] stream Stream used for insertion
   * @return Current accumulation state and publication counters
   */
  [[nodiscard]] dynamic_filter_accumulation_result contribute(std::uint64_t batch_id,
                                                              cudf::table_view const& build_view,
                                                              rmm::cuda_stream_view stream);

  /**
   * @brief Atomically aborts an incomplete accumulator
   *
   * @return The outcome when this call performs the abort, otherwise `std::nullopt`
   */
  [[nodiscard]] std::optional<dynamic_filter_publication_outcome> abort_if_incomplete() noexcept;

  /**
   * @brief Resolves the accumulator to its current terminal result, aborting it if incomplete
   *
   * @return `published` with the complete outcome, or `aborted` with the failure outcome
   */
  [[nodiscard]] dynamic_filter_accumulation_result abort_or_get_terminal() noexcept;

  [[nodiscard]] bool complete() const noexcept;
  [[nodiscard]] bool aborted() const noexcept;

 private:
  struct impl;
  std::unique_ptr<impl> _impl;
};

namespace detail {

struct dynamic_filter_publication_session_test_hooks {
  dynamic_filter_accumulator_test_hooks accumulator;
  std::function<void(dynamic_filter_accumulation_result::status)> after_accumulation_result;
  std::function<void(std::size_t)> before_one_shot_key;
};

}  // namespace detail

/**
 * @brief Coordinates one hash join's one-shot or accumulated dynamic-filter publication
 *
 * The immutable `dynamic_filter_publish_plan` must outlive the session. The statistics sink may be
 * null; a non-null sink must also outlive the session.
 *
 * Construction increments `dynamic_filter_stats::producers_enabled` when the plan is enabled and a
 * sink is present.
 *
 * Calls are thread-safe. Publication work runs without holding the session mutex. Each claimed
 * attempt that produces a `dynamic_filter_publication_outcome` folds it exactly once; accumulator
 * construction failure records failure without an outcome. A terminal transition releases the
 * session-owned accumulator; in-flight calls retain shared ownership until they return.
 */
class dynamic_filter_publication_session final {
 public:
  /**
   * @brief Creates a publication session
   *
   * @param[in] plan Immutable publication plan
   * @param[in] stats Optional non-owning statistics sink
   * @param[in] enable_multi_partition Whether exact-ID accumulation may claim the session
   */
  dynamic_filter_publication_session(dynamic_filter_publish_plan const& plan,
                                     dynamic_filter_stats* stats,
                                     bool enable_multi_partition);
  dynamic_filter_publication_session(
    dynamic_filter_publish_plan const& plan,
    dynamic_filter_stats* stats,
    bool enable_multi_partition,
    detail::dynamic_filter_publication_session_test_hooks test_hooks);

  [[nodiscard]] bool enabled() const noexcept { return _plan.enabled(); }
  [[nodiscard]] bool wants_multi_partition() const noexcept
  {
    return _enable_multi_partition && enabled();
  }
  [[nodiscard]] bool is_open() const noexcept;

  /**
   * @brief Installs an exact-ID accumulator if the publication window is still open
   *
   * @param[in] snapshot Complete build snapshot consumed by the accumulator
   * @return True when the accumulator was installed; false when multi-partition publication is
   * disabled, the snapshot is invalid, the session is no longer open, or construction fails.
   * Every call that finds the session open begins an attempt; only construction failure records
   * a failure.
   */
  [[nodiscard]] bool try_arm(complete_build_snapshot snapshot);

  /**
   * @brief Contributes one original build batch to an armed accumulator
   *
   * Invalid contributions fail the optional publication without failing query execution.
   *
   * @param[in] join_operator_id Stable producing hash-join identity for terminal observability
   * @param[in] batch_id Original pre-scatter batch ID
   * @param[in] build_view Batch containing the admitted build-key ordinals
   * @param[in] stream Stream used for insertion
   */
  void contribute(std::uint64_t join_operator_id,
                  std::uint64_t batch_id,
                  cudf::table_view const& build_view,
                  rmm::cuda_stream_view stream) noexcept;

  /**
   * @brief Claims and performs one-shot publication from a complete build table
   *
   * Does nothing after another path claims or closes the session. Device memory exhaustion marks
   * the session failed without rethrowing; other publication exceptions mark it failed and are
   * rethrown. Exceptional stream and input-lifetime behavior follows `publish_dynamic_filters()`.
   *
   * @param[in] complete_build Complete build table
   * @param[in] stream Durable construction stream
   */
  void publish_one_shot(cudf::table_view const& complete_build, rmm::cuda_stream_view stream);

  /**
   * @brief Closes an open session or aborts an incomplete accumulator
   */
  void finalize_or_abort() noexcept;

  /**
   * @brief Records one source-not-resident delivery without claiming the session
   *
   * Does nothing after the session leaves OPEN or when the statistics sink is null.
   */
  void record_source_not_resident() noexcept;

  /**
   * @brief Records one build-not-whole delivery without claiming the session
   *
   * Does nothing after the session leaves OPEN or when the statistics sink is null. The caller owns
   * once-per-join latching.
   */
  void record_build_not_whole() noexcept;

 private:
  enum class state : std::uint8_t { open, accumulating, publishing, finished, failed, closed };

  void commit_terminal_locked(state terminal,
                              dynamic_filter_publication_outcome const& outcome) noexcept;
  void commit_accumulation_terminal_locked(state terminal,
                                           dynamic_filter_accumulation_result const& result,
                                           std::uint64_t join_operator_id) noexcept;

  dynamic_filter_publish_plan const& _plan;
  dynamic_filter_stats* _stats;
  bool _enable_multi_partition;
  detail::dynamic_filter_publication_session_test_hooks _test_hooks;
  std::shared_ptr<dynamic_filter_accumulator> _accumulator;
  mutable std::mutex _mutex;
  state _state{state::open};
};

}  // namespace sirius::op
