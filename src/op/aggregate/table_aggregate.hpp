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

#include "cuda/agg_table/agg_table.hpp"
#include "op/aggregate/table_aggregate_pending_state.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace sirius::op {

/// What the local grouped aggregate asks of the table, in the engine's terms: the one key column
/// and the cuDF aggregates (AVG already decomposed into SUM and COUNT_VALID), each over one input
/// column.
struct table_aggregate_plan {
  int key_column = -1;
  cudf::data_type key_type{cudf::type_id::EMPTY};
  std::vector<cudf::aggregation::Kind> aggregates;
  std::vector<int> aggregate_columns;                  ///< Parallel to `aggregates`
  std::vector<cudf::data_type> aggregate_input_types;  ///< Parallel to `aggregates`
  /// Parallel to `aggregates`: plan-time bound on |unscaled value| of a SUM input, when proven.
  std::vector<std::optional<std::uint64_t>> input_max_abs;
  /// Parallel to `aggregates`: the SUM's partial may be INT64 or DECIMAL128 with scale 0.
  std::vector<bool> is_wide_integer_sum;
  std::optional<std::uint64_t> rows_hint;  ///< Estimated input rows, when the planner has one
};

struct table_aggregate_options {
  /// Share of one GPU's memory space a table may hold. The state is not evictable, so the
  /// ceiling is low (integration plan, section 3).
  double memory_fraction = 0.25;
  /// Cap on one table's bytes below that share; 0 = no cap.
  std::size_t max_budget_bytes = 0;
  /// Test hook: this many folds throw `rmm::out_of_memory` after the table took the batch and
  /// before the task has recorded the result, so that the task is replayed. The injections
  /// alternate between the two places a fold can fail after it changed the table: inside the
  /// table, between "mutated" and "done" (`agg_table::table::test_fail_after_mutation`), and
  /// after the table returned the result. The last injection is of the first kind.
  std::uint64_t inject_fold_oom = 0;
  /// Target uncompressed size of one flushed batch of partial rows.
  std::size_t flush_chunk_bytes = std::size_t{512} << 20;
};

/**
 * @brief What the table aggregations of this process have done so far, summed over all queries.
 *
 * Counters only grow. A test takes one snapshot before a query and one after it; the difference
 * says whether that query used a table at all, how much of its input the tables took and on how
 * many GPUs (`gpus_folded` of the difference).
 */
struct table_aggregate_stats {
  static constexpr std::size_t max_gpus = 64;

  std::uint64_t folded_batches = 0;  ///< Batches a table folded (each batch once)
  std::uint64_t folded_rows    = 0;  ///< Input rows the tables consumed
  std::uint64_t returned_rows  = 0;  ///< Rows of folded batches sent on to the cuDF path
  /// Partial rows a full table handed back for rows it consumed (`fold_result::unapplied_partial`)
  std::uint64_t returned_partial_rows = 0;
  std::uint64_t legacy_batches        = 0;  ///< Batches of a table aggregate that skipped the table
  std::uint64_t replayed_folds        = 0;  ///< Folds that failed after they changed the table
  std::uint64_t boundary_flushes      = 0;  ///< Tables flushed by a task at a batch boundary
  std::uint64_t end_of_input_flushes  = 0;  ///< Tables flushed by the consumer at end of input
  std::uint64_t flushed_rows          = 0;  ///< Partial rows that left the tables
  /// `folded_batches` by GPU id; ids of `max_gpus` and above share the last slot.
  std::array<std::uint64_t, max_gpus> folded_batches_by_gpu{};

  /// Number of GPUs with at least one folded batch.
  [[nodiscard]] std::size_t gpus_folded() const noexcept;
  /// Counter-wise difference: `*this` taken after @p before.
  [[nodiscard]] table_aggregate_stats since(table_aggregate_stats const& before) const noexcept;
};

/// Snapshot of the process-wide table aggregation counters. Thread-safe.
[[nodiscard]] table_aggregate_stats table_aggregate_process_stats() noexcept;

/**
 * @brief The tables of one grouped aggregate: one `agg_table::table` per GPU.
 *
 * The table sits inside the local (pre-merge) aggregate. Every task folds its batch into the
 * table of the GPU it runs on; the rows the table does not take go through the cuDF local
 * aggregation as before. Whatever a table holds leaves it as partial rows in the local
 * aggregate's own output schema and enters the PARTITION and merge stages like any other local
 * output, so a group that is split between two GPUs' tables, or between a table and the cuDF
 * path, is combined by the merge. Tables are never read across GPUs.
 *
 * A table is flushed
 * - by a task, at a batch boundary, when the GPU's memory space asks for memory back or the table
 *   has stopped absorbing rows: the partial rows are part of that task's output;
 * - at end of input, by a task of the consumer (`pending_state_source`), one per GPU.
 *
 * Thread safety: `aggregate_batch` may run concurrently for different batches, on any GPUs. Folds
 * into one table share a lock; creating, flushing and destroying it are exclusive.
 */
class table_aggregate_state final : public pending_state_source {
 public:
  /// Aggregates rows on the cuDF path and returns the local aggregate's output batch.
  using legacy_aggregate =
    std::function<std::shared_ptr<cucascade::data_batch>(cudf::table_view const&)>;

  /**
   * @brief Builds the state when the table supports @p plan; nullptr otherwise.
   *
   * @param why_not If not null, receives the reason for a nullptr
   */
  [[nodiscard]] static std::shared_ptr<table_aggregate_state> try_create(
    table_aggregate_plan plan, table_aggregate_options options, std::string* why_not = nullptr);

  ~table_aggregate_state() override;
  table_aggregate_state(table_aggregate_state const&)            = delete;
  table_aggregate_state& operator=(table_aggregate_state const&) = delete;

  /**
   * @brief Aggregates one input batch and appends its output batches to @p results.
   *
   * @p batch_id identifies the batch across replays of its task: a replay after
   * `rmm::out_of_memory` or a retryable launch failure passes the same id and never applies a row
   * to a table twice. Call `complete_batches` once the task's `execute` has returned.
   *
   * @param input The batch, on the GPU of @p space
   * @param batch_id `pipelineable_operator_data::original_batch_ids()` entry of the batch
   * @param stream The task's stream
   * @param space The GPU memory space of the task
   * @param telemetry_info The operator's `batch_telemetry()`
   * @param legacy The cuDF local aggregation
   * @param[out] results Receives the batches to publish
   * @throws rmm::out_of_memory if device memory runs out; the task is then replayed
   * @throws std::runtime_error if a batch that was partly folded on one GPU is replayed on another
   */
  void aggregate_batch(cudf::table_view const& input,
                       std::uint64_t batch_id,
                       ::cuda::stream_ref stream,
                       cucascade::memory::memory_space& space,
                       const telemetry::batch_telemetry_info& telemetry_info,
                       legacy_aggregate const& legacy,
                       std::vector<std::shared_ptr<cucascade::data_batch>>& results);

  /// Forgets the replay records of @p batch_ids. Call after `execute` has returned normally.
  void complete_batches(std::span<const std::uint64_t> batch_ids) noexcept;

  /// Sends every later batch down the cuDF path. Tables keep what they hold until flushed.
  void disable(std::string_view reason) noexcept;

  /// Whether new batches still go to the tables.
  [[nodiscard]] bool enabled() const noexcept { return !_disabled.load(std::memory_order_acquire); }

  /// Frees the tables that hold no rows. Called when the aggregate's pipeline has finished.
  void release_idle_tables() noexcept;

  // pending_state_source
  [[nodiscard]] bool has_unclaimed() const noexcept override;
  [[nodiscard]] totals unclaimed_totals() const override;
  [[nodiscard]] std::unique_ptr<operator_data> claim() override;
  [[nodiscard]] std::vector<std::shared_ptr<cucascade::data_batch>> emit(
    const operator_data& unit,
    ::cuda::stream_ref stream,
    const telemetry::batch_telemetry_info& telemetry_info) override;

 private:
  struct device_state;
  struct fold_entry;

  table_aggregate_state(table_aggregate_plan plan,
                        table_aggregate_options options,
                        agg_table::config config,
                        std::vector<int> value_columns,
                        std::vector<agg_table::column_bound> bounds);

  [[nodiscard]] std::shared_ptr<fold_entry> find_or_create_entry(std::uint64_t batch_id);
  [[nodiscard]] device_state& device(cucascade::memory::memory_space& space);
  [[nodiscard]] device_state* find_device(int device_id) const;

  /// Learns the local aggregate's partial schema from the cuDF path and checks that the table's
  /// partial rows can be merged with it. Disables the table path when they cannot.
  void check_partial_schema(cudf::table_view const& input,
                            legacy_aggregate const& legacy,
                            ::cuda::stream_ref stream);
  [[nodiscard]] bool batch_matches_plan(cudf::table_view const& input) const;
  [[nodiscard]] bool admit(cudf::table_view const& input,
                           device_state& dev,
                           fold_entry& entry,
                           ::cuda::stream_ref stream,
                           const telemetry::batch_telemetry_info& telemetry_info);
  [[nodiscard]] bool ensure_table(device_state& dev);
  [[nodiscard]] bool fold(cudf::table_view const& input,
                          device_state& dev,
                          fold_entry& entry,
                          ::cuda::stream_ref stream,
                          const telemetry::batch_telemetry_info& telemetry_info);
  /// Takes one injected fold failure: its 1-based number counted from the last one, 0 when none
  /// is left.
  [[nodiscard]] std::uint64_t take_injected_fold_oom() noexcept;
  void after_fold_failure(device_state& dev, fold_entry& entry, bool out_of_memory) noexcept;
  void adopt_on_device(fold_entry& entry,
                       int device_id,
                       ::cuda::stream_ref stream,
                       rmm::device_async_resource_ref mr);

  /// With `dev.table_mutex` held exclusively: empties the table into `dev.flushed_tables` and
  /// frees it together with its budget.
  void flush_locked(device_state& dev, ::cuda::stream_ref stream);
  /// With `dev.table_mutex` held exclusively: turns `dev.flushed_tables` into batches in the
  /// local aggregate's schema, appended to @p out.
  void drain_flushed_locked(device_state& dev,
                            ::cuda::stream_ref stream,
                            const telemetry::batch_telemetry_info& telemetry_info,
                            std::vector<std::shared_ptr<cucascade::data_batch>>& out);
  /// Partial rows of the table in the local aggregate's column types: a new table when a column
  /// had to be cast, nullptr when @p partial already conforms. @p partial is consumed only when a
  /// table is returned.
  [[nodiscard]] std::unique_ptr<cudf::table> conform_partial(cudf::table& partial,
                                                             ::cuda::stream_ref stream,
                                                             rmm::device_async_resource_ref mr);

  table_aggregate_plan _plan;
  table_aggregate_options _options;
  agg_table::config _config;
  std::vector<int> _value_columns;  ///< Input columns of the fold's `values` view, in order
  std::vector<agg_table::column_bound> _bounds;  ///< Parallel to `_value_columns`
  std::vector<cudf::data_type> _table_partial_schema;
  std::size_t _partial_row_bytes = 0;
  std::size_t _flush_chunk_rows  = 0;

  std::atomic<bool> _disabled{false};

  /// Guards `_devices`, `_entries` and the schema check.
  mutable std::mutex _mutex;
  std::map<int, std::unique_ptr<device_state>> _devices;
  std::unordered_map<std::uint64_t, std::shared_ptr<fold_entry>> _entries;
  bool _schema_checked = false;
  /// The local aggregate's partial schema, as its cuDF path produces it. Written once, under
  /// `_mutex`, before the first fold; read without the lock afterwards.
  std::vector<cudf::data_type> _engine_partial_schema;
  std::atomic<std::uint64_t> _inject_fold_oom{0};

  /// The sorted-keys check runs on every batch until it has failed this many times in a row
  /// without ever succeeding; see `admit`.
  static constexpr std::uint32_t sorted_check_patience = 8;
  std::atomic<std::uint32_t> _sorted_checks{0};
  std::atomic<std::uint32_t> _sorted_hits{0};
};

}  // namespace sirius::op
