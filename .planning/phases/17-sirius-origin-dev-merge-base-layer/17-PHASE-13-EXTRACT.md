# Phase 13 Stream-Lineage Extraction (MERGE-03 / D-C1)

**Source file (deleted by origin/dev #731):** `src/include/op/scan/sirius_parquet_metadata_scan_operator.hpp`
**Source SHA at extraction time:** `98cdea20691a53a84c03eb2463ffc5d1027fe2df`
**Extracted:** 2026-05-05
**Purpose:** Hold the Phase 13 stream-lineage design intent so Phase 20 SM-03 can re-attach it into the new Scan Manager world.

## Why this file holds Phase 13 intent

Phase 13 (commits 62e0517, 407d574, 833bb72) made `writer_stream` a required ctor argument on `cucascade::gpu_table_representation` and added `record_writer_event`/`get_writer_event` accessors. The producer side of stream lineage in Sirius is the parquet scan path: the metadata scan operator's `execute(stream)` flows the CUDA stream into the paired `sirius_gpu_parquet_scan_operator`, which constructs `gpu_table_representation(table, mem_space, stream)` — that ctor records the writer_event on `stream`. Cucascade's `convert_gpu_to_gpu` later reads that event via `cudaStreamWaitEvent` (verified at `cucascade/src/data/representation_converter.cpp:855` per Phase 16 verification).

When PR #731 deletes this header, the stream→writer_event production path is broken if not re-wired. Phase 20 (SM-03) is the dedicated re-attachment phase; this document captures what needs re-attaching.

## Full file content (232 lines, as of HEAD `98cdea20`)

```cpp
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

// sirius
#include <config.hpp>
#include <expression_executor/gpu_expression_translator_internal.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/scan/sirius_gpu_parquet_scan_operator.hpp>
#include <op/sirius_physical_operator.hpp>
#include <op/sirius_physical_operator_type.hpp>
#include <sirius_config.hpp>

// cucascade
#include <cucascade/data/disk_io_backend.hpp>

// duckdb
#include <duckdb/common/column_index.hpp>
#include <duckdb/common/multi_file/multi_file_data.hpp>
#include <duckdb/common/types.hpp>
#include <duckdb/common/vector.hpp>

// standard library
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace sirius::op::scan {

//===----------------------------------------------------------------------===//
// Parquet metadata scan operator
//===----------------------------------------------------------------------===//
/**
 * @brief Operator that parses parquet file metadata and produces row-group partitions.
 *
 * Pipeline role:
 *   - Both source and sink of the metadata-scan pipeline (pipeline 1), which is a
 *     single-op pipeline containing only this operator.
 *   - get_next_task_input_data() returns parquet_metadata_input (up to
 *     max_file_processed files per task).
 *   - execute() parses parquet footers, builds partitioned_parquet_metadata.
 *   - sink() forwards each produced partitioned_parquet_metadata into the paired
 *     sirius_gpu_parquet_scan_operator via its accumulate_metadata() entry point.
 *     This is a direct handoff — no inter-pipeline port or data repository is
 *     involved; the metadata never flows through the generic pipeline data path.
 *   - Completion of this pipeline is detected by the downstream gpu_scan operator
 *     via the standard "handoff" port / is_pipeline_finished() mechanism — no
 *     explicit finalize callback is needed.
 *
 * @pre The caller must validate before construction that:
 *   - The table function is NOT an in-out function (in_out_function == false).
 *   - There are no dynamic table filters (dynamic_filters == nullptr).
 *   - file_paths is non-empty.
 */
class sirius_parquet_metadata_scan_operator : public sirius_physical_operator {
 public:
  using translated_expression = gpu_expression_translator::translated_expression;

  /// The physical operator type for this operator.
  static constexpr SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::PARQUET_METADATA_SCAN;

  /// Default number of files processed per metadata-scan task.
  static constexpr size_t DEFAULT_MAX_FILE_PROCESSED = 8;

  //===----------Constructor----------===//
  /**
   * @brief Construct the metadata scan operator from the individual fields extracted from the
   *        physical parquet scan node (or equivalent source).
   *
   * @param gpu_scan                The downstream gpu scan operator into which to push partition
   *                                metadata. This is necessary in order to avoid the gpu scan
   *                                serving as sink operator in both the metadatascan pipeline and
   *                                in a subsequent standalone pipeline for which it is the source
   *                                operator.
   * @param types                   Output column types.
   * @param returned_types          The types of all columns in the source file.
   * @param estimated_cardinality   Estimated output row count.
   * @param file_paths              The list of parquet files to scan.
   * @param column_ids              Column ids exposed by the table function (used for column
   *                                selection; see detail::make_selected_column_indices).
   * @param projection_ids          Indices into column_ids that the planner has projected out
   *                                (empty = no projection, read all columns).
   * @param names                   All column names in schema order (used to build column-name
   *                                projections passed to the parquet reader).
   * @param table_filter_set        The table filter set for row-group pruning and filter pushdown
   *                                (optional; may be nullptr if no filters or filter translation
   *                                fails).
   * @param partition_indices       The hive partition indices, if any.
   * @param approximate_batch_size  Target uncompressed bytes per row-group partition.
   * @param max_file_processed      Maximum number of files handled by one metadata task.
   * @param io_backend              Per-GPU cucascade io backend used to construct the
   *                                sirius::io::cucascade_datasource adapters in execute().
   *                                Planning-time call site — caller resolves via
   *                                SiriusContext::get_gpu_io_backends() and passes the first
   *                                GPU's backend (research Pitfall 6 — correctness-neutral).
   *                                If nullptr, execute() throws at the footer-read site; it
   *                                is the caller's responsibility to supply a backend for
   *                                production code paths.
   *
   * @throws if projection_ids is nonempty or filter_expression is non-nullptr but names is empty
   *         (column names are required for both projection and filter pushdown).
   */
  sirius_parquet_metadata_scan_operator(
    sirius_gpu_parquet_scan_operator* gpu_scan,
    duckdb::vector<sirius::logical_type> types,
    duckdb::vector<sirius::logical_type> const& returned_types,
    duckdb::idx_t estimated_cardinality,
    std::vector<std::string> const& file_paths,
    duckdb::vector<duckdb::ColumnIndex> const& column_ids,
    duckdb::vector<duckdb::idx_t> const& projection_ids,
    duckdb::vector<std::string> const& names,
    duckdb::unique_ptr<duckdb::TableFilterSet> table_filter_set            = nullptr,
    duckdb::vector<duckdb::HivePartitioningIndex> const& partition_indices = {},
    std::size_t approximate_batch_size = sirius::config::DEFAULT_SCAN_TASK_BATCH_SIZE,
    std::size_t max_file_processed     = DEFAULT_MAX_FILE_PROCESSED,
    std::shared_ptr<cucascade::idisk_io_backend> io_backend = nullptr);

  //===----------Source interface----------===//
  bool is_source() const override { return true; }

  //===----------Scheduling interface----------===//
  /**
   * @return READY (pointing to itself) while there are unprocessed files,
   *         or nullopt when all files have been dispatched.
   */
  std::optional<task_creation_hint> get_next_task_hint() override;

  /**
   * @brief Returns true once all files have been dispatched to metadata tasks.
   *
   * @return true iff _next_file_idx >= _total_files (all files dispatched).
   * @note Overrides the default port-based check since this is a source operator with
   *       no input ports.
   */
  [[nodiscard]] bool all_ports_empty() override;

  /**
   * @brief Creates a parquet_metadata_input for the next batch of unprocessed files.
   *
   * Atomically advances the file-index counter and returns up to max_file_processed
   * file paths. Returns nullptr when all files have been consumed.
   */
  std::unique_ptr<operator_data> get_next_task_input_data() override;

  //===----------Execution----------===//
  /**
   * @brief Parse parquet metadata for the files in @p input_data and produce
   *        a partitioned_parquet_metadata.
   *
   * If a filter expression was provided at construction, this method attempts to translate it
   * into a cuDF AST for filter pushdown and row-group pruning. If column names were not provided
   * to the constructor, or if AST translation fails, the original DuckDB expression is stored on
   * the result for post-scan filtering in the GPU scan operator.
   *
   * @param input_data  Must be a parquet_metadata_input instance.
   * @param stream      CUDA stream used for AST filter translation and row-group pruning.
   * @return            A partitioned_parquet_metadata containing the parsed FileMetaData
   *                    objects, reader options, and row-group partitions.
   * @throws std::runtime_error if @p input_data is not a parquet_metadata_input.
   */
  std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                         rmm::cuda_stream_view stream) override;

  //===----------Sink interface (forwarding to gpu_scan)----------===//
  bool is_sink() const override { return true; }

  /**
   * @brief Forward a partitioned_parquet_metadata produced by execute() into the paired
   *        sirius_gpu_parquet_scan_operator.
   *
   * Invoked by the pipeline framework once per completed metadata task, from worker threads.
   * Delegates to gpu_scan::accumulate_metadata(), which is thread-safe; no ordering between
   * concurrent sink() calls is required or observed.
   *
   * @param input_data  Must dynamic_cast to partitioned_parquet_metadata.
   * @param stream      Unused; metadata accumulation is CPU-only.
   * @throws std::runtime_error if @p input_data is not a partitioned_parquet_metadata.
   */
  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

  //===----------Accessors----------===//
  [[nodiscard]] std::size_t get_total_files() const { return _total_files; }
  [[nodiscard]] std::size_t get_max_file_processed() const { return _max_file_processed; }
  [[nodiscard]] std::size_t get_approximate_batch_size() const { return _approximate_batch_size; }

 private:
  /// The list of parquet files to scan.
  std::vector<std::string> _file_paths;
  /// Canonical scan plan — data columns (D order), partition columns, output layout,
  /// and C→D filter map. Replaces the scattered bookkeeping that used to live here.
  scan_plan _plan;
  /// The coalesced DuckDB filter expression (AST translation attempted in execute()).
  /// Empty when no filters were translatable (after skipping partition-column filters).
  std::shared_ptr<duckdb::Expression> _duckdb_filter_expression;

  std::size_t _approximate_batch_size;
  std::size_t _max_file_processed;
  std::size_t _total_files;

  /// Per-GPU cucascade io backend used for constructing cucascade_datasource adapters.
  /// Supplied at construction (resolved by the caller via
  /// SiriusContext::get_gpu_io_backends()). May be null in isolated test contexts;
  /// if null, execute() throws at the datasource construction site.
  std::shared_ptr<cucascade::idisk_io_backend> _io_backend;

  /// Atomic file-batch counter; incremented by get_next_task_input_data().
  std::atomic<std::size_t> _next_file_idx{0};

  /// Paired GPU parquet scan operator — the source of the downstream pipeline. sink() forwards
  /// accumulated metadata into it via accumulate_metadata(). Set at construction; never null.
  sirius_gpu_parquet_scan_operator* _gpu_scan;
};

}  // namespace sirius::op::scan
```

## Extracted: stream-carrying methods

```cpp
// From sirius_parquet_metadata_scan_operator.hpp lines 173-180:
/**
 * @param stream      CUDA stream used for AST filter translation and row-group pruning.
 */
std::unique_ptr<operator_data> execute(const operator_data& input_data,
                                       rmm::cuda_stream_view stream) override;

// From lines 192-197:
/**
 * @param stream      Unused; metadata accumulation is CPU-only.
 */
void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;
```

## Extracted: paired-operator forwarding (where the writer_stream concept fans out)

```cpp
// From lines 227-229:
/// Paired GPU parquet scan operator — the source of the downstream pipeline. sink() forwards
/// accumulated metadata into it via accumulate_metadata(). Set at construction; never null.
sirius_gpu_parquet_scan_operator* _gpu_scan;
```

The actual `writer_stream` is recorded inside `_gpu_scan->execute()` when it constructs
`gpu_table_representation`. The metadata operator's `execute(stream)` is the
upstream-most place where the CUDA stream first appears in the Sirius parquet scan pipeline.
Under the new Scan Manager (PR #731), the metadata-scan-as-operator pattern is gone; metadata
reading moves into `parquet_split_provider::run_batch` driven by `sirius_scan_manager`'s
thread pool. The `writer_stream` argument has to thread through that new path.

## Stream-lineage in the new Scan Manager world (from origin/dev `parquet_split_provider.cpp`)

In `src/scan_manager/parquet_split_provider.cpp:184` (origin/dev HEAD), `run_batch` acquires
a stream via:

```cpp
void parquet_split_provider::run_batch(file_batch const& batch, split_connector& connector)
{
  auto stream = cudf::get_default_stream();
  // ...
  // stream is passed to gpu_expression_translator for AST translation (line 206):
  gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
  // ...
  // stream is also passed to filter_row_groups_with_stats (line 270):
  reader.filter_row_groups_with_stats(row_group_indices, *reader_options, stream)
```

Note: `cudf::get_default_stream()` is currently used here. This is the Phase 20 SM-03
re-attachment target — the `writer_stream` (an explicit task-level stream from the Sirius
pipeline executor) must replace `cudf::get_default_stream()` so that the writer_event is
recorded on the correct task-owned stream, enabling `convert_gpu_to_gpu` to issue
`cudaStreamWaitEvent(target_stream.value(), writer_event, 0)` correctly.

## Re-attachment target (for Phase 20 SM-03)

**Primary candidate:** `src/op/scan/sirius_gpu_parquet_scan_operator.cpp` — its `execute()`
is where `gpu_table_representation` is constructed in the post-#731 world (per ARCHITECTURE.md
Surface 5: "PR #731 moves filter translation entirely into `execute()` at task time"). The
`writer_stream` argument is `stream` (already passed to `execute(input_data, stream)`).

**Secondary candidate:** `src/scan_manager/parquet_split_provider.cpp` — if a writer_event
needs to be recorded earlier (e.g., for cached-table footer scans), the split provider's
`run_batch()` is the next-most-upstream stream site. Currently uses `cudf::get_default_stream()`
(line 184); Phase 20 must replace this with a task-level stream passed down from the executor
so that the writer_event is recorded on the right stream.

The re-attachment must ensure that wherever `gpu_table_representation` is constructed in the
post-#731 world, it receives a proper task-level `rmm::cuda_stream_view` — NOT
`cudf::get_default_stream()` — so the writer_event chain from producer to `convert_gpu_to_gpu`
(at `cucascade/src/data/representation_converter.cpp:855`) is intact.

## Phase 20 acceptance check (SM-03)

After Phase 20 completes, this command MUST return non-zero:
```
grep -rn "writer_stream\|record_writer_event" src/op/scan/
```
That grep is the regression gate codified in ROADMAP Phase 20 success criterion 1.

## Phase 13 commits (archaeology)

- `62e0517` (cucascade) — original writer_stream ctor requirement
- `407d574` (sirius)  — migrate all Sirius producers to writer_stream ctor argument
- `833bb72` (sirius)  — migrate all test producers to writer_stream ctor argument

(All three are now embedded in the Phase 16 cucascade pin `1c1e648` and the Sirius source
surface at HEAD.)
