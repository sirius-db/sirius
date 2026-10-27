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
#include "op/scan/parquet_scan_types.hpp"
namespace sirius::op::scan {
class parquet_scan_core {
 public:
  using metadata_scan_task_t           = gpu_ingestible::metadata_scan_task_t;
  using stoppable_metadata_scan_task_t = gpu_ingestible::stoppable_metadata_scan_task_t;
  void set_execution_completion(std::shared_ptr<pipeline::completion_handler> c)
  {
    _execution_completion = std::move(c);
  }
  /// Owns the Parquet bind data for the lifetime of the composing ingestible.
  /// @param device_prefetch_allowed Whether this scan's splits may be read ahead onto a GPU by the
  ///        readahead's device tier and decoded from there; stamped on every split as
  ///        parquet_split_info::device_prefetch_allowed.
  explicit parquet_scan_core(std::unique_ptr<parquet_ingestible_table_info> info,
                             bool device_prefetch_allowed = true);

  ~parquet_scan_core();

  std::unique_ptr<batch_coalescer> create_batch_coalescer() const;

  [[nodiscard]] bool device_prefetch_allowed() const noexcept { return _device_prefetch_allowed; }

  [[nodiscard]] bool has_processed_all_metadata() const;

  metadata_scan_task_t next_split_provider(io::ioctx_resolver resolve);
  metadata_scan_task_t next_split_provider(io::ioctx_resolver resolve, std::stop_token stop);
  stoppable_metadata_scan_task_t next_split_provider_with_stop(io::ioctx_resolver resolve);

  filtered_table materialize_metadata_to_table(
    scan_info const& info,
    const cucascade::memory::memory_space& mem_space,
    ::cuda::stream_ref stream,
    bool like_swar_fastpath,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache,
    expression_query_policy policy = {});

  std::unique_ptr<cudf::table> post_filter_and_project(
    filtered_table&& table,
    const cucascade::memory::memory_space& mem_space,
    ::cuda::stream_ref stream,
    bool like_swar_fastpath,
    std::shared_ptr<const sirius::like_multiliteral_cache> like_cache,
    std::unique_ptr<cudf::column>* survivors,
    std::span<std::size_t const> elided,
    expression_query_policy policy = {});

  [[nodiscard]] bool can_serve_cached_view() const noexcept;

  [[nodiscard]] std::optional<cached_view_layout> view_layout(
    filtered_table const& input, std::span<std::size_t const> elided) const;

  [[nodiscard]] const ingestible_table_info& table_info() const noexcept { return *_info; }

  [[nodiscard]] std::vector<std::size_t> materialized_column_order() const;

  [[nodiscard]] bool output_assembly_is_leading_identity() const noexcept;

  [[nodiscard]] std::optional<std::size_t> output_prefix_width() const noexcept;

  [[nodiscard]] bool has_row_filter() const noexcept
  {
    return _duckdb_filter_expression != nullptr;
  }

  /// post_filter_and_project routes its filter through
  /// expression_evaluator::select_with_survivors, which writes the surviving
  /// positions into the out-parameter.
  [[nodiscard]] bool can_report_survivors() const noexcept { return true; }

  [[nodiscard]] scan_filter_analysis const& filter_analysis() const { return _filter_analysis; }

 private:
  [[nodiscard]] std::unique_ptr<sirius::ast::node> remaining_residual(
    filtered_table const& input) const;

  std::shared_ptr<pipeline::completion_handler> _execution_completion;
  /// Read one file's footer, prune its row groups against the filter, and record
  /// per-row-group byte accounting. Returns a single @c parquet_file_scan_info.
  /// Runs on a scan-manager dispatcher thread (the task returned by
  /// @ref next_split_provider).
  std::unique_ptr<scan_info> build_file_scan_info(std::string const& file_path,
                                                  std::size_t file_index,
                                                  std::shared_ptr<io::ioctx> const& io_ctx,
                                                  std::stop_token stop = {});

  /// Add the carrier and user-requested virtual columns to a decoded parquet batch.
  [[nodiscard]] std::unique_ptr<cudf::table> append_virtual_columns(
    std::unique_ptr<cudf::table> decoded,
    cudf::io::parquet_reader_options const& reader_options,
    std::string const& file_path,
    std::size_t file_index,
    std::int64_t file_row_offset,
    ::cuda::stream_ref stream,
    rmm::device_async_resource_ref mr) const;

  [[nodiscard]] bool can_project_during_filter() const noexcept;

  std::unique_ptr<parquet_ingestible_table_info> _info;
  bool const _device_prefetch_allowed;
  // Built only when physical evidence is consumed, after finalize has published it.
  std::once_flag _evidence_index_once;
  std::vector<std::size_t> _evidence_index_by_file;

  // Canonical scan plan — built once in the constructor, shared by every
  // emitted split via its parquet_split_info::plan member.
  std::shared_ptr<scan_plan const> _plan;
  // Shared plan-level projection options. Filters are applied per split.
  std::shared_ptr<cudf::io::parquet_reader_options> _reader_options;
  // The same options without a column projection: stamped onto files whose
  // carrier column is unavailable (parquet_file_scan_info::carrier_unavailable).
  std::shared_ptr<cudf::io::parquet_reader_options> _natural_reader_options;
  // Coalesced DuckDB filter expression. Empty when no filters survived the
  // partition-column drop pass.
  std::shared_ptr<duckdb::Expression> _duckdb_filter_expression;
  std::unordered_map<duckdb::column_t, sirius::logical_type> _virtual_types;
  bool _has_virtual_filter = false;

  // This scan's pushed-down filter digested once at bind — what filter_analysis()
  // advertises. Empty when the scan has no pushed-down filter.
  scan_filter_analysis _filter_analysis;
  // The same filter as the predicate the scan must still evaluate after the
  // decode, decomposed into conjuncts at bind. Which conjuncts a given batch
  // still needs is a per-batch fact — a pinned compressed chunk answers a
  // column only when its plan can, and the same scan's disk splits never do —
  // so post_filter_and_project assembles the residual from what the batch
  // reports rather than from anything precomputed here.
  residual_filter _residual;

  // The part of _duckdb_filter_expression safe to push into the reader: the
  // full predicate minus any top-level AND conjunct that cuDF's row-group
  // stats filter cannot handle (a null test, or a bare column reference used
  // as the predicate). Surviving conjuncts still prune, so
  // `v IS NULL AND id > 3000` keeps pruning on `id`. Shares the full
  // expression when nothing needs stripping; null when none survive.
  //
  // Separate from disable_filter_pushdown, which also suppresses dynamic join
  // filters — those carry no null predicate and stay safe to push down.
  std::shared_ptr<duckdb::Expression> _static_pushdown_expression;
  // False when the above is a strict subset: the reader has then only
  // partially filtered, so the scan must NOT be reported ROW_FILTERED or the
  // dropped conjuncts would never be applied.
  bool _static_pushdown_is_complete = true;

  // The part of _static_pushdown_expression pushed into the per-task reader:
  // minus any top-level AND conjunct with an equality on a column whose bloom
  // filter probe cuDF before 26.12 gets wrong (has_unreliable_bloom_filter_probe).
  // The reader probes bloom filters, the row-group stats filter does not.
  std::shared_ptr<duckdb::Expression> _reader_pushdown_expression;
  // False when the reader filter is a strict subset of _duckdb_filter_expression,
  // like _static_pushdown_is_complete.
  bool _reader_pushdown_is_complete = true;

  // Only the simple `IS [NOT] NULL` over a bare column reference shape is
  // recorded; anything compound is left to the post-decode filter. Empty when
  // the predicate has no such conjunct.
  std::vector<null_prune_predicate> _null_prune_predicates;

  std::vector<std::string> _file_paths;

  // Per-file metadata-scan cursor. next_split_provider hands out one file index
  // per claim; the coalescer downstream batches files and chunks row groups.
  std::atomic<std::size_t> _next_file_idx{0};

  // Dynamic join filters shared with the producing hash join; null when none are wired.
  // AST-capable filters are ANDed into the parquet reader filter; membership filtering happens in
  // the downstream dynamic-filter operator.
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> _sirius_dynamic_filters;
};

}  // namespace sirius::op::scan
