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

#include "scan_manager/parquet_split_provider.hpp"

#include "expression_executor/gpu_expression_translator_internal.hpp"
#include "io/io_context.hpp"
#include "io/prefetching_cache.hpp"
#include "log/logging.hpp"
#include "op/scan/parquet_scan_operator_data.hpp"
#include "op/scan/parquet_schema_mapping.hpp"
#include "op/scan/scan_utils.hpp"
#include "scan_manager/parquet_metadata.hpp"
#include "scan_manager/sirius_scan_manager.hpp"

// Sirius IO framework includes. sirius_datasource declares the per-ioctx
// datasource factory; uring_reactor pulls in the concrete uring_io_object
// construction. uring_reactor MUST be included last among sirius headers —
// liburing's BLOCK_SIZE macro collides with blockingconcurrentqueue.h's
// static const BLOCK_SIZE member when both transitively land in the same TU.
#include <io/sirius_datasource.hpp>
// (other sirius headers above already included)
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <duckdb/common/hive_partitioning.hpp>
#include <io/uring/uring_reactor.hpp>

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace sirius::scan_manager {

namespace {

struct rg_accumulator {
  std::vector<op::scan::row_group_slice> slices;
  std::size_t total_uncompressed_bytes = 0;
  // Partition values for the files currently bundled, in scan_plan::partition_columns order.
  // nullopt until the first file is added. Bundling is only safe across files with identical
  // values: assemble_scan_output synthesizes constant scalar columns from this single vector on
  // behalf of every file in the bundle, so all files in the bundle must share those values.
  std::optional<std::vector<std::string>> partition_values;
};

}  // namespace

// Legacy public ctor — no scan_manager. Forwards to the private ctor with
// nullptr; run_batch falls through to cudf::io::datasource::create for each
// file (mirrors the pre-multi-ioctx single-ioctx-null behavior).
parquet_split_provider::parquet_split_provider(
  duckdb::vector<sirius::logical_type> const& returned_types,
  std::vector<std::string> const& file_paths,
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<duckdb::idx_t> const& projection_ids,
  duckdb::vector<std::string> const& names,
  std::size_t scan_output_arity,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filter_set,
  duckdb::vector<duckdb::HivePartitioningIndex> const& partition_indices,
  std::size_t approximate_batch_size,
  std::size_t max_file_processed,
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> gpu_ioctxs)
  : parquet_split_provider(returned_types,
                           file_paths,
                           column_ids,
                           projection_ids,
                           names,
                           scan_output_arity,
                           std::move(table_filter_set),
                           partition_indices,
                           approximate_batch_size,
                           max_file_processed,
                           static_cast<sirius_scan_manager*>(nullptr),
                           std::move(gpu_ioctxs))
{
}

// New public ctor — reference to scan_manager. Forwards to the private ctor
// with the reference's address.
parquet_split_provider::parquet_split_provider(
  duckdb::vector<sirius::logical_type> const& returned_types,
  std::vector<std::string> const& file_paths,
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<duckdb::idx_t> const& projection_ids,
  duckdb::vector<std::string> const& names,
  std::size_t scan_output_arity,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filter_set,
  duckdb::vector<duckdb::HivePartitioningIndex> const& partition_indices,
  std::size_t approximate_batch_size,
  std::size_t max_file_processed,
  sirius_scan_manager& scan_manager,
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> gpu_ioctxs)
  : parquet_split_provider(returned_types,
                           file_paths,
                           column_ids,
                           projection_ids,
                           names,
                           scan_output_arity,
                           std::move(table_filter_set),
                           partition_indices,
                           approximate_batch_size,
                           max_file_processed,
                           &scan_manager,
                           std::move(gpu_ioctxs))
{
}

// Private delegating ctor — actual init body lives here.
parquet_split_provider::parquet_split_provider(
  duckdb::vector<sirius::logical_type> const& returned_types,
  std::vector<std::string> const& file_paths,
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  duckdb::vector<duckdb::idx_t> const& projection_ids,
  duckdb::vector<std::string> const& names,
  std::size_t scan_output_arity,
  duckdb::unique_ptr<duckdb::TableFilterSet> table_filter_set,
  duckdb::vector<duckdb::HivePartitioningIndex> const& partition_indices,
  std::size_t approximate_batch_size,
  std::size_t max_file_processed,
  sirius_scan_manager* scan_manager,
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> gpu_ioctxs)
  : _file_paths(file_paths),
    _approximate_batch_size(approximate_batch_size),
    _max_file_processed(max_file_processed),
    _total_files(file_paths.size()),
    _scan_manager(scan_manager),
    _gpu_ioctxs(std::move(gpu_ioctxs))
{
  // Any non-trivial scan shape — reader-side projection, filter pushdown, or hive-partition
  // injection — needs column names for reader set_column_names / AST name resolution /
  // HivePartitioning::Parse lookups.
  bool const needs_names = !projection_ids.empty() ||
                           (table_filter_set && !table_filter_set->filters.empty()) ||
                           !partition_indices.empty();
  if (needs_names && names.empty()) {
    throw sirius::internal_exception(
      "[parquet_split_provider] Projection, filter pushdown, or hive partitions "
      "require column names to be provided.");
  }

  // Build the canonical scan plan
  _plan = std::make_shared<op::scan::scan_plan const>(op::scan::build_scan_plan(
    column_ids, projection_ids, names, returned_types, scan_output_arity, partition_indices));

  // Build the DuckDB filter expression. AST translation is deferred to execute() so that a
  // task-local CUDA stream can be used. Filters on hive-partition columns are dropped because
  // those columns aren't in the parquet file (DuckDB prunes them at the file-list level).
  if (table_filter_set && !table_filter_set->filters.empty()) {
    auto duckdb_expression =
      op::convert_table_filters_to_expression(*table_filter_set,
                                              column_ids,
                                              returned_types,
                                              _plan->batch_position_by_column_id,
                                              _plan->partition_primary_indices);
    if (duckdb_expression) { _duckdb_filter_expression = std::move(duckdb_expression); }
  }

  // Pre-decompose the file list into per-task batches once; next_split_provider() iterates this
  // list one batch at a time and hands each claimed batch to a worker for parallel processing.
  for (std::size_t start = 0; start < _total_files; start += _max_file_processed) {
    auto const end = std::min(start + _max_file_processed, _total_files);
    file_batch batch;
    batch.file_paths.assign(_file_paths.begin() + static_cast<std::ptrdiff_t>(start),
                            _file_paths.begin() + static_cast<std::ptrdiff_t>(end));
    _batches.push_back(std::move(batch));
  }
}

parquet_split_provider::~parquet_split_provider() = default;

bool parquet_split_provider::has_more_splits() const
{
  return _next_batch_idx.load(std::memory_order_relaxed) < _batches.size();
}

std::function<std::vector<std::unique_ptr<op::operator_data>>()>
parquet_split_provider::next_split_provider()
{
  // Atomic claim happens here so the (expensive) run_batch work captured
  // below can run in parallel on a worker pool — distinct callables operate
  // on distinct batch indices. fetch_add can briefly observe an index past
  // the end (when more workers run than batches); returning null in that
  // case signals "no work claimed" without forcing the caller to invoke a
  // dummy callable.
  auto const batch_idx = _next_batch_idx.fetch_add(1, std::memory_order_relaxed);
  if (batch_idx >= _batches.size()) { return nullptr; }
  return [this, batch_idx]() {
    std::vector<std::unique_ptr<op::operator_data>> out;
    run_batch(_batches[batch_idx], out);
    return out;
  };
}

void parquet_split_provider::run_batch(file_batch const& batch,
                                       std::vector<std::unique_ptr<op::operator_data>>& out)
{
  auto stream = cudf::get_default_stream();

  //===----------Build reader options----------===//
  auto const data_column_names = _plan->data_column_names();
  auto reader_options          = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());

  // Tell the parquet reader which columns to produce. Required whenever the scan
  // is projected / has hive partitions to remove.
  if (_plan->is_projected()) { reader_options->set_column_names(data_column_names); }

  // Route reads through sirius_datasource — cudf's bundled file_source uses
  // libkvikio which binds a single CUDA context per FileHandle, breaking
  // multi-GPU residency. Picking the first ioctx for planning is safe: footer
  // reads are small, and per-GPU placement of column data is decided later
  // by the scan operator's task affinity.
  //
  // Initialized HERE (before filter pushdown probe) so that the FLBA-decimal probe
  // below can use it. dev #732 forbids an empty gpu_ioctxs to keep local reads off
  // cudf's kvikio file_source (which binds one CUDA context per handle, breaking
  // multi-GPU residency). The relaxed form below permits an empty gpu_ioctxs only
  // when a scan_manager is wired in (s3:// paths route through its s3_ioctx and
  // local paths through its borrowed uring backend); both missing is fatal.
  if (_gpu_ioctxs.empty() && _scan_manager == nullptr) {
    throw std::runtime_error(
      "parquet_split_provider: gpu_ioctxs is empty and no scan_manager is wired — "
      "kvikio path is forbidden. Production callers receive gpu_ioctxs from "
      "SiriusContext::get_gpu_ioctxs(); test fixtures must inject via "
      "make_test_gpu_ioctxs() helper (test/cpp/scan/test_helpers_ioctx.hpp).");
  }
  auto const planning_ioctx_it = _gpu_ioctxs.begin();

  // Translate the filter to a cudf AST for reader-side pushdown, falling back to a post-read
  // DuckDB-expression evaluation when translation isn't possible. Partition-column filters
  // have already been dropped at construction; anything remaining references data columns.
  //
  // Pushdown safety guard: cudf's row-group stats filter (stats_filter_helpers.hpp:132)
  // throws "Invalid type and stats combination" when comparing a `fixed_point_scalar` AST
  // literal against parquet stats stored in physical type FIXED_LEN_BYTE_ARRAY (or
  // BYTE_ARRAY) — which TPC-H generators like tpchgen-rs emit at SF1000. INT32/INT64
  // decimal stats compare fine. If any file in this batch stores a decimal column with
  // FLBA/BYTE_ARRAY, we skip pushdown for row-group pruning; the filter still applies
  // post-decode in the operator's scan path (sirius_gpu_parquet_scan_operator.cpp:197).
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  // Propagated to each parquet_scan_data so the operator also skips its own set_filter
  // call (otherwise the same cudf crash happens at scan time instead of pruning time).
  bool skip_pushdown_due_to_flba = false;
  if (_duckdb_filter_expression) {
    // Resolver maps the BoundReferenceExpression's batch position (D) to the corresponding
    // parquet column name. scan_plan::batch_column_name is the single source of truth for
    // this D→name mapping.
    auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
      return _plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    ast_expression =
      translator.translate_expression_with_names(*_duckdb_filter_expression, name_resolver);
    if (ast_expression) {
      // Probe the first file's schema before committing to filter pushdown. The probe
      // result is reused below by the main file loop via the prefetch cache.
      //
      // The probe requires a gpu_ioctx (planning_ioctx_it). When gpu_ioctxs is empty
      // (scan_manager-only path) we cannot probe cheaply, so we default to skipping
      // pushdown — correctness over the row-group-pruning perf win.
      if (planning_ioctx_it == _gpu_ioctxs.end()) {
        skip_pushdown_due_to_flba = true;
      } else if (!batch.file_paths.empty()) {
        auto const& probe_path = batch.file_paths.front();
        try {
          auto probe_io_object = planning_ioctx_it->second->create_io_object(probe_path);
          auto probe_ds        = planning_ioctx_it->second->make_datasource(probe_io_object);
          std::shared_ptr<cudf::io::parquet::FileMetaData const> probe_meta;
          if (planning_ioctx_it->second->cache() != nullptr) {
            if (auto cached = planning_ioctx_it->second->cache()->get_metadata(*probe_io_object)) {
              if (auto pm = std::dynamic_pointer_cast<parquet_metadata>(cached)) {
                probe_meta = pm->file_metadata();
              }
            }
          }
          if (!probe_meta) {
            auto footer = cudf::io::parquet::fetch_footer_to_host(*probe_ds);
            op::scan::hybrid_scan_reader probe_reader(
              cudf::host_span<uint8_t const>(footer->data(), footer->size()), *reader_options);
            probe_meta = std::make_shared<cudf::io::parquet::FileMetaData const>(
              probe_reader.parquet_metadata());
          }
          for (auto const& elem : probe_meta->schema) {
            bool const is_decimal =
              (elem.converted_type.has_value() &&
               *elem.converted_type == cudf::io::parquet::ConvertedType::DECIMAL) ||
              (elem.logical_type.has_value() &&
               elem.logical_type->type == cudf::io::parquet::LogicalType::DECIMAL);
            if (!is_decimal) { continue; }
            if (elem.type == cudf::io::parquet::Type::FIXED_LEN_BYTE_ARRAY ||
                elem.type == cudf::io::parquet::Type::BYTE_ARRAY) {
              skip_pushdown_due_to_flba = true;
              break;
            }
          }
        } catch (std::exception const& e) {
          SIRIUS_LOG_DEBUG(
            "[parquet_split_provider] FLBA-decimal probe failed ({}); proceeding without "
            "pushdown",
            e.what());
          skip_pushdown_due_to_flba = true;
        }
      }

      if (!skip_pushdown_due_to_flba) {
        reader_options->set_filter(ast_expression->back());
        SIRIUS_LOG_DEBUG(
          "[parquet_split_provider] Translated filter expression for row group pruning.");
      } else {
        SIRIUS_LOG_DEBUG(
          "[parquet_split_provider] Skipping row-group pruning pushdown: file has "
          "FIXED_LEN_BYTE_ARRAY/BYTE_ARRAY decimal column(s) which cudf's stats filter does "
          "not support. Filter will still apply post-decode in the scan operator.");
      }
    } else {
      SIRIUS_LOG_DEBUG("[parquet_split_provider] AST translation failed for row group pruning.");
    }
  }

  // Loop over files to read footers, parse metadata, and compute row-group partitions.
  rg_accumulator accum;
  // flush() appends the bundled slices to `out` but does NOT reset partition_values. The file
  // loop owns partition_values and re-seeds it on every file iteration; clearing it here would
  // orphan the post-flush tail of a mid-file overflow.
  auto flush = [&]() {
    if (accum.slices.empty()) { return; }
    out.push_back(std::make_unique<op::scan::parquet_scan_data>(
      std::move(accum.slices),
      reader_options,
      _duckdb_filter_expression,
      _plan,
      accum.partition_values.value_or(std::vector<std::string>{}),
      skip_pushdown_due_to_flba));
    accum.slices.clear();
    accum.total_uncompressed_bytes = 0;
  };

  for (auto const& file_path : batch.file_paths) {
    // Partition compatibility: if the current accumulator already holds files with different
    // partition values, flush before starting this file. assemble_scan_output synthesizes
    // constants from one partition_values vector on behalf of the whole bundle, so mixing
    // partitions would produce wrong rows. Always (re-)seed partition_values for this file
    // afterward — the previous iteration may have flushed mid-file (byte-budget overflow),
    // leaving accum.partition_values intact but accum.slices empty.
    if (!_plan->partition_columns.empty()) {
      std::vector<std::string> file_partition_values;
      file_partition_values.reserve(_plan->partition_columns.size());
      auto parsed = duckdb::HivePartitioning::Parse(file_path);
      for (auto const& pc : _plan->partition_columns) {
        auto it = parsed.find(pc.name);
        file_partition_values.push_back(it != parsed.end() ? it->second : std::string{});
      }
      if (accum.partition_values && *accum.partition_values != file_partition_values) { flush(); }
      accum.partition_values = std::move(file_partition_values);
    }

    //===----------Read metadata footers----------===//
    // Merged dispatch (multi-GPU #732 × multi-backend-S3 PR4+5):
    //   * s3:// (any non-local scheme) → scan_manager's per-path dispatch. The
    //     s3_ioctx is a single shared network backend (not per-GPU) and carries
    //     the S3 prefetch cache.
    //   * local file → dev #732's per-GPU planning ioctx (gpu_ioctxs.begin()).
    //     Footer reads are GPU-agnostic; per-GPU column placement is decided
    //     downstream by the scan operator's task affinity, so any GPU's ioctx
    //     is safe for planning, and routing through io_uring (not cudf's kvikio
    //     file_source) preserves per-GPU CUDA-context binding.
    //   * neither (legacy test fixture with no scan_manager and empty
    //     gpu_ioctxs) → cudf::io::datasource::create.
    //
    // Path normalization: uring_reactor::supports / create_io_object only
    // accept bare absolute paths (they call is_regular_file on the raw
    // string). When the planner gives us a "file://" URI, strip the scheme
    // BEFORE dispatching. The "file://" match is case-insensitive so a
    // "FILE://" URI is still local. S3 and other schemes pass through unchanged.
    auto normalize_path = [](std::string const& p) -> std::string {
      static constexpr std::string_view kFile = "file://";
      if (p.size() > kFile.size()) {
        bool is_file_uri = true;
        for (std::size_t i = 0; i < kFile.size(); ++i) {
          if (std::tolower(static_cast<unsigned char>(p[i])) !=
              static_cast<unsigned char>(kFile[i])) {
            is_file_uri = false;
            break;
          }
        }
        if (is_file_uri) { return p.substr(kFile.size()); }
      }
      return p;
    };
    // A path still carrying a "://" after file:// stripping has a URI scheme
    // (s3://, http://, …) that cudf cannot read locally — used to pick the
    // failure mode and to decide s3 vs local routing.
    auto has_uri_scheme = [](std::string const& p) -> bool {
      return p.find("://") != std::string::npos;
    };
    auto const lookup_path = normalize_path(file_path);
    std::shared_ptr<sirius::io::sirius_ioctx> file_io_ctx;
    if (has_uri_scheme(lookup_path)) {
      // Non-local scheme → route through scan_manager (s3_ioctx etc.).
      if (_scan_manager != nullptr) { file_io_ctx = _scan_manager->io_ctx_shared_for(lookup_path); }
      if (!file_io_ctx) {
        throw std::runtime_error("[parquet_split_provider] no backend supports path: " + file_path);
      }
    } else if (planning_ioctx_it != _gpu_ioctxs.end()) {
      // Local file → dev #732's per-GPU planning ioctx.
      file_io_ctx = planning_ioctx_it->second;
    } else if (_scan_manager != nullptr) {
      // Local file, no gpu_ioctxs injected → scan_manager's uring backend
      // (still keeps the read off cudf's kvikio file_source).
      file_io_ctx = _scan_manager->io_ctx_shared_for(lookup_path);
    }
    std::shared_ptr<sirius::io::sirius_io_object> file_io_object;
    std::unique_ptr<cudf::io::datasource> datasource;
    if (file_io_ctx) {
      file_io_object = file_io_ctx->create_io_object(lookup_path);
      datasource     = file_io_ctx->make_datasource(file_io_object);
    } else {
      datasource = cudf::io::datasource::create(lookup_path);
    }

    //===----------Parse metadata (with prefetch-cache reuse)----------===//
    // If the prefetching cache already has a parquet_metadata entry for this
    // file (from a previous scan of the same path in this session), reuse the
    // parsed FileMetaData and skip the footer fetch.  Otherwise fetch + parse
    // and stash the result on the cache below.
    std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
    std::shared_ptr<parquet_metadata> cached_parquet_metadata;
    std::size_t footer_byte_len = 0;
    std::unique_ptr<op::scan::hybrid_scan_reader> reader_ptr;

    if (file_io_object && file_io_ctx && file_io_ctx->cache() != nullptr) {
      if (auto cached = file_io_ctx->cache()->get_metadata(*file_io_object)) {
        cached_parquet_metadata = std::dynamic_pointer_cast<parquet_metadata>(std::move(cached));
      }
    }

    if (cached_parquet_metadata) {
      file_metadata   = cached_parquet_metadata->file_metadata();
      footer_byte_len = cached_parquet_metadata->footer_byte_len();
      reader_ptr = std::make_unique<op::scan::hybrid_scan_reader>(*file_metadata, *reader_options);
    } else {
      auto footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
      footer_byte_len    = footer_buffer->size();
      reader_ptr         = std::make_unique<op::scan::hybrid_scan_reader>(
        cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()),
        *reader_options);
      file_metadata =
        std::make_shared<cudf::io::parquet::FileMetaData const>(reader_ptr->parquet_metadata());
    }
    auto& reader         = *reader_ptr;
    auto const& metadata = *file_metadata;

    //===----------Resolve selected DuckDB columns to parquet column chunk indices----------===//
    // row_group.columns is indexed in parquet schema-leaf order (preorder), which can differ from
    // DuckDB's logical column order. Resolve by name per file (chunk order is consistent across row
    // groups in a single file, but can vary across files).
    std::vector<std::size_t> selected_chunk_indices;
    std::unordered_set<std::size_t> pure_filter_chunk_indices;
    if (_plan->is_projected()) {
      auto const pure_filter_positions = _plan->pure_filter_batch_positions();
      selected_chunk_indices.reserve(data_column_names.size());
      for (std::size_t k = 0; k < data_column_names.size(); ++k) {
        auto leaves = op::scan::detail::leaf_indices_for_column(metadata, data_column_names[k]);
        if (leaves.empty()) {
          throw std::runtime_error("[parquet_split_provider] Projected column '" +
                                   data_column_names[k] +
                                   "' not found in parquet file: " + file_path);
        }
        bool const is_pure_filter = pure_filter_positions.count(k);
        for (auto const leaf : leaves) {
          selected_chunk_indices.push_back(leaf);
          if (is_pure_filter) { pure_filter_chunk_indices.insert(leaf); }
        }
      }
    }

    //===----------Row Group Partitioning----------===//
    auto row_group_indices = reader.all_row_groups(*reader_options);
    // Row group pruning with filter pushdown using metadata statistics.
    // Also skipped when the FLBA-decimal probe disabled pushdown — in that case
    // reader_options has no filter set and filter_row_groups_with_stats would
    // throw "Empty input filter expression encountered" (hybrid_scan_impl.cpp:217).
    if (ast_expression && !skip_pushdown_due_to_flba) {
      auto const row_groups_before_pruning = row_group_indices.size();
      // clang-format off
      SIRIUS_LOG_DEBUG("[parquet_split_provider] Row group pruning: file: {}\n" \
                       "                                                  before: {}",
                       file_path,
                       row_groups_before_pruning);
      // clang-format on
      // Prune row groups with filter pushdown using metadata statistics.
      row_group_indices =
        reader.filter_row_groups_with_stats(row_group_indices, *reader_options, stream);
      auto const row_groups_after_pruning = row_group_indices.size();
      auto const pruned_row_groups        = row_groups_before_pruning - row_groups_after_pruning;
      // clang-format off
      SIRIUS_LOG_DEBUG("[parquet_split_provider]                     after: {} (pruned {})",
                       row_groups_after_pruning,
                       pruned_row_groups);
      // clang-format on
    }

    //===----------Prefetch cache prewarm----------===//
    // When the ioctx has a cache, hand it the exact byte ranges scan_task
    // will request: PAR1 header + (merged) column-chunk ranges for every
    // surviving row group + footer/trailer.  insert() must use the same
    // merged ranges scan_task computes — the cache only serves reads that
    // are fully covered by an inserted range.
    if (file_io_object && file_io_ctx && file_io_ctx->cache() != nullptr &&
        !row_group_indices.empty()) {
      using range_t = cudf::io::text::byte_range_info;

      auto chunk_ranges = reader.all_column_chunks_byte_ranges(row_group_indices, *reader_options);

      // Inline merge: parquet_scan_task::detail::merge_byte_ranges is TU-local;
      // duplicating the ~10-line walk avoids cross-component coupling.
      std::sort(chunk_ranges.begin(), chunk_ranges.end(), [](auto const& a, auto const& b) {
        return a.offset() < b.offset();
      });
      std::vector<range_t> merged;
      merged.reserve(chunk_ranges.size());
      if (!chunk_ranges.empty()) {
        auto cur_start = chunk_ranges[0].offset();
        auto cur_end   = cur_start + chunk_ranges[0].size();
        for (auto const& r : chunk_ranges) {
          auto const rs = r.offset();
          auto const re = rs + r.size();
          if (rs <= cur_end) {
            cur_end = std::max(cur_end, re);
          } else {
            merged.emplace_back(cur_start, cur_end - cur_start);
            cur_start = rs;
            cur_end   = re;
          }
        }
        merged.emplace_back(cur_start, cur_end - cur_start);
      }

      // footer_offset / footer_size mirror parquet_scan_task's computation:
      // the trailer is 8 bytes (4 footer_len + 4 magic) and the footer body
      // length (excluding the trailer) is recorded on parquet_metadata so we
      // don't need the original footer buffer to recompute the range.
      constexpr std::size_t FOOTER_TAIL_SIZE = 8;
      auto const file_size                   = file_io_object->size();
      auto const footer_off  = static_cast<int64_t>(file_size - FOOTER_TAIL_SIZE - footer_byte_len);
      auto const footer_size = static_cast<int64_t>(FOOTER_TAIL_SIZE + footer_byte_len);

      std::vector<range_t> ranges;
      ranges.reserve(merged.size() + 2);
      ranges.emplace_back(0, 4);  // PAR1 header
      ranges.insert(ranges.end(), merged.begin(), merged.end());
      ranges.emplace_back(footer_off, footer_size);
      // Cache requires sorted-by-offset.  Header is at 0, footer is at file end,
      // and merged column chunks live in between — a defensive sort handles any
      // pathological layout where a column chunk starts before the header.
      std::sort(ranges.begin(), ranges.end(), [](auto const& a, auto const& b) {
        return a.offset() < b.offset();
      });

      // When the cache already had parquet_metadata for this file we reused it
      // and don't need to re-store it; otherwise we just parsed the footer and
      // hand the freshly-built parquet_metadata to the cache so the next scan
      // of this file can skip the footer fetch.
      // Use per-file `file_io_ctx` (PR4+5 per-path dispatch) so mixed-backend
      // batches each route metadata to the right backend's cache.
      std::shared_ptr<sirius::io::sirius_io_object_metadata> metadata_to_store =
        cached_parquet_metadata
          ? nullptr
          : std::static_pointer_cast<sirius::io::sirius_io_object_metadata>(
              std::make_shared<parquet_metadata>(file_metadata, footer_byte_len));
      // B1 Phase 1: prewarm of column-chunk byte ranges is gated by
      // scan_manager_config::enable_chunk_prewarm. When false, we still
      // insert metadata (so §24 describe_parquet's footer reuse keeps
      // working) but skip the per-chunk prefetch — letting the bench
      // compare prefetch overlap vs cost.
      bool const prewarm = (_scan_manager != nullptr) && _scan_manager->chunk_prewarm_enabled();
      if (prewarm) {
        file_io_ctx->cache()->insert(*file_io_object, std::move(metadata_to_store), ranges);
      } else {
        file_io_ctx->cache()->insert(*file_io_object, std::move(metadata_to_store), {});
      }
    }

    std::vector<cudf::size_type> cur_rgs;
    std::size_t cur_uncompressed_bytes = 0;
    std::size_t cur_compressed_bytes   = 0;

    auto seal_current_file = [&]() {
      if (cur_rgs.empty()) { return; }
      accum.slices.emplace_back(file_metadata,
                                file_path,
                                std::move(cur_rgs),
                                cur_uncompressed_bytes,
                                cur_compressed_bytes,
                                file_io_ctx,
                                file_io_object);
      // Promote the just-sealed slice's uncompressed bytes into the cross-file accumulator.
      accum.total_uncompressed_bytes += cur_uncompressed_bytes;
      cur_rgs.clear();
      cur_uncompressed_bytes = 0;
      cur_compressed_bytes   = 0;
    };

    // Compute the row group's contribution
    auto rg_contribution = [&](cudf::io::parquet::RowGroup const& row_group) {
      std::size_t rg_uncompressed = 0;
      std::size_t rg_compressed   = 0;
      auto add_chunk = [&](cudf::io::parquet::ColumnChunk const& chunk, bool is_pure_filter) {
        auto const& column_metadata = chunk.meta_data;
        // Pure-filter columns are not part of the scan result, so omit them from the
        // uncompressed byte count used for sizing partitions.
        if (!is_pure_filter) {
          rg_uncompressed += static_cast<std::size_t>(column_metadata.total_uncompressed_size);
        }
        rg_compressed += static_cast<std::size_t>(column_metadata.total_compressed_size);
      };
      if (_plan->is_projected()) {
        for (auto const chunk_idx : selected_chunk_indices) {
          add_chunk(row_group.columns[chunk_idx], pure_filter_chunk_indices.contains(chunk_idx));
        }
      } else {
        // Non-projected: all chunks contribute, no pure-filter pruning.
        for (auto const& chunk : row_group.columns) {
          add_chunk(chunk, false);
        }
      }
      return std::pair{rg_uncompressed, rg_compressed};
    };

    for (auto const rg_idx : row_group_indices) {
      auto const& row_group        = metadata.row_groups[rg_idx];
      auto const [rg_unc, rg_comp] = rg_contribution(row_group);

      // Ensure that a single oversized rg/file still gets through.
      if (!accum.slices.empty() || !cur_rgs.empty()) {
        if (accum.total_uncompressed_bytes + cur_uncompressed_bytes + rg_unc >
            _approximate_batch_size) {
          seal_current_file();
          flush();
        }
      }

      cur_uncompressed_bytes += rg_unc;
      cur_compressed_bytes += rg_comp;
      cur_rgs.push_back(rg_idx);
    }
    seal_current_file();
    // Multi-GPU only: emit at least one split per file so source pipelines
    // (GPU_PARQUET_SCAN -> ...) generate multiple gpu_pipeline_tasks when
    // scanning multiple files. The task_scheduler's round-robin counter then
    // distributes those tasks across GPUs. On a single-GPU system there is no
    // GPU to balance across, so we keep the BASE byte-budget bundling (more
    // tasks just add pipeline-start overhead without parallelism benefit).
    if (_gpu_ioctxs.size() > 1) { flush(); }
  }
  flush();
}

}  // namespace sirius::scan_manager
