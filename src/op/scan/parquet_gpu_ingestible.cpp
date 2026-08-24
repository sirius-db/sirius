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

// sirius
#include "op/scan/gpu_ingestible_types.hpp"
#include "op/scan/owning_table_view.hpp"

#include <expression/ast/from_duckdb.hpp>
#include <expression_evaluator/expression_evaluator.hpp>
#include <expression_evaluator/gpu_expression_translator_internal.hpp>
#include <io/io_context.hpp>
#include <io/sirius_datasource.hpp>
#include <log/logging.hpp>
#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/parquet_metadata.hpp>
#include <op/scan/parquet_schema_mapping.hpp>
#include <op/scan/scan_utils.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// duckdb
#include <duckdb/common/hive_partitioning.hpp>
#include <duckdb/planner/expression/bound_conjunction_expression.hpp>
#include <duckdb/planner/expression/bound_operator_expression.hpp>
#include <duckdb/planner/expression/bound_reference_expression.hpp>
#include <duckdb/planner/expression_iterator.hpp>
#include <duckdb/planner/filter/conjunction_filter.hpp>
#include <duckdb/planner/filter/null_filter.hpp>

// uring_reactor MUST be included last among sirius headers: liburing.h,
// pulled in transitively, defines a BLOCK_SIZE macro that collides with the
// BLOCK_SIZE static member in <blockingconcurrentqueue.h>.
#include <io/uring/uring_reactor.hpp>

// standard library
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

/// True when @p expr contains an IS NULL / IS NOT NULL predicate anywhere.
///
/// cuDF's row-group statistics filter rewrites every column reference into that
/// row group's (min, max) pair. A null test has no min/max analogue — only
/// null_count answers it — and handing one to filter_row_groups_with_stats
/// faults rather than merely mis-pruning.
///
/// IS NOT NULL counts: it lowers to IS_NULL + NOT in the cuDF AST, and
/// convert_table_filters_to_expression only drops it when it is a column's
/// top-level filter, so one nested in a conjunction still reaches here.
bool expression_has_null_predicate(duckdb::Expression const& expr)
{
  auto const expr_type = expr.GetExpressionType();
  if (expr_type == duckdb::ExpressionType::OPERATOR_IS_NULL ||
      expr_type == duckdb::ExpressionType::OPERATOR_IS_NOT_NULL) {
    return true;
  }
  bool found = false;
  duckdb::ExpressionIterator::EnumerateChildren(expr, [&found](duckdb::Expression const& child) {
    if (!found) { found = expression_has_null_predicate(child); }
  });
  return found;
}

/// True when @p expr cannot be handed to cuDF's row-group statistics filter as
/// a predicate in its own right.
///
/// Two shapes fault there, both because the stats rewrite replaces each column
/// reference with that row group's (min, max) pair:
///
///   - a null test, which has no min/max analogue (only null_count answers it);
///   - a BARE column reference used directly as the predicate, e.g.
///     `WHERE flag` on a BOOLEAN column, where the rewrite leaves a pair of
///     stats columns with no comparison to evaluate over them.
///
/// The bare-reference case is only unsafe when the reference IS the predicate.
/// `WHERE flag AND v > 5000` prunes fine, as do `NOT flag`, IN and BETWEEN, so
/// this deliberately does not recurse — over-rejecting would forfeit pruning
/// that demonstrably works.
bool is_unsafe_for_stats_filter(duckdb::Expression const& expr)
{
  return expr.GetExpressionType() == duckdb::ExpressionType::BOUND_REF ||
         expression_has_null_predicate(expr);
}

/// @p expr minus every top-level AND conjunct the stats filter cannot handle;
/// nullptr when none survive. Only meaningful when @p expr is itself unsafe —
/// callers check that first.
///
/// Sound because pruning on part of a conjunction can only keep extra row
/// groups, never drop wanted ones: nothing failing `id > 3000` can satisfy
/// `v IS NULL AND id > 3000`. The caller must still apply the full predicate
/// post-decode.
///
/// Only the top level is split: a compound conjunct (an OR, or a nested AND)
/// containing anything unsafe is dropped whole. Conservative, not wrong.
duckdb::unique_ptr<duckdb::Expression> stats_safe_conjuncts(duckdb::Expression const& expr)
{
  if (expr.GetExpressionType() != duckdb::ExpressionType::CONJUNCTION_AND) { return nullptr; }

  auto const& conjunction = expr.Cast<duckdb::BoundConjunctionExpression>();
  duckdb::vector<duckdb::unique_ptr<duckdb::Expression>> kept;
  for (auto const& child : conjunction.children) {
    if (!is_unsafe_for_stats_filter(*child)) { kept.push_back(child->Copy()); }
  }
  if (kept.empty()) { return nullptr; }
  if (kept.size() == 1) { return std::move(kept[0]); }

  auto out =
    duckdb::make_uniq<duckdb::BoundConjunctionExpression>(duckdb::ExpressionType::CONJUNCTION_AND);
  for (auto& child : kept) {
    out->children.push_back(std::move(child));
  }
  return out;
}

/// Every `<col> IS [NOT] NULL` filter usable for null_count row-group pruning.
///
/// Read from the TableFilterSet rather than from the converted expression,
/// because convert_table_filters_to_expression DROPS a column's top-level
/// IS_NOT_NULL before building the expression. Collecting downstream of that
/// would leave the ordinary `WHERE v IS NOT NULL` with no pruning at all, which
/// is the common form of the predicate.
///
/// Both a top-level filter and one nested in a conjunction qualify. A
/// conjunction is an AND of per-column filters, so each null test in it must
/// hold independently -- unlike an OR, where the other branch could still
/// match, so those are not collected.
std::vector<null_prune_predicate> collect_null_prune_predicates(
  duckdb::TableFilterSet const& filters,
  duckdb::vector<duckdb::ColumnIndex> const& column_ids,
  std::vector<std::optional<std::size_t>> const& batch_position_by_column_id,
  std::unordered_set<std::size_t> const& skip_primary_indices)
{
  std::vector<null_prune_predicate> out;

  auto classify = [](duckdb::TableFilterType type) -> std::optional<bool> {
    if (type == duckdb::TableFilterType::IS_NULL) { return true; }
    if (type == duckdb::TableFilterType::IS_NOT_NULL) { return false; }
    return std::nullopt;
  };

  for (auto const& [column_index, filter] : filters.filters) {
    // A partition column has no statistics in the file to prune on, and a
    // column that is not in the batch cannot be pruned by — both are simply
    // skipped here, unlike a conjunct that has to be evaluated.
    auto const column = sirius::op::resolve_filtered_column(
      column_index, column_ids, batch_position_by_column_id, skip_primary_indices);
    if (column.status != sirius::op::filter_column_status::usable) { continue; }
    auto const index = static_cast<duckdb::idx_t>(column.batch_position);

    if (auto expects_null = classify(filter->filter_type)) {
      out.push_back(null_prune_predicate{index, *expects_null});
    } else if (filter->filter_type == duckdb::TableFilterType::CONJUNCTION_AND) {
      for (auto const& child : filter->Cast<duckdb::ConjunctionAndFilter>().child_filters) {
        if (auto nested = classify(child->filter_type)) {
          out.push_back(null_prune_predicate{index, *nested});
        }
      }
    }
  }
  return out;
}

bool has_uri_scheme(std::string const& p) { return p.find("://") != std::string::npos; }

// Strip a leading, case-insensitive "file://" scheme, if present.
std::string strip_file_uri(std::string const& p)
{
  static constexpr std::string_view kFile = "file://";
  if (p.size() > kFile.size()) {
    bool is_file_uri = true;
    for (std::size_t i = 0; i < kFile.size(); ++i) {
      if (std::tolower(static_cast<unsigned char>(p[i])) != static_cast<unsigned char>(kFile[i])) {
        is_file_uri = false;
        break;
      }
    }
    if (is_file_uri) { return p.substr(kFile.size()); }
  }
  return p;
}

//===----------------------------------------------------------------------===//
// parquet_batch_coalescer
//===----------------------------------------------------------------------===//
/**
 * @brief Coalesces per-file metadata units into data-batch splits.
 *
 * Receives one @c parquet_file_scan_info per file (each already pruned and
 * byte-accounted by the metadata-scan task) and accumulates their row groups
 * into @c parquet_split_info batches sized to @c approximate_batch_size. A
 * single large file spans multiple splits (each with its own row_group_slice),
 * and several small files bundle into one split. Bundling across files is only
 * safe when they share hive-partition values and the same pushdown decision, so
 * a mismatch on either forces a flush.
 */
class parquet_batch_coalescer : public batch_coalescer {
 public:
  parquet_batch_coalescer(std::size_t cap,
                          std::shared_ptr<cudf::io::parquet_reader_options> reader_options,
                          std::shared_ptr<scan_plan const> plan)
    : _cap(cap),
      _reader_options(std::move(reader_options)),
      _plan(std::move(plan)),
      _needs_assembly(needs_output_assembly(*_plan))
  {
  }

  std::vector<std::unique_ptr<scan_info>> push(std::unique_ptr<scan_info> info) override
  {
    std::vector<std::unique_ptr<scan_info>> emitted;
    auto* file = dynamic_cast<parquet_file_scan_info*>(info.get());
    if (file == nullptr) { return emitted; }

    // Remember the first fully-pruned file. If the WHOLE source coalesces to
    // nothing, flush() emits one empty split built from it — zero splits mean
    // zero tasks, and the pipeline-completion accounting only fires from task
    // completion, hanging sirius_engine::execute().
    if (file->row_groups.empty() && !_empty_split_fallback) {
      _empty_split_fallback = fallback_file{
        file->file_metadata,
        file->file_path,
        file->datasource ? std::shared_ptr<io::sirius_datasource>(file->datasource->duplicate())
                         : std::shared_ptr<io::sirius_datasource>{},
        file->partition_values,
        file->disable_filter_pushdown};
    }

    if (!_slices.empty() && (_partition_values != file->partition_values ||
                             _disable_pushdown != file->disable_filter_pushdown)) {
      emitted.push_back(emit_current());
    }
    _partition_values = file->partition_values;
    _disable_pushdown = file->disable_filter_pushdown;

    std::vector<cudf::size_type> cur_rgs;
    std::size_t cur_output  = 0;
    std::size_t cur_working = 0;
    std::size_t cur_comp    = 0;
    int64_t cur_rows        = 0;
    auto seal_file          = [&]() {
      if (cur_rgs.empty()) { return; }
      // A file's row groups can span multiple splits, each sealed into its own
      // slice. fadvise stores a per-scan prefetch handle on the datasource, so
      // each slice gets its own datasource (sharing the io_object) — otherwise
      // a later split's fadvise would stomp an earlier one's handle.
      auto slice_ds = file->datasource
                                 ? std::shared_ptr<io::sirius_datasource>(file->datasource->duplicate())
                                 : std::shared_ptr<io::sirius_datasource>{};
      _slices.emplace_back(file->file_metadata,
                           file->file_path,
                           std::move(cur_rgs),
                           cur_output,
                           cur_working,
                           cur_comp,
                           std::move(slice_ds));
      _produced_any = true;
      _acc_working_bytes += cur_working;
      _acc_rows += cur_rows;
      cur_rgs.clear();
      cur_output  = 0;
      cur_working = 0;
      cur_comp    = 0;
      cur_rows    = 0;
    };

    // cuDF tables are limited to cudf::size_type (int32_t) rows per call.
    static constexpr int64_t cudf_max_rows = std::numeric_limits<cudf::size_type>::max();

    for (auto const& rg : file->row_groups) {
      bool const byte_cap_hit = (!_slices.empty() || !cur_rgs.empty()) && _cap > 0 &&
                                _acc_working_bytes + cur_working + rg.decode_working_bytes > _cap;
      bool const row_cap_hit = (!_slices.empty() || !cur_rgs.empty()) &&
                               _acc_rows + cur_rows + rg.num_rows > cudf_max_rows;
      if (byte_cap_hit || row_cap_hit) {
        seal_file();
        emitted.push_back(emit_current());
      }
      cur_output += rg.output_bytes;
      cur_working += rg.decode_working_bytes;
      cur_comp += rg.compressed_bytes;
      cur_rgs.push_back(rg.index);
      cur_rows += rg.num_rows;
    }
    seal_file();
    return emitted;
  }

  std::vector<std::unique_ptr<scan_info>> flush() override
  {
    std::vector<std::unique_ptr<scan_info>> out;
    if (!_slices.empty()) { out.push_back(emit_current()); }
    // Every file was stats-pruned to zero row groups: emit exactly one split
    // with a single zero-row-group slice so the scan still creates one task
    // (materialize_metadata_to_table short-circuits it to a schema-correct
    // empty table). Partial prunes never reach here — any surviving slice sets
    // _produced_any. Zero splits would mean zero tasks, and pipeline-completion
    // accounting only fires from task completion, hanging the query.
    if (!_produced_any && _empty_split_fallback) {
      _slices.emplace_back(_empty_split_fallback->file_metadata,
                           _empty_split_fallback->file_path,
                           std::vector<cudf::size_type>{},
                           /*estimated_output_bytes=*/0,
                           /*estimated_decode_working_bytes=*/0,
                           /*reserved_compressed_bytes=*/0,
                           _empty_split_fallback->datasource);
      _partition_values = _empty_split_fallback->partition_values;
      _disable_pushdown = _empty_split_fallback->disable_filter_pushdown;
      _produced_any     = true;
      out.push_back(emit_current());
    }
    return out;
  }

 private:
  std::unique_ptr<scan_info> emit_current()
  {
    auto split                     = std::make_unique<parquet_split_info>();
    split->rg_slices               = std::move(_slices);
    split->reader_options          = _reader_options;
    split->plan                    = _plan;
    split->disable_filter_pushdown = _disable_pushdown;
    split->needs_assembly          = _needs_assembly;
    split->partition_values        = _partition_values;
    _slices.clear();
    _acc_working_bytes = 0;
    _acc_rows          = 0;
    return split;
  }

  const std::size_t _cap;
  std::shared_ptr<cudf::io::parquet_reader_options> _reader_options;
  std::shared_ptr<scan_plan const> _plan;
  const bool _needs_assembly;

  std::vector<row_group_slice> _slices;
  std::size_t _acc_working_bytes = 0;
  int64_t _acc_rows              = 0;
  std::size_t _emit_count        = 0;  // [coalesce-debug] running count of emitted batches
  std::vector<std::string> _partition_values;
  bool _disable_pushdown = false;

  /// First fully-pruned file, kept as the source for flush()'s empty-split
  /// fallback when the whole scan produced no slice.
  struct fallback_file {
    std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
    std::string file_path;
    std::shared_ptr<io::sirius_datasource> datasource;
    std::vector<std::string> partition_values;
    bool disable_filter_pushdown;
  };
  std::optional<fallback_file> _empty_split_fallback;
  bool _produced_any = false;
};

/// Column-chunk byte ranges a read fetches for @p row_group_indices, honoring
/// @p options' column projection — the ranges materialize_table reads, used to
/// drive prefetch. Empty when there are no row groups.
std::vector<cudf::io::text::byte_range_info> column_chunk_ranges(
  cudf::io::parquet::FileMetaData const& metadata,
  cudf::io::parquet_reader_options const& options,
  std::vector<cudf::size_type> const& row_group_indices)
{
  if (row_group_indices.empty()) { return {}; }
  hybrid_scan_reader reader(metadata, options);
  return reader.all_column_chunks_byte_ranges(
    cudf::host_span<cudf::size_type const>(row_group_indices.data(), row_group_indices.size()),
    options);
}

}  // namespace

std::string canonical_scan_file_path(std::string const& raw)
{
  std::string p = strip_file_uri(raw);
  if (has_uri_scheme(p)) { return p; }  // s3://, gs://, http(s):// — local canon N/A
  std::error_code ec;
  auto c = std::filesystem::weakly_canonical(std::filesystem::path(p), ec);
  return ec ? std::filesystem::path(p).lexically_normal().string() : c.string();
}

void canonicalize_scan_file_paths(std::vector<std::string>& paths)
{
  for (auto& p : paths) {
    p = canonical_scan_file_path(p);
  }
}

//===----------------------------------------------------------------------===//
// scan_info fadvise_entries — prefetch byte ranges
//===----------------------------------------------------------------------===//
std::vector<scan_info::fadvise_entry> parquet_file_scan_info::fadvise_entries() const
{
  if (!file_metadata || !reader_options) { return {}; }
  std::vector<fadvise_entry> entries;
  append_fadvise_entry(entries, datasource, [this] {
    std::vector<cudf::size_type> rg_indices;
    rg_indices.reserve(row_groups.size());
    for (auto const& rg : row_groups) {
      rg_indices.push_back(rg.index);
    }
    return column_chunk_ranges(*file_metadata, *reader_options, rg_indices);
  });
  return entries;
}

std::vector<scan_info::fadvise_entry> parquet_split_info::fadvise_entries() const
{
  if (!reader_options) { return {}; }
  std::vector<fadvise_entry> entries;
  entries.reserve(rg_slices.size());
  for (auto const& slice : rg_slices) {
    if (!slice.file_metadata) { continue; }
    append_fadvise_entry(entries, slice.datasource, [&slice, this] {
      return column_chunk_ranges(*slice.file_metadata, *reader_options, slice.row_group_indices);
    });
  }
  return entries;
}

//===----------------------------------------------------------------------===//
// parquet_ingestible_table_info::make_ingestible
//===----------------------------------------------------------------------===//
std::shared_ptr<parquet_gpu_ingestible> make_ingestible(
  std::unique_ptr<parquet_ingestible_table_info> info)
{
  return std::make_shared<parquet_gpu_ingestible>(std::move(info));
}

//===----------------------------------------------------------------------===//
// parquet_gpu_ingestible — construction
//===----------------------------------------------------------------------===//
parquet_gpu_ingestible::parquet_gpu_ingestible(std::unique_ptr<parquet_ingestible_table_info> info)
  : _info(std::move(info))
{
  auto const& bind = static_cast<parquet_ingestible_table_info const&>(table_info());

  // Any non-trivial scan shape — reader-side projection (incl. a pruned/reordered
  // column_ids with empty projection_ids, the no-pushdown sirius_read_parquet
  // case), filter pushdown, or hive-partition injection — needs column names.
  bool const needs_names = !bind.projection_ids.empty() ||
                           (bind.table_filters && !bind.table_filters->filters.empty()) ||
                           !bind.partition_indices.empty() ||
                           column_ids_need_reader_projection(bind.column_ids, bind.names.size());
  if (needs_names && bind.names.empty()) {
    throw sirius::internal_exception(
      "[parquet_gpu_ingestible] Projection, filter pushdown, or hive partitions "
      "require column names to be provided.");
  }

  _plan = std::make_shared<scan_plan const>(build_scan_plan(bind.column_ids,
                                                            bind.projection_ids,
                                                            bind.names,
                                                            bind.returned_types,
                                                            bind.scan_output_arity,
                                                            bind.partition_indices));

  // AST translation deferred to materialize_table so a task-local stream is used.
  // Filters on hive-partition columns are dropped — those columns aren't in the
  // parquet file (DuckDB prunes them at the file-list level already).
  if (bind.table_filters && !bind.table_filters->filters.empty()) {
    // Collected from the filters themselves, not the expression built below --
    // see collect_null_prune_predicates. Independent of whether any part of the
    // predicate survives into reader-side pushdown.
    _null_prune_predicates = collect_null_prune_predicates(*bind.table_filters,
                                                           bind.column_ids,
                                                           _plan->batch_position_by_column_id,
                                                           _plan->partition_primary_indices);

    auto duckdb_expression =
      sirius::op::convert_table_filters_to_expression(*bind.table_filters,
                                                      bind.column_ids,
                                                      bind.returned_types,
                                                      _plan->batch_position_by_column_id,
                                                      _plan->partition_primary_indices);
    if (duckdb_expression) {
      // Validate before scan tasks retranslate and dereference the predicate.
      if (sirius::ast::from_duckdb(*duckdb_expression) == nullptr) {
        throw duckdb::InvalidInputException(
          "parquet scan: cannot evaluate pushed-down predicate on GPU: %s",
          duckdb_expression->ToString());
      }
      _duckdb_filter_expression = std::move(duckdb_expression);

      if (!is_unsafe_for_stats_filter(*_duckdb_filter_expression)) {
        // Nothing to strip — push the whole predicate, sharing it rather than
        // copying.
        _static_pushdown_expression = _duckdb_filter_expression;
      } else {
        _static_pushdown_is_complete = false;
        _static_pushdown_expression  = stats_safe_conjuncts(*_duckdb_filter_expression);
        // Whatever survives must itself be safe, or the crash returns.
        D_ASSERT(!_static_pushdown_expression ||
                 !is_unsafe_for_stats_filter(*_static_pushdown_expression));
        SIRIUS_LOG_DEBUG(
          "[parquet_gpu_ingestible] Predicate is unsupported by the row-group stats filter; "
          "pushing only the safe conjuncts ({}), full predicate applied post-decode: {}",
          _static_pushdown_expression ? _static_pushdown_expression->ToString() : "none",
          _duckdb_filter_expression->ToString());
      }
    }

    // Digest the pushed-down filter once: which pure-filter string columns a
    // dictionary-compressed source could answer without decoding them, and
    // which conjuncts a decoder could use to drop rows. This records only what
    // the scan is WILLING to accept; whether a given batch actually took the
    // offer is decided per batch (post_filter_and_project).
    //
    // Pure-filter columns only for the equality answers: the answer REPLACES
    // the column's values, which a projected column could not survive.
    if (_duckdb_filter_expression) {
      std::unordered_set<std::size_t> filter_only;
      std::unordered_set<std::size_t> answerable_positions;
      for (auto const batch_pos : _plan->pure_filter_batch_positions()) {
        auto const primary_idx = _plan->data_columns.at(batch_pos).primary_idx;
        if (_plan->partition_primary_indices.count(primary_idx)) { continue; }
        filter_only.insert(primary_idx);
      }
      _filter_analysis = sirius::op::analyze_scan_filters(*bind.table_filters,
                                                          bind.column_ids,
                                                          bind.returned_types,
                                                          _plan->partition_primary_indices,
                                                          filter_only);
      for (auto const batch_pos : _plan->pure_filter_batch_positions()) {
        auto const primary_idx = _plan->data_columns.at(batch_pos).primary_idx;
        if (_filter_analysis.equality_sets.count(primary_idx)) {
          answerable_positions.insert(batch_pos);
        }
      }
      // The residual the scan evaluates after the decode, decomposed now so no
      // batch has to rebuild a filter expression: each conjunct is lowered to
      // Sirius AST once, and a conjunct whose column can arrive as the answer
      // records where to read it instead.
      _residual = sirius::op::residual_filter(
        sirius::op::decompose_table_filters(*bind.table_filters,
                                            bind.column_ids,
                                            bind.returned_types,
                                            _plan->batch_position_by_column_id,
                                            _plan->partition_primary_indices),
        answerable_positions);
    }
  }

  // Shared reader options — column projection only. set_filter is never applied
  // here: it is a per-split decision (FLBA files disable it) made in
  // materialize_table on a copy of these options.
  _reader_options = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());
  // Never hand cuDF an empty column list — a zero-column read over live row groups
  // hangs. is_projected() already excludes it; this pins the invariant here.
  if (_plan->is_projected() && !_plan->data_columns.empty()) {
    _reader_options->set_column_names(_plan->data_column_names());
  }

  _sirius_dynamic_filters = bind.sirius_dynamic_filters;

  // Hive-partition columns are path-derived constants, not decoded parquet columns, so they must
  // not receive post-decode dynamic filters.
  if (_sirius_dynamic_filters && _plan->has_partitions()) {
    std::vector<std::size_t> partition_cols;
    for (std::size_t i = 0; i < _plan->output_layout.size(); ++i) {
      if (_plan->output_layout[i].source == scan_plan::output_entry::PARTITION) {
        partition_cols.push_back(i);
      }
    }
    _sirius_dynamic_filters->ignore_columns(partition_cols);
  }

  _file_paths = bind.resolved_file_paths;
}

parquet_gpu_ingestible::~parquet_gpu_ingestible() = default;

//===----------------------------------------------------------------------===//
// coalescer / post-filter factories
//===----------------------------------------------------------------------===//
std::unique_ptr<batch_coalescer> parquet_gpu_ingestible::create_batch_coalescer() const
{
  return std::make_unique<parquet_batch_coalescer>(
    _info->approximate_batch_size, _reader_options, _plan);
}

//===----------------------------------------------------------------------===//
// split-provider interface
//===----------------------------------------------------------------------===//
bool parquet_gpu_ingestible::has_processed_all_metadata() const
{
  return _next_file_idx.load(std::memory_order_relaxed) >= _file_paths.size();
}

std::function<std::unique_ptr<op::scan::scan_info>()> parquet_gpu_ingestible::next_split_provider(
  io::ioctx_resolver resolve)
{
  if (!resolve) { throw std::runtime_error("parquet_gpu_ingestible: no scan_manager is wired."); }
  auto const idx = _next_file_idx.fetch_add(1, std::memory_order_relaxed);
  if (idx >= _file_paths.size()) { return nullptr; }  // lost the race for the final file

  // Route each file to its own backend (s3:// -> rest, local -> uring/kvikio) so a
  // mixed-scheme scan opens every file on the right ioctx.  One metadata-scan task
  // per file; row-group chunking and file bundling happen downstream in
  // parquet_batch_coalescer.
  auto const& file_path = _file_paths[idx];
  // The resolver returns a valid ioctx or throws if no backend supports the path.
  auto io_ctx = resolve(file_path);
  return [this, file_path, io_ctx = std::move(io_ctx)]() -> std::unique_ptr<scan_info> {
    return build_file_scan_info(file_path, io_ctx);
  };
}

//===----------------------------------------------------------------------===//
// build_file_scan_info — per-file footer read + row-group pruning
//===----------------------------------------------------------------------===//
std::unique_ptr<scan_info> parquet_gpu_ingestible::build_file_scan_info(
  std::string const& file_path, std::shared_ptr<io::sirius_ioctx> const& io_ctx)
{
  auto stream = cudf::get_default_stream();

  // Resolve the file to a sirius_datasource (own io backend, prefetch cache and
  // cached metadata). The parquet_footer_probe hint collapses the S3 footer read
  // to one suffix-range GET that resolves the size and stashes the footer, so
  // cuDF's footer reads are served locally (no HEAD, no separate trailer/body
  // GETs). Fall back to a plain cudf datasource only for local paths no sirius
  // backend claims.
  std::shared_ptr<io::sirius_datasource> sirius_ds =
    io_ctx->open_datasource(file_path, io::open_hint::parquet_footer_probe);
  if (!sirius_ds && has_uri_scheme(file_path)) {
    throw std::runtime_error("[parquet_gpu_ingestible] no backend supports path: " + file_path);
  }

  // Local copy of the shared options; the per-file filter pushdown decision is
  // applied here, never on _reader_options.
  auto opts = *_reader_options;

  // Obtain footer metadata — from the datasource's cached parquet_metadata when
  // present, else by fetching and parsing the footer.
  std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
  if (sirius_ds) {
    if (auto cached = sirius_ds->metadata()) {
      if (auto pm = std::dynamic_pointer_cast<parquet_metadata>(std::move(cached))) {
        file_metadata = pm->file_metadata();
      }
    }
  }
  if (!file_metadata) {
    auto footer           = cudf::io::parquet::fetch_footer_to_host(*sirius_ds);
    auto const footer_len = footer->size();
    hybrid_scan_reader footer_reader(cudf::host_span<uint8_t const>(footer->data(), footer->size()),
                                     opts);
    file_metadata =
      std::make_shared<cudf::io::parquet::FileMetaData const>(footer_reader.parquet_metadata());
    // Park the parse in the ioctx metadata store so a later scan of the same
    // file skips the footer fetch + Thrift parse (the read above already
    // dereferences *sirius_ds, so it is non-null here). Best-effort.
    [[maybe_unused]] auto const stored =
      sirius_ds->store_metadata(std::make_shared<parquet_metadata>(file_metadata, footer_len));
  }
  auto const& metadata = *file_metadata;

  // FLBA-decimal pushdown probe: cudf's row-group stats filter cannot compare a
  // fixed_point_scalar AST literal against FLBA / BYTE_ARRAY decimal stats, so
  // reader-side pushdown is disabled when such a decimal is among the columns
  // this scan reads (the filter still applies post-decode).
  bool const restrict_to_scanned = _plan->is_projected();
  std::unordered_set<std::string> scanned_column_names;
  if (restrict_to_scanned) {
    auto const names = _plan->data_column_names();
    scanned_column_names.insert(names.begin(), names.end());
  }
  bool disable_filter_pushdown = false;
  for (auto const& elem : metadata.schema) {
    if (restrict_to_scanned && !scanned_column_names.contains(elem.name)) { continue; }
    bool const is_decimal = (elem.converted_type.has_value() &&
                             *elem.converted_type == cudf::io::parquet::ConvertedType::DECIMAL) ||
                            (elem.logical_type.has_value() &&
                             elem.logical_type->type == cudf::io::parquet::LogicalType::DECIMAL);
    if (!is_decimal) { continue; }
    if (elem.type == cudf::io::parquet::Type::FIXED_LEN_BYTE_ARRAY ||
        elem.type == cudf::io::parquet::Type::BYTE_ARRAY) {
      disable_filter_pushdown = true;
      break;
    }
  }

  // Translate the filter for reader-side row-group pruning unless disabled. The
  // translated cuDF AST must outlive filter_row_groups_with_stats below.
  // Pushes _static_pushdown_expression, not the full predicate.
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  if (_static_pushdown_expression && !disable_filter_pushdown) {
    auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
      return _plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_static_pushdown_expression);
    D_ASSERT(sirius_filter_ast != nullptr);
    ast_expression = translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
    if (ast_expression) { opts.set_filter(ast_expression->back()); }
  }

  hybrid_scan_reader reader(metadata, opts);

  // Per-file leaf-column selection for byte accounting. Pure-filter columns are
  // part of the decode working set but not the projected-column estimate.

  // DuckDB schema types (P-space), indexed by scan_plan::data_column::primary_idx.
  // Used below to estimate the decoded (GPU-resident) byte size of each projected
  // column when partitioning row groups into batches — see rg_contribution.
  auto const& returned_types   = _info->returned_types;
  auto const data_column_names = _plan->data_column_names();
  std::vector<std::size_t> selected_chunk_indices;
  // Parallel to selected_chunk_indices: the decoded (GPU) byte width of each
  // selected leaf chunk's column, or 0 for VARCHAR / nested / unknown types
  // (which fall back to the parquet encoded-uncompressed size in rg_contribution).
  std::vector<std::size_t> selected_chunk_decoded_width;
  std::unordered_set<std::size_t> pure_filter_chunk_indices;
  if (_plan->is_projected()) {
    auto const pure_filter_positions = _plan->pure_filter_batch_positions();
    bool has_data_output             = false;
    for (auto const& output : _plan->output_layout) {
      if (output.source == scan_plan::output_entry::DATA) {
        has_data_output = true;
        break;
      }
    }
    selected_chunk_indices.reserve(data_column_names.size());
    selected_chunk_decoded_width.reserve(data_column_names.size());
    for (std::size_t k = 0; k < data_column_names.size(); ++k) {
      auto leaves = detail::leaf_indices_for_column(metadata, data_column_names[k]);
      if (leaves.empty()) {
        throw std::runtime_error("[parquet_gpu_ingestible] Projected column '" +
                                 data_column_names[k] +
                                 "' not found in parquet file: " + file_path);
      }
      // Decoded byte width for this data column: fixed-width types use their
      // cuDF decoded width; VARCHAR (fixed_width_byte_size()==0) and nested
      // types (which throw) get 0, signalling rg_contribution to fall back to
      // the encoded-uncompressed byte size.
      std::size_t decoded_width = 0;
      if (k < _plan->data_columns.size()) {
        auto const primary_idx = _plan->data_columns[k].primary_idx;
        if (primary_idx < returned_types.size()) {
          try {
            decoded_width = returned_types[primary_idx].fixed_width_byte_size();
          } catch (...) {
            decoded_width = 0;  // VARCHAR/LIST/STRUCT/etc — fall back to encoded size
          }
        }
      }
      // When no data column is projected, use the decoded columns as a nonzero
      // history basis. This covers count-style and partition-only outputs.
      bool const is_pure_filter = has_data_output && pure_filter_positions.count(k);
      for (auto const leaf : leaves) {
        selected_chunk_indices.push_back(leaf);
        selected_chunk_decoded_width.push_back(decoded_width);
        if (is_pure_filter) { pure_filter_chunk_indices.insert(leaf); }
      }
    }
  }

  auto row_group_indices = reader.all_row_groups(opts);
  if (ast_expression && !disable_filter_pushdown) {
    auto const rgs_before = row_group_indices.size();
    row_group_indices     = reader.filter_row_groups_with_stats(row_group_indices, opts, stream);
    SIRIUS_LOG_DEBUG("[parquet_gpu_ingestible] Row group pruning {}: {} -> {} row group(s)",
                     file_path,
                     rgs_before,
                     row_group_indices.size());
  }

  // Prune from null_count for the conjuncts cuDF could not take. `IS NULL`
  // excludes a row group whose column has no nulls; `IS NOT NULL` excludes one
  // that is entirely null. Both are exact, so this loses no matching rows.
  //
  // null_count is optional in the parquet spec — an absent value means unknown,
  // never zero, so the row group is kept.
  if (!_null_prune_predicates.empty() && !row_group_indices.empty()) {
    struct resolved_predicate {
      std::size_t chunk_index;
      bool expects_null;
    };
    std::vector<resolved_predicate> resolved;
    resolved.reserve(_null_prune_predicates.size());
    for (auto const& pred : _null_prune_predicates) {
      // Only a scalar top-level column qualifies. For anything nested, the leaf
      // statistic counts repeated values or leaf-level nulls, neither of which
      // is the top-level column's nullness -- pruning on it can drop row groups
      // that do contain matching rows.
      //
      // The DECLARED type is what decides this, not the parquet encoding. A
      // single leaf does not imply scalar (a one-field struct has one), and
      // neither does a one-component path: legacy parquet allows a top-level
      // REPEATED primitive, whose path is just the column name, and DuckDB
      // surfaces that as a LIST.
      auto const batch_index = static_cast<std::size_t>(pred.batch_index);
      if (batch_index >= _plan->data_columns.size()) { continue; }
      auto const primary_idx = _plan->data_columns[batch_index].primary_idx;
      if (primary_idx >= _info->returned_types.size()) { continue; }
      auto const& column_type = _info->returned_types[primary_idx];
      // MAP is normalized to Sirius LIST by from_duckdb(). DuckDB UNION is not
      // supported by that conversion, so it cannot reach this scan path.
      if (column_type.id() == sirius::type_id::LIST || column_type.id() == sirius::type_id::ARRAY ||
          column_type.id() == sirius::type_id::STRUCT) {
        continue;
      }

      auto const leaves =
        detail::leaf_indices_for_column(metadata, _plan->batch_column_name(pred.batch_index));
      if (leaves.size() != 1) { continue; }
      auto const chunk_index = leaves.front();
      auto const& first_rg   = metadata.row_groups.front();
      if (chunk_index >= first_rg.columns.size() ||
          first_rg.columns[chunk_index].meta_data.path_in_schema.size() != 1) {
        continue;
      }
      resolved.push_back({chunk_index, pred.expects_null});
    }

    if (!resolved.empty()) {
      auto const rgs_before = row_group_indices.size();
      auto kept             = decltype(row_group_indices){};
      kept.reserve(row_group_indices.size());
      for (auto const rg_idx : row_group_indices) {
        auto const& rg_meta = metadata.row_groups[static_cast<std::size_t>(rg_idx)];
        bool keep           = true;
        for (auto const& pred : resolved) {
          if (pred.chunk_index >= rg_meta.columns.size()) { continue; }
          auto const& null_count =
            rg_meta.columns[pred.chunk_index].meta_data.statistics.null_count;
          if (!null_count.has_value()) { continue; }
          bool const provably_empty =
            pred.expects_null ? (*null_count == 0) : (*null_count == rg_meta.num_rows);
          if (provably_empty) {
            keep = false;
            break;
          }
        }
        if (keep) { kept.push_back(rg_idx); }
      }
      row_group_indices = std::move(kept);
      if (row_group_indices.size() != rgs_before) {
        SIRIUS_LOG_DEBUG(
          "[parquet_gpu_ingestible] null_count row group pruning {}: {} -> {} row group(s)",
          file_path,
          rgs_before,
          row_group_indices.size());
      }
    }
  }

  struct row_group_size_estimate {
    std::size_t output_bytes         = 0;
    std::size_t decode_working_bytes = 0;
    std::size_t compressed_bytes     = 0;
  };

  // Estimate the decoded output and full decode working set for one row group.
  auto rg_contribution = [&](cudf::io::parquet::RowGroup const& row_group) {
    row_group_size_estimate estimate;
    auto const row_count = static_cast<std::size_t>(row_group.num_rows);
    auto add_chunk       = [&](cudf::io::parquet::ColumnChunk const& chunk,
                         bool is_pure_filter,
                         std::size_t decoded_width) {
      auto const& column_metadata = chunk.meta_data;
      std::size_t decoded_bytes   = 0;
      if (decoded_width > 0) {
        // Fixed-width column: row_count x decoded width, plus a validity mask.
        decoded_bytes = row_count * decoded_width + row_count / 8;
      } else {
        // VARCHAR / nested / unknown. Dictionary/RLE encoding can make the
        // encoded chunk many times smaller than its decoded char buffer, so
        // prefer SizeStatistics::unencoded_byte_array_data_bytes (the exact
        // decoded BYTE_ARRAY size) when the writer recorded it, else fall back
        // to the encoded-uncompressed size (under-counts dictionary data).
        std::size_t const char_bytes =
          (column_metadata.size_statistics &&
           column_metadata.size_statistics->unencoded_byte_array_data_bytes)
                  ? static_cast<std::size_t>(
                *column_metadata.size_statistics->unencoded_byte_array_data_bytes)
                  : static_cast<std::size_t>(column_metadata.total_uncompressed_size);
        // Plus the cuDF string column's offsets (one int32 per row) and validity.
        decoded_bytes = char_bytes + row_count * sizeof(std::uint32_t) + row_count / 8;
      }
      estimate.decode_working_bytes += decoded_bytes;
      if (!is_pure_filter) { estimate.output_bytes += decoded_bytes; }
      estimate.compressed_bytes += static_cast<std::size_t>(column_metadata.total_compressed_size);
    };
    if (_plan->is_projected()) {
      for (std::size_t i = 0; i < selected_chunk_indices.size(); ++i) {
        auto const chunk_idx = selected_chunk_indices[i];
        add_chunk(row_group.columns[chunk_idx],
                  pure_filter_chunk_indices.contains(chunk_idx),
                  selected_chunk_decoded_width[i]);
      }
    } else if (returned_types.size() == row_group.columns.size()) {
      // Unprojected (identity) scan: the reader materializes every file column
      // in order, so column ci aligns 1:1 with returned_types[ci]. Estimate
      // decoded bytes per column the same way as the projected path — fixed
      // widths from the type, VARCHAR/nested falling back to encoded size.
      for (std::size_t ci = 0; ci < row_group.columns.size(); ++ci) {
        std::size_t decoded_width = 0;
        try {
          decoded_width = returned_types[ci].fixed_width_byte_size();
        } catch (...) {
          decoded_width = 0;
        }
        add_chunk(row_group.columns[ci], /*is_pure_filter=*/false, decoded_width);
      }
    } else {
      // Column count does not match returned_types (cannot safely align types to
      // chunks): keep the original parquet encoded-uncompressed sizing.
      for (auto const& chunk : row_group.columns) {
        auto const uncompressed = static_cast<std::size_t>(chunk.meta_data.total_uncompressed_size);
        estimate.output_bytes += uncompressed;
        estimate.decode_working_bytes += uncompressed;
        estimate.compressed_bytes +=
          static_cast<std::size_t>(chunk.meta_data.total_compressed_size);
      }
    }
    return estimate;
  };

  auto out                     = std::make_unique<parquet_file_scan_info>();
  out->file_metadata           = file_metadata;
  out->file_path               = file_path;
  out->datasource              = std::move(sirius_ds);
  out->reader_options          = _reader_options;
  out->disable_filter_pushdown = disable_filter_pushdown;
  out->row_groups.reserve(row_group_indices.size());
  for (auto const rg_idx : row_group_indices) {
    auto const estimate = rg_contribution(metadata.row_groups[rg_idx]);
    out->row_groups.push_back({rg_idx,
                               estimate.output_bytes,
                               estimate.decode_working_bytes,
                               estimate.compressed_bytes,
                               metadata.row_groups[rg_idx].num_rows});
  }

  // Hive partition values for this file, in scan_plan::partition_columns order.
  if (!_plan->partition_columns.empty()) {
    out->partition_values.reserve(_plan->partition_columns.size());
    auto parsed = duckdb::HivePartitioning::Parse(file_path);
    for (auto const& pc : _plan->partition_columns) {
      auto it = parsed.find(pc.name);
      out->partition_values.push_back(it != parsed.end() ? it->second : std::string{});
    }
  }

  return out;
}

//===----------------------------------------------------------------------===//
// materialize_table — ports read_table_from_metadata
//===----------------------------------------------------------------------===//
filtered_table parquet_gpu_ingestible::materialize_metadata_to_table(
  op::scan::scan_info const& info,
  const cucascade::memory::memory_space& mem_space,
  rmm::cuda_stream_view stream)
{
  auto const& split = static_cast<parquet_split_info const&>(info);

  std::vector<std::unique_ptr<cudf::io::datasource>> sources;
  std::vector<cudf::io::parquet::FileMetaData> metadatas;
  std::vector<std::vector<cudf::size_type>> rg_per_src;
  sources.reserve(split.rg_slices.size());
  metadatas.reserve(split.rg_slices.size());
  rg_per_src.reserve(split.rg_slices.size());

  for (auto const& slice : split.rg_slices) {
    if (slice.datasource) {
      sources.push_back(cudf::io::datasource::create(slice.datasource.get()));
    } else {
      sources.push_back(cudf::io::datasource::create(slice.file_path));
    }
    metadatas.push_back(*slice.file_metadata);
    rg_per_src.push_back(slice.row_group_indices);
  }
  // All-pruned fallback split (parquet_batch_coalescer::flush): every slice
  // carries zero row groups.
  // Don't express that via set_row_groups — the meaning of an empty per-source
  // vector has flipped between cudf versions ("all row groups" vs "none").
  // Instead bound the read to zero rows against the footer metadata alone:
  // cudf builds the schema-correct empty table without touching data pages,
  // and it flows through the normal filter / partition / projection assembly
  // below.
  bool const all_slices_pruned =
    !split.rg_slices.empty() &&
    std::all_of(split.rg_slices.begin(), split.rg_slices.end(), [](row_group_slice const& s) {
      return s.row_group_indices.empty();
    });
  auto opts = *split.reader_options;
  if (all_slices_pruned) {
    opts.set_num_rows(0);
  } else {
    opts.set_row_groups(std::move(rg_per_src));
  }

  // Per-task AST translation for reader-side row-group + row pushdown. set_filter
  // is gated on translation success AND on the per-batch disable_filter_pushdown
  // flag (set when the FLBA-decimal probe failed). When pushdown does not engage
  // — disabled, translation fails, or the split is the all-pruned zero-row
  // fallback (zero rows need no reader filter; skipping keeps GPU AST
  // translation off that path) — the row filter is left for
  // post_filter_and_project to apply post-decode. The translated cuDF AST
  // (`ast_expression`) must outlive read_parquet; the borrowed Sirius AST and
  // the translator are only needed during translation.
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  std::optional<gpu_expression_translator::translated_expression> dynamic_ast_expression =
    std::nullopt;
  cudf::ast::expression const* reader_filter_root = nullptr;

  // Null-free conjuncts only; the dynamic-filter block below is unaffected.
  if (_static_pushdown_expression && !split.disable_filter_pushdown && !all_slices_pruned) {
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_static_pushdown_expression);
    D_ASSERT(sirius_filter_ast != nullptr);
    auto name_resolver = [plan = split.plan](duckdb::idx_t ref_index) -> std::string {
      return plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    ast_expression = translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
    if (ast_expression) { reader_filter_root = &ast_expression->back(); }
  }

  if (!split.disable_filter_pushdown && _sirius_dynamic_filters &&
      _sirius_dynamic_filters->has_filters()) {
    if (ast_expression) {
      reader_filter_root = merge_dynamic_filters_into_ast(ast_expression->tree,
                                                          reader_filter_root,
                                                          *_sirius_dynamic_filters,
                                                          *split.plan,
                                                          mem_space.get_device_id());
    } else {
      dynamic_ast_expression.emplace();
      reader_filter_root = merge_dynamic_filters_into_ast(dynamic_ast_expression->tree,
                                                          /*existing_root=*/nullptr,
                                                          *_sirius_dynamic_filters,
                                                          *split.plan,
                                                          mem_space.get_device_id());
      if (!reader_filter_root) { dynamic_ast_expression.reset(); }
    }
  }

  if (reader_filter_root) { opts.set_filter(*reader_filter_root); }

  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());
  auto [table, _] =
    cudf::io::read_parquet(std::move(sources), std::move(metadatas), opts, stream, mr_ref);

  // Hive-partition scans assemble inline here: partition_values are per-split
  // (carried on parquet_split_info) and do not travel to the pipeline-shared
  // post_filter info. Apply the row filter first when pushdown did not, then
  // inject the partition columns and project to the output layout, so the
  // result is fully ROW_FILTERED_AND_PROJECTED and post_filter_and_project is
  // skipped. `sirius_filter_ast` must outlive `exec` — the evaluator borrows it.

  // The reader discharged the row filter only if it pushed the WHOLE predicate.
  // After a partial push the dropped conjuncts still have to be applied below.
  bool const reader_applied_full_filter =
    ast_expression.has_value() && _static_pushdown_is_complete;

  if (_plan->has_partitions()) {
    owning_table_view view{std::move(table)};
    if (!reader_applied_full_filter && _duckdb_filter_expression) {
      auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
      sirius::expression_evaluator exec(sirius_filter_ast.get(), mr_ref, stream);
      auto const data_positions = output_data_positions(*_plan);
      view = data_positions.empty() ? owning_table_view{exec.select(view.view())}
                                    : owning_table_view{exec.select(view.view(), data_positions)};
    }
    auto assembled = assemble_scan_output(*_plan, std::move(view), split.partition_values, stream);
    return op::scan::filtered_table{std::move(assembled),
                                    op::scan::filter_state::ROW_FILTERED_AND_PROJECTED};
  }

  auto const state = reader_applied_full_filter ? op::scan::filter_state::ROW_FILTERED
                                                : op::scan::filter_state::UNFILTERED;
  return op::scan::filtered_table{owning_table_view{std::move(table)}, state};
}

//===----------------------------------------------------------------------===//
// post_filter_and_project — post-decode filter + non-partition projection
//===----------------------------------------------------------------------===//
// Hive-partition scans are fully assembled in materialize_table (it owns the
// per-split partition values) and return ROW_FILTERED_AND_PROJECTED, so they
// never reach here. This path therefore only applies a pending row filter and a
// non-partition projection; partition injection is unreachable.
std::unique_ptr<cudf::table> parquet_gpu_ingestible::post_filter_and_project(
  filtered_table&& input,
  ::cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream)
{
  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());

  // Apply the row filter post-decode when materialization did not — reader-side
  // pushdown was disabled (FLBA-decimal file) or AST translation failed. A
  // ROW_FILTERED / ROW_FILTERED_AND_PROJECTED state means the reader already
  // applied it. `sirius_filter_ast` must outlive `exec` — the evaluator only
  // borrows the AST.
  if (input.state != filter_state::ROW_FILTERED &&
      input.state != filter_state::ROW_FILTERED_AND_PROJECTED && !_residual.empty()) {
    // The decoder may have answered some pure-filter columns for this batch.
    // Where it also APPLIED those answers, the surviving rows already satisfy
    // the conjunct and it leaves the residual entirely; where it only answered
    // them, the conjunct becomes a reference to the BOOL8 rather than a
    // re-comparison. The residual owns the per-conjunct forms and must outlive
    // `exec`, which only borrows the AST.
    auto sirius_filter_ast = _residual.against(input.predicate_columns, input.predicates_enforced);
    if (sirius_filter_ast) {
      sirius::expression_evaluator exec(sirius_filter_ast.get(), mr_ref, stream);
      auto const data_positions = output_data_positions(*_plan);
      auto filtered             = data_positions.empty() ? exec.select(input.table.view())
                                                         : exec.select(input.table.view(), data_positions);
      // The select only enqueued its reads; record before the reassignment drops the read lock.
      input.table.record_reader_event(stream);
      input = filtered_table{owning_table_view{std::move(filtered)}, filter_state::ROW_FILTERED};
      SIRIUS_LOG_DEBUG(
        "[parquet_gpu_ingestible::post_filter_and_project] Applied the residual filter "
        "post-decode.");
    } else {
      // Nothing left to evaluate: the decode enforced every conjunct of this
      // scan's filter, so these rows are already the answer. NOT "no filtering
      // needed" — the rows were filtered, just not here.
      input.state = filter_state::ROW_FILTERED;
      SIRIUS_LOG_DEBUG(
        "[parquet_gpu_ingestible::post_filter_and_project] The decode applied every conjunct; "
        "no residual filter to run.");
    }
  }

  // Project / reorder the reader's D-order batch to the plan's output layout
  // (non-owning select_columns, no GPU copy). No partitions reach this path, so
  // partition_values is unused. The release below moves the surviving column
  // buffers out.
  auto assembled =
    assemble_scan_output(*_plan, std::move(input.table), /*partition_values=*/{}, stream);
  SIRIUS_LOG_DEBUG(
    "[parquet_gpu_ingestible::post_filter_and_project] Assembled scan output to plan layout.");
  return assembled.release(stream, mr_ref);
}

bool parquet_gpu_ingestible::output_assembly_is_leading_identity() const noexcept
{
  // !needs_output_assembly means assemble_scan_output is a pass-through: no
  // partition columns to synthesize and output_layout reads data columns
  // 0..N-1 in order. (Its other false case, the empty count(*) layout, can
  // never reach the transactional steal: such scans have no carrier targets.)
  return !needs_output_assembly(*_plan);
}

//===----------------------------------------------------------------------===//
// materialized_column_order
//===----------------------------------------------------------------------===//
std::vector<std::size_t> parquet_gpu_ingestible::materialized_column_order() const
{
  // The reader materializes columns in _plan->data_columns order (output columns first,
  // pure-filter columns trailing; partition/virtual excluded) — exactly the layout
  // post_filter_and_project's filter refs (batch_position_by_column_id) and output_layout
  // assume. Expose it as primary/storage indices for the pinned-cache path.
  std::vector<std::size_t> order;
  order.reserve(_plan->data_columns.size());
  for (auto const& dc : _plan->data_columns) {
    order.push_back(dc.primary_idx);
  }
  return order;
}

}  // namespace sirius::op::scan
