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
#include <op/dynamic_filter/dynamic_filter_stats.hpp>
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
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

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
    }
  }

  // Shared reader options — column projection only. set_filter is never applied
  // here: it is a per-split decision (BYTE_ARRAY-decimal files disable it) made in
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

  // BYTE_ARRAY-decimal pushdown probe: reader-side pushdown is disabled when a decimal stored in
  // the variable-length BYTE_ARRAY physical type is among the columns this scan reads (the filter
  // still applies post-decode). This probe once covered FIXED_LEN_BYTE_ARRAY decimals as well,
  // because cudf's row-group stats filter threw "Invalid type and stats combination" when
  // comparing a fixed_point_scalar AST literal against them. At the pinned cudf that comparison
  // succeeds and prunes correctly at each stored width, including negative values and nulls, so
  // those decimals are pushed down -- which matters because DuckDB stores every DECIMAL wider
  // than 18 digits as FIXED_LEN_BYTE_ARRAY and Arrow-based writers store every decimal that way.
  // `test/cpp/scan/test_parquet_decimal_pushdown.cpp` covers the surviving row groups at widths
  // 4, 8 and 16, and `test/sql/parquet_decimal_pushdown.test` covers the values they decode to.
  //
  // Two consequences worth stating. Neither the stats filter below nor the read in
  // materialize_metadata_to_table is wrapped in a try/catch, so while it was in force this probe
  // also served as an exception net: anything the reader threw on a FIXED_LEN_BYTE_ARRAY decimal
  // propagated out of the scan instead of being caught, and that is now the behaviour for those
  // files. And BYTE_ARRAY decimals stay disabled only because no writer available here emits
  // them, so the fix could not be confirmed for that encoding; they are rare enough that leaving
  // pushdown off costs nothing.
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
    if (elem.type == cudf::io::parquet::Type::BYTE_ARRAY) {
      disable_filter_pushdown = true;
      break;
    }
  }

  // Translate the filter for reader-side row-group pruning unless disabled. The
  // translated cuDF AST must outlive filter_row_groups_with_stats below.
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  if (_duckdb_filter_expression && !disable_filter_pushdown) {
    auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
      return _plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
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
  // flag (set when the BYTE_ARRAY-decimal probe matched). When pushdown does not engage
  // — disabled, translation fails, or the split is the all-pruned zero-row
  // fallback (zero rows need no reader filter; skipping keeps GPU AST
  // translation off that path) — the row filter is left for
  // post_filter_and_project to apply post-decode. The translated cuDF AST
  // (`ast_expression`) must outlive read_parquet; the borrowed Sirius AST and
  // the translator are only needed during translation.
  //
  // `filters_snapshot` must outlive read_parquet for the same reason, and is declared here rather
  // than where it is taken so that it does. A tree owns its AST nodes, but the device scalars a
  // dynamic filter's literals point at are owned by the filter, and the snapshot's owning copies
  // are what keep a *superseded* filter alive while a consumer still references it. A refinement
  // slot replaces its filter on every accepted revision, so releasing the snapshot before the
  // reader has finished walking the AST can drop the last owner of the filter whose scalars
  // `opts` still points at -- which cuDF then dereferences on the device. Append-only join
  // filters never expose this, because the channel keeps its own reference forever. The snapshot
  // is declared before the two AST trees so reverse destruction order alone destroys every
  // literal-bearing tree before the scalars' last owner: the outlives-referents property holds
  // through destruction by construction, not by relying on the trees' destructors dereferencing
  // nothing.
  sirius::op::dynamic_filter_snapshot filters_snapshot;
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  std::optional<gpu_expression_translator::translated_expression> dynamic_ast_expression =
    std::nullopt;
  cudf::ast::expression const* reader_filter_root = nullptr;

  if (_duckdb_filter_expression && !split.disable_filter_pushdown && !all_slices_pruned) {
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    D_ASSERT(sirius_filter_ast != nullptr);
    auto name_resolver = [plan = split.plan](duckdb::idx_t ref_index) -> std::string {
      return plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    ast_expression = translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
    if (ast_expression) { reader_filter_root = &ast_expression->back(); }
  }

  bool dynamic_filters_merged    = false;
  std::uint64_t merge_generation = 0;

  if (!split.disable_filter_pushdown && !all_slices_pruned && _sirius_dynamic_filters &&
      _sirius_dynamic_filters->has_filters()) {
    // WI-0b: the dynamic merge -- never the static predicate -- is gated per scan by the
    // reader_pruning_gate on observed row-group pruning. The advisory generation() feeds only
    // the gate check; the predicate is still built from one coherent snapshot, whose generation
    // tags the sample.
    if (_reader_gate.applicable(_sirius_dynamic_filters->generation())) {
      // One coherent snapshot per checkpoint: taken after the lock-free fast path, before any
      // other lock, and used for every predicate built for this split.
      filters_snapshot           = _sirius_dynamic_filters->snapshot();
      merge_generation           = filters_snapshot.generation;
      auto const* pre_merge_root = reader_filter_root;
      if (ast_expression) {
        reader_filter_root = merge_dynamic_filters_into_ast(ast_expression->tree,
                                                            reader_filter_root,
                                                            filters_snapshot,
                                                            *split.plan,
                                                            mem_space.get_device_id());
      } else {
        dynamic_ast_expression.emplace();
        reader_filter_root = merge_dynamic_filters_into_ast(dynamic_ast_expression->tree,
                                                            /*existing_root=*/nullptr,
                                                            filters_snapshot,
                                                            *split.plan,
                                                            mem_space.get_device_id());
        if (!reader_filter_root) { dynamic_ast_expression.reset(); }
      }
      dynamic_filters_merged = reader_filter_root != pre_merge_root;
    } else if (_info->stats != nullptr) {
      _info->stats->reader_gate_merges_skipped.fetch_add(1, std::memory_order_relaxed);
    }
  }

  if (reader_filter_root) { opts.set_filter(*reader_filter_root); }

  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());
  auto [table, read_metadata] =
    cudf::io::read_parquet(std::move(sources), std::move(metadatas), opts, stream, mr_ref);

  // WI-0b sample: only splits whose reader AST carried merged dynamic conjuncts are evidence,
  // and only when the reader reported its row-group accounting. Attribution note: the split's
  // row groups were already pruned against the static conjuncts at metadata time, so stats-stage
  // pruning here is the dynamic filters'; bloom-stage over-attribution only keeps the gate on.
  if (dynamic_filters_merged) {
    auto const& m                                  = read_metadata;
    std::optional<cudf::size_type> const remaining = m.num_row_groups_after_bloom_filter.has_value()
                                                       ? m.num_row_groups_after_bloom_filter
                                                       : m.num_row_groups_after_stats_filter;
    if (remaining.has_value()) {
      _reader_gate.record_sample(static_cast<std::size_t>(m.num_input_row_groups),
                                 static_cast<std::size_t>(*remaining),
                                 merge_generation,
                                 _info->stats);
    }
  }

  // Hive-partition scans assemble inline here: partition_values are per-split
  // (carried on parquet_split_info) and do not travel to the pipeline-shared
  // post_filter info. Apply the row filter first when pushdown did not, then
  // inject the partition columns and project to the output layout, so the
  // result is fully ROW_FILTERED_AND_PROJECTED and post_filter_and_project is
  // skipped. `sirius_filter_ast` must outlive `exec` — the evaluator borrows it.
  if (_plan->has_partitions()) {
    owning_table_view view{std::move(table)};
    if (!ast_expression.has_value() && _duckdb_filter_expression) {
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

  auto const state = ast_expression.has_value() ? op::scan::filter_state::ROW_FILTERED
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
  // pushdown was disabled (BYTE_ARRAY-decimal file) or AST translation failed. A
  // ROW_FILTERED / ROW_FILTERED_AND_PROJECTED state means the reader already
  // applied it. `sirius_filter_ast` must outlive `exec` — the evaluator only
  // borrows the AST.
  if (input.state != filter_state::ROW_FILTERED &&
      input.state != filter_state::ROW_FILTERED_AND_PROJECTED && _duckdb_filter_expression) {
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    sirius::expression_evaluator exec(sirius_filter_ast.get(), mr_ref, stream);
    auto const data_positions = output_data_positions(*_plan);
    auto filtered             = data_positions.empty() ? exec.select(input.table.view())
                                                       : exec.select(input.table.view(), data_positions);
    input = filtered_table{owning_table_view{std::move(filtered)}, filter_state::ROW_FILTERED};
    SIRIUS_LOG_DEBUG(
      "[parquet_gpu_ingestible::post_filter_and_project] Applied duckdb filter expression "
      "post-decode.");
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
