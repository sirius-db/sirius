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
#include <expression_executor/gpu_expression_executor.hpp>
#include <expression_executor/gpu_expression_translator_internal.hpp>
#include <io/io_context.hpp>
#include <io/sirius_datasource.hpp>
#include <log/logging.hpp>
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

// uring_reactor MUST be included last among sirius headers — see
// parquet_split_provider.cpp for the BLOCK_SIZE macro-collision rationale.
#include <io/uring/uring_reactor.hpp>

// standard library
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

bool has_uri_scheme(std::string const& p) { return p.find("://") != std::string::npos; }

//===----------------------------------------------------------------------===//
// parquet_batch_coalecer
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
class parquet_batch_coalecer : public batch_coalecer {
 public:
  parquet_batch_coalecer(std::size_t cap,
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

    if (!_slices.empty() && (_partition_values != file->partition_values ||
                             _disable_pushdown != file->disable_filter_pushdown)) {
      emitted.push_back(emit_current());
    }
    _partition_values = file->partition_values;
    _disable_pushdown = file->disable_filter_pushdown;

    std::vector<cudf::size_type> cur_rgs;
    std::size_t cur_unc  = 0;
    std::size_t cur_comp = 0;
    auto seal_file       = [&]() {
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
                           cur_unc,
                           cur_comp,
                           std::move(slice_ds));
      _acc_bytes += cur_unc;
      cur_rgs.clear();
      cur_unc  = 0;
      cur_comp = 0;
    };

    for (auto const& rg : file->row_groups) {
      if ((!_slices.empty() || !cur_rgs.empty()) && _cap > 0 &&
          _acc_bytes + cur_unc + rg.uncompressed_bytes > _cap) {
        seal_file();
        emitted.push_back(emit_current());
      }
      cur_unc += rg.uncompressed_bytes;
      cur_comp += rg.compressed_bytes;
      cur_rgs.push_back(rg.index);
    }
    seal_file();
    return emitted;
  }

  std::vector<std::unique_ptr<scan_info>> flush() override
  {
    std::vector<std::unique_ptr<scan_info>> out;
    if (!_slices.empty()) { out.push_back(emit_current()); }
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
    _acc_bytes = 0;
    return split;
  }

  const std::size_t _cap;
  std::shared_ptr<cudf::io::parquet_reader_options> _reader_options;
  std::shared_ptr<scan_plan const> _plan;
  const bool _needs_assembly;

  std::vector<row_group_slice> _slices;
  std::size_t _acc_bytes = 0;
  std::vector<std::string> _partition_values;
  bool _disable_pushdown = false;
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

//===----------------------------------------------------------------------===//
// scan_info fadvise_entries — prefetch byte ranges
//===----------------------------------------------------------------------===//
std::vector<scan_info::fadvise_entry> parquet_file_scan_info::fadvise_entries() const
{
  if (!datasource || !file_metadata || !reader_options) { return {}; }
  std::vector<cudf::size_type> rg_indices;
  rg_indices.reserve(row_groups.size());
  for (auto const& rg : row_groups) {
    rg_indices.push_back(rg.index);
  }
  auto ranges = column_chunk_ranges(*file_metadata, *reader_options, rg_indices);
  if (ranges.empty()) { return {}; }
  fadvise_entry entry;
  entry.datasource = datasource;
  entry.ranges     = std::move(ranges);
  std::vector<fadvise_entry> out;
  out.push_back(std::move(entry));
  return out;
}

std::vector<scan_info::fadvise_entry> parquet_split_info::fadvise_entries() const
{
  if (!reader_options) { return {}; }
  std::vector<fadvise_entry> entries;
  entries.reserve(rg_slices.size());
  for (auto const& slice : rg_slices) {
    if (!slice.datasource || !slice.file_metadata) { continue; }
    auto ranges =
      column_chunk_ranges(*slice.file_metadata, *reader_options, slice.row_group_indices);
    if (ranges.empty()) { continue; }
    fadvise_entry entry;
    entry.datasource = slice.datasource;
    entry.ranges     = std::move(ranges);
    entries.push_back(std::move(entry));
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

  // Any non-trivial scan shape — reader-side projection, filter pushdown, or hive-partition
  // injection — needs column names. Matches parquet_split_provider's ctor invariant.
  bool const needs_names = !bind.projection_ids.empty() ||
                           (bind.table_filters && !bind.table_filters->filters.empty()) ||
                           !bind.partition_indices.empty();
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
    if (duckdb_expression) { _duckdb_filter_expression = std::move(duckdb_expression); }
  }

  // Shared reader options — column projection only. set_filter is never applied
  // here: it is a per-split decision (FLBA files disable it) made in
  // materialize_table on a copy of these options.
  _reader_options = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());
  if (_plan->is_projected()) { _reader_options->set_column_names(_plan->data_column_names()); }

  _file_paths = bind.resolved_file_paths;
}

parquet_gpu_ingestible::~parquet_gpu_ingestible() = default;

//===----------------------------------------------------------------------===//
// coalescer / post-filter factories
//===----------------------------------------------------------------------===//
std::unique_ptr<batch_coalecer> parquet_gpu_ingestible::create_batch_coalecer() const
{
  return std::make_unique<parquet_batch_coalecer>(
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
  std::shared_ptr<io::sirius_ioctx> io_ctx)
{
  if (io_ctx == nullptr) {
    throw std::runtime_error("parquet_gpu_ingestible: no scan_manager is wired.");
  }
  auto const idx = _next_file_idx.fetch_add(1, std::memory_order_relaxed);
  if (idx >= _file_paths.size()) { return nullptr; }  // lost the race for the final file

  // One metadata-scan task per file. Row-group chunking and file bundling happen
  // downstream in parquet_batch_coalecer.
  return [this, file_path = _file_paths[idx], io_ctx = std::move(io_ctx)]()
           -> std::unique_ptr<scan_info> { return build_file_scan_info(file_path, io_ctx); };
}

//===----------------------------------------------------------------------===//
// build_file_scan_info — per-file footer read + row-group pruning
//===----------------------------------------------------------------------===//
std::unique_ptr<scan_info> parquet_gpu_ingestible::build_file_scan_info(
  std::string const& file_path, std::shared_ptr<io::sirius_ioctx> const& io_ctx)
{
  auto stream = cudf::get_default_stream();

  // Resolve the file to a sirius_datasource (own io backend, prefetch cache and
  // cached metadata). Fall back to a plain cudf datasource only for local paths
  // no sirius backend claims.
  std::shared_ptr<io::sirius_datasource> sirius_ds = io_ctx->open_datasource(file_path);
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
  // pushdown is disabled for such files (the filter still applies post-decode).
  bool disable_filter_pushdown = false;
  for (auto const& elem : metadata.schema) {
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
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  if (_duckdb_filter_expression && !disable_filter_pushdown) {
    auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
      return _plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    ast_expression = translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
    if (ast_expression) { opts.set_filter(ast_expression->back()); }
  }

  hybrid_scan_reader reader(metadata, opts);

  // Per-file leaf-column selection for byte accounting. Pure-filter columns are
  // read for filter evaluation but excluded from the uncompressed accounting.
  auto const data_column_names = _plan->data_column_names();
  std::vector<std::size_t> selected_chunk_indices;
  std::unordered_set<std::size_t> pure_filter_chunk_indices;
  if (_plan->is_projected()) {
    auto const pure_filter_positions = _plan->pure_filter_batch_positions();
    selected_chunk_indices.reserve(data_column_names.size());
    for (std::size_t k = 0; k < data_column_names.size(); ++k) {
      auto leaves = detail::leaf_indices_for_column(metadata, data_column_names[k]);
      if (leaves.empty()) {
        throw std::runtime_error("[parquet_gpu_ingestible] Projected column '" +
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

  auto row_group_indices = reader.all_row_groups(opts);
  if (ast_expression && !disable_filter_pushdown) {
    auto const rgs_before = row_group_indices.size();
    row_group_indices     = reader.filter_row_groups_with_stats(row_group_indices, opts, stream);
    SIRIUS_LOG_DEBUG("[parquet_gpu_ingestible] Row group pruning {}: {} -> {} row group(s)",
                     file_path,
                     rgs_before,
                     row_group_indices.size());
  }

  auto rg_contribution = [&](cudf::io::parquet::RowGroup const& row_group) {
    std::size_t rg_uncompressed = 0;
    std::size_t rg_compressed   = 0;
    auto add_chunk = [&](cudf::io::parquet::ColumnChunk const& chunk, bool is_pure_filter) {
      auto const& column_metadata = chunk.meta_data;
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
      for (auto const& chunk : row_group.columns) {
        add_chunk(chunk, false);
      }
    }
    return std::pair{rg_uncompressed, rg_compressed};
  };

  auto out                     = std::make_unique<parquet_file_scan_info>();
  out->file_metadata           = file_metadata;
  out->file_path               = file_path;
  out->datasource              = std::move(sirius_ds);
  out->reader_options          = _reader_options;
  out->disable_filter_pushdown = disable_filter_pushdown;
  out->row_groups.reserve(row_group_indices.size());
  for (auto const rg_idx : row_group_indices) {
    auto const [rg_unc, rg_comp] = rg_contribution(metadata.row_groups[rg_idx]);
    out->row_groups.push_back({rg_idx, rg_unc, rg_comp});
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
  auto opts = *split.reader_options;
  opts.set_row_groups(std::move(rg_per_src));

  // Per-task AST translation for reader-side row-group + row pushdown. set_filter
  // is gated on translation success AND on the per-batch disable_filter_pushdown
  // flag (set when the FLBA-decimal probe failed). When pushdown does not engage
  // — disabled, or translation fails — the row filter is left for
  // post_filter_and_project to apply post-decode. The translated cuDF AST
  // (`ast_expression`) must outlive read_parquet; the borrowed Sirius AST and
  // the translator are only needed during translation.
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  if (_duckdb_filter_expression && !split.disable_filter_pushdown) {
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    auto name_resolver     = [plan = split.plan](duckdb::idx_t ref_index) -> std::string {
      return plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    ast_expression = translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
    if (ast_expression) { opts.set_filter(ast_expression->back()); }
  }

  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());
  auto [table, _] =
    cudf::io::read_parquet(std::move(sources), std::move(metadatas), opts, stream, mr_ref);

  // Hive-partition scans assemble inline here: partition_values are per-split
  // (carried on parquet_split_info) and do not travel to the pipeline-shared
  // post_filter info. Apply the row filter first when pushdown did not, then
  // inject the partition columns and project to the output layout, so the
  // result is fully ROW_FILTERED_AND_PROJECTED and post_filter_and_project is
  // skipped. `sirius_filter_ast` must outlive `exec` — the executor borrows it.
  if (_plan->has_partitions()) {
    owning_table_view view{std::move(table)};
    if (!ast_expression.has_value() && _duckdb_filter_expression) {
      auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
      sirius::gpu_expression_executor exec(sirius_filter_ast.get(), mr_ref, stream);
      view = owning_table_view{exec.select(view.view())};
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
  // pushdown was disabled (FLBA-decimal file) or AST translation failed. A
  // ROW_FILTERED / ROW_FILTERED_AND_PROJECTED state means the reader already
  // applied it. `sirius_filter_ast` must outlive `exec` — the executor only
  // borrows the AST.
  if (input.state != filter_state::ROW_FILTERED &&
      input.state != filter_state::ROW_FILTERED_AND_PROJECTED && _duckdb_filter_expression) {
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    sirius::gpu_expression_executor exec(sirius_filter_ast.get(), mr_ref, stream);
    auto filtered = exec.select(input.table.view());
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

}  // namespace sirius::op::scan
