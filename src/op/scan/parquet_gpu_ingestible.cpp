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
#include <expression/ast/from_duckdb.hpp>
#include <expression_executor/gpu_expression_executor.hpp>
#include <expression_executor/gpu_expression_translator_internal.hpp>
#include <io/io_context.hpp>
#include <io/prefetching_cache.hpp>
#include <io/sirius_datasource.hpp>
#include <log/logging.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/parquet_schema_mapping.hpp>
#include <op/scan/scan_utils.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/parquet_metadata.hpp>
#include <scan_manager/sirius_scan_manager.hpp>

// cudf
#include <cudf/io/datasource.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_io_utils.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// duckdb
#include <duckdb/common/hive_partitioning.hpp>

// uring_reactor MUST be included last among sirius headers — see
// parquet_split_provider.cpp for the BLOCK_SIZE macro-collision rationale.
#include <io/uring/uring_reactor.hpp>

// standard library
#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

struct rg_accumulator {
  std::vector<row_group_slice> slices;
  std::size_t total_uncompressed_bytes = 0;
  // Partition values for the files currently bundled, in scan_plan::partition_columns order.
  // nullopt until the first file is added. Bundling is only safe across files with identical
  // values: post_filter_and_project synthesizes constant scalar columns from this single vector
  // on behalf of every file in the bundle, so all files in the bundle must share those values.
  std::optional<std::vector<std::string>> partition_values;
};

bool has_uri_scheme(std::string const& p) { return p.find("://") != std::string::npos; }

// Case-insensitively strip a leading "file://" so explicit local URIs behave
// exactly like bare paths (mirrors the scan_manager's normalize_path): the
// scheme check must not classify file:// as object-store, and cudf's bundled
// datasource wants a plain filesystem path.
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

}  // namespace

//===----------------------------------------------------------------------===//
// parquet_ingestible_table_info::make_ingestible
//===----------------------------------------------------------------------===//
std::shared_ptr<io::gpu_ingestible> parquet_ingestible_table_info::make_ingestible(
  std::unique_ptr<io::ingestible_table_info> self, scan_manager::sirius_scan_manager const& mgr)
{
  return std::make_shared<parquet_gpu_ingestible>(std::move(self), mgr);
}

//===----------------------------------------------------------------------===//
// parquet_gpu_ingestible — construction
//===----------------------------------------------------------------------===//
parquet_gpu_ingestible::parquet_gpu_ingestible(std::unique_ptr<io::ingestible_table_info> info,
                                               scan_manager::sirius_scan_manager const& mgr)
  : io::gpu_ingestible(std::move(info)), _scan_manager(&mgr)
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

  _file_paths             = bind.resolved_file_paths;
  _approximate_batch_size = bind.approximate_batch_size;
  _max_file_processed     = bind.max_file_processed;
  _total_files            = _file_paths.size();

  for (std::size_t start = 0; start < _total_files; start += _max_file_processed) {
    auto const end = std::min(start + _max_file_processed, _total_files);
    file_batch batch;
    batch.file_paths.assign(_file_paths.begin() + static_cast<std::ptrdiff_t>(start),
                            _file_paths.begin() + static_cast<std::ptrdiff_t>(end));
    _batches.push_back(std::move(batch));
  }
}

parquet_gpu_ingestible::~parquet_gpu_ingestible() = default;

//===----------------------------------------------------------------------===//
// split-provider interface
//===----------------------------------------------------------------------===//
bool parquet_gpu_ingestible::has_more_splits() const
{
  return _next_batch_idx.load(std::memory_order_relaxed) < _batches.size();
}

std::function<std::vector<std::unique_ptr<op::operator_data>>()>
parquet_gpu_ingestible::next_split_provider()
{
  auto const batch_idx = _next_batch_idx.fetch_add(1, std::memory_order_relaxed);
  if (batch_idx >= _batches.size()) { return nullptr; }
  return [this, batch_idx]() {
    std::vector<std::unique_ptr<op::operator_data>> out;
    run_batch(_batches[batch_idx], out);
    return out;
  };
}

//===----------------------------------------------------------------------===//
// run_batch — ports parquet_split_provider::run_batch
//===----------------------------------------------------------------------===//
void parquet_gpu_ingestible::run_batch(file_batch const& batch,
                                       std::vector<std::unique_ptr<op::operator_data>>& out)
{
  auto stream = cudf::get_default_stream();

  auto const data_column_names = _plan->data_column_names();
  auto reader_options          = std::make_shared<cudf::io::parquet_reader_options>(
    cudf::io::parquet_reader_options::builder().build());

  if (_plan->is_projected()) { reader_options->set_column_names(data_column_names); }

  if (_scan_manager == nullptr) {
    throw std::runtime_error("parquet_gpu_ingestible: no scan_manager is wired.");
  }

  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  bool skip_pushdown_due_to_flba                                                 = false;
  if (_duckdb_filter_expression) {
    auto name_resolver = [this](duckdb::idx_t ref_index) -> std::string {
      return _plan->batch_column_name(ref_index);
    };
    gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    ast_expression = translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
    if (ast_expression) {
      // FLBA-decimal pushdown probe — see parquet_split_provider.cpp:276-326.
      if (!batch.file_paths.empty()) {
        auto const& probe_path = batch.file_paths.front();
        try {
          // Resolve the probe file to a datasource (carries its own io backend,
          // cache and metadata) instead of reaching into io_context. Local
          // paths left unclaimed (use_sirius_datasource=false) probe through a
          // plain cudf datasource so filter pushdown is not lost.
          auto probe_ds = _scan_manager->create_datasource(probe_path);
          std::unique_ptr<cudf::io::datasource> probe_fallback;
          cudf::io::datasource* probe_src = probe_ds.get();
          if (probe_src == nullptr) {
            auto const probe_local = strip_file_uri(probe_path);
            if (has_uri_scheme(probe_local)) {
              throw std::runtime_error("no backend supports path: " + probe_path);
            }
            probe_fallback = cudf::io::datasource::create(probe_local);
            probe_src      = probe_fallback.get();
          }
          std::shared_ptr<cudf::io::parquet::FileMetaData const> probe_meta;
          if (probe_ds) {
            if (auto cached = probe_ds->metadata()) {
              if (auto pm = std::dynamic_pointer_cast<scan_manager::parquet_metadata>(cached)) {
                probe_meta = pm->file_metadata();
              }
            }
          }
          if (!probe_meta) {
            auto footer = cudf::io::parquet::fetch_footer_to_host(*probe_src);
            hybrid_scan_reader probe_reader(
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
            "[parquet_gpu_ingestible] FLBA-decimal probe failed ({}); proceeding without "
            "pushdown",
            e.what());
          skip_pushdown_due_to_flba = true;
        }
      }

      if (!skip_pushdown_due_to_flba) {
        reader_options->set_filter(ast_expression->back());
        SIRIUS_LOG_DEBUG(
          "[parquet_gpu_ingestible] Translated filter expression for row group pruning.");
      } else {
        SIRIUS_LOG_DEBUG(
          "[parquet_gpu_ingestible] Skipping row-group pruning pushdown: FLBA-decimal file.");
      }
    } else {
      SIRIUS_LOG_DEBUG("[parquet_gpu_ingestible] AST translation failed for row group pruning.");
    }
  }

  rg_accumulator accum;
  bool const needs_post_processing = needs_output_assembly(*_plan);

  auto build_post_filter_info =
    [&accum, needs_post_processing]() -> std::unique_ptr<io::post_filter_and_projection_info> {
    if (!needs_post_processing) { return nullptr; }
    auto info              = std::make_unique<parquet_post_filter_and_projection_info>();
    info->partition_values = accum.partition_values.value_or(std::vector<std::string>{});
    return info;
  };

  auto flush = [&](std::shared_ptr<cudf::io::parquet_reader_options> shared_opts,
                   std::shared_ptr<scan_plan const> shared_plan) {
    if (accum.slices.empty()) { return; }
    auto split_info                     = std::make_unique<parquet_split_info>();
    split_info->rg_slices               = std::move(accum.slices);
    split_info->reader_options          = std::move(shared_opts);
    split_info->plan                    = std::move(shared_plan);
    split_info->disable_filter_pushdown = skip_pushdown_due_to_flba;
    split_info->needs_assembly          = needs_post_processing;
    split_info->partition_values = accum.partition_values.value_or(std::vector<std::string>{});
    auto metadata = std::make_unique<io::scan_and_filter_metadata>(std::move(split_info),
                                                                   build_post_filter_info());
    out.push_back(std::make_unique<scan_operator_input>(std::move(metadata)));
    accum.slices.clear();
    accum.total_uncompressed_bytes = 0;
  };

  for (auto const& file_path : batch.file_paths) {
    if (!_plan->partition_columns.empty()) {
      std::vector<std::string> file_partition_values;
      file_partition_values.reserve(_plan->partition_columns.size());
      auto parsed = duckdb::HivePartitioning::Parse(file_path);
      for (auto const& pc : _plan->partition_columns) {
        auto it = parsed.find(pc.name);
        file_partition_values.push_back(it != parsed.end() ? it->second : std::string{});
      }
      if (accum.partition_values && *accum.partition_values != file_partition_values) {
        flush(reader_options, _plan);
      }
      accum.partition_values = std::move(file_partition_values);
    }

    // Resolve the file to a sirius_datasource — it carries its own io backend,
    // prefetch cache and cached metadata, so the ingestible no longer reaches
    // into io_context. Fall back to a plain cudf datasource only for local
    // paths no sirius backend claims.
    auto sirius_ds = _scan_manager->create_datasource(file_path);
    // file:// counts as local: create_datasource strips it before deciding the
    // local fallback, so the scheme check here must strip it too.
    auto const local_file_path = strip_file_uri(file_path);
    if (!sirius_ds && has_uri_scheme(local_file_path)) {
      throw std::runtime_error("[parquet_gpu_ingestible] no backend supports path: " + file_path);
    }

    std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
    std::shared_ptr<scan_manager::parquet_metadata> cached_parquet_metadata;
    std::size_t footer_byte_len = 0;
    std::unique_ptr<hybrid_scan_reader> reader_ptr;

    if (sirius_ds) {
      if (auto cached = sirius_ds->metadata()) {
        cached_parquet_metadata =
          std::dynamic_pointer_cast<scan_manager::parquet_metadata>(std::move(cached));
      }
    }

    if (cached_parquet_metadata) {
      file_metadata   = cached_parquet_metadata->file_metadata();
      footer_byte_len = cached_parquet_metadata->footer_byte_len();
      reader_ptr      = std::make_unique<hybrid_scan_reader>(*file_metadata, *reader_options);
    } else {
      // Local paths left unclaimed (use_sirius_datasource=false) read the
      // footer through a plain cudf datasource; the slice itself stays
      // datasource-less so materialize falls back to cudf/KvikIO.
      std::unique_ptr<cudf::io::datasource> footer_fallback;
      cudf::io::datasource* footer_src = sirius_ds.get();
      if (footer_src == nullptr) {
        footer_fallback = cudf::io::datasource::create(local_file_path);
        footer_src      = footer_fallback.get();
      }
      auto footer_buffer = cudf::io::parquet::fetch_footer_to_host(*footer_src);
      footer_byte_len    = footer_buffer->size();
      reader_ptr         = std::make_unique<hybrid_scan_reader>(
        cudf::host_span<uint8_t const>(footer_buffer->data(), footer_buffer->size()),
        *reader_options);
      file_metadata =
        std::make_shared<cudf::io::parquet::FileMetaData const>(reader_ptr->parquet_metadata());
    }
    auto& reader         = *reader_ptr;
    auto const& metadata = *file_metadata;

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

    auto row_group_indices = reader.all_row_groups(*reader_options);
    if (ast_expression && !skip_pushdown_due_to_flba) {
      auto const rgs_before = row_group_indices.size();
      SIRIUS_LOG_DEBUG(
        "[parquet_gpu_ingestible] Row group pruning: file: {}\n"
        "                                                  before: {}",
        file_path,
        rgs_before);
      row_group_indices =
        reader.filter_row_groups_with_stats(row_group_indices, *reader_options, stream);
      SIRIUS_LOG_DEBUG("[parquet_gpu_ingestible]                     after: {} (pruned {})",
                       row_group_indices.size(),
                       rgs_before - row_group_indices.size());
    }

    // Scan-side chunk prewarm: projection and row-group pruning are final for
    // this file, so hand the prefetch cache the merged column-chunk byte
    // ranges (plus parquet magic + footer) to stage while slices are built.
    // describe_parquet()'s insert stays metadata-only — this is the only
    // place ranges enter the cache, and only when the knob is on.
    if (sirius_ds && _scan_manager->chunk_prewarm_enabled() && !row_group_indices.empty()) {
      if (auto* cache = sirius_ds->io_ctx() ? sirius_ds->io_ctx()->cache() : nullptr) {
        using range_t = cudf::io::text::byte_range_info;

        auto chunk_ranges =
          reader.all_column_chunks_byte_ranges(row_group_indices, *reader_options);
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

        constexpr std::size_t FOOTER_TAIL_SIZE = 8;
        auto const file_size                   = sirius_ds->io_object()->size();
        auto const footer_off =
          static_cast<int64_t>(file_size - FOOTER_TAIL_SIZE - footer_byte_len);
        auto const footer_size = static_cast<int64_t>(FOOTER_TAIL_SIZE + footer_byte_len);

        std::vector<range_t> ranges;
        ranges.reserve(merged.size() + 2);
        ranges.emplace_back(0, 4);
        ranges.insert(ranges.end(), merged.begin(), merged.end());
        ranges.emplace_back(footer_off, footer_size);
        std::sort(ranges.begin(), ranges.end(), [](auto const& a, auto const& b) {
          return a.offset() < b.offset();
        });

        std::shared_ptr<sirius::io::sirius_io_object_metadata> metadata_to_store =
          cached_parquet_metadata
            ? nullptr
            : std::static_pointer_cast<sirius::io::sirius_io_object_metadata>(
                std::make_shared<scan_manager::parquet_metadata>(file_metadata, footer_byte_len));
        cache->insert(*sirius_ds->io_object(), std::move(metadata_to_store), ranges);
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
                                sirius_ds);
      accum.total_uncompressed_bytes += cur_uncompressed_bytes;
      cur_rgs.clear();
      cur_uncompressed_bytes = 0;
      cur_compressed_bytes   = 0;
    };

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

    for (auto const rg_idx : row_group_indices) {
      auto const& row_group        = metadata.row_groups[rg_idx];
      auto const [rg_unc, rg_comp] = rg_contribution(row_group);

      if (!accum.slices.empty() || !cur_rgs.empty()) {
        if (accum.total_uncompressed_bytes + cur_uncompressed_bytes + rg_unc >
            _approximate_batch_size) {
          seal_current_file();
          flush(reader_options, _plan);
        }
      }

      cur_uncompressed_bytes += rg_unc;
      cur_compressed_bytes += rg_comp;
      cur_rgs.push_back(rg_idx);
    }
    seal_current_file();
  }
  flush(reader_options, _plan);
}

//===----------------------------------------------------------------------===//
// materialize_table — ports read_table_from_metadata
//===----------------------------------------------------------------------===//
io::filtered_table parquet_gpu_ingestible::materialize_table(
  io::scan_info const& info,
  ::cucascade::memory::memory_space const& mem_space,
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
      // Unclaimed local slice (use_sirius_datasource=false): cudf's bundled
      // datasource wants a plain path, so strip an explicit file:// scheme.
      sources.push_back(cudf::io::datasource::create(strip_file_uri(slice.file_path)));
    }
    metadatas.push_back(*slice.file_metadata);
    rg_per_src.push_back(slice.row_group_indices);
  }
  auto opts = *split.reader_options;
  opts.set_row_groups(std::move(rg_per_src));

  // Per-task AST translation. set_filter is gated on translation success AND on
  // the per-batch disable_filter_pushdown flag (set when the FLBA-decimal probe
  // failed). `sirius_filter_ast` is hoisted so the post-decode fallback can
  // reuse it on a pushdown miss.
  std::unique_ptr<sirius::ast::node> sirius_filter_ast;
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  if (_duckdb_filter_expression) {
    sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    if (!split.disable_filter_pushdown) {
      auto name_resolver = [plan = split.plan](duckdb::idx_t ref_index) -> std::string {
        return plan->batch_column_name(ref_index);
      };
      gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
      ast_expression =
        translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
      if (ast_expression) { opts.set_filter(ast_expression->back()); }
    }
  }

  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());
  auto [table, _] =
    cudf::io::read_parquet(std::move(sources), std::move(metadatas), opts, stream, mr_ref);

  SIRIUS_LOG_DEBUG(
    "[parquet_gpu_ingestible::materialize_table] Read {} file(s) (first: {}) — {} rows, {} "
    "columns",
    split.rg_slices.size(),
    split.rg_slices.empty() ? "<none>" : split.rg_slices.front().file_path,
    table->num_rows(),
    table->num_columns());

  // Determine filter state. When pushdown engaged the reader applied the
  // filter; otherwise we apply post-decode here. `sirius_filter_ast` must
  // outlive `exec` — the executor only borrows the AST.
  io::filter_state state = io::filter_state::UNFILTERED;
  if (sirius_filter_ast) {
    if (!ast_expression) {
      sirius::gpu_expression_executor exec(
        sirius_filter_ast.get(), cudf::get_current_device_resource_ref(), stream);
      auto input = std::move(table);
      table      = exec.select(input->view());
      SIRIUS_LOG_DEBUG(
        "[parquet_gpu_ingestible::materialize_table] Applied duckdb filter expression "
        "post-decode.");
    }
    state = io::filter_state::ROW_FILTERED;
  }

  // Reader-side pushdown succeeded and the plan needs assembly — inline it
  // here so the scan operator can skip post_filter_and_project entirely.
  // (post-decode fallback keeps assembly external because re-allocating after
  // exec.select is the same shape either way.)
  if (state == io::filter_state::ROW_FILTERED && ast_expression && split.needs_assembly) {
    table = assemble_scan_output(*_plan, std::move(table), split.partition_values, stream);
    state = io::filter_state::ROW_FILTERED_AND_PROJECTED;
    SIRIUS_LOG_DEBUG(
      "[parquet_gpu_ingestible::materialize_table] Assembled inline on reader-side pushdown path.");
  }

  return io::filtered_table{std::move(table), state};
}

//===----------------------------------------------------------------------===//
// post_filter_and_project — assembly only
//===----------------------------------------------------------------------===//
std::unique_ptr<cudf::table> parquet_gpu_ingestible::post_filter_and_project(
  std::unique_ptr<cudf::table> input,
  io::post_filter_and_projection_info const& info,
  ::cucascade::memory::memory_space const& /*mem_space*/,
  rmm::cuda_stream_view stream)
{
  auto const& pf = static_cast<parquet_post_filter_and_projection_info const&>(info);
  // The per-batch assembly call. The ingestible only emits a non-null
  // post_filter_and_projection_info when needs_output_assembly(*_plan) is true,
  // so this is unconditionally meaningful.
  auto out = assemble_scan_output(*_plan, std::move(input), pf.partition_values, stream);
  SIRIUS_LOG_DEBUG(
    "[parquet_gpu_ingestible::post_filter_and_project] Assembled scan output to plan layout.");
  return out;
}

}  // namespace sirius::op::scan
