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
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/scan/parquet_gpu_ingestible.hpp>
#include <op/scan/parquet_schema_mapping.hpp>
#include <op/scan/scan_utils.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <op/sirius_dynamic_filter.hpp>
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

}  // namespace

//===----------------------------------------------------------------------===//
// parquet_ingestible_table_info::make_ingestible
//===----------------------------------------------------------------------===//
std::shared_ptr<io::gpu_ingestible> parquet_ingestible_table_info::make_ingestible(
  std::unique_ptr<io::ingestible_table_info> self,
  scan_manager::sirius_scan_manager const& mgr,
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> const& gpu_ioctxs)
{
  // self is *this; we route it through gpu_ingestible's base constructor so
  // table_info() remains valid for accessors on the constructed ingestible.
  return std::make_shared<parquet_gpu_ingestible>(std::move(self), mgr, gpu_ioctxs);
}

//===----------------------------------------------------------------------===//
// parquet_gpu_ingestible — construction
//===----------------------------------------------------------------------===//
parquet_gpu_ingestible::parquet_gpu_ingestible(
  std::unique_ptr<io::ingestible_table_info> info,
  scan_manager::sirius_scan_manager const& mgr,
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>> gpu_ioctxs)
  : io::gpu_ingestible(std::move(info)), _scan_manager(&mgr), _gpu_ioctxs(std::move(gpu_ioctxs))
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
  _sirius_dynamic_filters = bind.sirius_dynamic_filters;

  // Producers reference probe columns in DuckDB's column_ids space; the AST merge and the
  // post-decode apply both key by output-column position. Install the translation so push_filter
  // remaps before storing. Wiring-time setup, before the producing build publishes.
  if (_sirius_dynamic_filters) {
    _sirius_dynamic_filters->set_consumer_column_remap(_plan->output_position_by_column_id);
  }

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

  if (_gpu_ioctxs.empty() && _scan_manager == nullptr) {
    throw std::runtime_error(
      "parquet_gpu_ingestible: gpu_ioctxs is empty and no scan_manager is wired — "
      "kvikio path is forbidden.");
  }
  auto const planning_ioctx_it = _gpu_ioctxs.begin();

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
              if (auto pm = std::dynamic_pointer_cast<scan_manager::parquet_metadata>(cached)) {
                probe_meta = pm->file_metadata();
              }
            }
          }
          if (!probe_meta) {
            auto footer = cudf::io::parquet::fetch_footer_to_host(*probe_ds);
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
    auto has_uri_scheme = [](std::string const& p) -> bool {
      return p.find("://") != std::string::npos;
    };
    auto const lookup_path = normalize_path(file_path);
    std::shared_ptr<sirius::io::sirius_ioctx> file_io_ctx;
    if (has_uri_scheme(lookup_path)) {
      if (_scan_manager != nullptr) { file_io_ctx = _scan_manager->io_ctx_shared_for(lookup_path); }
      if (!file_io_ctx) {
        throw std::runtime_error("[parquet_gpu_ingestible] no backend supports path: " + file_path);
      }
    } else if (planning_ioctx_it != _gpu_ioctxs.end()) {
      file_io_ctx = planning_ioctx_it->second;
    } else if (_scan_manager != nullptr) {
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

    std::shared_ptr<cudf::io::parquet::FileMetaData const> file_metadata;
    std::shared_ptr<scan_manager::parquet_metadata> cached_parquet_metadata;
    std::size_t footer_byte_len = 0;
    std::unique_ptr<hybrid_scan_reader> reader_ptr;

    if (file_io_object && file_io_ctx && file_io_ctx->cache() != nullptr) {
      if (auto cached = file_io_ctx->cache()->get_metadata(*file_io_object)) {
        cached_parquet_metadata =
          std::dynamic_pointer_cast<scan_manager::parquet_metadata>(std::move(cached));
      }
    }

    if (cached_parquet_metadata) {
      file_metadata   = cached_parquet_metadata->file_metadata();
      footer_byte_len = cached_parquet_metadata->footer_byte_len();
      reader_ptr      = std::make_unique<hybrid_scan_reader>(*file_metadata, *reader_options);
    } else {
      auto footer_buffer = cudf::io::parquet::fetch_footer_to_host(*datasource);
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

    // Prefetch cache prewarm — see parquet_split_provider.cpp:521-604.
    if (file_io_object && file_io_ctx && file_io_ctx->cache() != nullptr &&
        !row_group_indices.empty()) {
      using range_t = cudf::io::text::byte_range_info;

      auto chunk_ranges = reader.all_column_chunks_byte_ranges(row_group_indices, *reader_options);
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
      auto const file_size                   = file_io_object->size();
      auto const footer_off  = static_cast<int64_t>(file_size - FOOTER_TAIL_SIZE - footer_byte_len);
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
    // Per-file flush — see comment in parquet_split_provider::run_batch.
    if (_gpu_ioctxs.size() > 1) { flush(reader_options, _plan); }
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

  bool const kvikio_fallback_mode = _gpu_ioctxs.empty();
  std::unordered_map<int, std::shared_ptr<sirius::io::sirius_ioctx>>::const_iterator ioctx_it =
    _gpu_ioctxs.end();
  if (!kvikio_fallback_mode) {
    int const target_device_id = mem_space.get_device_id();
    ioctx_it                   = _gpu_ioctxs.find(target_device_id);
    if (ioctx_it == _gpu_ioctxs.end()) {
      throw std::out_of_range(
        "[sirius_gpu_parquet_scan_operator::read_table_from_metadata] no sirius_ioctx for "
        "device_id=" +
        std::to_string(target_device_id) + ".");
    }
  }

  // Hold uring_io_object shared_ptrs alive for the duration of the read — see
  // sirius_gpu_parquet_scan_operator::read_table_from_metadata for rationale.
  std::vector<std::shared_ptr<sirius::io::sirius_io_object>> io_objects;
  io_objects.reserve(split.rg_slices.size());
  for (auto const& slice : split.rg_slices) {
    if (!slice.io_ctx) {
      // No sirius backend minted this slice (local file, single-GPU
      // use_sirius_datasource=false): use cudf's bundled datasource (kvikio).
      // Slices that carry an io_ctx (e.g. the shared s3_ioctx for s3://) fall
      // through to the make_datasource path below even when kvikio_fallback_mode.
      sources.push_back(cudf::io::datasource::create(slice.file_path));
    } else {
      // Two-dimensional ioctx selection (multi-GPU #732 × multi-backend-S3).
      // slice.io_ctx is the backend that minted slice.io_object:
      //   * per-GPU LOCAL backend (one of _gpu_ioctxs) → rebind the read to the
      //     *target* GPU's local ioctx so it binds to the executing GPU's CUDA
      //     context (dev #732 residency); the planning-time GPU may differ.
      //   * shared REMOTE backend (e.g. the single s3_ioctx) → read through
      //     slice.io_ctx directly; S3 is network→host and not per-GPU.
      auto const is_per_gpu_local = [&] {
        for (auto const& [dev, ctx] : _gpu_ioctxs) {
          if (ctx == slice.io_ctx) { return true; }
        }
        return false;
      }();
      auto const ds_ioctx = (slice.io_ctx && !is_per_gpu_local) ? slice.io_ctx : ioctx_it->second;
      if (slice.io_object) {
        sources.push_back(ds_ioctx->make_datasource(slice.io_object));
      } else {
        auto io_object = ds_ioctx->create_io_object(slice.file_path);
        sources.push_back(ds_ioctx->make_datasource(io_object));
        io_objects.push_back(std::move(io_object));
      }
    }
    metadatas.push_back(*slice.file_metadata);
    rg_per_src.push_back(slice.row_group_indices);
  }
  auto opts = *split.reader_options;

  opts.set_row_groups(std::move(rg_per_src));

  // Per-task reader filter. Translate DuckDB's static filter first, then AND any AST-capable
  // dynamic filters (zone maps) into the same cuDF AST passed to read_parquet.
  // `sirius_filter_ast` is hoisted so the static post-decode fallback can reuse it if translation
  // fails or pushdown is disabled for this split.
  std::unique_ptr<sirius::ast::node> sirius_filter_ast;
  std::optional<gpu_expression_translator::translated_expression> ast_expression = std::nullopt;
  std::optional<gpu_expression_translator::translated_expression> dynamic_ast_expression =
    std::nullopt;
  cudf::ast::expression const* reader_filter_root = nullptr;

  if (_duckdb_filter_expression) {
    sirius_filter_ast = sirius::ast::from_duckdb(*_duckdb_filter_expression);
    if (!split.disable_filter_pushdown) {
      auto name_resolver = [plan = split.plan](duckdb::idx_t ref_index) -> std::string {
        return plan->batch_column_name(ref_index);
      };
      gpu_expression_translator translator(stream, cudf::get_current_device_resource_ref());
      ast_expression =
        translator.translate_expression_with_names(*sirius_filter_ast, name_resolver);
      if (ast_expression) { reader_filter_root = &ast_expression->back(); }
    }
  }

  if (!split.disable_filter_pushdown && _sirius_dynamic_filters &&
      _sirius_dynamic_filters->has_filters()) {
    if (ast_expression) {
      reader_filter_root = merge_dynamic_filters_into_ast(
        ast_expression->tree, reader_filter_root, *_sirius_dynamic_filters, *split.plan);
    } else {
      dynamic_ast_expression.emplace();
      reader_filter_root = merge_dynamic_filters_into_ast(dynamic_ast_expression->tree,
                                                          /*existing_root=*/nullptr,
                                                          *_sirius_dynamic_filters,
                                                          *split.plan);
      if (!reader_filter_root) { dynamic_ast_expression.reset(); }
    }
  }

  if (reader_filter_root) { opts.set_filter(*reader_filter_root); }

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

  // Batches leave in output layout; the downstream dynamic-filter operator applies membership
  // filters. AST-capable dynamic filters, when present, already rode the reader filter above.
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
  // Assembly only: reshape the decoded table to output layout. The downstream dynamic-filter
  // operator applies any membership filters.
  auto out = assemble_scan_output(*_plan, std::move(input), pf.partition_values, stream);
  SIRIUS_LOG_DEBUG(
    "[parquet_gpu_ingestible::post_filter_and_project] Assembled scan output to plan layout.");
  return out;
}

}  // namespace sirius::op::scan
