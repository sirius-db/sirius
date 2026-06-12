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
#include <helper/utils.hpp>
#include <io/io_context.hpp>
#include <io/sirius_datasource.hpp>
#include <log/logging.hpp>
#include <op/scan/duckdb_native_batch_coalescer.hpp>
#include <op/scan/duckdb_native_decoder.hpp>
#include <op/scan/duckdb_native_gpu_ingestible.hpp>
#include <op/scan/scan_utils.hpp>
#include <op/scan/sirius_gpu_scan_operator_data.hpp>
#include <scan_manager/sirius_scan_manager.hpp>
#include <scan_manager/split_connector.hpp>

// cudf
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

// cucascade
#include <cucascade/memory/memory_space.hpp>

// standard library
#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sirius::op::scan {

namespace {

// Number of row groups a worker claims and walks per range. Overridable via
// SIRIUS_METADATA_PARSE_CHUNK.
constexpr std::size_t DEFAULT_PARSE_CHUNK_ROW_GROUPS = 8;

std::size_t parse_chunk_row_groups()
{
  if (char const* env = std::getenv("SIRIUS_METADATA_PARSE_CHUNK")) {
    try {
      auto const v = std::stoull(env);
      if (v > 0) { return static_cast<std::size_t>(v); }
    } catch (...) {
      // Unparsable value → fall through to the default.
    }
  }
  return DEFAULT_PARSE_CHUNK_ROW_GROUPS;
}

}  // namespace

//===----------------------------------------------------------------------===//
// duckdb_native_ingestible_table_info::make_ingestible
//===----------------------------------------------------------------------===//
std::shared_ptr<io::gpu_ingestible> duckdb_native_ingestible_table_info::make_ingestible(
  std::unique_ptr<io::ingestible_table_info> self, scan_manager::sirius_scan_manager const& mgr)
{
  return std::make_shared<duckdb_native_gpu_ingestible>(std::move(self), mgr);
}

//===----------------------------------------------------------------------===//
// duckdb_native_gpu_ingestible — construction
//===----------------------------------------------------------------------===//
duckdb_native_gpu_ingestible::duckdb_native_gpu_ingestible(
  std::unique_ptr<io::ingestible_table_info> info, scan_manager::sirius_scan_manager const& mgr)
  : io::gpu_ingestible(std::move(info))
{
  auto const& bind = static_cast<duckdb_native_ingestible_table_info const&>(table_info());

  if (bind.storage == nullptr) {
    throw std::invalid_argument(
      "[duckdb_native_gpu_ingestible] table_info.storage must be non-null");
  }
  if (bind.context == nullptr) {
    throw std::invalid_argument(
      "[duckdb_native_gpu_ingestible] table_info.context must be non-null");
  }
  if (bind.projected_cols.size() != bind.projected_types.size()) {
    throw std::invalid_argument(
      "[duckdb_native_gpu_ingestible] projected_cols and projected_types must be parallel");
  }

  // Table-global metadata parse. Pushed-down filters drive row-group pruning in
  // the range walks; column_ids maps each filter to its storage index.
  _plan = prepare_duckdb_native_walk(*bind.storage,
                                     *bind.context,
                                     bind.projected_cols,
                                     bind.projected_types,
                                     bind.table_filters.get(),
                                     &bind.column_ids);
  if (!_plan.viable) {
    SPDLOG_DEBUG("[duckdb_native_gpu_ingestible] non-viable: {}", _plan.viability_failure_reason);
    throw std::runtime_error("duckdb-native scan rejected query: " +
                             _plan.viability_failure_reason);
  }
  if (_plan.n_row_groups == 0) {
    // Nothing to scan; reject so the table falls back to DuckDB CPU.
    throw std::runtime_error("duckdb-native scan rejected query: no row groups in table");
  }

  // Resolve the .db file to a datasource when the manager exposes a backend for
  // db_path, and derive the io_ctx + io_object from it. io_ctx stays null when no
  // backend supports db_path; the decoder rejects the scan at decode time then.
  auto db_datasource = !bind.db_path.empty() ? mgr.create_datasource(bind.db_path) : nullptr;
  _io_ctx            = db_datasource ? db_datasource->io_ctx() : nullptr;
  _db_io_object      = db_datasource ? db_datasource->io_object() : nullptr;

  // Pre-build the coalesced filter expression once.
  if (bind.table_filters && !bind.table_filters->filters.empty()) {
    duckdb::vector<duckdb::idx_t> source_ids_fallback;
    if (bind.projection_ids.empty()) {
      source_ids_fallback.reserve(bind.column_ids.size());
      for (duckdb::idx_t i = 0; i < bind.column_ids.size(); ++i) {
        source_ids_fallback.push_back(i);
      }
    }
    auto const& source_ids =
      bind.projection_ids.empty() ? source_ids_fallback : bind.projection_ids;

    std::vector<std::optional<std::size_t>> emission_order_map(bind.column_ids.size());
    for (std::size_t k = 0; k < source_ids.size(); ++k) {
      emission_order_map[source_ids[k]] = k;
    }

    auto filter_expr_duckdb = sirius::op::convert_table_filters_to_expression(
      *bind.table_filters, bind.column_ids, bind.returned_types, emission_order_map);
    if (filter_expr_duckdb) {
      _filter_expression = std::shared_ptr<duckdb::Expression>(filter_expr_duckdb.release());
    }
  }

  // Decoder emits one column per source_id; projection-down is needed when
  // the planner injected pure-filter trailing columns beyond output_types.
  std::size_t const decoded_cols =
    bind.projection_ids.empty() ? bind.column_ids.size() : bind.projection_ids.size();
  _output_arity        = bind.output_types.size();
  _projection_required = (_output_arity > 0) && (decoded_cols > _output_arity);

  // Coalescer that packs parsed ranges into cap-sized batches with one tail
  // batch per scan.
  _coalescer = std::make_unique<batch_coalescer>(bind.approximate_batch_size, bind.projected_types);

  // Divide the row groups into ranges of _chunk_row_groups each.
  _chunk_row_groups = parse_chunk_row_groups();
  _num_ranges       = utils::ceil_div(_plan.n_row_groups, _chunk_row_groups);
}

duckdb_native_gpu_ingestible::~duckdb_native_gpu_ingestible() = default;

//===----------------------------------------------------------------------===//
// split-provider interface
//===----------------------------------------------------------------------===//
bool duckdb_native_gpu_ingestible::has_more_splits() const
{
  return _next_range_idx.load(std::memory_order_relaxed) < _num_ranges;
}

std::function<std::vector<std::unique_ptr<op::operator_data>>()>
duckdb_native_gpu_ingestible::next_split_provider()
{
  auto const idx = _next_range_idx.fetch_add(1, std::memory_order_relaxed);
  if (idx >= _num_ranges) { return {}; }

  auto const rg_begin = idx * _chunk_row_groups;
  auto const rg_end   = std::min(rg_begin + _chunk_row_groups, _plan.n_row_groups);

  // Runs on a worker thread: walk this range and emit one raw range carrier for
  // the consumer to coalesce. Reads the read-only _plan.
  return [this, rg_begin, rg_end]() -> std::vector<std::unique_ptr<op::operator_data>> {
    auto range = walk_duckdb_native_row_group_range(_plan, rg_begin, rg_end);
    if (!range.viable) {
      // Surfaced to the consumer through the connector; rejects the query.
      throw std::runtime_error("duckdb-native scan rejected query: " +
                               range.viability_failure_reason);
    }

    auto carrier        = std::make_unique<duckdb_native_range_input>();
    carrier->row_groups = std::move(range.row_groups);

    std::vector<std::unique_ptr<op::operator_data>> out;
    out.push_back(std::move(carrier));
    return out;
  };
}

//===----------------------------------------------------------------------===//
// consumer-side coalescing
//===----------------------------------------------------------------------===//
std::unique_ptr<op::operator_data> duckdb_native_gpu_ingestible::make_batch(
  std::vector<duckdb_row_group_metadata> row_groups)
{
  auto split_info = std::make_unique<duckdb_native_split_info>();
  split_info->payload.table_info =
    &static_cast<duckdb_native_ingestible_table_info const&>(table_info());
  split_info->payload.io_ctx       = _io_ctx;
  split_info->payload.db_io_object = _db_io_object;
  split_info->payload.row_groups   = std::move(row_groups);

  std::unique_ptr<io::post_filter_and_projection_info> filter_info;
  bool const apply_filter = static_cast<bool>(_filter_expression);
  if (apply_filter || _projection_required) {
    auto pf          = std::make_unique<duckdb_native_post_filter_and_projection_info>();
    pf->apply_filter = apply_filter;
    pf->output_arity = _projection_required ? _output_arity : 0;
    filter_info      = std::move(pf);
  }

  auto metadata =
    std::make_unique<io::scan_and_filter_metadata>(std::move(split_info), std::move(filter_info));
  return std::make_unique<scan_operator_input>(std::move(metadata));
}

std::unique_ptr<op::operator_data> duckdb_native_gpu_ingestible::consume_next_input(
  scan_manager::split_connector& connector)
{
  // Coalesce parsed ranges into cap-sized batches as they arrive, so early
  // batches decode while later ranges are still being walked. Rowids are
  // absolute per row group, so packing order is unconstrained. get_next_split
  // blocks until a range is ready.
  for (;;) {
    if (_coalescer->has_ready()) { return make_batch(_coalescer->pop_ready()); }

    auto next = connector.get_next_split();
    if (!next.has_value()) {
      // Connector closed and drained: emit the single tail batch, if any.
      auto tail = _coalescer->flush();
      if (!tail.empty()) { return make_batch(std::move(tail)); }
      return nullptr;
    }

    auto* range = dynamic_cast<duckdb_native_range_input*>(next->get());
    if (range == nullptr) {
      throw std::runtime_error(
        "[duckdb_native_gpu_ingestible::consume_next_input] unexpected operator_data type from "
        "split connector; expected duckdb_native_range_input.");
    }
    for (auto& rg : range->row_groups) {
      _coalescer->push(std::move(rg));
    }
  }
}

bool duckdb_native_gpu_ingestible::consumer_drained() const
{
  // The scan operator conjoins this with split_connector::is_closed(). Gating on
  // an empty coalescer ensures a single-split scan (small table, or chunk >=
  // row-group count) still serves every queued batch, including the tail.
  return _coalescer == nullptr || _coalescer->empty();
}

//===----------------------------------------------------------------------===//
// materialize_table
//===----------------------------------------------------------------------===//
io::filtered_table duckdb_native_gpu_ingestible::materialize_table(
  io::scan_info const& info,
  ::cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream)
{
  auto const& split = static_cast<duckdb_native_split_info const&>(info);
  // Decoder takes mem_space by non-const ref; the const-ref input here is a
  // formality (the ingestible interface preserves immutability conceptually,
  // but the decoder mutates the allocator state on the space).
  auto& mem_space_mut = const_cast<::cucascade::memory::memory_space&>(mem_space);
  auto table          = decode_duckdb_native_split(split.payload, mem_space_mut, stream);
  SIRIUS_LOG_DEBUG(
    "[duckdb_native_gpu_ingestible::materialize_table] decoded split: row_groups={} rows={} "
    "cols={}",
    split.payload.row_groups.size(),
    table->num_rows(),
    table->num_columns());
  // duckdb-native applies filter + projection inside post_filter_and_project,
  // never during materialization — always UNFILTERED here.
  return io::filtered_table{std::move(table), io::filter_state::UNFILTERED};
}

//===----------------------------------------------------------------------===//
// post_filter_and_project — filter eval + projection to output arity
//===----------------------------------------------------------------------===//
std::unique_ptr<cudf::table> duckdb_native_gpu_ingestible::post_filter_and_project(
  std::unique_ptr<cudf::table> input,
  io::post_filter_and_projection_info const& info,
  ::cucascade::memory::memory_space const& mem_space,
  rmm::cuda_stream_view stream)
{
  auto const& pf = static_cast<duckdb_native_post_filter_and_projection_info const&>(info);

  rmm::device_async_resource_ref mr_ref(mem_space.get_default_allocator());

  if (pf.apply_filter && _filter_expression) {
    auto sirius_filter_ast = sirius::ast::from_duckdb(*_filter_expression);
    sirius::gpu_expression_executor exec(sirius_filter_ast.get(), mr_ref, stream);
    auto src = std::move(input);
    input    = exec.select(src->view());
  }

  if (pf.output_arity > 0 && static_cast<std::size_t>(input->num_columns()) > pf.output_arity) {
    auto cols = input->release();
    std::vector<std::unique_ptr<cudf::column>> selected;
    selected.reserve(pf.output_arity);
    for (std::size_t i = 0; i < pf.output_arity; ++i) {
      selected.push_back(std::move(cols[i]));
    }
    input = std::make_unique<cudf::table>(std::move(selected));
  }

  return input;
}

}  // namespace sirius::op::scan
