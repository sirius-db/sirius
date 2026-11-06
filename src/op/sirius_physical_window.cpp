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

#include "op/sirius_physical_window.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "op/key_preparation.hpp"
#include "pipeline/sirius_pipeline.hpp"  // IWYU pragma: keep
#include "telemetry/nvtx.hpp"

#include <cudf/utilities/traits.hpp>

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace sirius::op {

// A window of aggregates over the whole partition.
sirius_physical_window::sirius_physical_window(duckdb::vector<logical_type> output_types,
                                               duckdb::vector<logical_type> prepared_types,
                                               std::size_t input_width,
                                               std::vector<cudf::size_type> partition_columns,
                                               std::vector<whole_partition_aggregate> aggregates,
                                               std::size_t estimated_cardinality)
  : sirius_physical_operator(TYPE, std::move(output_types), estimated_cardinality),
    _prepared_types(std::move(prepared_types)),
    _input_width(input_width),
    _partition_columns(std::move(partition_columns)),
    _aggregates(std::move(aggregates))
{
  auto const valid_slot = [&](cudf::size_type slot) {
    return slot >= 0 && static_cast<std::size_t>(slot) < _prepared_types.size();
  };
  if (_aggregates.empty() ||
      !std::all_of(_partition_columns.begin(), _partition_columns.end(), valid_slot) ||
      !std::all_of(_aggregates.begin(), _aggregates.end(), [&](auto const& aggregate) {
        return valid_slot(aggregate.argument_column);
      })) {
    throw std::invalid_argument("Window key or argument is outside its prepared schema");
  }
  if (_input_width > _prepared_types.size() || types.size() != _input_width + _aggregates.size()) {
    throw std::invalid_argument("Window output schema does not match its prepared input");
  }
}

// A window of ranking functions over the window ORDER BY.
sirius_physical_window::sirius_physical_window(duckdb::vector<logical_type> output_types,
                                               duckdb::vector<logical_type> prepared_types,
                                               std::size_t input_width,
                                               std::vector<cudf::size_type> partition_columns,
                                               ranking_order order,
                                               std::vector<ranking_function> ranking_functions,
                                               std::size_t estimated_cardinality)
  : sirius_physical_operator(TYPE, std::move(output_types), estimated_cardinality),
    _prepared_types(std::move(prepared_types)),
    _input_width(input_width),
    _partition_columns(std::move(partition_columns)),
    _ranking_order(std::move(order)),
    _ranking_functions(std::move(ranking_functions))
{
  auto const valid_slot = [&](cudf::size_type slot) {
    return slot >= 0 && static_cast<std::size_t>(slot) < _prepared_types.size();
  };
  if (_ranking_functions.empty() || _partition_columns.empty() || _ranking_order.columns.empty() ||
      _ranking_order.directions.size() != _ranking_order.columns.size() ||
      _ranking_order.nulls.size() != _ranking_order.columns.size() ||
      !std::all_of(_partition_columns.begin(), _partition_columns.end(), valid_slot) ||
      !std::all_of(_ranking_order.columns.begin(), _ranking_order.columns.end(), valid_slot)) {
    throw std::invalid_argument("Window key or order key is outside its prepared schema");
  }
  if (_input_width > _prepared_types.size() ||
      types.size() != _input_width + _ranking_functions.size()) {
    throw std::invalid_argument("Window output schema does not match its prepared input");
  }
}

// A window of running aggregates over the window ORDER BY.
sirius_physical_window::sirius_physical_window(duckdb::vector<logical_type> output_types,
                                               duckdb::vector<logical_type> prepared_types,
                                               std::size_t input_width,
                                               std::vector<cudf::size_type> partition_columns,
                                               ranking_order order,
                                               std::vector<cumulative_aggregate> aggregates,
                                               std::size_t estimated_cardinality)
  : sirius_physical_operator(TYPE, std::move(output_types), estimated_cardinality),
    _prepared_types(std::move(prepared_types)),
    _input_width(input_width),
    _partition_columns(std::move(partition_columns)),
    _ranking_order(std::move(order)),
    _cumulative_aggregates(std::move(aggregates))
{
  auto const valid_slot = [&](cudf::size_type slot) {
    return slot >= 0 && static_cast<std::size_t>(slot) < _prepared_types.size();
  };
  if (_cumulative_aggregates.empty() || _ranking_order.columns.empty() ||
      _ranking_order.directions.size() != _ranking_order.columns.size() ||
      _ranking_order.nulls.size() != _ranking_order.columns.size() ||
      !std::all_of(_partition_columns.begin(), _partition_columns.end(), valid_slot) ||
      !std::all_of(_ranking_order.columns.begin(), _ranking_order.columns.end(), valid_slot) ||
      !std::all_of(_cumulative_aggregates.begin(),
                   _cumulative_aggregates.end(),
                   [&](auto const& aggregate) { return valid_slot(aggregate.argument_column); })) {
    throw std::invalid_argument("Window key, order key or argument is outside its prepared schema");
  }
  if (_input_width > _prepared_types.size() ||
      types.size() != _input_width + _cumulative_aggregates.size()) {
    throw std::invalid_argument("Window output schema does not match its prepared input");
  }
}

// The next complete bucket from CONCAT, or nullptr when none is ready.
std::unique_ptr<operator_data> sirius_physical_window::get_next_task_input_data()
{
  std::lock_guard<std::mutex> guard(lock);
  auto const* port = get_port("default");
  if (port->type == MemoryBarrierType::FULL && port->src_pipeline &&
      !port->src_pipeline->is_pipeline_finished()) {
    return nullptr;
  }
  // CONCAT preserves bucket indices in its ordinary publication. The base acquisition only
  // pops repository partition 0. Drain all buckets here, including gaps, one assembled batch
  // per task. Completion and task accounting still belong to the ordinary pipeline lifecycle.
  for (std::size_t partition = 0; partition < port->repo->num_partitions(); ++partition) {
    if (auto batch = port->repo->pop_next_data_batch(partition)) {
      return std::make_unique<pipelineable_operator_data>(
        std::vector<std::shared_ptr<cucascade::data_batch>>{std::move(batch)});
    }
  }
  return nullptr;
}

std::size_t sirius_physical_window::workspace_bytes(std::size_t rows, std::size_t input_bytes) const
{
  // Cold-start estimate, not a skew guarantee. Allow copied group keys (including strings),
  // groupby and distinct-join hash tables, the row-to-group map, restored/widened arguments,
  // checked SUM structs, counts/AVG temporaries, and expanded output columns. Worst case: one
  // group per row. History refines this after execution, and native OOM follows ordinary retries.
  // A ranking window needs about 24 bytes per row for each function: the INT32 scan result, the
  // scattered result and the INT64 cast. The sort order is shared by all functions.
  auto per_row = memory::saturating_add(128, memory::saturating_mul(_aggregates.size(), 128));
  per_row = memory::saturating_add(per_row, memory::saturating_mul(_ranking_functions.size(), 24));
  // A running aggregate holds a widened argument, the scan result, the filled result and the
  // scattered result.
  per_row =
    memory::saturating_add(per_row, memory::saturating_mul(_cumulative_aggregates.size(), 128));
  per_row = memory::saturating_add(per_row, memory::saturating_mul(_partition_columns.size(), 32));
  per_row =
    memory::saturating_add(per_row, memory::saturating_mul(_ranking_order.columns.size(), 32));
  return memory::saturating_add(memory::saturating_mul(input_bytes, 2),
                                memory::saturating_mul(rows, per_row));
}

std::size_t sirius_physical_window::no_history_peak_memory_estimate(input_stats const& stats) const
{
  // Each nonempty prepared row has at least one byte per column, even with narrow carriers.
  auto rows = stats.bytes / std::max<std::size_t>(_prepared_types.size(), 1);
  return workspace_bytes(rows, stats.bytes);
}

// Evaluates each input bucket and publishes its columns followed by the window results.
std::unique_ptr<operator_data> sirius_physical_window::execute(operator_data const& input,
                                                               ::cuda::stream_ref stream)
{
  nvtx_scoped_range range{"sirius_physical_window::execute"};
  auto const& data = dynamic_cast<pipelineable_operator_data const&>(input);
  auto batches     = data.get_read_only_batches();
  std::vector<std::shared_ptr<cucascade::data_batch>> output;
  for (auto& batch : batches) {
    auto table  = get_cudf_table_view(batch);
    auto& space = *batch.get_memory_space();
    auto mr     = space.get_default_allocator();
    if (static_cast<std::size_t>(table.num_columns()) != _prepared_types.size()) {
      throw std::invalid_argument("Window prepared input width mismatch");
    }
    // Throws only when a known lower bound (input bytes plus fixed-width result bytes) exceeds
    // the memory space's maximum. Size estimates are not used to reject a bucket.
    std::size_t result_width = 0;
    for (std::size_t i = _input_width; i < types.size(); ++i) {
      auto const type = require_cudf_type(types[i]);
      // String extrema retain variable-width values. Offsets alone are a safe lower bound.
      // Their character storage and scratch still use ordinary allocation/OOM accounting.
      auto const width = cudf::is_fixed_width(type) ? cudf::size_of(type) : sizeof(cudf::size_type);
      result_width     = memory::saturating_add(result_width, width);
    }
    auto const input_bytes = batch.get_data()->get_size_in_bytes();
    auto const required    = memory::saturating_add(
      input_bytes,
      memory::saturating_mul(static_cast<std::size_t>(table.num_rows()), result_width));
    if (required > space.get_max_memory()) {
      throw std::runtime_error("Window input and results exceed GPU memory capacity");
    }

    auto columns = std::vector<cudf::column_view>(table.begin(), table.end());
    std::vector<std::unique_ptr<cudf::column>> restored;
    auto restore = [&](cudf::size_type slot) {
      auto const& logical = _prepared_types[slot];
      auto const target   = require_cudf_type(logical);
      auto column         = restore_prepared_key(columns[slot], target, stream, mr);
      if (column) {
        columns[slot] = column->view();
        restored.push_back(std::move(column));
      }
    };
    // Ranking only reads the equality and the order of its keys, which an exact narrower carrier
    // keeps, so its keys are used as they are. The same holds for the keys of a running aggregate.
    // Its arguments, and the keys of a whole partition aggregate, are restored.
    bool const ranking    = !_ranking_functions.empty();
    bool const cumulative = !_cumulative_aggregates.empty();
    if (!ranking && !cumulative) {
      for (auto slot : _partition_columns) {
        restore(slot);
      }
      for (auto const& aggregate : _aggregates) {
        restore(aggregate.argument_column);
      }
    }
    if (cumulative) {
      for (auto const& aggregate : _cumulative_aggregates) {
        restore(aggregate.argument_column);
      }
    }
    std::shared_ptr<cudf::table> results;
    if (ranking) {
      results = evaluate_ranking_window(cudf::table_view(columns),
                                        _partition_columns,
                                        _ranking_order,
                                        _ranking_functions,
                                        stream,
                                        mr);
    } else if (cumulative) {
      results = evaluate_cumulative_window(cudf::table_view(columns),
                                           _partition_columns,
                                           _ranking_order,
                                           _cumulative_aggregates,
                                           stream,
                                           mr);
    } else {
      results = evaluate_whole_partition_window(
        cudf::table_view(columns), _partition_columns, _aggregates, stream, mr);
    }

    // Forward payload in its original carrier and row order, followed by the window results. The
    // hidden columns after the payload are not published.
    std::vector<cudf::size_type> payload(_input_width);
    std::iota(payload.begin(), payload.end(), 0);
    cudf::table_view const visible{{table.select(payload), results->view()}};
    struct window_owner {
      std::shared_ptr<cudf::table> results;
      cucascade::read_only_data_batch input;
    };
    auto bytes = memory::saturating_add(input_bytes, results->alloc_size());
    window_owner owner{std::move(results), std::move(batch)};
    output.push_back(make_data_batch_from_view(
      visible, std::move(owner), bytes, space, stream, batch_telemetry()));
  }
  return std::make_unique<pipelineable_operator_data>(std::move(output));
}

}  // namespace sirius::op
