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

#include "op/sirius_physical_limit.hpp"

#include "data/data_batch_utils.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/copying.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>

namespace sirius {
namespace op {

sirius_physical_streaming_limit::sirius_physical_streaming_limit(
  duckdb::vector<sirius::logical_type> types,
  duckdb::BoundLimitNode limit_val_p,
  duckdb::BoundLimitNode offset_val_p,
  std::size_t estimated_cardinality,
  bool parallel)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::STREAMING_LIMIT, std::move(types), estimated_cardinality),
    limit_val(std::move(limit_val_p)),
    offset_val(std::move(offset_val_p)),
    parallel(parallel),
    _remaining_offset(0),
    _remaining_limit(0)
{
  if (limit_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
    _remaining_limit = limit_val.GetConstantValue();
  }
  if (offset_val.Type() == duckdb::LimitNodeType::CONSTANT_VALUE) {
    _remaining_offset = offset_val.GetConstantValue();
  }
}

std::unique_ptr<operator_data> sirius_physical_streaming_limit::get_next_task_input_data()
{
  std::lock_guard<std::mutex> guard(_claim_mutex);
  if (!is_source_pipeline_finished()) {
    throw internal_exception("STREAMING_LIMIT requires a completed producer");
  }
  if (limit_val.Type() != duckdb::LimitNodeType::CONSTANT_VALUE ||
      (offset_val.Type() != duckdb::LimitNodeType::CONSTANT_VALUE &&
       offset_val.Type() != duckdb::LimitNodeType::UNSET)) {
    throw not_implemented_exception("Streaming limit requires constant limit/offset");
  }
  auto* repo = get_port("default")->repo;
  if (!_positions && repo->get_ordering() == cucascade::data_repository_ordering::ORDERED) {
    _positions = repo->get_positions();
  }
  std::shared_ptr<cucascade::data_batch> batch;
  std::optional<std::size_t> position;
  if (_positions) {
    if (_next_position == _positions->size()) { return nullptr; }
    position = (*_positions)[_next_position];
    batch    = repo->pop_data_batch_by_position(*position);
    if (!batch) { throw internal_exception("STREAMING_LIMIT input position disappeared"); }
  } else {
    batch = repo->pop_next_data_batch();
    if (!batch) { return nullptr; }
  }
  auto rows = sirius::get_batch_rows_and_bytes(*batch).rows;
  if (!rows) { throw internal_exception("Cannot determine batch row count for STREAMING_LIMIT"); }
  auto skip = std::min(_remaining_offset, *rows);
  auto take = std::min(_remaining_limit, *rows - skip);
  _remaining_offset -= skip;
  _remaining_limit -= take;
  if (_positions) { ++_next_position; }
  // Retain zero-row outputs, including LIMIT 0, so downstream ungrouped aggregates see input.
  // Every input is consumed; no early-exhaustion flag can close the pipeline before task creation.
  return std::make_unique<limiting_operator_data>(
    std::vector{std::move(batch)}, skip, take, position);
}

std::unique_ptr<operator_data> sirius_physical_streaming_limit::execute(
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_streaming_limit::execute"};
  auto const& input = dynamic_cast<const limiting_operator_data&>(input_data);
  auto batches      = input.get_read_only_batches();
  if (batches.size() != 1) { throw internal_exception("Limit task requires one input batch"); }
  auto const& batch = batches.front();
  auto view         = sirius::get_cudf_table_view(batch);
  if (input.slice_begin > static_cast<std::size_t>(view.num_rows()) ||
      input.slice_count > static_cast<std::size_t>(view.num_rows()) - input.slice_begin) {
    throw internal_exception("Limit slice exceeds input rows");
  }
  if (input.slice_begin == 0 && input.slice_count == static_cast<std::size_t>(view.num_rows())) {
    return std::make_unique<pipelineable_operator_data>(input.get_data_batches(),
                                                        input.get_position());
  }
  auto slices = cudf::slice(view,
                            {static_cast<cudf::size_type>(input.slice_begin),
                             static_cast<cudf::size_type>(input.slice_begin + input.slice_count)},
                            stream);
  auto table  = std::make_unique<cudf::table>(
    slices.front(), stream, batch.get_memory_space()->get_default_allocator());
  auto output =
    sirius::make_data_batch(std::move(table), *batch.get_memory_space(), stream, batch_telemetry());
  return std::make_unique<pipelineable_operator_data>(std::vector{std::move(output)},
                                                      input.get_position());
}

}  // namespace op
}  // namespace sirius
