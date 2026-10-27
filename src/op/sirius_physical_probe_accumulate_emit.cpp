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

#include "op/sirius_physical_probe_accumulate_emit.hpp"

#include "cudf/cudf_utils.hpp"
#include "data/data_batch_utils.hpp"
#include "helper/type_conversions.hpp"
#include "log/logging.hpp"
#include "memory/size_arithmetic.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>

#include <mutex>
#include <utility>
#include <vector>

namespace sirius::op {

sirius_physical_probe_accumulate_emit::sirius_physical_probe_accumulate_emit(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<probe_accumulate_context> context)
  : sirius_physical_operator(TYPE, std::move(types), estimated_cardinality),
    _context(std::move(context))
{
  if (_context == nullptr) {
    throw sirius::internal_exception("probe_accumulate_emit requires a shared context");
  }
}

std::string sirius_physical_probe_accumulate_emit::params_to_string() const
{
  return std::to_string(_context->state.specs().size()) + " accumulator(s)";
}

std::size_t sirius_physical_probe_accumulate_emit::no_history_peak_memory_estimate(
  input_stats const& /*stats*/) const
{
  std::lock_guard<std::mutex> guard(_context->mutex);
  return sirius::memory::saturating_mul(
    2, _context->state.emit_peak_bytes(_context->state.num_build_rows()));
}

std::optional<task_creation_hint> sirius_physical_probe_accumulate_emit::get_next_task_hint()
{
  std::lock_guard<std::mutex> guard(lock);
  if (_emitted || ports.empty()) { return std::nullopt; }
  for (auto const& port : _ports_list) {
    if (port->src_pipeline && !port->src_pipeline->is_pipeline_finished()) {
      auto* producer = &(port->src_pipeline->get_operators()[0].get());
      return task_creation_hint{TaskCreationHint::WAITING_FOR_INPUT_DATA, producer};
    }
  }
  return task_creation_hint{TaskCreationHint::READY, this};
}

std::unique_ptr<operator_data> sirius_physical_probe_accumulate_emit::get_next_task_input_data()
{
  std::lock_guard<std::mutex> guard(lock);
  if (_emitted) { return nullptr; }
  _emitted = true;
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches;
  for (auto& [name, port] : ports) {
    if (!port->repo) { continue; }
    while (auto batch = port->repo->pop_next_data_batch()) {
      batches.push_back(std::move(batch));
    }
  }
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

std::unique_ptr<operator_data> sirius_physical_probe_accumulate_emit::execute(
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_probe_accumulate_emit::execute"};
  auto const& input        = dynamic_cast<pipelineable_operator_data const&>(input_data);
  auto const input_batches = input.get_read_only_batches();

  // The state is final: the child join's pipeline has finished and synced its streams.
  bool initialized                         = false;
  ::cucascade::memory::memory_space* space = nullptr;
  {
    std::lock_guard<std::mutex> guard(_context->mutex);
    initialized = _context->state.initialized();
    space       = _context->space;
  }
  if (space == nullptr && !input_batches.empty()) { space = input_batches[0].get_memory_space(); }
  if (space == nullptr) {
    // Neither a build table nor a delivered batch: there is nothing to place a result on.
    return std::make_unique<pipelineable_operator_data>(
      std::vector<std::shared_ptr<::cucascade::data_batch>>{});
  }
  auto mr = space->get_default_allocator();

  std::unique_ptr<cudf::table> output;
  {
    std::lock_guard<std::mutex> guard(_context->mutex);
    if (_context->emitted == nullptr) {
      if (initialized) {
        auto const build_view = sirius::get_cudf_table_view(*_context->build_batch);
        _context->emitted =
          _context->state.emit(build_view.column(_context->build_key_column), stream, mr);
      } else {
        // The build side was empty, so no table was ever built and nothing can have matched.
        std::vector<std::unique_ptr<cudf::column>> columns;
        columns.reserve(types.size());
        for (auto const& type : types) {
          columns.push_back(cudf::make_empty_column(sirius::require_cudf_type(type)));
        }
        _context->emitted = std::make_unique<cudf::table>(std::move(columns));
      }
      SIRIUS_LOG_INFO("[probe_accumulate] emitted {} group row(s)", _context->emitted->num_rows());
    }
    // A copy, so the cached table survives a retry of this task.
    output = std::make_unique<cudf::table>(_context->emitted->view(), stream, mr);
  }

  std::vector<std::shared_ptr<::cucascade::data_batch>> results;
  results.push_back(sirius::make_data_batch(std::move(output), *space, stream, batch_telemetry()));
  return std::make_unique<pipelineable_operator_data>(std::move(results));
}

}  // namespace sirius::op
