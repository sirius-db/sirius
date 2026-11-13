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

#include "op/sirius_physical_generator.hpp"

#include "data/data_batch_utils.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/input_free_operator_data.hpp"
#include "op/scan/generator/generator_detail.cuh"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/stream>

#include <cucascade/memory/memory_space.hpp>

#include <limits>
#include <string>
#include <typeinfo>
#include <vector>

namespace sirius::op {

namespace {

constexpr auto max_bytes = std::numeric_limits<std::size_t>::max();

sirius::type_id output_type_id(scan::generator::value_type type)
{
  using scan::generator::value_type;
  switch (type) {
    case value_type::int32: return sirius::type_id::INTEGER;
    case value_type::int64: return sirius::type_id::BIGINT;
    case value_type::float64: return sirius::type_id::DOUBLE;
    case value_type::varchar: return sirius::type_id::VARCHAR;
  }
  throw sirius::internal_exception("[sirius_physical_generator] unknown generator value type");
}

}  // namespace

sirius_physical_generator::sirius_physical_generator(duckdb::vector<sirius::logical_type> types,
                                                     scan::generator::generator_spec spec,
                                                     duckdb::idx_t estimated_cardinality)
  : sirius_physical_generator(
      std::move(types), spec, scan::generator::all_columns(spec), estimated_cardinality)
{
}

sirius_physical_generator::sirius_physical_generator(
  duckdb::vector<sirius::logical_type> types,
  scan::generator::generator_spec spec,
  std::vector<std::size_t> output_columns,
  duckdb::idx_t estimated_cardinality,
  std::shared_ptr<sirius::transparent::read_view_registry> read_views,
  scan::scan_contract_id contract_id)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::GENERATOR, std::move(types), estimated_cardinality),
    _spec(std::move(spec)),
    _output_columns(std::move(output_columns)),
    _read_views(std::move(read_views)),
    _contract_id(contract_id)
{
  // generate_batch emits one column per output column, in that order; the declared output
  // types must describe exactly that table. A table without columns cannot carry a row count.
  if (_output_columns.empty()) {
    throw sirius::internal_exception("[sirius_physical_generator] no output columns");
  }
  if (this->types.size() != _output_columns.size()) {
    throw sirius::internal_exception(
      "[sirius_physical_generator] output has " + std::to_string(this->types.size()) +
      " types but the generator produces " + std::to_string(_output_columns.size()) + " columns");
  }
  for (std::size_t i = 0; i < _output_columns.size(); ++i) {
    auto const column = _output_columns[i];
    if (column >= _spec.columns.size()) {
      throw sirius::internal_exception("[sirius_physical_generator] output column " +
                                       std::to_string(column) + " is not one of the spec's " +
                                       std::to_string(_spec.columns.size()) + " columns");
    }
    if (this->types[i].id() != output_type_id(_spec.columns[column].type)) {
      throw sirius::internal_exception("[sirius_physical_generator] output type of column " +
                                       std::to_string(i) + " ('" + _spec.columns[column].name +
                                       "') does not match the generator spec");
    }
  }
}

//===----------------------------------------------------------------------===//
// Source / scheduling interface
//===----------------------------------------------------------------------===//
std::optional<task_creation_hint> sirius_physical_generator::get_next_task_hint()
{
  // READY while batches remain, like a scan with open splits: the task creator then drains
  // get_next_task_input_data() until all_ports_empty(). The tasks hold no data until they run,
  // so creating all of them up front needs no throttling.
  if (all_ports_empty()) { return std::nullopt; }
  return task_creation_hint{TaskCreationHint::READY, this};
}

bool sirius_physical_generator::all_ports_empty()
{
  return _next_batch.load(std::memory_order_acquire) >= _spec.num_batches;
}

std::unique_ptr<operator_data> sirius_physical_generator::get_next_task_input_data()
{
  // The check before fetch_add keeps calls after exhaustion from growing the counter; racing
  // claims can overshoot num_batches by at most one per thread, which the second check rejects.
  if (all_ports_empty()) { return nullptr; }
  auto const batch = _next_batch.fetch_add(1, std::memory_order_acq_rel);
  if (batch >= _spec.num_batches) { return nullptr; }
  return std::make_unique<input_free_operator_data>(
    scan::generator::estimated_batch_bytes(_spec, _output_columns), batch);
}

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//
std::unique_ptr<operator_data> sirius_physical_generator::execute(const operator_data& input_data,
                                                                  ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_generator::execute"};

  auto const* generator_input = dynamic_cast<const input_free_operator_data*>(&input_data);
  if (!generator_input) {
    throw sirius::internal_exception(
      "[sirius_physical_generator::execute] expected input of type input_free_operator_data; "
      "got " +
      std::string(typeid(input_data).name()));
  }
  auto const batch_index = generator_input->sequence();
  if (!batch_index.has_value()) {
    throw sirius::internal_exception(
      "[sirius_physical_generator::execute] input carries no batch index");
  }
  auto* mem_space = &generator_input->get_gpu_memory_space();
  auto mr         = mem_space->get_default_allocator();

  // The batch depends only on (spec, output columns, batch index), so an OOM retry of the same
  // input regenerates identical data.
  auto output_table = scan::generator::generate_batch(
    _spec, _output_columns, *batch_index, rmm::cuda_stream_view{stream.get()}, mr);

  auto batch =
    sirius::make_data_batch(std::move(output_table), *mem_space, stream, batch_telemetry());
  std::vector<std::shared_ptr<::cucascade::data_batch>> batches{std::move(batch)};
  return std::make_unique<pipelineable_operator_data>(std::move(batches));
}

//===----------------------------------------------------------------------===//
// Memory estimation
//===----------------------------------------------------------------------===//
std::size_t sirius_physical_generator::no_history_peak_memory_estimate(const input_stats&) const
{
  // Columns are generated one at a time, so the peak is the finished columns plus one column's
  // temporaries: the draw buffers, and for VARCHAR the offsets and chars being built. 2x the
  // batch covers that.
  return sirius::memory::saturating_mul(
    scan::generator::estimated_batch_bytes(_spec, _output_columns), 2);
}

std::optional<std::size_t> sirius_physical_generator::total_source_input_bytes() const
{
  if (_spec.num_batches > max_bytes) { return max_bytes; }
  return sirius::memory::saturating_mul(
    static_cast<std::size_t>(_spec.num_batches),
    scan::generator::estimated_batch_bytes(_spec, _output_columns));
}

sirius::OrderPreservationType sirius_physical_generator::output_order_preservation() const
{
  // Batches are generated by independent tasks and arrive in any order.
  return sirius::OrderPreservationType::NO_ORDER;
}

}  // namespace sirius::op
