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

#include "op/sirius_physical_replicate.hpp"

#include "data/data_batch_utils.hpp"
#include "sirius/exception.hpp"
#include "telemetry/nvtx.hpp"

#include <cudf/copying.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>

namespace sirius {
namespace op {

sirius_physical_replicate::sirius_physical_replicate(duckdb::vector<sirius::logical_type> types,
                                                     cudf::size_type count_column,
                                                     gpu_replicate_impl::limits output_limits,
                                                     std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::REPLICATE, std::move(types), estimated_cardinality),
    _count_column(count_column),
    _output_limits(output_limits)
{
  auto const input_width = static_cast<cudf::size_type>(this->types.size()) + 1;
  // A table with no columns has no row count to repeat.
  if (this->types.empty() || count_column < 0 || count_column >= input_width) {
    throw internal_exception(
      "REPLICATE: count column {} is not one of {} input columns", count_column, input_width);
  }
  if (output_limits.max_rows <= 0 || output_limits.max_bytes == 0) {
    throw internal_exception("REPLICATE: output limits must be positive");
  }
  _data_columns.reserve(this->types.size());
  for (cudf::size_type column = 0; column < input_width; ++column) {
    if (column != count_column) { _data_columns.push_back(column); }
  }
}

std::string sirius_physical_replicate::params_to_string() const
{
  return " (count_column=" + std::to_string(_count_column) +
         ", max_rows=" + std::to_string(_output_limits.max_rows) +
         ", max_bytes=" + std::to_string(_output_limits.max_bytes) + ")";
}

std::unique_ptr<operator_data> sirius_physical_replicate::execute(const operator_data& input_data,
                                                                  ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_replicate::execute"};
  auto const& input         = dynamic_cast<const pipelineable_operator_data&>(input_data);
  auto const& input_batches = input.get_read_only_batches();

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());
  for (auto const& batch : input_batches) {
    auto const view =
      batch.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
    if (view.num_columns() != static_cast<cudf::size_type>(_data_columns.size()) + 1) {
      throw internal_exception("REPLICATE: input batch has {} columns, expected {}",
                               view.num_columns(),
                               _data_columns.size() + 1);
    }
    auto& space     = *batch.get_memory_space();
    auto const mr   = space.get_default_allocator();
    auto const data = view.select(_data_columns);

    auto const expansion =
      gpu_replicate_impl::plan_slices(data, view.column(_count_column), _output_limits, stream, mr);
    if (expansion.slices.empty()) {
      output_batches.push_back(
        sirius::make_data_batch(cudf::empty_like(data), space, stream, batch_telemetry()));
      continue;
    }
    for (auto const& part : expansion.slices) {
      output_batches.push_back(
        sirius::make_data_batch(gpu_replicate_impl::materialize(data, expansion, part, stream, mr),
                                space,
                                stream,
                                batch_telemetry()));
    }
  }
  return std::make_unique<pipelineable_operator_data>(output_batches);
}

}  // namespace op
}  // namespace sirius
