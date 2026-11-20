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

#include <vector>

namespace sirius {
namespace op {

sirius_physical_replicate::sirius_physical_replicate(duckdb::vector<sirius::logical_type> types,
                                                     gpu_replicate_impl::limits output_limits,
                                                     std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::REPLICATE, std::move(types), estimated_cardinality),
    _output_limits(output_limits)
{
  // A table with no columns has no row count to repeat.
  if (this->types.empty()) { throw internal_exception("REPLICATE: no output columns"); }
  if (output_limits.max_rows <= 0 || output_limits.max_bytes == 0) {
    throw internal_exception("REPLICATE: output limits must be positive");
  }
}

std::string sirius_physical_replicate::params_to_string() const
{
  return " (max_rows=" + std::to_string(_output_limits.max_rows) +
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
    auto const width = static_cast<cudf::size_type>(types.size());
    if (view.num_columns() != width + 1) {
      throw internal_exception(
        "REPLICATE: input batch has {} columns, expected {}", view.num_columns(), width + 1);
    }
    auto& space   = *batch.get_memory_space();
    auto const mr = space.get_default_allocator();
    cudf::table_view const data{std::vector<cudf::column_view>(view.begin(), view.end() - 1)};

    auto const expansion =
      gpu_replicate_impl::plan_slices(data, view.column(width), _output_limits, stream, mr);
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
