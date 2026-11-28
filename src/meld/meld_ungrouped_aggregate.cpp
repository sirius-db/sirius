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

#include "meld/meld_ungrouped_aggregate.hpp"

#include "data/data_batch_utils.hpp"
#include "meld/meld_unit.hpp"
#include "telemetry/data_batch_probe.hpp"
#include "telemetry/nvtx.hpp"

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>

#include <utility>
#include <vector>

namespace sirius::meld {

meld_ungrouped_aggregate::meld_ungrouped_aggregate(
  op::sirius_physical_ungrouped_aggregate&& aggregate, std::shared_ptr<meld_unit const> unit)
  : op::sirius_physical_ungrouped_aggregate(std::move(aggregate.types),
                                            std::move(aggregate.aggregates),
                                            aggregate.estimated_cardinality,
                                            duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES,
                                            std::move(aggregate.aggregate_input_max_abs)),
    _unit(std::move(unit))
{
  children = std::move(aggregate.children);
}

std::string meld_ungrouped_aggregate::get_name() const { return "MELD_UNGROUPED_AGGREGATE"; }

std::unique_ptr<op::operator_data> meld_ungrouped_aggregate::execute(
  const op::operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"meld_ungrouped_aggregate::execute"};
  auto& input               = dynamic_cast<const op::pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  std::vector<std::shared_ptr<cucascade::data_batch>> outputs;
  outputs.reserve(input_batches.size());
  for (auto const& batch : input_batches) {
    auto* space = batch.get_memory_space();
    if (!space) { continue; }
    auto view = batch.get_data()->cast<cucascade::gpu_table_representation>().get_table_view();
    std::unique_ptr<cucascade::idata_representation> output_data =
      std::make_unique<cucascade::gpu_table_representation>(
        _unit->run(view, stream, space->get_default_allocator()), *space, stream);
    auto const batch_id = ::sirius::get_next_batch_id();
    outputs.push_back(cucascade::data_batch::make(
      batch_id,
      std::move(output_data),
      telemetry::quent_data_batch_probe::create(batch_telemetry(), batch_id)));
  }
  return std::make_unique<op::pipelineable_operator_data>(outputs);
}

}  // namespace sirius::meld
