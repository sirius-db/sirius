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
#include <data/data_batch_utils.hpp>
#include <op/scan/dynamic_filter_merge.hpp>
#include <op/scan/scan_output_operator_data.hpp>
#include <op/scan/sirius_physical_dynamic_filter.hpp>
#include <telemetry/nvtx.hpp>

// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>

// standard library
#include <cstdint>
#include <ranges>
#include <utility>
#include <vector>

namespace sirius::op::scan {

sirius_physical_dynamic_filter::sirius_physical_dynamic_filter(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<dynamic_filter_consumer> consumer)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::DYNAMIC_FILTER, std::move(types), estimated_cardinality),
    _consumer(std::move(consumer)),
    _bindings(identity_bindings(this->types.size()))
{
}

void sirius_physical_dynamic_filter::on_finalize_operator()
{
  if (_consumer) { _consumer->channel()->close_for_new_filters(); }
}

std::unique_ptr<operator_data> sirius_physical_dynamic_filter::execute(
  const operator_data& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range nvtx_range{"sirius_physical_dynamic_filter::execute"};
  auto const& input = dynamic_cast<const pipelineable_operator_data&>(input_data);
  if (!_consumer) { return std::make_unique<pipelineable_operator_data>(input.get_data_batches()); }

  auto const* scan_output = dynamic_cast<scan_output_operator_data const*>(&input);
  auto const* prior       = scan_output ? &scan_output->receipt() : nullptr;
  if (prior && prior->endpoint == _consumer->channel()) {
    D_ASSERT(input.original_batch_ids() == std::vector<std::uint64_t>{prior->original_batch_id});
  }

  auto const ro_batches = input.get_read_only_batches();
  // The read-only leases skip null batches, so pair them with the non-null idle batches.
  auto idle_batches = input.get_data_batches() |
                      std::views::filter([](auto const& batch) { return batch != nullptr; });
  auto idle = std::ranges::begin(idle_batches);

  std::vector<std::shared_ptr<::cucascade::data_batch>> output_batches;
  std::vector<application_result> pending;
  output_batches.reserve(ro_batches.size());
  pending.reserve(ro_batches.size());
  for (auto const& source : ro_batches) {
    auto const& unchanged = *idle++;
    auto const view       = sirius::get_cudf_table_view(source);
    // prepare_for_processing places every input on the task's reservation device.
    auto* space  = source.get_memory_space();
    auto program = _consumer->prepare({.source      = view,
                                       .bindings    = _bindings,
                                       .input_bytes = source.get_data()->get_size_in_bytes(),
                                       .prior       = prior,
                                       .device_id   = space->get_device_id()});
    if (!program) {
      output_batches.push_back(unchanged);
      continue;
    }
    // The result keeps no GPU scratch or lease, only the observations committed below. A reader
    // event recorded by `apply` makes any later mutable access to the batch wait for its reads.
    auto result =
      _consumer->apply(std::move(*program), source, stream, space->get_default_allocator());
    auto table = result.take_output();
    output_batches.push_back(
      table ? sirius::make_data_batch(std::move(table), *space, stream, batch_telemetry())
            : unchanged);
    pending.push_back(std::move(result));
  }

  auto output = std::make_unique<pipelineable_operator_data>(std::move(output_batches));
  // No allocating output operation follows the first commit, including for later batches.
  for (auto& result : pending) {
    _consumer->commit(std::move(result));
  }
  return output;
}

}  // namespace sirius::op::scan
