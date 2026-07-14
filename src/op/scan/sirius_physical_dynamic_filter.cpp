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
#include <op/scan/sirius_physical_dynamic_filter.hpp>

// nvtx
#include <nvtx3/nvtx3.hpp>

// cucascade
#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>

namespace sirius::op::scan {

sirius_physical_dynamic_filter::sirius_physical_dynamic_filter(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::shared_ptr<sirius::op::sirius_dynamic_filter_set> filters,
  double gate_keep_threshold,
  dynamic_filter_apply_mode mode)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::DYNAMIC_FILTER, std::move(types), estimated_cardinality),
    _filters(std::move(filters)),
    _gate(gate_keep_threshold),
    _mode(mode)
{
}

void sirius_physical_dynamic_filter::on_finalize_operator()
{
  if (_filters) { _filters->close_for_new_filters(); }
}

std::unique_ptr<operator_data> sirius_physical_dynamic_filter::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_dynamic_filter::execute"};
  auto& input = dynamic_cast<const pipelineable_operator_data&>(input_data);

  // An immediate BUILD_PROBE target runs after publication, but a scan reached through an
  // intervening join can race it. Keep the no-filter fast path for that transitive case and for an
  // intentionally empty publication. The gate remains filter-count-aware if later splits observe
  // additional filters.
  if (!_filters || !_gate.applicable(*_filters)) {
    return std::make_unique<pipelineable_operator_data>(input.get_data_batches());
  }

  auto const& idle_batches = input.get_data_batches();
  auto const ro_batches    = input.get_read_only_batches();

  std::vector<std::shared_ptr<::cucascade::data_batch>> output_batches;
  output_batches.reserve(ro_batches.size());

  for (std::size_t i = 0; i < ro_batches.size(); ++i) {
    auto const& ro = ro_batches[i];
    // A null result means nothing was dropped — the gate declined, or no published filter matched.
    // Forward the batch unchanged (zero-copy; its columns stay co-owned via the idle shared_ptr).
    auto filtered = apply_dynamic_filters_gated_view(sirius::get_cudf_table_view(ro),
                                                     *_filters,
                                                     _gate,
                                                     stream,
                                                     _mode,
                                                     ro.get_memory_space()->get_device_id());
    if (filtered) {
      output_batches.push_back(sirius::make_data_batch(
        std::move(filtered), *ro.get_memory_space(), stream, batch_telemetry()));
    } else {
      output_batches.push_back(idle_batches[i]);
    }
  }
  return std::make_unique<pipelineable_operator_data>(std::move(output_batches));
}

}  // namespace sirius::op::scan
