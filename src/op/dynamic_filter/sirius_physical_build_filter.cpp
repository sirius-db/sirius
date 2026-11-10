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

#include "op/dynamic_filter/sirius_physical_build_filter.hpp"

#include "op/dynamic_filter/dynamic_filter_publisher.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "telemetry/nvtx.hpp"

#include <rmm/aligned.hpp>

#include <algorithm>
#include <utility>

namespace sirius::op {

sirius_physical_build_filter::sirius_physical_build_filter(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  dynamic_filter_publication_session& session)
  : sirius_physical_operator(TYPE, std::move(types), estimated_cardinality),
    _session(session),
    _certifier(*this, session)
{
}

MemoryBarrierType sirius_physical_build_filter::input_barrier_for(
  sirius_physical_operator const& /*producer*/) const
{
  return MemoryBarrierType::FULL;
}

void sirius_physical_build_filter::build_pipelines(pipeline::sirius_pipeline& current,
                                                   pipeline::sirius_meta_pipeline& meta_pipeline)
{
  D_ASSERT(children.size() == 1);
  D_ASSERT(children.front()->is_sink());
  auto& publication_meta = meta_pipeline.create_child_meta_pipeline(current, *this);
  children.front()->build_pipelines(*publication_meta.get_base_pipeline(), publication_meta);
}

void sirius_physical_build_filter::on_input_batch_pushed(std::string_view port_id,
                                                         cucascade::data_batch& batch)
{
  _certifier.on_arrival(port_id, batch);
}

std::unique_ptr<operator_data> sirius_physical_build_filter::get_next_task_input_data()
{
  if (!_certifier.decide()) { return nullptr; }
  return sirius_physical_operator::get_next_task_input_data();
}

std::unique_ptr<operator_data> sirius_physical_build_filter::execute(
  operator_data const& input_data, ::cuda::stream_ref stream)
{
  nvtx_scoped_range range{"sirius_physical_build_filter::execute"};
  auto const& input  = dynamic_cast<pipelineable_operator_data const&>(input_data);
  auto output        = std::make_unique<pipelineable_operator_data>(input.get_data_batches());
  auto const batches = input.get_read_only_batches();
  if (batches.empty()) { return output; }
  auto const& batch = batches.front();
  // The output forwards the input unchanged, so the final contributor publishes at once.
  if (auto const id = _certifier.contribute(input, batch, stream);
      id && batch.get_memory_space() != nullptr) {
    _session.publish_if_final(*id, *batch.get_memory_space(), stream);
  }
  return output;
}

void sirius_physical_build_filter::on_finalize_operator()
{
  _session.finalize_input();
  D_ASSERT(!_session.plan().enabled() || _session.terminal());
}

std::size_t sirius_physical_build_filter::no_history_peak_memory_estimate(
  input_stats const& stats) const
{
  if (!port_directive().empty()) {
    return sirius_physical_operator::no_history_peak_memory_estimate(stats);
  }
  // The storage leased when Construction begins is reserved separately; a task allocates only what
  // `task_scratch_bound` counts. A positive floor avoids the task's zero-to-2x fallback.
  return std::max<std::size_t>(rmm::CUDA_ALLOCATION_ALIGNMENT, _session.task_scratch_bound());
}

}  // namespace sirius::op
