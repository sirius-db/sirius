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

#include "op/sirius_physical_passthrough_sink.hpp"

#include "sirius/exception.hpp"

#include <nvtx3/nvtx3.hpp>

namespace sirius {
namespace op {

sirius_physical_passthrough_sink::sirius_physical_passthrough_sink(
  duckdb::vector<sirius::logical_type> types,
  std::size_t estimated_cardinality,
  std::string port_label)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::PASSTHROUGH_SINK, std::move(types), estimated_cardinality),
    _union_port_label(std::move(port_label))
{
}

std::string sirius_physical_passthrough_sink::get_name() const { return "PASSTHROUGH_SINK"; }

bool sirius_physical_passthrough_sink::is_source() const { return true; }

bool sirius_physical_passthrough_sink::is_sink() const { return true; }

std::unique_ptr<operator_data> sirius_physical_passthrough_sink::execute(
  const operator_data& input_data, rmm::cuda_stream_view /*stream*/)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_passthrough_sink::execute"};
  // Re-wrap as the base `pipelineable_operator_data`, not a `partitioned_operator_data`: the
  // absence of a partition index is what lets the task creator route the downstream UNION task by
  // data locality. Forwarding the read-only accessors keeps the shared read lock held across the
  // handoff.
  const auto* pipelineable = dynamic_cast<const pipelineable_operator_data*>(&input_data);
  if (pipelineable == nullptr) {
    throw internal_exception(
      "sirius_physical_passthrough_sink::execute: expected pipelineable_operator_data");
  }
  return std::make_unique<pipelineable_operator_data>(pipelineable->get_read_only_batches(false));
}

}  // namespace op
}  // namespace sirius
