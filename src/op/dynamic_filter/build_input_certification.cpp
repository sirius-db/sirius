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

#include "op/dynamic_filter/build_input_certification.hpp"

#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"  // IWYU pragma: keep

#include <cucascade/data/data_repository.hpp>

namespace sirius::op {

certification_decision certify_decision(build_port_facts const& facts) noexcept
{
  if (facts.data_ports != 1 || !facts.single_port_is_default || facts.repository_partitions != 1) {
    return construction_decline::input_shape;
  }
  if (facts.barrier == MemoryBarrierType::PIPELINE) {
    return construction_decline::pipeline_barrier;
  }
  if (facts.barrier != MemoryBarrierType::FULL || !facts.has_source_pipeline) {
    return construction_decline::input_shape;
  }
  if (facts.has_port_directive) { return construction_decline::port_directive; }
  if (!facts.source_finished) { return certification_pending{}; }
  return certification_proceed{};
}

build_port_facts read_build_port_facts(sirius_physical_operator const& site)
{
  build_port_facts facts{.data_ports             = 0,
                         .single_port_is_default = false,
                         .repository_partitions  = 0,
                         .barrier                = MemoryBarrierType::FULL,
                         .has_source_pipeline    = false,
                         .source_finished        = false,
                         .repository_batches     = 0,
                         .has_port_directive     = !site.port_directive().empty()};
  sirius_physical_operator::port const* input = nullptr;
  for (auto const& candidate : site.get_ports()) {
    if (candidate->repo != nullptr) {
      ++facts.data_ports;
      input = candidate.get();
    }
  }
  if (facts.data_ports != 1) { return facts; }
  facts.single_port_is_default = site.try_get_port("default") == input;
  facts.repository_partitions  = input->repo->num_partitions();
  facts.barrier                = input->type;
  facts.has_source_pipeline    = input->src_pipeline != nullptr;
  facts.source_finished = facts.has_source_pipeline && input->src_pipeline->is_pipeline_finished();
  // Read after the source finished: no later batch can arrive, so the count is final.
  if (facts.repository_partitions >= 1) { facts.repository_batches = input->repo->size(0); }
  return facts;
}

}  // namespace sirius::op
