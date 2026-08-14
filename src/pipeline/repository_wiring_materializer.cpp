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

#include "op/sirius_physical_operator.hpp"
#include "op/sirius_physical_operator_type.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>

#include <memory>

namespace sirius {
namespace pipeline {

namespace {

//! Number @p op if it has not been numbered yet, advancing @p next_id.
void assign_if_unassigned(op::sirius_physical_operator& op, size_t& next_id)
{
  if (!op.has_operator_id()) { op.operator_id = next_id++; }
}

}  // namespace

size_t assign_operator_ids(const duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines)
{
  size_t next_id = 0;
  for (const auto& pipeline : pipelines) {
    if (!pipeline) { continue; }

    // source and sink are visited explicitly rather than relying on get_operators() to
    // contain them: a pipeline can carry a sink with an empty operators list (the
    // destination resolution below special-cases exactly that shape).
    if (auto source = pipeline->get_source()) { assign_if_unassigned(*source, next_id); }
    for (auto& op : pipeline->get_operators()) {
      assign_if_unassigned(op.get(), next_id);
    }
    if (auto sink = pipeline->get_sink()) { assign_if_unassigned(*sink, next_id); }
  }
  return next_id;
}

void materialize_repository_wiring(const std::vector<repository_wiring>& wirings,
                                   cucascade::shared_data_repository_manager& data_repo_manager)
{
  for (const auto& wiring : wirings) {
    auto& dest_pipeline = wiring.dest_pipeline;
    auto* next_op       = dest_pipeline->get_operators().size() == 0
                            ? dest_pipeline->get_sink().get()
                            : &dest_pipeline->get_operators()[0].get();
    const size_t op_id  = next_op->get_operator_id();

    data_repo_manager.add_new_repository(
      op_id, wiring.port_id, std::make_unique<::cucascade::shared_data_repository>());

    // Locked, lifetime-safe accessor (B9). The port stores a raw pointer as
    // before: the per-query manager outlives the wiring it materializes.
    auto* repo = data_repo_manager.get_repository_shared(op_id, wiring.port_id).get();

    next_op->add_port(wiring.port_id,
                      std::make_unique<op::sirius_physical_operator::port>(
                        wiring.barrier_type, repo, wiring.source_pipeline, dest_pipeline));
    wiring.source_op->add_next_port_after_sink({next_op, wiring.port_id});
  }
}

}  // namespace pipeline
}  // namespace sirius
