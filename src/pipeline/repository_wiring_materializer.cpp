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
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <cucascade/data/data_repository.hpp>
#include <cucascade/data/data_repository_manager.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace sirius {
namespace pipeline {

namespace {

//! Number @p op if it has not been numbered yet, advancing @p next_id.
void assign_if_unassigned(op::sirius_physical_operator& op, size_t& next_id)
{
  if (!op.has_operator_id()) { op.operator_id = next_id++; }
}

//! The operator that owns @p wiring's port: the destination pipeline's first operator, or its
//! sink if it has none.
op::sirius_physical_operator& destination_operator(const repository_wiring& wiring)
{
  auto& dest_pipeline = *wiring.dest_pipeline;
  return dest_pipeline.get_operators().empty() ? *dest_pipeline.get_sink().get()
                                               : dest_pipeline.get_operators()[0].get();
}

struct producer {
  std::weak_ptr<sirius_pipeline> pipeline;
  std::int64_t position;
};

}  // namespace

size_t assign_operator_ids(const std::vector<std::shared_ptr<sirius_pipeline>>& pipelines)
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
    auto* next_op      = &destination_operator(wiring);
    const size_t op_id = next_op->get_operator_id();

    data_repo_manager.add_new_repository(
      op_id, wiring.port_id, std::make_unique<::cucascade::shared_data_repository>());

    auto* repo = data_repo_manager.get_repository(op_id, wiring.port_id).get();

    next_op->add_port(wiring.port_id,
                      std::make_unique<op::sirius_physical_operator::port>(
                        wiring.barrier_type, repo, wiring.source_pipeline, wiring.dest_pipeline));
    wiring.source_op->add_next_port_after_sink({next_op, wiring.port_id});
  }
}

data::data_repository_manager_registry::spill_rank_fn make_spill_rank(
  const std::vector<repository_wiring>& wirings,
  const std::vector<std::shared_ptr<sirius_pipeline>>& pipelines,
  cucascade::shared_data_repository_manager& data_repo_manager)
{
  using repository = data::data_repository_manager_registry::repository_type;

  std::unordered_map<const sirius_pipeline*, std::int64_t> positions;
  for (std::size_t i = 0; i < pipelines.size(); ++i) {
    positions.emplace(pipelines[i].get(), static_cast<std::int64_t>(i));
  }
  // A producer outside the scheduled set is treated as running last.
  auto const position_of = [&](const sirius_pipeline* pipeline) {
    auto it = positions.find(pipeline);
    return it == positions.end() ? static_cast<std::int64_t>(pipelines.size()) : it->second;
  };

  std::unordered_map<const sirius_pipeline*, std::shared_ptr<std::vector<producer>>> producers;
  for (const auto& wiring : wirings) {
    auto& list = producers[wiring.dest_pipeline.get()];
    if (!list) { list = std::make_shared<std::vector<producer>>(); }
    list->push_back({wiring.source_pipeline, position_of(wiring.source_pipeline.get())});
  }

  std::unordered_map<const repository*, std::shared_ptr<const std::vector<producer>>> consumers;
  for (const auto& wiring : wirings) {
    auto const op_id = destination_operator(wiring).get_operator_id();
    if (auto& repo = data_repo_manager.get_repository(op_id, wiring.port_id)) {
      consumers.emplace(repo.get(), producers.at(wiring.dest_pipeline.get()));
    }
  }

  return [consumers = std::move(consumers)](const repository* repo) -> std::int64_t {
    std::int64_t rank = -1;
    auto it           = consumers.find(repo);
    if (it == consumers.end()) { return rank; }
    for (auto const& p : *it->second) {
      auto pipeline = p.pipeline.lock();
      if (pipeline && !pipeline->is_pipeline_finished()) { rank = std::max(rank, p.position); }
    }
    return rank;
  };
}

}  // namespace pipeline
}  // namespace sirius
