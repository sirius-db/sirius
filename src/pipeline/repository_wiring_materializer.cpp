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

void materialize_repository_wiring(const std::vector<repository_wiring>& wirings,
                                   cucascade::shared_data_repository_manager& data_repo_manager)
{
  for (const auto& wiring : wirings) {
    auto& dest_pipeline = wiring.dest_pipeline;
    auto* next_op       = dest_pipeline->get_operators().size() == 0
                            ? dest_pipeline->get_sink().get()
                            : &dest_pipeline->get_operators()[0].get();
    const size_t op_id  = next_op->operator_id;

    data_repo_manager.add_new_repository(
      op_id, wiring.port_id, std::make_unique<::cucascade::shared_data_repository>());

    auto* repo = data_repo_manager.get_repository(op_id, wiring.port_id).get();

    next_op->add_port(wiring.port_id,
                      std::make_unique<op::sirius_physical_operator::port>(
                        wiring.barrier_type, repo, wiring.source_pipeline, dest_pipeline));
    wiring.source_op->add_next_port_after_sink({next_op, wiring.port_id});
  }
}

}  // namespace pipeline
}  // namespace sirius
