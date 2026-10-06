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

#include "pipeline/sirius_pipeline_itask.hpp"

#include "telemetry/telemetry_context.hpp"

#include <memory>
#include <utility>

namespace sirius::pipeline {

sirius_pipeline_itask::sirius_pipeline_itask(
  uint64_t task_id,
  std::unique_ptr<sirius_pipeline_task_local_state> local_state,
  std::shared_ptr<sirius_pipeline_task_global_state> global_state)
  : itask(task_id, std::move(local_state), global_state),
    _telemetry_fsm(global_state->get_telemetry_context()
                     .context()
                     .task_observer()
                     ->handle()
                     .created({.pipeline_uuid = quent::operator_::OperatorId(
                                 global_state->get_pipeline()->pipeline_uuid())

                     })
                     .into_dynamic())
{
}

sirius_pipeline_itask::~sirius_pipeline_itask()
{
  if (_telemetry_finalized) { return; }

  _telemetry_fsm.finalizing({
    .success = false,
  });
  _telemetry_fsm.exit();
}

}  // namespace sirius::pipeline
