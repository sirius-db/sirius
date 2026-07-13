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

#include "telemetry-bridge/gen/uuid.rs.h"
#include "telemetry/telemetry_context.hpp"

#include <format>
#include <memory>
#include <utility>

namespace sirius::pipeline {

sirius_pipeline_itask::sirius_pipeline_itask(
  uint64_t task_id,
  std::unique_ptr<sirius_pipeline_task_local_state> local_state,
  std::shared_ptr<sirius_pipeline_task_global_state> global_state)
  : itask(task_id, std::move(local_state), global_state),
    _telemetry_task_handle(
      quent::task::create(global_state->get_telemetry_context().context(),
                          {
                            .instance_name = std::format("task-{}", task_id),
                            .pipeline_uuid = global_state->get_pipeline()
                                               ? global_state->get_pipeline()->pipeline_uuid()
                                               : uuid::new_nil(),
                          }))
{
}

sirius_pipeline_itask::~sirius_pipeline_itask()
{
  if (_telemetry_finalized) { return; }

  _telemetry_task_handle->finalizing({
    .instance_name = "",
    .success       = false,
  });
  _telemetry_task_handle->exit();
}

quent::task::TaskHandle& sirius_pipeline_itask::telemetry_handle() noexcept
{
  return *_telemetry_task_handle;
}

}  // namespace sirius::pipeline
