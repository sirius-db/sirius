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
#include <string>
#include <utility>

namespace sirius::pipeline {
namespace {

rust::Box<quent::task::TaskHandle> create_task_handle(
  uint64_t task_id, const std::shared_ptr<sirius_pipeline_task_global_state>& global_state)
{
  const auto instance_name      = std::string("task-") + std::to_string(task_id);
  const auto* pipeline          = global_state->get_pipeline();
  const auto pipeline_uuid      = pipeline ? pipeline->pipeline_uuid() : uuid::new_nil();
  const auto& telemetry_context = global_state->telemetry_context();
  return quent::task::create(telemetry_context.context(),
                             {
                               .instance_name = instance_name,
                               .pipeline_uuid = pipeline_uuid,
                             });
}

}  // namespace

sirius_pipeline_itask::sirius_pipeline_itask(
  uint64_t task_id,
  std::unique_ptr<sirius_pipeline_task_local_state> local_state,
  std::shared_ptr<sirius_pipeline_task_global_state> global_state)
  : itask(task_id, std::move(local_state), global_state),
    _telemetry_task_handle(create_task_handle(task_id, global_state))
{
}

sirius_pipeline_itask::~sirius_pipeline_itask()
{
  if (_telemetry_finalized) { return; }

  _telemetry_task_handle->finalizing({
    .instance_name = std::string(),
    .success       = false,
  });
  _telemetry_task_handle->exit();
}

quent::task::TaskHandle* sirius_pipeline_itask::telemetry_handle() const noexcept
{
  return const_cast<quent::task::TaskHandle*>(&*_telemetry_task_handle);
}

}  // namespace sirius::pipeline
