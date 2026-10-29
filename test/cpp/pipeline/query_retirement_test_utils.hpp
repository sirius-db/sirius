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

#pragma once

#include "creator/task_creator.hpp"
#include "exec/query_lifecycle_registry.hpp"
#include "pipeline/task_scheduler.hpp"

namespace sirius::test {

/// Standalone scheduler fixtures have no SiriusContext or scan producers. Perform the owner's
/// barriers explicitly; component queue disposal/validation deliberately does not join work.
inline void retire_scheduler_query(exec::query_lifecycle_registry& lifecycle,
                                   pipeline::task_scheduler& scheduler,
                                   query_id_t query,
                                   bool completed                 = false,
                                   creator::task_creator* creator = nullptr)
{
  lifecycle.quiesce_and_wait_for_submissions(query);
  if (completed) { scheduler.validate_query_complete(query); }
  if (creator) { creator->discard_pending_tasks(query); }
  if (!completed) { scheduler.discard_query_tasks(query); }
  lifecycle.wait_for_work(query);
}

}  // namespace sirius::test
