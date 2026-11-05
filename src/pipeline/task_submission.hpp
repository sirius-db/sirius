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

#pragma once

#include "exec/query_lifecycle_registry.hpp"
#include "log/logging.hpp"
#include "pipeline/gpu_pipeline_task.hpp"

#include <memory>
#include <string_view>

namespace sirius::pipeline {

/// Admit a publisher and diagnose a missing registration. Resolve the completion handler
/// only on an unknown ID: creator-state lookup must not add a lock to accepted submissions.
/// This component adapter keeps task/completion/logging dependencies out of the registry.
template <typename CompletionResolver>
[[nodiscard]] exec::query_lifecycle_registry::submission_guard begin_submission(
  exec::query_lifecycle_registry& lifecycle,
  query_id_t query_id,
  std::string_view missing_registration_message,
  CompletionResolver&& resolve_completion)
{
  auto submission = lifecycle.try_begin_submission(query_id);
  if (submission.status() == exec::query_submission_status::unknown) {
    if (auto completion = resolve_completion()) {
      completion->report_error(missing_registration_message);
    }
    try {
      SIRIUS_LOG_ERROR("{} (unknown query {})", missing_registration_message, query_id);
    } catch (...) {
      // Also used by spill-return destructors. Diagnostics must not interrupt disposal.
    }
  }
  return submission;
}

/// Admit a task handoff and give first-time publications a work lease. retain_work() preserves
/// an existing lease across queue transfers. On refusal the task and its existing lease remain
/// with the caller, which must dispose of or otherwise settle the task.
///
/// Declare the submission guard BEFORE the owned task and retain it through push or disposal:
///   submission_guard submission;
///   auto task = std::move(input);
///   submission = begin_submission(registry, *task, diagnostic);
/// This also keeps the publisher counted through task destruction on exceptions/rejected pushes.
[[nodiscard]] inline exec::query_lifecycle_registry::submission_guard begin_submission(
  exec::query_lifecycle_registry& lifecycle,
  parallel::itask& task,
  std::string_view missing_registration_message)
{
  auto submission =
    begin_submission(lifecycle,
                     make_query_id(index_keys_for(task).query_id),
                     missing_registration_message,
                     [&task]() noexcept -> std::shared_ptr<completion_handler> {
                       auto* gpu_task = dynamic_cast<gpu_pipeline_task*>(&task);
                       return gpu_task ? gpu_task->get_completion_handler() : nullptr;
                     });
  if (submission) { task.retain_work(submission.take_work_lease()); }
  return submission;
}

}  // namespace sirius::pipeline
