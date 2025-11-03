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

#include "downgrade/downgrade_task.hpp"
#include "downgrade/downgrade_executor.hpp"

namespace sirius {
namespace parallel {

void downgrade_task::execute() {
    mark_task_completion();
}

void downgrade_task::mark_task_completion() {
    // notify task_creator about task completion
    uint64_t task_id = _local_state->cast<downgrade_task_local_state>()._task_id;
    uint64_t pipeline_id = _local_state->cast<downgrade_task_local_state>()._pipeline_id;
    auto message = sirius::make_unique<sirius::task_completion_message>();
    message->task_id = task_id;
    message->pipeline_id = pipeline_id;
    message->source = sirius::Source::PIPELINE;
    _global_state->cast<downgrade_task_global_state>()._message_queue.enqueue_message(std::move(message));
}

uint64_t downgrade_task::get_task_id() const {
    return _local_state->cast<downgrade_task_local_state>()._task_id;
}

} // namespace parallel
} // namespace sirius