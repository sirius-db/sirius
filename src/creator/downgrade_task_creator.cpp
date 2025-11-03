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

#include "downgrade/downgrade_task_creator.hpp"

namespace sirius {

void downgrade_task_creator::schedule(sirius::unique_ptr<parallel::downgrade_task> downgrade_task) {
    // Downgrade-specific scheduling logic
    // Schedule the downgrade task using the downgrade_task_queue
    _downgrade_exec.schedule(std::move(downgrade_task));
}

void downgrade_task_creator::worker_loop() {

}

} // namespace sirius