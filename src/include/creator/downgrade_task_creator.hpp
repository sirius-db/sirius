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
#include "downgrade/downgrade_executor.hpp"
#include "creator/task_creator.hpp"

namespace sirius {

/**
 * @brief Task creator specialized for managing downgrade operations in memory tier hierarchies.
 * 
 * This class inherits from itask_creator and is responsible for creating and scheduling
 * downgrade tasks that move data between different memory tiers (e.g., GPU to CPU,
 * CPU to disk). It coordinates with the downgrade_executor to manage the execution
 * of these memory management operations.
 */
class downgrade_task_creator : public itask_creator {
public:
    /**
     * @brief Constructs a new downgrade_task_creator object
     * 
     * @param data_repo_mgr Reference to the data repository manager for accessing and storing data batches
     * @param downgrade_exec Reference to the downgrade executor for task execution
     */
    downgrade_task_creator(
        data_repository_manager& data_repo_mgr, parallel::downgrade_executor &downgrade_exec)
        : itask_creator(), _data_repo_mgr(data_repo_mgr), _downgrade_exec(downgrade_exec) {}

    /**
     * @brief Destructor for the downgrade_task_creator
     */
    ~downgrade_task_creator() = default;

    // Non-copyable but movable
    downgrade_task_creator(const downgrade_task_creator&) = delete;
    downgrade_task_creator& operator=(const downgrade_task_creator&) = delete;
    downgrade_task_creator(downgrade_task_creator&&) = default;
    downgrade_task_creator& operator=(downgrade_task_creator&&) = default;

    /**
     * @brief Main worker loop for the downgrade task creator
     * 
     * This method continuously monitors for memory pressure and creates downgrade
     * tasks as needed to move data from higher-tier to lower-tier memory.
     */
    void worker_loop() override;

    /**
     * @brief Schedules a downgrade task for execution
     * 
     * @param downgrade_task The downgrade task to schedule for execution
     */
    void schedule(sirius::unique_ptr<parallel::downgrade_task> downgrade_task);

private:
    data_repository_manager& _data_repo_mgr;                    ///< Reference to the data repository for data access
    parallel::downgrade_executor& _downgrade_exec;    ///< Reference to the downgrade executor for task execution
};

} // namespace sirius