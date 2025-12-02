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
#include "gpu_pipeline.hpp"
#include "data/data_repository.hpp"
#include "data/data_batch_view.hpp"
#include "parallel/task_executor.hpp"
#include "helper/helper.hpp"
#include "creator/task_completion.hpp"
#include "config.hpp"

namespace sirius {
namespace parallel {

/**
 * @brief Global state shared across all GPU pipeline tasks in an execution context.
 * 
 * This class maintains resources and state that are shared among multiple tasks
 * within the same execution context. It provides access to the data repository
 * for retrieving input data and a message queue for notifying the TaskCreator
 * about task completion events.
 */
class gpu_pipeline_task_global_state : public itask_global_state {
public:
    /**
     * @brief Construct a new gpu_pipeline_task_global_state object
     * 
     * @param pipeline_id Identifier for the pipeline this task belongs to
     * @param pipeline Shared pointer to the GPU pipeline to execute
     * @param data_repo_mgr Reference to the data repository manager for accessing input data
     * @param message_queue Reference to the message queue for task completion notifications
     */
    explicit gpu_pipeline_task_global_state(uint64_t pipeline_id, 
        duckdb::shared_ptr<duckdb::GPUPipeline> pipeline,
        data_repository_manager& data_repo_mgr, 
        task_completion_message_queue& message_queue
        ) : 
        _pipeline_id(pipeline_id), _pipeline(std::move(pipeline)),
        _data_repo_mgr(data_repo_mgr), 
        _message_queue(message_queue) 
        {}

    data_repository_manager& _data_repo_mgr; ///< Reference to the data repository manager
    task_completion_message_queue& _message_queue; ///< Message queue to notify TaskCreator about task completion
    duckdb::shared_ptr<duckdb::GPUPipeline> _pipeline; ///< Shared pointer to the GPU pipeline to execute
    uint64_t _pipeline_id; ///< Identifier for the pipeline this task belongs to
};


/**
 * @brief Local state specific to an individual GPU pipeline task instance.
 * 
 * This class encapsulates the state and data that is unique to a single task
 * execution. It holds the task and pipeline identifiers, the GPU pipeline to
 * execute, and the data batch views that serve as input to the pipeline.
 */
class gpu_pipeline_task_local_state : public itask_local_state {
public:
    /**
     * @brief Construct a new gpu_pipeline_task_local_state object
     * 
     * @param task_id Unique identifier for this task
     * @param batch_views Vector of data batch views serving as input to the pipeline
     */
    explicit gpu_pipeline_task_local_state(uint64_t task_id, 
        sirius::vector<sirius::unique_ptr<data_batch_view>> batch_views,
        sirius::unique_ptr<sirius::memory::reservation> res = nullptr) : 
        _task_id(task_id), _batch_views(std::move(batch_views)), 
        _reservation(std::move(res)) {}
    
    uint64_t _task_id; ///< Unique identifier for this task
    sirius::vector<sirius::unique_ptr<data_batch_view>> _batch_views; ///< Input data batch views for the pipeline

    void set_reservation(sirius::unique_ptr<sirius::memory::reservation> res) {
        _reservation = std::move(res);
    }
private:
    sirius::unique_ptr<sirius::memory::reservation> _reservation; ///< Memory reservation for GPU resources
    // TODO: for now, reservation is passed as a local state, will be null when the task is first created, and will be set when reservation is made
};

/**
 * @brief A task representing a unit of work in a GPU pipeline.
 * 
 * This class encapsulates the necessary information to execute a task within a pipeline on the GPU. 
 * These task will be created by the TaskCreator and be scheduled for execution on the gpu_pipeline_executor.
 * 
 * Note that this class will be further derived to represent specific types of tasks such as build, aggregation, etc..
 */
class gpu_pipeline_task : public itask {
public:
    /**
     * @brief Construct a new gpu_pipeline_task object
     * 
     * @param local_state The local state specific to this task
     * @param global_state The global state shared across multiple tasks
     */
    gpu_pipeline_task(sirius::unique_ptr<itask_local_state> local_state,
                        sirius::shared_ptr<itask_global_state> global_state);

        /**
         * @brief Method to actually execute the task
         */
        void execute() override;

    /**
     * @brief Get the unique identifier for this task
     * 
     * @return uint64_t The task ID
     */
    uint64_t get_task_id() const;

    /**
     * @brief Get the GPU pipeline associated with this task
     * 
     * @return const duckdb::GPUPipeline* Pointer to the GPU pipeline
     */
    const duckdb::GPUPipeline* get_pipeline() const;

    /**
     * @brief Method to mark that this task is completed
     * 
     * This method informs that TaskCreator that the task is completed so that it can start scheduling
     * tasks that were dependent on this task. This method should be called after pushing the output
     * of this task to the Data Repository.
     */
    void mark_task_completion();

    /**
     * @brief Method to push the output of this task to the Data Repository
     * 
     * @param batch The data batch to push
     * @param pipeline_id The id of the pipeline that produced this data batch
     */
    void push_data_batch(sirius::unique_ptr<data_batch> batch, uint64_t pipeline_id);
};

} // namespace parallel
} // namespace sirius
