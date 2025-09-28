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
#include "pipeline/gpu_pipeline_task.hpp"
#include "memory/memory_reservation.hpp"
#include "data_repository.hpp"
#include <ctpl_stl.h> // Include the CTPL header

namespace sirius {

// Manages a pool of threads to execute GPU pipeline tasks, assuming that we are using CTPL to manage thread pool
class GPUPipelineExecutor {
public:
    GPUPipelineExecutor(GPUPipelineTaskQueue &task_queue, DataRepository &data_repository, sirius::memory::MemoryReservationManager &reservation_manager, size_t num_threads);
    ~GPUPipelineExecutor();

private:
    // Function run by each worker thread, it will pull tasks from the queue, acquire memory reservation, and execute the task
    void workerThreadFunction(int id);

    // Execute a given PipelineTask by invoking each operator in a pipeline
    void executeTask(std::unique_ptr<GPUPipelineTask> task);

    // pull a task from the PipelineTaskQueue
    std::unique_ptr<GPUPipelineTask> pullTask();

    // push the output data batch to the Data Repository
    void pushPipelineOutput(std::unique_ptr<DataBatch> data_batch, size_t pipeline_id, size_t idx);

    // reschedule a task by putting it back to the PipelineTaskQueue
    bool rescheduleTask(std::unique_ptr<GPUPipelineTask> task);

    GPUPipelineTaskQueue &task_queue_;
    DataRepository &data_repository_;
    sirius::memory::MemoryReservationManager &reservation_manager_;

    size_t num_threads_;
    ctpl::thread_pool thread_pool_; // Assume using CTPL for the thread pool
};

} // namespace sirius