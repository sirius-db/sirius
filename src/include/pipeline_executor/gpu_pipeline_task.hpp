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
#include <queue>
#include "gpu_pipeline.hpp"
#include "data_batch.hpp"

namespace sirius {

class GPUPipelineTask {
public:
    GPUPipelineTask(std::shared_ptr<duckdb::GPUPipeline> pipeline, uint64_t task_id, std::unique_ptr<DataBatch> data_batch);
private:
    uint64_t task_id_;
    std::unique_ptr<DataBatch> data_batch_;
    std::shared_ptr<duckdb::GPUPipeline> pipeline_;

    // Other members and methods as needed
};

class GPUPipelineTaskQueue {
public:
    GPUPipelineTaskQueue() = default; 

    // Queue to hold tasks
    std::queue<std::unique_ptr<GPUPipelineTask>> task_queue_;

    // Add a new task to the queue
    void EnqueueTask(std::unique_ptr<GPUPipelineTask> gpu_pipeline_task) {
    }

    // Retrieve and remove a task from the queue
    std::unique_ptr<GPUPipelineTask> DequeueTask() {
        return nullptr;
    }

    // Check if the queue is empty
    bool IsEmpty() {
        return task_queue_.empty();
    }
};

} // namespace sirius