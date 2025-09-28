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
#include "scan/scan_task.hpp"
#include "data/data_repository.hpp"
#include "config.hpp"
#include "operator/gpu_physical_table_scan.hpp"
#include "gpu_buffer_manager.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/execution/execution_context.hpp"
#include "duckdb/parallel/task_executor.hpp"

namespace sirius {

// This executor is just handling out the task to duckdb scheduler, and converting the duckdb output chunk to a data batch
// TODO: making sure that ScanExecutor can do this in batches, instead scanning all at once.
class ScanExecutor {
    ScanExecutor(duckdb::TaskExecutor &executor, duckdb::TableFunction& function_p, duckdb::ExecutionContext& context_p,
                       duckdb::GPUPhysicalTableScan& op_p, DataRepository& data_repository) :
        task_executor_(executor), function_(function_p), context_(context_p), op_(op_p), data_repository_(data_repository) {}

    ~ScanExecutor() = default;

    // Create a ScanTask and schedule it to duckdb task scheduler (see example in gpu_physical_table_scan.cpp)
    void createAndScheduleTask();

    // Tell DuckDB scheduler to work on the scheduled task until it's done (see example in gpu_physical_table_scan.cpp)
    void workOnTask();

    // Convert the output chunk from duckdb to a DataBatch
    void convertToDataBatch();

    // Push the output DataBatch to Data Repository
    void pushScanOutput(std::unique_ptr<DataBatch> data_batch, size_t pipeline_id, size_t idx);

private:
    DataRepository& data_repository_;
    duckdb::TaskExecutor &task_executor_;
    duckdb::TableFunction& function_;
    duckdb::ExecutionContext& context_;
    duckdb::GPUPhysicalTableScan& op_;
};

} // namespace sirius