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

#include "duckdb/common/common.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/execution/task_error_manager.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_meta_pipeline.hpp"
#include "operator/gpu_physical_result_collector.hpp"
#include "gpu_buffer_manager.hpp"
#include "duckdb/execution/executor.hpp"
namespace duckdb {

class ClientContext;
class GPUContext;

class GPUExecutor {
	friend class GPUPipeline;
	friend class GPUPipelineBuildState;

public:
	explicit GPUExecutor(ClientContext &context, GPUContext &gpu_context)
	    : context(context), gpu_context(gpu_context) {
		gpuBufferManager = &(GPUBufferManager::GetInstance());
		executor = new Executor(context);
	};
	~GPUExecutor() {
		delete executor;
	}

	GPUBufferManager* gpuBufferManager;
	ClientContext &context;
	GPUContext &gpu_context;
	optional_ptr<GPUPhysicalOperator> gpu_physical_plan;
	unique_ptr<GPUPhysicalOperator> gpu_owned_plan;

	//! All pipelines of the query plan
	vector<shared_ptr<GPUPipeline>> pipelines;
	//! The root pipelines of the query
	vector<shared_ptr<GPUPipeline>> root_pipelines;
	vector<shared_ptr<GPUPipeline>> scheduled;
	//! The recursive CTE's in this query plan
	vector<reference<GPUPhysicalOperator>> recursive_ctes;
	//! The current root pipeline index
	idx_t root_pipeline_idx;
	//! The amount of completed pipelines of the query
	atomic<idx_t> completed_pipelines;
	//! The total amount of pipelines in the query
	idx_t total_pipelines;
	
	//! Whether or not the root of the pipeline is a result collector object
	bool HasResultCollector();
	//! Returns the query result - can only be used if `HasResultCollector` returns true
	unique_ptr<QueryResult> GetResult();
	void CancelTasks();

	void Initialize(unique_ptr<GPUPhysicalOperator> physical_plan);
	void InitializeInternal(GPUPhysicalOperator &physical_result_collector);
	void Execute();
	void Reset();
	shared_ptr<GPUPipeline> CreateChildPipeline(GPUPipeline &current, GPUPhysicalOperator &op);

	Executor* executor;

	//! Convert the DuckDB physical plan to a GPU physical plan

};

class GPUExecutionContext {
public:
	GPUExecutionContext(ClientContext &client_p, ThreadContext &thread_p, optional_ptr<GPUPipeline> pipeline_p)
	    : client(client_p), thread(thread_p), pipeline(pipeline_p) {
	}

	//! The client-global context; caution needs to be taken when used in parallel situations
	ClientContext &client;
	//! The thread-local context for this execution
	ThreadContext &thread;
	//! Reference to the pipeline for this execution, can be used for example by operators determine caching strategy
	optional_ptr<GPUPipeline> pipeline;
};

} // namespace duckdb
