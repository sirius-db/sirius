#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/mutex.hpp"
#include "duckdb/common/pair.hpp"
#include "duckdb/common/reference_map.hpp"
#include "duckdb/execution/task_error_manager.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_meta_pipeline.hpp"
// #include "duckdb/execution/operator/helper/physical_result_collector.hpp"
#include "operator/gpu_physical_result_collector.hpp"

namespace duckdb {

class ClientContext;
class GPUContext;

class GPUExecutor {
	friend class GPUPipeline;
	friend class GPUPipelineBuildState;

public:
	explicit GPUExecutor(ClientContext &context, GPUContext &gpu_context)
	    : context(context), gpu_context(gpu_context) {
	};
	// ~GPUExecutor();

	ClientContext &context;
	GPUContext &gpu_context;
	optional_ptr<PhysicalOperator> physical_plan;
	unique_ptr<PhysicalOperator> owned_plan;
	optional_ptr<GPUPhysicalOperator> gpu_physical_plan;
	unique_ptr<GPUPhysicalOperator> gpu_owned_plan;

	//! All pipelines of the query plan
	vector<shared_ptr<GPUPipeline>> pipelines;
	//! The root pipelines of the query
	vector<shared_ptr<GPUPipeline>> root_pipelines;
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

	void Initialize(unique_ptr<GPUPhysicalResultCollector> physical_plan);
	void InitializeInternal(GPUPhysicalResultCollector &physical_result_collector);
	void Reset();
	shared_ptr<GPUPipeline> CreateChildPipeline(GPUPipeline &current, GPUPhysicalOperator &op);

	//! Convert the DuckDB physical plan to a GPU physical plan

};

} // namespace duckdb
