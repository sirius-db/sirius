#pragma once

#include "duckdb/common/atomic.hpp"
#include "duckdb/common/unordered_set.hpp"
#include "duckdb/common/set.hpp"
#include "gpu_physical_operator.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb/common/reference_map.hpp"

namespace duckdb {

class GPUExecutor;
class GPUPipeline;
class GPUMetaPipeline;

class GPUPipelineBuildState{
public:
	//! How much to increment batch indexes when multiple pipelines share the same source
	constexpr static idx_t BATCH_INCREMENT = 10000000000000;

public:
	//! Duplicate eliminated join scan dependencies
	reference_map_t<const GPUPhysicalOperator, reference<GPUPipeline>> delim_join_dependencies;
	//! Materialized CTE scan dependencies
	// reference_map_t<const GPUPhysicalOperator, reference<GPUPipeline>> cte_dependencies;

public:
	void SetPipelineSource(GPUPipeline &pipeline, GPUPhysicalOperator &op);
	void SetPipelineSink(GPUPipeline &pipeline, optional_ptr<GPUPhysicalOperator> op, idx_t sink_pipeline_count);
	void SetPipelineOperators(GPUPipeline &pipeline, vector<reference<GPUPhysicalOperator>> operators);
	void AddPipelineOperator(GPUPipeline &pipeline, GPUPhysicalOperator &op);
	shared_ptr<GPUPipeline> CreateChildPipeline(GPUExecutor &executor, GPUPipeline &pipeline, GPUPhysicalOperator &op);

	optional_ptr<GPUPhysicalOperator> GetPipelineSource(GPUPipeline &pipeline);
	optional_ptr<GPUPhysicalOperator> GetPipelineSink(GPUPipeline &pipeline);
	vector<reference<GPUPhysicalOperator>> GetPipelineOperators(GPUPipeline &pipeline);
};

//! The Pipeline class represents an execution pipeline starting at a
class GPUPipeline : public enable_shared_from_this<GPUPipeline>{
	friend class GPUExecutor;
	friend class GPUPipelineBuildState;
	friend class GPUMetaPipeline;

public:
	explicit GPUPipeline(GPUExecutor &execution_context);

	GPUExecutor &executor;

public:
	ClientContext &GetClientContext();

	void AddDependency(shared_ptr<GPUPipeline> &pipeline);

	void Ready();
	void Reset();
	void ResetSink();
	void ResetSource(bool force);
	void ClearSource();
	void Schedule(shared_ptr<Event> &event);

	// string ToString() const;
	// void Print() const;
	// void PrintDependencies() const;

	//! Returns query progress
	// bool GetProgress(double &current_percentage, idx_t &estimated_cardinality);

	//! Returns a list of all operators (including source and sink) involved in this pipeline
	vector<reference<GPUPhysicalOperator>> GetOperators();
	vector<const_reference<GPUPhysicalOperator>> GetOperators() const;

	optional_ptr<GPUPhysicalOperator> GetSink() {
		return sink;
	}

	optional_ptr<GPUPhysicalOperator> GetSource() {
		return source;
	}

	//! Returns whether any of the operators in the pipeline care about preserving order
	bool IsOrderDependent() const;

	//! Registers a new batch index for a pipeline executor - returns the current minimum batch index
	idx_t RegisterNewBatchIndex();

	//! Updates the batch index of a pipeline (and returns the new minimum batch index)
	idx_t UpdateBatchIndex(idx_t old_index, idx_t new_index);

	//! The dependencies of this pipeline
	// vector<weak_ptr<GPUPipeline>> dependencies;
	vector<shared_ptr<GPUPipeline>> dependencies;

private:
	//! Whether or not the pipeline has been readied
	bool ready;
	//! Whether or not the pipeline has been initialized
	atomic<bool> initialized;
	//! The source of this pipeline
	optional_ptr<GPUPhysicalOperator> source;
	//! The chain of intermediate operators
	vector<reference<GPUPhysicalOperator>> operators;
	//! The sink (i.e. destination) for data; this is e.g. a hash table to-be-built
	optional_ptr<GPUPhysicalOperator> sink;

	//! The global source state
	unique_ptr<GlobalSourceState> source_state;

	//! The parent pipelines (i.e. pipelines that are dependent on this pipeline to finish)
	vector<weak_ptr<GPUPipeline>> parents;

	//! The base batch index of this pipeline
	idx_t base_batch_index = 0;
	//! Lock for accessing the set of batch indexes
	mutex batch_lock;
	//! The set of batch indexes that are currently being processed
	//! Despite batch indexes being unique - this is a multiset
	//! The reason is that when we start a new pipeline we insert the current minimum batch index as a placeholder
	//! Which leads to duplicate entries in the set of active batch indexes
	multiset<idx_t> batch_indexes;

private:
	void ScheduleSequentialTask(shared_ptr<Event> &event);
	bool LaunchScanTasks(shared_ptr<Event> &event, idx_t max_threads);

	bool ScheduleParallel(shared_ptr<Event> &event);
};

} // namespace duckdb