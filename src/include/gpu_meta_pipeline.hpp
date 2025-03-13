#pragma once

#include "duckdb/common/reference_map.hpp"
#include "gpu_physical_operator.hpp"
// #include "gpu_pipeline.hpp"

namespace duckdb {

//! GPUMetaPipeline represents a set of pipelines that all have the same sink
// class GPUMetaPipeline : public enable_shared_from_this<GPUMetaPipeline> {
class GPUMetaPipeline : public enable_shared_from_this<GPUMetaPipeline>{
	//! We follow these rules when building:
	//! 1. For joins, build out the blocking side before going down the probe side
	//!     - The current streaming pipeline will have a dependency on it (dependency across MetaPipelines)
	//!     - Unions of this streaming pipeline will automatically inherit this dependency
	//! 2. Build child pipelines last (e.g., Hash Join becomes source after probe is done: scan HT for FULL OUTER JOIN)
	//!     - 'last' means after building out all other pipelines associated with this operator
	//!     - The child pipeline automatically has dependencies (within this GPUMetaPipeline) on:
	//!         * The 'current' streaming pipeline
	//!         * And all pipelines that were added to the GPUMetaPipeline after 'current'
public:
	//! Create a GPUMetaPipeline with the given sink
	GPUMetaPipeline(GPUExecutor &gpu_executor, GPUPipelineBuildState &state, optional_ptr<GPUPhysicalOperator> sink);

public:
	//! Get the GPUExecutor for this GPUMetaPipeline
	GPUExecutor &GetExecutor() const;
	//! Get the PipelineBuildState for this GPUMetaPipeline
	GPUPipelineBuildState &GetState() const;
	//! Get the sink operator for this GPUMetaPipeline
	optional_ptr<GPUPhysicalOperator> GetSink() const;

	//! Get the initial pipeline of this GPUMetaPipeline
	shared_ptr<GPUPipeline> &GetBasePipeline();
	//! Get the pipelines of this GPUMetaPipeline
	void GetPipelines(vector<shared_ptr<GPUPipeline>> &result, bool recursive);
	//! Get the GPUMetaPipeline children of this GPUMetaPipeline
	void GetMetaPipelines(vector<shared_ptr<GPUMetaPipeline>> &result, bool recursive, bool skip);
	//! Get the dependencies (within this GPUMetaPipeline) of the given Pipeline
	optional_ptr<const vector<reference<GPUPipeline>>> GetDependencies(GPUPipeline &dependant) const;
	//! Whether this GPUMetaPipeline has a recursive CTE
	bool HasRecursiveCTE() const;
	//! Set the flag that this GPUMetaPipeline is a recursive CTE pipeline
	void SetRecursiveCTE();
	//! Assign a batch index to the given pipeline
	void AssignNextBatchIndex(GPUPipeline &pipeline);
	//! Let 'dependant' depend on all pipeline that were created since 'start',
	//! where 'including' determines whether 'start' is added to the dependencies
	void AddDependenciesFrom(GPUPipeline &dependant, GPUPipeline &start, bool including);
	//! Make sure that the given pipeline has its own PipelineFinishEvent (e.g., for IEJoin - double Finalize)
	void AddFinishEvent(GPUPipeline &pipeline);
	//! Whether the pipeline needs its own PipelineFinishEvent
	bool HasFinishEvent(GPUPipeline &pipeline) const;
	//! Whether this pipeline is part of a PipelineFinishEvent
	optional_ptr<GPUPipeline> GetFinishGroup(GPUPipeline &pipeline) const;

	void BuildGPUPipelines(GPUPhysicalOperator &node, GPUPipeline &current);

public:
	//! Build the GPUMetaPipeline with 'op' as the first operator (excl. the shared sink)
	void Build(GPUPhysicalOperator &op);
	//! Ready all the pipelines (recursively)
	void Ready();

	//! Create an empty pipeline within this GPUMetaPipeline
	GPUPipeline &CreatePipeline();
	//! Create a union pipeline (clone of 'current')
	GPUPipeline &CreateUnionPipeline(GPUPipeline &current, bool order_matters);
	//! Create a child pipeline op 'current' starting at 'op',
	//! where 'last_pipeline' is the last pipeline added before building out 'current'
	void CreateChildPipeline(GPUPipeline &current, GPUPhysicalOperator &op, GPUPipeline &last_pipeline);
	//! Create a GPUMetaPipeline child that 'current' depends on
	GPUMetaPipeline &CreateChildMetaPipeline(GPUPipeline &current, GPUPhysicalOperator &op);

private:
	//! The executor for all MetaPipelines in the query plan
	GPUExecutor &executor;
	//! The PipelineBuildState for all MetaPipelines in the query plan
	GPUPipelineBuildState &state;
	//! The sink of all pipelines within this GPUMetaPipeline
	optional_ptr<GPUPhysicalOperator> sink;
	//! Whether this GPUMetaPipeline is a the recursive pipeline of a recursive CTE
	bool recursive_cte;
	//! All pipelines with a different source, but the same sink
	vector<shared_ptr<GPUPipeline>> pipelines;
	//! Dependencies within this GPUMetaPipeline
	reference_map_t<GPUPipeline, vector<reference<GPUPipeline>>> dependencies;
	//! Other MetaPipelines that this GPUMetaPipeline depends on
	vector<shared_ptr<GPUMetaPipeline>> children;
	//! Next batch index
	idx_t next_batch_index;
	//! Pipelines (other than the base pipeline) that need their own PipelineFinishEvent (e.g., for IEJoin)
	reference_set_t<GPUPipeline> finish_pipelines;
	//! Mapping from pipeline (e.g., child or union) to finish pipeline
	reference_map_t<GPUPipeline, GPUPipeline &> finish_map;
};

} // namespace duckdb
