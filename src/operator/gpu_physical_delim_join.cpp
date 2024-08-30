#include "operator/gpu_physical_delim_join.hpp"
#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_meta_pipeline.hpp"
#include "operator/gpu_physical_hash_join.hpp"
#include "operator/gpu_physical_column_data_scan.hpp"
#include "duckdb/execution/operator/join/physical_left_delim_join.hpp"
#include "duckdb/execution/operator/join/physical_right_delim_join.hpp"

namespace duckdb {

class GPULeftDelimJoinGlobalState : public GlobalSinkState {
public:
	explicit GPULeftDelimJoinGlobalState(ClientContext &context, const GPUPhysicalLeftDelimJoin &delim_join)
	    : lhs_data(context, delim_join.children[0]->GetTypes()) {
		D_ASSERT(!delim_join.delim_scans.empty());
		// set up the delim join chunk to scan in the original join
		auto &cached_chunk_scan = delim_join.join->children[0]->Cast<GPUPhysicalColumnDataScan>();
		cached_chunk_scan.collection = &lhs_data;
	}

	ColumnDataCollection lhs_data;
	mutex lhs_lock;

	void Merge(ColumnDataCollection &input) {
		lock_guard<mutex> guard(lhs_lock);
		lhs_data.Combine(input);
	}
};

class GPULeftDelimJoinLocalState : public LocalSinkState {
public:
	explicit GPULeftDelimJoinLocalState(ClientContext &context, const GPUPhysicalLeftDelimJoin &delim_join)
	    : lhs_data(context, delim_join.children[0]->GetTypes()) {
		lhs_data.InitializeAppend(append_state);
	}

	unique_ptr<LocalSinkState> distinct_state;
	ColumnDataCollection lhs_data;
	ColumnDataAppendState append_state;

	void Append(DataChunk &input) {
		lhs_data.Append(input);
	}
};

class GPURightDelimJoinGlobalState : public GlobalSinkState {};

class GPURightDelimJoinLocalState : public LocalSinkState {
public:
	unique_ptr<LocalSinkState> join_state;
	unique_ptr<LocalSinkState> distinct_state;
};

GPUPhysicalDelimJoin::GPUPhysicalDelimJoin(PhysicalOperatorType type, vector<LogicalType> types,
                                     unique_ptr<GPUPhysicalOperator> original_join,
                                     vector<const_reference<GPUPhysicalOperator>> delim_scans, idx_t estimated_cardinality)
    : GPUPhysicalOperator(type, std::move(types), estimated_cardinality), join(std::move(original_join)),
      delim_scans(std::move(delim_scans)) {
	D_ASSERT(type == PhysicalOperatorType::LEFT_DELIM_JOIN || type == PhysicalOperatorType::RIGHT_DELIM_JOIN);
}

GPUPhysicalRightDelimJoin::GPUPhysicalRightDelimJoin(vector<LogicalType> types, unique_ptr<GPUPhysicalOperator> original_join,
                                               vector<const_reference<GPUPhysicalOperator>> delim_scans,
                                               idx_t estimated_cardinality)
    : GPUPhysicalDelimJoin(PhysicalOperatorType::RIGHT_DELIM_JOIN, std::move(types), std::move(original_join),
                        std::move(delim_scans), estimated_cardinality) {

}

GPUPhysicalLeftDelimJoin::GPUPhysicalLeftDelimJoin(vector<LogicalType> types, unique_ptr<GPUPhysicalOperator> original_join,
                                             vector<const_reference<GPUPhysicalOperator>> delim_scans,
                                             idx_t estimated_cardinality)
    : GPUPhysicalDelimJoin(PhysicalOperatorType::LEFT_DELIM_JOIN, std::move(types), std::move(original_join),
                        std::move(delim_scans), estimated_cardinality) {
}

SinkResultType 
GPUPhysicalRightDelimJoin::Sink(ExecutionContext &context, GPUIntermediateRelation &input_relation,
                                            OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<GPURightDelimJoinLocalState>();

	OperatorSinkInput join_sink_input {*join->sink_state, *lstate.join_state, input.interrupt_state};
	join->Sink(context, input_relation, join_sink_input);

	OperatorSinkInput distinct_sink_input {*distinct->sink_state, *lstate.distinct_state, input.interrupt_state};
	distinct->Sink(context, input_relation, distinct_sink_input);

	return SinkResultType::FINISHED;
}

SinkResultType 
GPUPhysicalLeftDelimJoin::Sink(ExecutionContext &context, GPUIntermediateRelation &input_relation,
                                           OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<GPULeftDelimJoinLocalState>();
	// lstate.lhs_data.Append(lstate.append_state, chunk);
	OperatorSinkInput distinct_sink_input {*distinct->sink_state, *lstate.distinct_state, input.interrupt_state};
	distinct->Sink(context, input_relation, distinct_sink_input);
	return SinkResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void 
GPUPhysicalLeftDelimJoin::BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) {
	op_state.reset();
	sink_state.reset();

	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	child_meta_pipeline.Build(*children[0]);

	D_ASSERT(type == PhysicalOperatorType::LEFT_DELIM_JOIN);
	// recurse into the actual join
	// any pipelines in there depend on the main pipeline
	// any scan of the duplicate eliminated data on the RHS depends on this pipeline
	// we add an entry to the mapping of (PhysicalOperator*) -> (Pipeline*)
	auto &state = meta_pipeline.GetState();
	for (auto &delim_scan : delim_scans) {
		state.delim_join_dependencies.insert(
		    make_pair(delim_scan, reference<GPUPipeline>(*child_meta_pipeline.GetBasePipeline())));
	}
	join->BuildPipelines(current, meta_pipeline);
}

void GPUPhysicalRightDelimJoin::BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) {
	op_state.reset();
	sink_state.reset();

	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	child_meta_pipeline.Build(*children[0]);

	D_ASSERT(type == PhysicalOperatorType::RIGHT_DELIM_JOIN);
	// recurse into the actual join
	// any pipelines in there depend on the main pipeline
	// any scan of the duplicate eliminated data on the LHS depends on this pipeline
	// we add an entry to the mapping of (PhysicalOperator*) -> (Pipeline*)
	auto &state = meta_pipeline.GetState();
	for (auto &delim_scan : delim_scans) {
		state.delim_join_dependencies.insert(
		    make_pair(delim_scan, reference<GPUPipeline>(*child_meta_pipeline.GetBasePipeline())));
	}

	// Build join pipelines without building the RHS (already built in the Sink of this op)
	GPUPhysicalHashJoin::BuildJoinPipelines(current, meta_pipeline, *join, false);
}

}
