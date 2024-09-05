#include "operator/gpu_physical_column_data_scan.hpp"
#include "operator/gpu_physical_delim_join.hpp"
#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_meta_pipeline.hpp"

namespace duckdb {

GPUPhysicalColumnDataScan::GPUPhysicalColumnDataScan(vector<LogicalType> types, PhysicalOperatorType op_type,
                                               idx_t estimated_cardinality, optionally_owned_ptr<ColumnDataCollection> collection_p)
    : GPUPhysicalOperator(op_type, std::move(types), estimated_cardinality), collection(std::move(collection_p)) {
}

GPUPhysicalColumnDataScan::GPUPhysicalColumnDataScan(vector<LogicalType> types, PhysicalOperatorType op_type,
                                               idx_t estimated_cardinality, idx_t cte_index)
    : GPUPhysicalOperator(op_type, std::move(types), estimated_cardinality), collection(nullptr), cte_index(cte_index) {
}

// SourceResultType 
// GPUPhysicalColumnDataScan::GetData(ExecutionContext &context, GPUIntermediateRelation &output_relation,
//                                                  OperatorSourceInput &input) const {
SourceResultType 
GPUPhysicalColumnDataScan::GetData(GPUIntermediateRelation &output_relation) const {
	// auto &state = input.global_state.Cast<PhysicalColumnDataScanState>();
	// if (collection->Count() == 0) {
	// 	return SourceResultType::FINISHED;
	// }
	// if (!state.initialized) {
	// 	collection->InitializeScan(state.scan_state);
	// 	state.initialized = true;
	// }
	// collection->Scan(state.scan_state, chunk);

	// return chunk.size() == 0 ? SourceResultType::FINISHED : SourceResultType::HAVE_MORE_OUTPUT;

	printf("Reading data from column data collection\n");
	for (int col_idx = 0; col_idx < output_relation.columns.size(); col_idx++) {
		output_relation.columns[col_idx] = intermediate_relation->columns[col_idx];
	}

	return SourceResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void 
GPUPhysicalColumnDataScan::BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) {
	// check if there is any additional action we need to do depending on the type
	auto &state = meta_pipeline.GetState();
	switch (type) {
	case PhysicalOperatorType::DELIM_SCAN: {
		auto entry = state.delim_join_dependencies.find(*this);
		D_ASSERT(entry != state.delim_join_dependencies.end());
		// this chunk scan introduces a dependency to the current pipeline
		// namely a dependency on the duplicate elimination pipeline to finish
		auto delim_dependency = entry->second.get().shared_from_this();
		auto delim_sink = state.GetPipelineSink(*delim_dependency);
		D_ASSERT(delim_sink);
		D_ASSERT(delim_sink->type == PhysicalOperatorType::LEFT_DELIM_JOIN ||
		         delim_sink->type == PhysicalOperatorType::RIGHT_DELIM_JOIN);
		auto &delim_join = delim_sink->Cast<GPUPhysicalDelimJoin>();
		current.AddDependency(delim_dependency);
		state.SetPipelineSource(current, delim_join.distinct->Cast<GPUPhysicalOperator>());
		return;
	}
	case PhysicalOperatorType::CTE_SCAN: {
        throw NotImplementedException("CTE scan not implemented for GPU");
		// auto entry = state.cte_dependencies.find(*this);
		// D_ASSERT(entry != state.cte_dependencies.end());
		// // this chunk scan introduces a dependency to the current pipeline
		// // namely a dependency on the CTE pipeline to finish
		// auto cte_dependency = entry->second.get().shared_from_this();
		// auto cte_sink = state.GetPipelineSink(*cte_dependency);
		// (void)cte_sink;
		// D_ASSERT(cte_sink);
		// D_ASSERT(cte_sink->type == PhysicalOperatorType::CTE);
		// current.AddDependency(cte_dependency);
		// state.SetPipelineSource(current, *this);
		// return;
	}
	case PhysicalOperatorType::RECURSIVE_CTE_SCAN:
        throw NotImplementedException("Recursive CTE scan not implemented for GPU");
		if (!meta_pipeline.HasRecursiveCTE()) {
			throw InternalException("Recursive CTE scan found without recursive CTE node");
		}
		break;
	default:
		break;
	}
	D_ASSERT(children.empty());
	state.SetPipelineSource(current, *this);
}

} // namespace duckdb