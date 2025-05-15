#include "duckdb/execution/operator/set/physical_cte.hpp"

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/aggregate_hashtable.hpp"
#include "duckdb/parallel/event.hpp"
// #include "duckdb/parallel/meta_pipeline.hpp"
// #include "duckdb/parallel/pipeline.hpp"

#include "gpu_buffer_manager.hpp"
#include "gpu_pipeline.hpp"
#include "gpu_meta_pipeline.hpp"
#include "gpu_physical_cte.hpp"

namespace duckdb {

GPUPhysicalCTE::GPUPhysicalCTE(string ctename, idx_t table_index, vector<LogicalType> types, unique_ptr<GPUPhysicalOperator> top,
                         unique_ptr<GPUPhysicalOperator> bottom, idx_t estimated_cardinality)
    : GPUPhysicalOperator(PhysicalOperatorType::CTE, std::move(types), estimated_cardinality), table_index(table_index),
      ctename(std::move(ctename)) {
	children.push_back(std::move(top));
	children.push_back(std::move(bottom));
}

GPUPhysicalCTE::~GPUPhysicalCTE() {
}

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
// class CTEGlobalState : public GlobalSinkState {
// public:
// 	explicit CTEGlobalState(ClientContext &context, const PhysicalCTE &op) : working_table_ref(op.working_table.get()) {
// 	}
// 	optional_ptr<ColumnDataCollection> working_table_ref;

// 	mutex lhs_lock;

// 	void MergeIT(ColumnDataCollection &input) {
// 		lock_guard<mutex> guard(lhs_lock);
// 		working_table_ref->Combine(input);
// 	}
// };

// class CTELocalState : public LocalSinkState {
// public:
// 	explicit CTELocalState(ClientContext &context, const PhysicalCTE &op)
// 	    : lhs_data(context, op.working_table->Types()) {
// 		lhs_data.InitializeAppend(append_state);
// 	}

// 	unique_ptr<LocalSinkState> distinct_state;
// 	ColumnDataCollection lhs_data;
// 	ColumnDataAppendState append_state;

// 	void Append(DataChunk &input) {
// 		lhs_data.Append(input);
// 	}
// };

// unique_ptr<GlobalSinkState> PhysicalCTE::GetGlobalSinkState(ClientContext &context) const {
// 	working_table->Reset();
// 	return make_uniq<CTEGlobalState>(context, *this);
// }

// unique_ptr<LocalSinkState> PhysicalCTE::GetLocalSinkState(ExecutionContext &context) const {
// 	auto state = make_uniq<CTELocalState>(context.client, *this);
// 	return std::move(state);
// }

SinkResultType GPUPhysicalCTE::Sink(GPUIntermediateRelation &input_relation) const {
	// auto &lstate = input.local_state.Cast<CTELocalState>();
	// lstate.lhs_data.Append(lstate.append_state, chunk);

	// return SinkResultType::NEED_MORE_INPUT;
	printf("Sinking data into CTE\n");
	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	for (int col_idx = 0; col_idx < input_relation.columns.size(); col_idx++) {
		working_table_gpu->columns[col_idx] = make_shared_ptr<GPUColumn>(input_relation.columns[col_idx]->column_length, input_relation.columns[col_idx]->data_wrapper.type, input_relation.columns[col_idx]->data_wrapper.data);
		working_table_gpu->columns[col_idx]->is_unique = input_relation.columns[col_idx]->is_unique;
		gpuBufferManager->lockAllocation(working_table_gpu->columns[col_idx]->data_wrapper.data, 0);
		gpuBufferManager->lockAllocation(working_table_gpu->columns[col_idx]->row_ids, 0);
		// If the column type is VARCHAR, also lock the offset allocation
		if (working_table_gpu->columns[col_idx]->data_wrapper.type == ColumnType::VARCHAR) {
			gpuBufferManager->lockAllocation(working_table_gpu->columns[col_idx]->data_wrapper.offset, 0);
		}
	}
    return SinkResultType::FINISHED;
}

// SinkCombineResultType PhysicalCTE::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
// 	auto &lstate = input.local_state.Cast<CTELocalState>();
// 	auto &gstate = input.global_state.Cast<CTEGlobalState>();
// 	gstate.MergeIT(lstate.lhs_data);

// 	return SinkCombineResultType::FINISHED;
// }

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void GPUPhysicalCTE::BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) {
	D_ASSERT(children.size() == 2);
	op_state.reset();
	sink_state.reset();

	auto &state = meta_pipeline.GetState();

	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	child_meta_pipeline.Build(*children[0]);

	for (auto &cte_scan : cte_scans) {
		state.cte_dependencies.insert(make_pair(cte_scan, reference<GPUPipeline>(*child_meta_pipeline.GetBasePipeline())));
	}

	children[1]->BuildPipelines(current, meta_pipeline);
}

vector<const_reference<GPUPhysicalOperator>> GPUPhysicalCTE::GetSources() const {
	return children[1]->GetSources();
}

// InsertionOrderPreservingMap<string> GPUPhysicalCTE::ParamsToString() const {
// 	InsertionOrderPreservingMap<string> result;
// 	result["CTE Name"] = ctename;
// 	result["Table Index"] = StringUtil::Format("%llu", table_index);
// 	return result;
// }

} // namespace duckdb
