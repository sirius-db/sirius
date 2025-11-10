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
#include "log/logging.hpp"

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

SinkResultType GPUPhysicalCTE::Sink(GPUIntermediateRelation &input_relation) const {
	// auto &lstate = input.local_state.Cast<CTELocalState>();
	// lstate.lhs_data.Append(lstate.append_state, chunk);

	// return SinkResultType::NEED_MORE_INPUT;
	SIRIUS_LOG_DEBUG("Sinking data into CTE");
	GPUBufferManager* gpuBufferManager = &(GPUBufferManager::GetInstance());
	for (int col_idx = 0; col_idx < input_relation.columns.size(); col_idx++) {
		// working_table_gpu->columns[col_idx] = make_shared_ptr<GPUColumn>(input_relation.columns[col_idx]->column_length, input_relation.columns[col_idx]->data_wrapper.type, input_relation.columns[col_idx]->data_wrapper.data);
		// working_table_gpu->columns[col_idx]->is_unique = input_relation.columns[col_idx]->is_unique;
		working_table_gpu->columns[col_idx] = make_shared_ptr<GPUColumn>(input_relation.columns[col_idx]);
		gpuBufferManager->lockAllocation(working_table_gpu->columns[col_idx]->data_wrapper.data, 0);
		gpuBufferManager->lockAllocation(working_table_gpu->columns[col_idx]->row_ids, 0);
		// If the column type is VARCHAR, also lock the offset allocation
		if (working_table_gpu->columns[col_idx]->data_wrapper.type.id() == GPUColumnTypeId::VARCHAR) {
			gpuBufferManager->lockAllocation(working_table_gpu->columns[col_idx]->data_wrapper.offset, 0);
		}
	}
    return SinkResultType::FINISHED;
}

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

} // namespace duckdb
