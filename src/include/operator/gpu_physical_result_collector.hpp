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

#include "gpu_physical_operator.hpp"
#include "duckdb/common/enums/statement_type.hpp"
#include "gpu_buffer_manager.hpp"
#include "gpu_query_result.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"

namespace duckdb {

class GPUPreparedStatementData;

class GPUPhysicalResultCollector : public GPUPhysicalOperator {

public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::RESULT_COLLECTOR;

public:
	explicit GPUPhysicalResultCollector(GPUPreparedStatementData &data);

	StatementType statement_type;
	StatementProperties properties;
	GPUPhysicalOperator &plan;
	vector<string> names;
	GPUBufferManager *gpuBufferManager;

// public:
// 	static unique_ptr<PhysicalResultCollector> GetResultCollector(ClientContext &context, PreparedStatementData &data);

public:
	// //! The final method used to fetch the query result from this operator
	virtual unique_ptr<QueryResult> GetResult(GlobalSinkState &state) = 0;
	
	bool IsSink() const override {
		return true;
	}

public:
	vector<const_reference<GPUPhysicalOperator>> GetChildren() const override;
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) override;

	bool IsSource() const override {
		return true;
	}
};


class GPUPhysicalMaterializedCollector : public GPUPhysicalResultCollector {
public:
	GPUPhysicalMaterializedCollector(GPUPreparedStatementData &data);
	unique_ptr<GPUResultCollection> result_collection;

public:
	unique_ptr<QueryResult> GetResult(GlobalSinkState &state) override;

public:
	// Sink interface
	static SinkResultType ConvertGPUTableToCPUCollection(
		GPUIntermediateRelation& input_relation, const vector<LogicalType>& types,
		GPUResultCollection* result_collection, GPUBufferManager *gpuBufferManager);
	SinkResultType Sink(GPUIntermediateRelation &input_relation) const override;

	unique_ptr<LocalSinkState> GetLocalSinkState(ExecutionContext &context) const override;
	unique_ptr<GlobalSinkState> GetGlobalSinkState(ClientContext &context) const override;

	template <typename T> static void FinalMaterializeInternal(GPUIntermediateRelation input_relation, GPUIntermediateRelation& output_relation, size_t col);
	static void FinalMaterializeString(GPUIntermediateRelation input_relation, GPUIntermediateRelation& output_relation, size_t col);
	static size_t FinalMaterialize(GPUIntermediateRelation input_relation, GPUIntermediateRelation& output_relation, size_t col);

};

} // namespace duckdb