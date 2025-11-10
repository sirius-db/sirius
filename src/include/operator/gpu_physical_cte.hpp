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

#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "gpu_physical_operator.hpp"

namespace duckdb {

// class RecursiveCTEState;

class GPUPhysicalCTE : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::CTE;

public:
	GPUPhysicalCTE(string ctename, idx_t table_index, vector<LogicalType> types, unique_ptr<GPUPhysicalOperator> top,
	            unique_ptr<GPUPhysicalOperator> bottom, idx_t estimated_cardinality);
	~GPUPhysicalCTE() override;

	vector<const_reference<GPUPhysicalOperator>> cte_scans;

	shared_ptr<ColumnDataCollection> working_table;

	shared_ptr<GPUIntermediateRelation> working_table_gpu;

	idx_t table_index;
	string ctename;

public:
	// Sink interface
	SinkResultType Sink(GPUIntermediateRelation &input_relation) const override;

	bool IsSink() const override {
		return true;
	}

	bool ParallelSink() const override {
		return true;
	}

	bool SinkOrderDependent() const override {
		return false;
	}

public:
	void BuildPipelines(GPUPipeline &current, GPUMetaPipeline &meta_pipeline) override;

	vector<const_reference<GPUPhysicalOperator>> GetSources() const override;
};

} // namespace duckdb
