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

#include "duckdb/execution/physical_operator.hpp"
#include "gpu_physical_operator.hpp"

namespace duckdb {

class GPUPhysicalEmptyResult : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EMPTY_RESULT;

public:
	explicit GPUPhysicalEmptyResult(vector<LogicalType> types, idx_t estimated_cardinality)
	    : GPUPhysicalOperator(PhysicalOperatorType::EMPTY_RESULT, std::move(types), estimated_cardinality) {
	}

public:
	SourceResultType GetData(GPUIntermediateRelation& output_relation) const override;

	bool IsSource() const override {
		return true;
	}
};
} // namespace duckdb
