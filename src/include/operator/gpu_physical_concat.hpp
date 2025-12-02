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
#include "operator/gpu_physical_hash_join.hpp"
#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "operator/gpu_physical_order.hpp"
#include "operator/gpu_physical_top_n.hpp"

namespace duckdb {

class GPUPhysicalConcat : public GPUPhysicalOperator {
public:
	static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::INVALID;

	explicit GPUPhysicalConcat(vector<LogicalType> types, idx_t estimated_cardinality);

  	string GetName() const override;

	bool IsSource() const override;

    bool IsSink() const override;

private:
    vector<idx_t> _partition_keys;
	idx_t _num_partitions;
};
} // namespace duckdb