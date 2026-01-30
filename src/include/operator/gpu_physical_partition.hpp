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
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "gpu_physical_operator.hpp"
#include "operator/gpu_physical_grouped_aggregate.hpp"
#include "operator/gpu_physical_hash_join.hpp"
#include "operator/gpu_physical_order.hpp"
#include "operator/gpu_physical_top_n.hpp"

#define PARTITION_SIZE 10000000

namespace duckdb {

class GPUPhysicalPartition : public GPUPhysicalOperator {
 public:
  static constexpr const PhysicalOperatorType TYPE = PhysicalOperatorType::EXTENSION;

  explicit GPUPhysicalPartition(vector<LogicalType> types,
                                idx_t estimated_cardinality,
                                GPUPhysicalOperator* parent_op,
                                bool is_build = false);

  string GetName() const override;

  bool IsSource() const override;

  bool IsSink() const override;

  bool isBuildPartition();

  //! Get the parent operator (e.g., HASH_JOIN for build partition)
  GPUPhysicalOperator* getParentOp() const { return _parent_op; }

 private:
  void GetPartitionKeys(GPUPhysicalOperator* op, bool is_build = false);
  GPUPhysicalOperator* _parent_op;
  vector<idx_t> _partition_keys;
  idx_t _num_partitions;
  bool _is_build;
};
}  // namespace duckdb
