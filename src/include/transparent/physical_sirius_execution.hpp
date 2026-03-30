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

#include "op/sirius_physical_operator.hpp"

#include <duckdb/common/enums/physical_operator_type.hpp>
#include <duckdb/execution/physical_operator.hpp>

namespace sirius::transparent {

/// \brief A DuckDB PhysicalOperator that transparently wraps Sirius GPU execution.
///
/// This operator replaces DuckDB's normal physical plan when transparent GPU execution is
/// enabled. It acts as a source operator: DuckDB's executor calls GetData() to retrieve
/// results, which are produced by executing the Sirius physical plan on the GPU.
///
/// Created by SiriusContext::OnFinalizePrepare when the query is GPU-acceleratable.
class PhysicalSiriusExecution : public duckdb::PhysicalOperator {
 public:
  static constexpr const duckdb::PhysicalOperatorType TYPE =
    duckdb::PhysicalOperatorType::EXTENSION;

  PhysicalSiriusExecution(duckdb::PhysicalPlan& physical_plan,
                          duckdb::unique_ptr<sirius::op::sirius_physical_operator> sirius_plan,
                          duckdb::vector<duckdb::LogicalType> types,
                          duckdb::vector<std::string> names,
                          duckdb::idx_t estimated_cardinality);

  // Source operator interface
  bool IsSource() const override { return true; }
  duckdb::unique_ptr<duckdb::GlobalSourceState> GetGlobalSourceState(
    duckdb::ClientContext& context) const override;
  duckdb::unique_ptr<duckdb::LocalSourceState> GetLocalSourceState(
    duckdb::ExecutionContext& context, duckdb::GlobalSourceState& gstate) const override;
  duckdb::SourceResultType GetData(duckdb::ExecutionContext& context,
                                   duckdb::DataChunk& chunk,
                                   duckdb::OperatorSourceInput& input) const override;

  std::string GetName() const override { return "SIRIUS_GPU_EXECUTION"; }

  /// The Sirius physical plan. Mutable because GetGlobalSourceState is const but we
  /// need to move the plan into the execution state.
  mutable duckdb::unique_ptr<sirius::op::sirius_physical_operator> sirius_plan_;

  /// Output column names (needed for result construction).
  duckdb::vector<std::string> result_names_;
};

}  // namespace sirius::transparent
