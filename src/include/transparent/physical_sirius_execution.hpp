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
#include <duckdb/planner/logical_operator.hpp>

#include <atomic>

namespace duckdb {
class PreparedStatementData;
}  // namespace duckdb

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
                          duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan,
                          std::string query_sql,
                          duckdb::vector<duckdb::LogicalType> types,
                          duckdb::vector<std::string> names,
                          duckdb::shared_ptr<duckdb::PreparedStatementData> cpu_fallback_prepared,
                          bool cpu_plan_reads_s3,
                          duckdb::idx_t estimated_cardinality);

  // Source operator interface
  bool IsSource() const override { return true; }
  duckdb::unique_ptr<duckdb::GlobalSourceState> GetGlobalSourceState(
    duckdb::ClientContext& context) const override;
  duckdb::unique_ptr<duckdb::LocalSourceState> GetLocalSourceState(
    duckdb::ExecutionContext& context, duckdb::GlobalSourceState& gstate) const override;
  duckdb::SourceResultType GetDataInternal(duckdb::ExecutionContext& context,
                                           duckdb::DataChunk& chunk,
                                           duckdb::OperatorSourceInput& input) const override;

  std::string GetName() const override { return "SIRIUS_GPU_EXECUTION"; }

 private:
  /// A reusable copy of the optimized logical plan.
  /// DuckDB can execute the same prepared physical operator multiple times —
  /// and, with streaming results, from more than one thread of one connection
  /// — so this member is IMMUTABLE after construction (register E4): every
  /// execution takes its own copy_logical_plan() into per-execution state and
  /// nothing ever resets the shared template (a reset would destroy the plan
  /// under a concurrent execution's copy). May be null when the plan contains
  /// a non-Copy()-able LogicalGet (a table function whose bind_data has no
  /// serializer) — then every execution re-plans from `query_sql_`.
  duckdb::unique_ptr<duckdb::LogicalOperator> logical_plan_;

  /// Set (never cleared) when an execution discovers Copy() throws
  /// NotImplementedException, so later executions skip straight to the SQL
  /// replan path. A monotonic atomic flag: concurrent executions may race to
  /// set it, both write `true`, and a stale `false` read only costs one more
  /// failed Copy() attempt — never a use-after-free.
  mutable std::atomic<bool> plan_copy_unsupported_{false};

  /// Original SQL string used to re-plan when `logical_plan_` cannot be
  /// copied (e.g. queries against table functions whose bind_data does not
  /// implement serialization). Captured up-front because
  /// PreparedStatementData::unbound_statement is not yet populated when
  /// OnFinalizePrepare runs.
  std::string query_sql_;

  /// Output column names (needed for result construction).
  duckdb::vector<std::string> result_names_;

  /// DuckDB's CPU physical plan, wrapped in a minimal PreparedStatementData and
  /// stashed at OnFinalizePrepare before this operator replaced it. On a GPU
  /// execution failure, GetDataInternal runs it on a private duckdb::Executor bound
  /// to the same ClientContext (same transaction/MVCC snapshot). shared_ptr so the
  /// const source state can keep it alive across the nested run.
  duckdb::shared_ptr<duckdb::PreparedStatementData> cpu_fallback_prepared_;

  /// Whether the CPU fallback plan reads s3:// data. S3 is GPU-only (DuckDB's CPU
  /// read_parquet cannot serve Sirius-owned s3://), so a runtime GPU failure on an
  /// s3 query surfaces a clear error instead of falling back to CPU.
  bool cpu_plan_reads_s3_ = false;
};

}  // namespace sirius::transparent
