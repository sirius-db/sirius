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

#include "duckdb/main/client_context.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/logical_operator.hpp"

#include <set>

namespace sirius::util {

/// RAII guard that temporarily disables DuckDB optimizers incompatible with Sirius
/// and restores the original settings on destruction.
class optimizer_guard {
 public:
  optimizer_guard(duckdb::ClientContext& context, bool enable_optimizer) : context_(context)
  {
    original_config_ = context.config;
    original_disabled_optimizers_ =
      duckdb::DBConfig::GetConfig(context).options.disabled_optimizers;

    context.config.enable_optimizer = enable_optimizer;

    auto disabled = duckdb::DBConfig::GetConfig(context).options.disabled_optimizers;
    disabled.insert(duckdb::OptimizerType::IN_CLAUSE);
    disabled.insert(duckdb::OptimizerType::COMPRESSED_MATERIALIZATION);
    // STATISTICS_PROPAGATION folds ungrouped MIN/MAX aggregates into constant
    // expressions using partition statistics, producing EXPRESSION_GET + DUMMY_SCAN.
    // The GPU pipeline cannot schedule COLUMN_DATA_SCAN sources, so disable this
    // to keep the query on the scan -> aggregate path where the GPU can execute it.
    disabled.insert(duckdb::OptimizerType::STATISTICS_PROPAGATION);
#ifdef DEBUG
    disabled.insert(duckdb::OptimizerType::COLUMN_LIFETIME);
#endif
    duckdb::DBConfig::GetConfig(context).options.disabled_optimizers = disabled;
  }

  ~optimizer_guard()
  {
    duckdb::DBConfig::GetConfig(context_).options.disabled_optimizers =
      original_disabled_optimizers_;
    context_.config = original_config_;
  }

  optimizer_guard(const optimizer_guard&)            = delete;
  optimizer_guard& operator=(const optimizer_guard&) = delete;

 private:
  duckdb::ClientContext& context_;
  duckdb::ClientConfig original_config_;
  std::set<duckdb::OptimizerType> original_disabled_optimizers_;
};

/// Parse, plan, optimize, and resolve column bindings for a SQL query.
/// Returns the optimized logical plan ready for physical plan generation.
duckdb::unique_ptr<duckdb::LogicalOperator> extract_optimized_plan(duckdb::ClientContext& context,
                                                                   const std::string& query,
                                                                   bool enable_optimizer = true);

}  // namespace sirius::util
