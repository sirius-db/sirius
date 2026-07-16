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

#include "duckdb/common/unordered_map.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "telemetry-bridge/gen/uuid.rs.h"
#include "telemetry/telemetry_context.hpp"

namespace sirius {
class sirius_engine;
}  // namespace sirius

namespace sirius::planner {

/**
 * @brief Represents a query execution plan with its associated pipelines.
 *
 * The query class manages the execution order of pipelines required to complete
 * a query. It provides methods to access scan operators, retrieve pipelines by
 * operator, and get all pipelines in execution order.
 */
class query {
 public:
  /**
   * @brief Construct a new query object.
   *
   * @param pipelines The ordered pipelines required to execute this query.
   * @param telemetry_info Info useful for emitting identifiable telemetry.
   */
  query(duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> pipelines,
        const quent::Context& context,
        telemetry::query_telemetry_info telemetry_info);

  ~query() = default;

  // Non-copyable
  query(const query&)            = delete;
  query& operator=(const query&) = delete;

  // Movable
  query(query&&)            = default;
  query& operator=(query&&) = default;

  /**
   * @brief Get the scan operators in pipeline execution order.
   *
   * @return Reference to the vector of pointers to scan operators in execution order.
   */
  [[nodiscard]] std::span<op::sirius_physical_operator* const> get_scan_operators() const;

  /**
   * @brief Get the pipeline containing a specific physical operator.
   *
   * @param op Pointer to the physical operator to look up.
   * @return Shared pointer to the pipeline containing the operator,
   *         or nullptr if not found.
   */
  duckdb::shared_ptr<pipeline::sirius_pipeline> get_pipeline(op::sirius_physical_operator* op);

  /**
   * @brief Get all pipelines in execution order.
   *
   * @return Reference to the vector of pipelines in execution order.
   */
  [[nodiscard]] const duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>>& get_pipelines()
    const;

 private:
  //! Builds the internal data structures from the pipelines
  void build_indices();

  //! Unique ID for this plan
  uuid::UUID _plan_id;
  //! Pipelines and the order in which they must be executed in order to successfully complete the
  // query.
  duckdb::vector<duckdb::shared_ptr<pipeline::sirius_pipeline>> _pipelines;
  //! Cached scan operators in pipeline execution order
  duckdb::vector<op::sirius_physical_operator*> _scan_operators;
  //! Map from operator pointer to its containing pipeline
  duckdb::unordered_map<op::sirius_physical_operator*,
                        duckdb::shared_ptr<pipeline::sirius_pipeline>>
    _operator_to_pipeline;
};

}  // namespace sirius::planner
