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

#include "duckdb/common/enums/statement_type.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "gpu_query_result.hpp"
#include "op/sirius_physical_operator.hpp"

namespace duckdb {
class SiriusPreparedStatementData;
}  // namespace duckdb

namespace sirius {

namespace pipeline {
class sirius_pipeline;
class sirius_meta_pipeline;
}  // namespace pipeline

namespace op {

class sirius_physical_result_collector : public sirius_physical_operator {
 public:
  static constexpr const duckdb::PhysicalOperatorType TYPE =
    duckdb::PhysicalOperatorType::RESULT_COLLECTOR;

 public:
  explicit sirius_physical_result_collector(duckdb::SiriusPreparedStatementData& data);

  duckdb::StatementType statement_type;
  duckdb::StatementProperties properties;
  sirius_physical_operator& plan;
  duckdb::vector<std::string> names;

 public:
  // //! The final method used to fetch the query result from this operator
  virtual duckdb::unique_ptr<duckdb::QueryResult> get_result(duckdb::GlobalSinkState& state) = 0;

  bool is_sink() const override { return true; }

 public:
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> get_children() const override;
  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;

  bool is_source() const override { return true; }
};

class sirius_physical_materialized_collector : public sirius_physical_result_collector {
 public:
  sirius_physical_materialized_collector(duckdb::SiriusPreparedStatementData& data);
  duckdb::unique_ptr<duckdb::GPUResultCollection> result_collection;

 public:
  duckdb::unique_ptr<duckdb::QueryResult> get_result(duckdb::GlobalSinkState& state) override;
};

}  // namespace op
}  // namespace sirius
