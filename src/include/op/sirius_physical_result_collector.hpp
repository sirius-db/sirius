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

// sirius
#include <gpu_query_result.hpp>
#include <op/sirius_physical_operator.hpp>
#include <sirius_context.hpp>

// duckdb
#include <duckdb/common/enums/statement_type.hpp>
#include <duckdb/common/types/column/column_data_collection.hpp>
#include <duckdb/main/client_context.hpp>

namespace sirius {
class sirius_prepared_statement_data;

namespace pipeline {
class sirius_pipeline;
class sirius_meta_pipeline;
}  // namespace pipeline

namespace op {

class sirius_physical_result_collector : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE =
    SiriusPhysicalOperatorType::RESULT_COLLECTOR;

 public:
  explicit sirius_physical_result_collector(::sirius::sirius_prepared_statement_data& data);

  operator_data execute(const operator_data& input_data, rmm::cuda_stream_view stream) override;

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
  sirius_physical_materialized_collector(::sirius::sirius_prepared_statement_data& data,
                                         duckdb::ClientContext& client_ctx);
  duckdb::unique_ptr<duckdb::ColumnDataCollection> result_collection;

 public:
  /**
   * @brief Fetch the final query result from the result collection
   *
   * @param[in] state The global sink state (currently unused)
   * @return The query result
   */
  duckdb::unique_ptr<duckdb::QueryResult> get_result(duckdb::GlobalSinkState& state) override;

  /**
   * @brief Sink a data batch into the result collection
   *
   * @param[in] input_batch The input data batch
   * @throws InvalidInputException if input_batches is empty, if any data_batch has no data, or if
   * any data_batch currently resides in the DISK tier
   * @throws InternalException if the memory manager is not initialized, if the reservation fails,
   * or if the memory space for the reservation is invalid
   * @note For now, we assume that the input batch, if in the HOST tier, is always in the
   * host_table_representation. If it is in the GPU tier, we convert it to the
   * host_table_representation. In the future, we should register converters for other specialized
   * data representations and invoke one such here.
   */
  void sink(const operator_data& input_data, rmm::cuda_stream_view stream) override;

 private:
  duckdb::ClientContext& _client_ctx;
};

}  // namespace op
}  // namespace sirius
