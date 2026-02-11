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

#include "duckdb/common/optionally_owned_ptr.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/execution/physical_operator.hpp"
#include "op/sirius_physical_operator.hpp"

namespace sirius {

namespace pipeline {
class sirius_pipeline;
class sirius_meta_pipeline;
}  // namespace pipeline

namespace op {

//! The sirius_physical_column_data_scan scans a ColumnDataCollection
class sirius_physical_column_data_scan : public sirius_physical_operator {
 public:
  static constexpr const SiriusPhysicalOperatorType TYPE = SiriusPhysicalOperatorType::INVALID;

 public:
  sirius_physical_column_data_scan(
    duckdb::vector<duckdb::LogicalType> types,
    SiriusPhysicalOperatorType op_type,
    duckdb::idx_t estimated_cardinality,
    duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection> collection);

  sirius_physical_column_data_scan(duckdb::vector<duckdb::LogicalType> types,
                                   SiriusPhysicalOperatorType op_type,
                                   duckdb::idx_t estimated_cardinality,
                                   duckdb::idx_t cte_index);

  //! (optionally owned) column data collection to scan
  duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection> collection;

  duckdb::idx_t cte_index;

  duckdb::optional_idx delim_index;

  std::vector<std::shared_ptr<cucascade::data_batch>> execute(
    const std::vector<std::shared_ptr<cucascade::data_batch>>& input_batches,
    rmm::cuda_stream_view stream) override;

 public:
  bool is_source() const override { return true; }

  void build_pipelines(pipeline::sirius_pipeline& current,
                       pipeline::sirius_meta_pipeline& meta_pipeline) override;
};

}  // namespace op
}  // namespace sirius
