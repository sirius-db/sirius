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

#include "duckdb/common/common.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

namespace sirius {

class sirius_engine;
struct operator_params;

namespace pipeline {

//! Result of converting meta-pipelines into execution-ready pipelines
struct pipeline_conversion_result {
  //! The execution-ready pipelines in dependency order
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> scheduled_pipelines;
  //! Ownership container for operators created during splitting (PARTITION, CONCAT, MERGE, etc.)
  duckdb::vector<duckdb::unique_ptr<op::sirius_physical_operator>> pipeline_breakers;
  //! Number of meta-pipelines (used for progress tracking)
  duckdb::idx_t meta_pipeline_count;
};

//! Converts DuckDB-style meta-pipelines into Sirius execution-ready pipelines.
//!
//! This class handles the full pipeline conversion process:
//! 1. Schedule meta-pipelines in dependency order and deep-copy them
//! 2. Split pipelines by operator type (insert PARTITION/CONCAT/MERGE/SORT operators)
//! 3. Wire data repositories between pipelines
//! 4. Set up parent-child pipeline dependencies
//! 5. Finalize pipeline structure (merge sink into operators vector)
//! 6. Link hash join sibling partition operators
class sirius_pipeline_converter {
 public:
  sirius_pipeline_converter(sirius_engine& engine, const sirius::operator_params& op_params);

  //! Convert meta-pipelines into execution-ready pipelines
  pipeline_conversion_result convert(sirius_meta_pipeline& root_pipeline);

 private:
  // Phase 1: Schedule and deep-copy
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> schedule_and_copy_pipelines(
    sirius_meta_pipeline& root_pipeline);

  // Phase 2: Split pipelines by operator type
  void split_pipelines(duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled);
  void split_table_scan_source(duckdb::shared_ptr<sirius_pipeline>& current_pipeline);
  void split_intermediate_joins(duckdb::shared_ptr<sirius_pipeline>& current_pipeline);
  void split_join_sink(duckdb::shared_ptr<sirius_pipeline>& current_pipeline);
  void split_group_aggregate_sink(
    duckdb::shared_ptr<sirius_pipeline>& current_pipeline,
    duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled,
    size_t pipeline_idx);
  void split_order_by_sink(duckdb::shared_ptr<sirius_pipeline>& current_pipeline,
                           duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled,
                           size_t pipeline_idx);
  void split_top_n_sink(duckdb::shared_ptr<sirius_pipeline>& current_pipeline,
                        duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled,
                        size_t pipeline_idx);
  void split_delim_join_sink(duckdb::shared_ptr<sirius_pipeline>& current_pipeline,
                             duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled,
                             size_t pipeline_idx);

  // Phase 3: Wire data repositories
  void wire_data_repositories();

  // Phase 4: Set up dependencies
  void setup_pipeline_parents();
  void finalize_pipeline_structure();
  void link_join_partition_siblings();

  // Phase 5: Debug logging
  void log_pipeline_debug_info() const;

  sirius_engine& engine_;
  const sirius::operator_params& op_params_;

  // Internal state built during convert(), moved into result
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> scheduled_;
  duckdb::vector<duckdb::unique_ptr<op::sirius_physical_operator>> pipeline_breakers_;
  duckdb::idx_t meta_pipeline_count_ = 0;
};

}  // namespace pipeline
}  // namespace sirius
