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
#include "op/scan/iceberg_metadata_reader.hpp"
#include "op/sirius_physical_operator.hpp"
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace duckdb {
class ClientContext;
}  // namespace duckdb

namespace sirius {

struct operator_params;

namespace pipeline {

//! Result of converting meta-pipelines into execution-ready pipelines
struct pipeline_conversion_result {
  //! The execution-ready pipelines in dependency order
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> scheduled_pipelines;
  //! Ownership container for operators inserted during splitting (PARTITION, CONCAT, MERGE,
  //! and source-side operators such as DUCKDB_SCAN, GPU_SCAN, CPU_SOURCE).
  duckdb::vector<duckdb::unique_ptr<op::sirius_physical_operator>> inserted_operators;
  //! Plan-time wiring descriptors. Materialized into runtime repositories and ports by
  //! `materialize_repository_wiring()` after the converter returns.
  std::vector<repository_wiring> repository_wirings;
  //! Number of meta-pipelines (used for progress tracking)
  std::size_t meta_pipeline_count;
};

//! Converts DuckDB-style meta-pipelines into Sirius execution-ready pipelines.
//!
//! This class handles the full pipeline conversion process:
//! 1. Schedule meta-pipelines in dependency order and deep-copy them
//! 2. Split pipelines by operator type (insert PARTITION/CONCAT/MERGE/SORT operators)
//! 3. Compute plan-time wiring descriptors
//! 4. Set up parent-child pipeline dependencies
//! 5. Finalize pipeline structure (merge sink into operators vector)
//! 6. Link hash join sibling partition operators

//! Construct a Sirius-specific operator (scan, merge) from a generic physical operator.
//! This is a pure factory function with no engine/context dependency.
duckdb::unique_ptr<op::sirius_physical_operator> construct_sirius_specific_operator(
  op::sirius_physical_operator& physical_op,
  const std::unordered_map<std::string, std::shared_ptr<const op::scan::IcebergDeleteData>>*
    iceberg_cache);

//! Deterministic, line-oriented serialization of a pipeline_conversion_result, used for
//! differential testing between the flag-OFF (legacy) and flag-ON (tree-based) conversion
//! paths: equivalent graphs produce byte-identical output regardless of emission order.
std::string dump_pipeline_conversion_result(const pipeline_conversion_result& result);

//! Reorders `pipelines` in place into a deterministic, strictly-topological schedule
//! (every pipeline after all of its `dependencies`), renumbers `pipeline_id` to the new
//! positions, and re-sorts each pipeline's `dependencies` ascending by the new ids.
//! Only the tree-based path needs this; the legacy path already emits topologically.
//!
//! Producers are visited in `dependencies` slot order, so with join dependencies kept
//! build-first (see `finalize_pipeline_structure`) every join's build subtree is
//! scheduled before its probe subtree. Pipelines launch in id order, so this is what
//! lets a join publish its dynamic filters before the probe-side scans they prune are
//! launched — probe-first numbering silently degrades those scans to full, unfiltered
//! reads.
void reorder_pipelines_topologically(
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines);

class sirius_pipeline_converter {
 public:
  sirius_pipeline_converter(
    const pipeline_build_context& ctx,
    const sirius::operator_params& op_params,
    const std::unordered_map<std::string, std::shared_ptr<const op::scan::IcebergDeleteData>>*
      iceberg_cache                       = nullptr,
    duckdb::ClientContext* client_context = nullptr);

  //! Convert meta-pipelines into execution-ready pipelines.
  //! No runtime state required: wiring is emitted as descriptors in the result;
  //! `materialize_repository_wiring()` performs the runtime side after this returns.
  pipeline_conversion_result convert(sirius_meta_pipeline& root_pipeline);

 private:
  // Phase 1: Schedule and deep-copy
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> schedule_and_copy_pipelines(
    sirius_meta_pipeline& root_pipeline);

  // Phase 2: Split pipelines by operator type
  void split_pipelines(duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& copied_scheduled);
  void insert_parquet_scan_operator(duckdb::shared_ptr<sirius_pipeline>& current_pipeline);
  void insert_duckdb_native_scan_operator(duckdb::shared_ptr<sirius_pipeline>& current_pipeline);
  void split_table_scan_source(duckdb::shared_ptr<sirius_pipeline>& current_pipeline);
  void split_cpu_source(duckdb::shared_ptr<sirius_pipeline>& current_pipeline);
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

  // Compute plan-time wiring descriptors. Runtime materialization is done by
  // `materialize_repository_wiring()` from `pipeline/repository_wiring.hpp`.
  void compute_repository_wiring(sirius_pipeline_build_state& state);
  // Tree-parent based wiring (USE_TREE_BASED_PIPELINE_BUILD). Assumes post-`is_ready`
  // pipelines and `_parent_op` populated by the plan generator's `set_parent_ops` pass.
  void compute_repository_wiring_tree_based(sirius_pipeline_build_state& state);
  static std::string_view resolve_port_id(const op::sirius_physical_operator& sink,
                                          const op::sirius_physical_operator& parent);
  static op::MemoryBarrierType resolve_barrier(const op::sirius_physical_operator& sink,
                                               const sirius_pipeline& dest);

  // Phase 4: Set up dependencies
  void setup_pipeline_parents();
  void finalize_pipeline_structure();
  void link_join_partition_siblings();

  // Phase 5: Debug logging
  void log_pipeline_debug_info() const;

  // Configure every sirius_physical_partition operator with the multi-GPU
  // partition floor so partition-consumer tasks (hash_join, merge_group_by)
  // have at least num_gpus partitions to spread across devices for big
  // inputs. Small tables stay at their natural partition count. Reads
  // num_gpus from build_ctx_; no-op when num_gpus <= 1.
  void configure_partition_min_partitions();

  const pipeline_build_context build_ctx_;
  const sirius::operator_params& op_params_;
  const std::unordered_map<std::string, std::shared_ptr<const op::scan::IcebergDeleteData>>*
    iceberg_cache_;
  duckdb::ClientContext* client_context_ = nullptr;

  // Internal state built during convert(), moved into result
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> scheduled_;
  duckdb::vector<duckdb::unique_ptr<op::sirius_physical_operator>> inserted_operators_;
  std::vector<repository_wiring> repository_wirings_;
  std::size_t meta_pipeline_count_ = 0;
};

}  // namespace pipeline
}  // namespace sirius
