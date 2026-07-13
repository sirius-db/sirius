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
#include "pipeline/repository_wiring.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <string>
#include <vector>

namespace sirius {

struct operator_params;

namespace pipeline {

//! Result of converting meta-pipelines into execution-ready pipelines
struct pipeline_conversion_result {
  //! The execution-ready pipelines in dependency order
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> scheduled_pipelines;
  //! Plan-time wiring descriptors. Materialized into runtime repositories and ports by
  //! `materialize_repository_wiring()` after the converter returns.
  std::vector<repository_wiring> repository_wirings;
  //! Number of meta-pipelines (used for progress tracking)
  std::size_t meta_pipeline_count;
};

//! Canonical, line-oriented serialization for debugging: pipelines are signature-sorted,
//! so equivalent graphs print byte-identically regardless of emission order — which also
//! means it cannot show schedule-order differences; use `dump_pipeline_schedule_raw`
//! for those.
std::string dump_pipeline_conversion_result(const pipeline_conversion_result& result);

//! The unsorted counterpart: pipelines in true scheduled order, each with `dependencies`
//! as scheduled indices, wirings keyed by scheduled indices. Comparing two conversions of
//! the same query catches schedule-order nondeterminism, which matters because scan
//! registration and `task_scheduler::start_query` consume pipelines in scheduled order.
std::string dump_pipeline_schedule_raw(const pipeline_conversion_result& result);

//! Reorders `pipelines` in place into a deterministic, strictly-topological schedule
//! (every pipeline after all of its `dependencies`), renumbers `pipeline_id` to the new
//! positions, and re-sorts each pipeline's `dependencies` ascending by the new ids.
//! The raw meta-sweep emission is not topological, so this must run last in `convert()`.
//!
//! Producers are visited in `dependencies` slot order, so with join dependencies kept
//! build-first (see `finalize_pipeline_structure`) every join's build subtree is
//! scheduled before its probe subtree. Pipelines launch in id order, so this is what
//! lets a join publish its dynamic filters before the probe-side scans they prune are
//! launched — probe-first numbering silently degrades those scans to full, unfiltered
//! reads.
void reorder_pipelines_topologically(
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>>& pipelines);

//! Converts the meta-pipeline tree into Sirius execution-ready pipelines.
//!
//! The plan generator already placed every GPU operator in the physical plan tree and
//! per-operator `build_pipelines` produced final-shape pipelines, so conversion is a pure
//! topology pass:
//! 1. Schedule meta-pipelines in dependency order
//! 2. Compute plan-time wiring descriptors via tree-parent lookup
//! 3. Set up parent/dependency pipeline edges
//! 4. Link hash join sibling partition operators and multi-GPU partition floors
//! 5. Reorder into the canonical strictly-topological schedule
class sirius_pipeline_converter {
 public:
  sirius_pipeline_converter(const pipeline_build_context& ctx,
                            const sirius::operator_params& op_params);

  //! Convert meta-pipelines into execution-ready pipelines.
  //! No runtime state required: wiring is emitted as descriptors in the result;
  //! `materialize_repository_wiring()` performs the runtime side after this returns.
  pipeline_conversion_result convert(sirius_meta_pipeline& root_pipeline);

 private:
  //! Schedule the meta-pipeline tree's pipelines in dependency order.
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> schedule_pipelines(
    sirius_meta_pipeline& root_pipeline);

  //! Compute plan-time wiring descriptors from the operator tree. Assumes post-`is_ready`
  //! pipelines and `_parent_op` populated by the plan generator's `set_parent_ops` pass.
  //! Runtime materialization is done by `materialize_repository_wiring()` from
  //! `pipeline/repository_wiring.hpp`.
  void compute_repository_wiring(sirius_pipeline_build_state& state);
  static std::string_view resolve_port_id(const op::sirius_physical_operator& sink,
                                          const op::sirius_physical_operator& parent);
  static op::MemoryBarrierType resolve_barrier(const op::sirius_physical_operator& sink,
                                               const sirius_pipeline& dest);

  //! Set up parent/dependency pipeline edges from the wiring descriptors.
  void setup_pipeline_parents();
  void finalize_pipeline_structure();
  void link_join_partition_siblings();

  // Configure every sirius_physical_partition operator with the multi-GPU
  // partition floor so partition-consumer tasks (hash_join, merge_group_by)
  // have at least num_gpus partitions to spread across devices for big
  // inputs. Small tables stay at their natural partition count. Reads
  // num_gpus from build_ctx_; no-op when num_gpus <= 1.
  void configure_partition_min_partitions();

  const pipeline_build_context build_ctx_;
  const sirius::operator_params& op_params_;

  // Internal state built during convert(), moved into result
  duckdb::vector<duckdb::shared_ptr<sirius_pipeline>> scheduled_;
  std::vector<repository_wiring> repository_wirings_;
  std::size_t meta_pipeline_count_ = 0;
};

}  // namespace pipeline
}  // namespace sirius
