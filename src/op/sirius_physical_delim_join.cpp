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

#include "op/sirius_physical_delim_join.hpp"

#include "config.hpp"
#include "op/sirius_physical_column_data_scan.hpp"
#include "op/sirius_physical_dummy_scan.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "op/sirius_physical_hash_join.hpp"
#include "op/sirius_physical_nested_loop_join.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"

#include <nvtx3/nvtx3.hpp>

namespace sirius {
namespace op {

class sirius_left_delim_join_local_state : public duckdb::LocalSinkState {
 public:
  duckdb::unique_ptr<duckdb::LocalSinkState> distinct_state;
  // duckdb::shared_ptr<GPUIntermediateRelation> lhs_data;
  duckdb::ColumnDataAppendState append_state;
};

class sirius_right_delim_join_local_state : public duckdb::LocalSinkState {
 public:
  duckdb::unique_ptr<duckdb::LocalSinkState> join_state;
  duckdb::unique_ptr<duckdb::LocalSinkState> distinct_state;
};

sirius_physical_delim_join::sirius_physical_delim_join(
  SiriusPhysicalOperatorType type,
  duckdb::vector<sirius::logical_type> types,
  duckdb::unique_ptr<sirius_physical_operator> original_join,
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> delim_scans,
  std::size_t estimated_cardinality,
  duckdb::optional_idx delim_idx)
  : sirius_physical_operator(type, std::move(types), estimated_cardinality),
    join(std::move(original_join)),
    delim_scans(std::move(delim_scans))
{
  D_ASSERT(type == SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
           type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
}

sirius_physical_right_delim_join::sirius_physical_right_delim_join(
  duckdb::vector<sirius::logical_type> types,
  duckdb::unique_ptr<sirius_physical_operator> original_join,
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> delim_scans,
  std::size_t estimated_cardinality,
  duckdb::optional_idx delim_idx)
  : sirius_physical_delim_join(SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN,
                               std::move(types),
                               std::move(original_join),
                               std::move(delim_scans),
                               estimated_cardinality,
                               delim_idx)
{
  D_ASSERT(join->children.size() == 2);

  // B.2 (#604): the inner join becomes the `delim.join` of this RIGHT_DELIM_JOIN —
  // owned by us, executed inline by our `sink()`, and never a standalone pipeline
  // sink. Tag it explicitly so the join's `is_sink()` and (future) build-side
  // externalization gate skip the rule that would otherwise treat it as one.
  // Scoped to RIGHT_DELIM_JOIN only; LEFT_DELIM_JOIN's inner join feeds a real
  // build subtree and must keep standard externalization.
  if (auto* hj = dynamic_cast<sirius_physical_hash_join*>(join.get())) {
    hj->set_delim_join_inner(true);
  } else if (auto* nlj = dynamic_cast<sirius_physical_nested_loop_join*>(join.get())) {
    nlj->set_delim_join_inner(true);
  }

  children.push_back(std::move(join->children[1]));

  // B.3+B.4+B.6 (#604): mark the synthetic DUMMY_SCAN so plan-gen's wrap_cpu_source
  // skips attaching a CPU_SOURCE leaf below it. The placeholder carries no runtime
  // data flow (RIGHT_DELIM_JOIN::sink invokes partition_join inline), so the
  // CPU_SOURCE wrap would be plan-time scaffolding that materializes a phantom
  // [CPU_SOURCE] pipeline with no legacy counterpart.
  auto dummy_placeholder =
    duckdb::make_uniq<sirius_physical_dummy_scan>(children[0]->get_types(), estimated_cardinality);
  dummy_placeholder->set_delim_join_placeholder(true);
  join->children[1] = std::move(dummy_placeholder);
}

sirius_physical_left_delim_join::sirius_physical_left_delim_join(
  duckdb::vector<sirius::logical_type> types,
  duckdb::unique_ptr<sirius_physical_operator> original_join,
  duckdb::vector<duckdb::const_reference<sirius_physical_operator>> delim_scans,
  std::size_t estimated_cardinality,
  duckdb::optional_idx delim_idx)
  : sirius_physical_delim_join(SiriusPhysicalOperatorType::LEFT_DELIM_JOIN,
                               std::move(types),
                               std::move(original_join),
                               std::move(delim_scans),
                               estimated_cardinality,
                               delim_idx)
{
  D_ASSERT(join->children.size() == 2);
  children.push_back(std::move(join->children[0]));

  auto cached_chunk_scan = duckdb::make_uniq<sirius_physical_column_data_scan>(
    children[0]->get_types(),
    SiriusPhysicalOperatorType::COLUMN_DATA_SCAN,
    estimated_cardinality,
    nullptr);
  if (delim_idx.IsValid()) { cached_chunk_scan->cte_index = delim_idx.GetIndex(); }
  join->children[0] = std::move(cached_chunk_scan);
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_left_delim_join::build_pipelines(pipeline::sirius_pipeline& current,
                                                      pipeline::sirius_meta_pipeline& meta_pipeline)
{
  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(*children[0]);

  D_ASSERT(type == SiriusPhysicalOperatorType::LEFT_DELIM_JOIN);
  // recurse into the actual join
  // any pipelines in there depend on the main pipeline
  // any scan of the duplicate eliminated data on the RHS depends on this pipeline
  // we add an entry to the mapping of (PhysicalOperator*) -> (Pipeline*)
  auto& state = meta_pipeline.get_state();
  for (auto& delim_scan : delim_scans) {
    state.delim_join_dependencies.insert(duckdb::make_pair(
      delim_scan,
      duckdb::reference<pipeline::sirius_pipeline>(*child_meta_pipeline.get_base_pipeline())));
  }
  join->build_pipelines(current, meta_pipeline);

  if (duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD && distinct_root) {
    // Phase 3.2 (#604): spawn a child meta_pipeline rooted at the distinct chain.
    // After wrap_delim_distinct, distinct_root holds
    // `DISTINCT_MERGE -> PARTITION_DISTINCT -> original DISTINCT`; building from it
    // produces the chain's three pipelines via the standard recursive walk. Mirrors
    // split_delim_join_sink's external chain construction (converter:820-841) but
    // reachable via the plan tree. Sibling of child_meta_pipeline under meta_pipeline;
    // data dependency on the original DISTINCT's per-thread output (populated by
    // child_meta_pipeline's LHS) sequences this meta after child_meta_pipeline.
    // Under flag OFF, distinct_root holds the bare DISTINCT and the legacy converter
    // still owns chain construction — skip the spawn there.
    //
    // distinct_meta is created with distinct_root pre-populated as its sink, so walk
    // into distinct_root's child rather than re-building distinct_root itself — calling
    // build_pipelines on distinct_root would re-trigger its sink protocol and produce a
    // duplicate child meta with the same sink, doubling the MERGE_GROUP_BY pipeline.
    auto& distinct_meta = meta_pipeline.create_child_meta_pipeline(current, *distinct_root);
    if (!distinct_root->children.empty()) { distinct_meta.build(*distinct_root->children[0]); }
  }
}

void sirius_physical_right_delim_join::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  auto& child_meta_pipeline = meta_pipeline.create_child_meta_pipeline(current, *this);
  child_meta_pipeline.build(*children[0]);

  D_ASSERT(type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
  // recurse into the actual join
  // any pipelines in there depend on the main pipeline
  // any scan of the duplicate eliminated data on the LHS depends on this pipeline
  // we add an entry to the mapping of (PhysicalOperator*) -> (Pipeline*)
  auto& state = meta_pipeline.get_state();
  for (auto& delim_scan : delim_scans) {
    state.delim_join_dependencies.insert(duckdb::make_pair(
      delim_scan,
      duckdb::reference<pipeline::sirius_pipeline>(*child_meta_pipeline.get_base_pipeline())));
  }

  // Phase 3.2 (#604) Path 3b: under USE_TREE_BASED_PIPELINE_BUILD, Sub-phase
  // B.5's wrap_delim_join recursed into delim->join and Sub-phase B.4's
  // wrap_join inserted CONCAT_build → PARTITION_build → original_build as
  // internal_join.children[1]. The modified build_join_pipelines (build_rhs=
  // true) consumes that chain into build_meta + partition_meta + deeper_meta
  // — handling the internal join's build side via the plan tree rather than
  // via the legacy converter's runtime partition_join construction. Under
  // flag OFF, build_rhs=false preserves today's behavior (legacy converter
  // builds partition_join at runtime).
  sirius_physical_hash_join::build_join_pipelines(
    current, meta_pipeline, *join, duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD);

  if (duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD && distinct_root) {
    // Phase 3.2 (#604): spawn a child meta_pipeline rooted at the distinct chain.
    // See LEFT_DELIM_JOIN's identical comment block above for the full reasoning,
    // including the rationale for walking distinct_root's child rather than
    // distinct_root itself (avoids duplicate-sink meta under the new is_sink protocol).
    auto& distinct_meta = meta_pipeline.create_child_meta_pipeline(current, *distinct_root);
    if (!distinct_root->children.empty()) { distinct_meta.build(*distinct_root->children[0]); }
  }
}

std::unique_ptr<operator_data> sirius_physical_right_delim_join::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_right_delim_join::execute"};
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_read_only_batches(false));
}

void sirius_physical_right_delim_join::sink(const operator_data& input_data,
                                            rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_right_delim_join::sink"};
  // partition_join stays inline (still part of the delim join)
  auto partition_join_output = partition_join->execute(input_data, stream);
  // distinct stays inline (still part of the delim join)
  auto distinct_output = distinct->execute(input_data, stream);

  stream.synchronize();

  partition_join->sink(*partition_join_output, stream);
  // partition_distinct is external — push distinct output via distinct's next_port_after_sink
  distinct->sink(*distinct_output, stream);
}

std::unique_ptr<operator_data> sirius_physical_right_delim_join::get_next_task_input_data()
{
  return partition_join->get_next_task_input_data();
}

std::unique_ptr<operator_data> sirius_physical_left_delim_join::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_left_delim_join::execute"};
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_read_only_batches(false));
}

void sirius_physical_left_delim_join::sink(const operator_data& input_data,
                                           rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_left_delim_join::sink"};
  // column_data_scan stays inline (still part of the delim join)
  auto column_data_scan_output = column_data_scan->execute(input_data, stream);
  // distinct stays inline (still part of the delim join)
  auto distinct_output = distinct->execute(input_data, stream);

  stream.synchronize();

  column_data_scan->sink(*column_data_scan_output, stream);
  // partition_distinct is external — push distinct output via distinct's next_port_after_sink
  distinct->sink(*distinct_output, stream);
}

}  // namespace op
}  // namespace sirius
