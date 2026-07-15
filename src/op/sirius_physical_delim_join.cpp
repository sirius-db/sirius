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

  // The inner join is the source of the delim join's output pipeline (built via
  // build_join_pipelines), not a standalone pipeline sink — tag it so is_sink() returns false.
  // RIGHT_DELIM_JOIN only: LEFT's inner join feeds a real build subtree.
  if (auto* hj = dynamic_cast<sirius_physical_hash_join*>(join.get())) {
    hj->set_delim_join_inner(true);
  } else if (auto* nlj = dynamic_cast<sirius_physical_nested_loop_join*>(join.get())) {
    nlj->set_delim_join_inner(true);
  }

  children.push_back(std::move(join->children[1]));

  // Mark the placeholder so replace_with_gpu_values skips it: it carries no runtime data
  // (the delim join fans its input directly to PARTITION_build), so a GPU_VALUES replacement
  // would only materialize a phantom source.
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

  // Record the cached chunk scan now — after wrap_join, internal_join->children[0] becomes a
  // CONCAT and the reference is unrecoverable. It becomes a first-class source pipeline fed by
  // the delim join's fan-out sink.
  column_data_scan = cached_chunk_scan.get();

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

  if (distinct_root) {
    // Spawn a child meta rooted at the distinct chain
    // (DISTINCT_MERGE -> PARTITION_DISTINCT -> DISTINCT); the standard walk produces its
    // three pipelines, sequenced after child_meta_pipeline by the data dependency on the
    // original DISTINCT's per-thread output. distinct_root is pre-populated as the meta's
    // sink, so build from its child — re-building distinct_root would re-trigger its sink
    // protocol and duplicate the MERGE_GROUP_BY pipeline.
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

  // wrap_join already inserted CONCAT_build → PARTITION_build → original_build as the
  // inner join's children[1]; build_join_pipelines consumes that chain via the plan tree.
  //
  // Nested RIGHT_DELIM_JOIN: `current` may already carry a barrier op (RDJ/PARTITION) as
  // its pre-populated sink; routing *join through it would fuse the inner join into that
  // pipeline. Split it into its own child meta instead — fusing with non-barrier ops
  // (PROJECTION, FILTER, ...) is correct.
  auto current_ops = current.get_operators();
  const bool needs_split =
    !current_ops.empty() &&
    (current_ops.front().get().type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN ||
     current_ops.front().get().type == SiriusPhysicalOperatorType::PARTITION);

  if (needs_split) {
    auto& join_meta = meta_pipeline.create_child_meta_pipeline(current, *join);
    auto& join_base = *join_meta.get_base_pipeline();
    // Clear the pre-populated [join] so build_join_pipelines re-adds it within join_meta.
    meta_pipeline.get_state().set_pipeline_operators(join_base, {});
    sirius_physical_hash_join::build_join_pipelines(join_base, join_meta, *join);
  } else {
    sirius_physical_hash_join::build_join_pipelines(current, meta_pipeline, *join);
  }

  if (distinct_root) {
    // Spawn the distinct-chain child meta; see LEFT_DELIM_JOIN above for why we build
    // from distinct_root's child rather than distinct_root itself.
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

std::unique_ptr<operator_data> sirius_physical_left_delim_join::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_left_delim_join::execute"};
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_read_only_batches(false));
}

}  // namespace op
}  // namespace sirius
