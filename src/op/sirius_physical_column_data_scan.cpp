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

#include "op/sirius_physical_column_data_scan.hpp"

#include "config.hpp"
#include "op/sirius_physical_delim_join.hpp"
#include "op/sirius_physical_grouped_aggregate.hpp"
#include "pipeline/sirius_meta_pipeline.hpp"
#include "pipeline/sirius_pipeline.hpp"
#include "sirius/exception.hpp"

#include <nvtx3/nvtx3.hpp>

namespace sirius {
namespace op {

sirius_physical_column_data_scan::sirius_physical_column_data_scan(
  duckdb::vector<sirius::logical_type> types,
  SiriusPhysicalOperatorType op_type,
  std::size_t estimated_cardinality,
  duckdb::optionally_owned_ptr<duckdb::ColumnDataCollection> collection_p)
  : sirius_physical_operator(op_type, std::move(types), estimated_cardinality),
    collection(std::move(collection_p)),
    cte_index(duckdb::DConstants::INVALID_INDEX)
{
}

sirius_physical_column_data_scan::sirius_physical_column_data_scan(
  duckdb::vector<sirius::logical_type> types,
  SiriusPhysicalOperatorType op_type,
  std::size_t estimated_cardinality,
  std::size_t cte_index)
  : sirius_physical_operator(op_type, std::move(types), estimated_cardinality),
    collection(nullptr),
    cte_index(cte_index)
{
}

//===--------------------------------------------------------------------===//
// Pipeline Construction
//===--------------------------------------------------------------------===//
void sirius_physical_column_data_scan::build_pipelines(
  pipeline::sirius_pipeline& current, pipeline::sirius_meta_pipeline& meta_pipeline)
{
  // check if there is any additional action we need to do depending on the type
  auto& state = meta_pipeline.get_state();
  switch (type) {
    case SiriusPhysicalOperatorType::DELIM_SCAN: {
      auto entry = state.delim_join_dependencies.find(*this);
      D_ASSERT(entry != state.delim_join_dependencies.end());
      // this chunk scan introduces a dependency to the current pipeline
      // namely a dependency on the duplicate elimination pipeline to finish
      auto delim_dependency = entry->second.get().shared_from_this();
      auto delim_sink       = state.get_pipeline_sink(*delim_dependency);
      D_ASSERT(delim_sink);
      D_ASSERT(delim_sink->type == SiriusPhysicalOperatorType::LEFT_DELIM_JOIN ||
               delim_sink->type == SiriusPhysicalOperatorType::RIGHT_DELIM_JOIN);
      auto& delim_join = delim_sink->Cast<sirius_physical_delim_join>();
      current.add_dependency(delim_dependency);
      if (duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {
        // B.1' (#604): DELIM_SCAN is routing-only under flag ON (mirrors CTE_SCAN
        // post-Phase A). The distinct chain top carries `_owning_delim_join`,
        // which redirects the uniform tree-parent walk in
        // compute_repository_wiring_tree_based to wire the merge-top into each
        // delim_scan's consumer pipeline (inner-HJ probe partition) instead of
        // the DELIM_JOIN itself — mirrors legacy split_delim_join_sink's
        // retarget loop. `is_ready` overwrites this pipeline's source to
        // operators[0] = PARTITION (the inner-HJ probe partition that `current`
        // was created with), so we don't append anything to operators[] here.
        D_ASSERT(delim_join.distinct_root);
        (void)delim_join;
      } else {
        // Flag OFF: legacy converter externalizes the chain and retargets downstream
        // sources from the bare DISTINCT to merge_distinct. Set the source to the
        // bare DISTINCT here and let the converter retarget it.
        state.set_pipeline_source(current, delim_join.distinct->Cast<sirius_physical_operator>());
      }
      return;
    }
    case SiriusPhysicalOperatorType::CTE_SCAN: {
      // throw NotImplementedException("CTE scan not implemented for GPU");
      auto entry = state.cte_dependencies.find(*this);
      D_ASSERT(entry != state.cte_dependencies.end());
      // this chunk scan introduces a dependency to the current pipeline
      // namely a dependency on the CTE pipeline to finish
      auto cte_dependency = entry->second.get().shared_from_this();
      auto cte_sink       = state.get_pipeline_sink(*cte_dependency);
      (void)cte_sink;
      D_ASSERT(cte_sink);
      D_ASSERT(cte_sink->type == SiriusPhysicalOperatorType::CTE);
      current.add_dependency(cte_dependency);
      // CTE_SCAN is a routing-only marker — never materialize into operators[].
      // Legacy's `finalize_pipeline_structure` drops the source and starts
      // operators[] at the next real operator (e.g. PROJECTION). The runtime
      // executor relies on this contract (`update_pipeline_status` and
      // PARTITION's sibling sequencing both assume `pipeline->source` is a
      // real producer like DUCKDB_SCAN, never CTE_SCAN). Tree must match.
      //
      // `set_pipeline_source` is still called so `compute_repository_wiring`
      // (legacy path) can use `source_to_pipelines` to find the consumer.
      // Under flag ON, `dest_for_op` won't contain CTE_SCAN (it's never in
      // any operators[0]), so the tree-based wiring uses
      // `state.cte_scan_consumers` instead — populated here.
      state.set_pipeline_source(current, *this);
      if (duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {
        state.cte_scan_consumers.insert(
          duckdb::make_pair(duckdb::reference<const sirius_physical_operator>(*this),
                            duckdb::reference<pipeline::sirius_pipeline>(current)));
      }
      return;
    }
    case SiriusPhysicalOperatorType::RECURSIVE_RECURRING_CTE_SCAN:
    case SiriusPhysicalOperatorType::RECURSIVE_CTE_SCAN:
      throw not_implemented_exception("Recursive CTE scan not implemented for GPU");
      if (!meta_pipeline.has_recursive_cte()) {
        throw internal_exception("Recursive CTE scan found without recursive CTE node");
      }
      break;
    default: break;
  }
  D_ASSERT(children.empty());
  if (duckdb::Config::USE_TREE_BASED_PIPELINE_BUILD) {
    // B.2 (#604): a leaf scan whose tree parent is RIGHT_DELIM_JOIN (or PARTITION) is
    // a pipeline sink — give it its own single-op pipeline rather than absorbing it
    // into `current`'s operators[]. Mirrors the leaf-sink branch in
    // sirius_physical_operator::build_pipelines; we need to replicate the check here
    // because this override exists to handle the DELIM_SCAN/CTE_SCAN special cases
    // above and would otherwise unconditionally add to the current pipeline.
    if (is_sink()) {
      meta_pipeline.create_child_meta_pipeline(current, *this);
    } else {
      state.add_pipeline_operator(current, *this);
    }
  } else {
    state.set_pipeline_source(current, *this);
  }
}

std::unique_ptr<operator_data> sirius_physical_column_data_scan::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_column_data_scan::execute"};
  return std::make_unique<pipelineable_operator_data>(
    dynamic_cast<const pipelineable_operator_data&>(input_data).get_read_only_batches(false));
}

}  // namespace op
}  // namespace sirius
