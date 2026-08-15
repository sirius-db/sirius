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

#include "op/sirius_physical_grouped_aggregate.hpp"

#include "config.hpp"
#include "data/data_batch_utils.hpp"
#include "log/logging.hpp"
#include "op/aggregate/aggregate_op_util.hpp"
#include "op/aggregate/gpu_aggregate_impl.hpp"
#include "op/dynamic_filter/top_n_group_key_producer.hpp"

#include <nvtx3/nvtx3.hpp>

#include <exception>
#include <memory>
#include <utility>
#include <vector>

namespace sirius {
namespace op {

sirius_physical_grouped_aggregate::sirius_physical_grouped_aggregate(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> groups_p,
  std::size_t estimated_cardinality)
  : sirius_physical_grouped_aggregate(std::move(types),
                                      std::move(expressions),
                                      std::move(groups_p),
                                      {},
                                      {},
                                      estimated_cardinality,
                                      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES,
                                      duckdb::TupleDataValidityType::CAN_HAVE_NULL_VALUES)
{
}

// expressions is the list of aggregates to be computed. Each aggregates has a bound_ref expression
// to a column groups_p is the list of group by columns. Each group by column is a bound_ref
// expression to a column grouping_sets_p is the list of grouping set. Each grouping set is a set of
// indexes to the group by columns. Seems like DuckDB group the groupby columns into several sets
// and for every grouping set there is one radix_table grouping_functions_p is a list of indexes to
// the groupby expressions (groups_p) for each grouping_sets. The first level of the vector is the
// grouping set and the second level is the indexes to the groupby expression for that set.
sirius_physical_grouped_aggregate::sirius_physical_grouped_aggregate(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> expressions,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> groups_p,
  duckdb::vector<duckdb::GroupingSet> grouping_sets_p,
  duckdb::vector<duckdb::unsafe_vector<std::size_t>> grouping_functions_p,
  std::size_t estimated_cardinality,
  duckdb::TupleDataValidityType /*group_validity*/,
  duckdb::TupleDataValidityType /*distinct_validity*/)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::HASH_GROUP_BY, std::move(types), estimated_cardinality),
    grouping_sets(std::move(grouping_sets_p))
{
  auto cudf_defs                    = convert_duckdb_aggregates_to_cudf(groups_p, expressions);
  group_idx                         = std::move(cudf_defs.group_idx);
  cudf_aggregates                   = std::move(cudf_defs.cudf_aggregates);
  cudf_aggregate_idx                = std::move(cudf_defs.cudf_aggregate_idx);
  cudf_aggregate_struct_col_indices = std::move(cudf_defs.cudf_aggregate_struct_col_indices);
  aggregate_slots                   = std::move(cudf_defs.aggregate_slots);
  has_avg                           = cudf_defs.has_avg;
  has_count_distinct                = cudf_defs.has_count_distinct;
}

sirius_physical_grouped_aggregate::~sirius_physical_grouped_aggregate() = default;

std::unique_ptr<operator_data> sirius_physical_grouped_aggregate::execute(
  const operator_data& input_data, rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_grouped_aggregate::execute"};
  auto& input               = dynamic_cast<const pipelineable_operator_data&>(input_data);
  const auto& input_batches = input.get_read_only_batches();
  std::vector<std::shared_ptr<::cucascade::data_batch>> results;
  for (auto const& input_batch : input_batches) {
    auto* space = input_batch.get_memory_space();
    if (!space) { continue; }
    if (!top_n_producer) {
      results.push_back(
        gpu_aggregate_impl::local_grouped_aggregate(input_batch,
                                                    group_idx,
                                                    cudf_aggregates,
                                                    cudf_aggregate_idx,
                                                    cudf_aggregate_struct_col_indices,
                                                    stream,
                                                    *space,
                                                    batch_telemetry()));
      continue;
    }

    // Top-N group-key producer seam, witness-first: the batch's best distinct grouping keys are
    // offered before the prefilter runs, so the boundary they establish or tighten can prune this
    // very batch ahead of the hash insert. The order is load-bearing under coarse batching, where
    // an aggregate's whole input can arrive as one batch -- a prefilter that ran first would never
    // see a boundary at all (measured: TPC-H Q18 at SF1000 prefiltered zero rows under the old
    // order). Witnessing the unpruned batch costs no evidence -- keys worse than the boundary can
    // never displace the set's best K -- and rows the boundary excludes belong to groups that
    // cannot reach the final K, so dropping them changes no surviving group's value regardless of
    // when the boundary formed. Both steps are optional work, so a failure degrades to a plain
    // aggregation.
    auto const memory_resource = space->get_default_allocator();
    auto input_table           = get_cudf_table_view(input_batch);
    try {
      top_n_producer->witness(input_table, stream, memory_resource);
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN(
        "[sirius_physical_grouped_aggregate] Top-N group-key witness skipped one batch: {}. "
        "The boundary simply stops tightening from this batch.",
        e.what());
    }
    // Prefiltering is a separate failure domain from witnessing: the offer above already banked
    // this batch's evidence, so a failed prefilter costs pruning and nothing else.
    std::unique_ptr<cudf::table> prefiltered;  // backs input_table when rows were dropped
    try {
      auto result = top_n_producer->prefilter(input_table, stream, memory_resource);
      // A null result is the all-pass fast path: keep the original view, copy-free.
      if (result.filtered) {
        prefiltered = std::move(result.filtered);
        input_table = prefiltered->view();
      }
    } catch (std::exception const& e) {
      SIRIUS_LOG_WARN(
        "[sirius_physical_grouped_aggregate] Top-N group-key prefilter skipped one batch: {}. "
        "Aggregation continues over the whole batch.",
        e.what());
      prefiltered.reset();
      input_table = get_cudf_table_view(input_batch);
    }

    results.push_back(gpu_aggregate_impl::local_grouped_aggregate(input_table,
                                                                  group_idx,
                                                                  cudf_aggregates,
                                                                  cudf_aggregate_idx,
                                                                  cudf_aggregate_struct_col_indices,
                                                                  stream,
                                                                  *space,
                                                                  batch_telemetry()));
  }
  return std::make_unique<pipelineable_operator_data>(results);
}

void sirius_physical_grouped_aggregate::on_finalize_operator()
{
  // Producer-side drain once this aggregate's input pipeline has completed: no further batch can
  // arrive, so no further distinct key can be witnessed. Idempotent, and it blocks only this
  // finalization task, never a scan consumer.
  if (top_n_producer) { top_n_producer->coordinator().finish(); }
}
}  // namespace op
}  // namespace sirius
