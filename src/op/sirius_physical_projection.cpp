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

#include "op/sirius_physical_projection.hpp"

#include "config.hpp"
#include "data/data_batch_utils.hpp"
#include "expression/ast/reference.hpp"
#include "expression_executor/gpu_expression_executor.hpp"

#include <cudf/types.hpp>

#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <duckdb/common/exception.hpp>

#include <memory>
#include <optional>
#include <vector>

namespace sirius {
namespace op {

namespace {

/// Where an output column of the projection comes from.
enum class projection_source { passthrough, evaluated };

/// Per-output-column plan: a pure BOUND_REF passthrough (index = input column index) or an
/// evaluated expression (index = position within the evaluated output table).
struct projection_column_plan {
  projection_source kind;
  cudf::size_type index;
};

}  // namespace

sirius_physical_projection::sirius_physical_projection(
  duckdb::vector<sirius::logical_type> types,
  duckdb::vector<std::unique_ptr<sirius::ast::node>> select_list,
  std::size_t estimated_cardinality)
  : sirius_physical_operator(
      SiriusPhysicalOperatorType::PROJECTION, std::move(types), estimated_cardinality),
    select_list(std::move(select_list))
{
}

std::unique_ptr<operator_data> sirius_physical_projection::execute(const operator_data& input_data,
                                                                   rmm::cuda_stream_view stream)
{
  nvtx3::scoped_range nvtx_range{"sirius_physical_projection::execute"};
  auto& input = dynamic_cast<const pipelineable_operator_data&>(input_data);
  // Mutable local: we move each read-only lock out of this vector into an output batch's owner.
  auto input_batches = input.get_read_only_batches();

  // Classify select_list once (batch-independent): pure BOUND_REF entries (sirius::ast::reference)
  // are passthroughs we can expose as zero-copy column_views of the input; everything else must be
  // evaluated. A pure reference never needs a cast (DuckDB wraps type changes in a cast node, which
  // is not a reference), so the input column type already matches the output type.
  std::vector<projection_column_plan> column_plan;
  column_plan.reserve(select_list.size());
  std::vector<const sirius::ast::node*> evaluated_exprs;
  for (auto const& expr : select_list) {
    if (expr->holds<sirius::ast::reference>()) {
      column_plan.push_back(
        {projection_source::passthrough,
         static_cast<cudf::size_type>(expr->get<sirius::ast::reference>().column_index)});
    } else {
      column_plan.push_back(
        {projection_source::evaluated, static_cast<cudf::size_type>(evaluated_exprs.size())});
      evaluated_exprs.push_back(expr.get());
    }
  }
  bool const all_passthrough = evaluated_exprs.empty();
  bool const all_evaluated   = evaluated_exprs.size() == select_list.size();

  /// TODO: the operator should choose the execution strategy based on statistics and a deeper
  /// understand of the trade-offs between the different strategies. See:
  /// https://github.com/sirius-db/sirius/issues/636
  // Construct the executor once (reused across batches; execute() resets its state each call).
  std::optional<sirius::gpu_expression_executor> gpu_expression_executor;
  if (!all_passthrough) {
    gpu_expression_executor.emplace(
      evaluated_exprs, cudf::get_current_device_resource_ref(), stream);
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  for (auto& input_ro : input_batches) {
    auto input_view = sirius::get_cudf_table_view(input_ro);
    auto& mem = *input_ro.get_memory_space();  // owned by the memory manager (outlives input_ro)

    // ---- Path 1: every output column is an evaluated expression ----
    if (all_evaluated) {
      auto projected_table = gpu_expression_executor->execute(input_view);
      output_batches.push_back(sirius::make_data_batch(std::move(projected_table), mem, stream));
      continue;
    }

    // ---- Path 2: every output column is a pure passthrough — zero device copies. ----
    if (all_passthrough) {
      std::vector<cudf::column_view> cols;
      cols.reserve(column_plan.size());
      for (auto const& p : column_plan) {
        cols.push_back(input_view.column(p.index));
      }
      cudf::table_view out_view(cols);
      // We pin the whole input batch alive (its read-only lock is the owner below), so charge its
      // full size. No new columns are allocated.
      std::size_t const referenced_bytes = input_ro.get_data()->get_size_in_bytes();
      // Owner = the input read-only lock: keeps the source columns alive AND pinned read-only for
      // the output batch's lifetime, so the view can never be freed/downgraded out from under us.
      output_batches.push_back(sirius::make_data_batch_from_view(
        out_view, std::move(input_ro), referenced_bytes, mem, stream));
      continue;
    }

    // ---- Path 3: mix of evaluated columns and passthroughs. ----
    auto evaluated_owned = gpu_expression_executor->execute(input_view);
    // Charge the full input batch (pinned alive for the passthrough columns) plus the real size of
    // the freshly-evaluated columns we just allocated.
    std::size_t const referenced_bytes =
      input_ro.get_data()->get_size_in_bytes() + evaluated_owned->alloc_size();
    // Move the evaluated table into a shared_ptr so it can live inside the (copy-constructible)
    // std::any owner alongside the input lock.
    std::shared_ptr<cudf::table> evaluated(std::move(evaluated_owned));
    auto eval_view = evaluated->view();

    std::vector<cudf::column_view> cols(column_plan.size());
    for (std::size_t i = 0; i < column_plan.size(); ++i) {
      auto const& p = column_plan[i];
      cols[i]       = (p.kind == projection_source::passthrough) ? input_view.column(p.index)
                                                                 : eval_view.column(p.index);
    }
    cudf::table_view out_view(cols);

    // Composite owner keeps BOTH lifetimes alive: the freshly-evaluated columns (shared_ptr) and
    // the input read-only lock (for the passthrough columns). Both members are copy-constructible,
    // as std::any requires. eval_view's column pointers stay valid because `owner.evaluated` still
    // owns the columns after the move.
    struct projection_owner {
      std::shared_ptr<cudf::table> evaluated;
      cucascade::read_only_data_batch input_lock;
    };
    projection_owner owner{std::move(evaluated), std::move(input_ro)};
    output_batches.push_back(
      sirius::make_data_batch_from_view(out_view, std::move(owner), referenced_bytes, mem, stream));
  }
  return std::make_unique<pipelineable_operator_data>(output_batches);
}

}  // namespace op
}  // namespace sirius
