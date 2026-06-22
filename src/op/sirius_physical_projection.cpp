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
#include "log/logging.hpp"

#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda_runtime.h>
#include <nvtx3/nvtx3.hpp>

#include <cucascade/cudf/gpu_data_representation.hpp>
#include <cucascade/data/data_batch.hpp>
#include <duckdb/common/exception.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace sirius {
namespace op {

namespace {

/// Conservative, sync-free estimate of the device bytes referenced by a column_view, used for
/// the owning_table_view's alloc_size (memory accounting). Counts the validity bitmask and
/// fixed-width data buffers, and recurses into children (string/list offsets, struct fields).
/// Variable-length leaf payloads (string chars) are intentionally NOT device-read here so the
/// projection hot path stays free of stream synchronization; the result is an estimate.
std::size_t estimate_column_view_bytes(cudf::column_view const& col)
{
  std::size_t bytes = 0;
  if (col.nullable()) { bytes += cudf::bitmask_allocation_size_bytes(col.size()); }
  if (cudf::is_fixed_width(col.type()) && col.size() > 0) {
    bytes += static_cast<std::size_t>(col.size()) * cudf::size_of(col.type());
  }
  for (cudf::size_type i = 0; i < col.num_children(); ++i) {
    bytes += estimate_column_view_bytes(col.child(i));
  }
  return bytes;
}

/// Where an output column of the projection comes from.
enum class projection_source { passthrough, evaluated };

/// Per-output-column plan: a pure BOUND_REF passthrough (index = input column index) or an
/// evaluated expression (index = position within the evaluated output table).
struct projection_column_plan {
  projection_source kind;
  std::uint32_t index;
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
  // Bind to a non-const local (default get_read_only_batches() returns a fresh, uncached vector):
  // we move each read_only lock into the owner of any zero-copy output batch we produce.
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
        {projection_source::passthrough, expr->get<sirius::ast::reference>().column_index});
    } else {
      column_plan.push_back(
        {projection_source::evaluated, static_cast<std::uint32_t>(evaluated_exprs.size())});
      evaluated_exprs.push_back(expr.get());
    }
  }
  bool const all_passthrough = evaluated_exprs.empty();
  bool const all_evaluated   = evaluated_exprs.size() == select_list.size();

  /// TODO: the operator should choose the execution strategy based on statistics and a deeper
  /// understand of the trade-offs between the different strategies. See:
  /// https://github.com/sirius-db/sirius/issues/636
  // Construct the executor once (reused across batches; execute() resets its state each call).
  // Path 2 (all passthrough) never evaluates anything, so skip building it.
  std::optional<sirius::gpu_expression_executor> gpu_expression_executor;
  if (!all_passthrough) {
    gpu_expression_executor.emplace(
      evaluated_exprs, cudf::get_current_device_resource_ref(), stream);
  }

  std::vector<std::shared_ptr<cucascade::data_batch>> output_batches;
  output_batches.reserve(input_batches.size());

  for (auto& input_ro : input_batches) {
    auto input_view = sirius::get_cudf_table_view(input_ro);
    auto& mem       = *input_ro.get_memory_space();  // owned by the manager; valid after the move

    // ---- Path 1: every output column is an evaluated expression (legacy behavior). ----
    if (all_evaluated) {
      auto projected_table = gpu_expression_executor->execute(input_view);
      output_batches.push_back(sirius::make_data_batch(std::move(projected_table), mem, stream));
      continue;
    }

    // Paths 2 & 3 expose input memory through a view. Order `stream` after the input's writes so
    // the writer event recorded on the output batch correctly bounds the referenced data.
    if (auto* ev = input_ro.get_writer_event(); ev != nullptr) {
      if (auto err = cudaStreamWaitEvent(stream.value(), ev, 0); err != cudaSuccess) {
        SIRIUS_LOG_WARN("[projection] cudaStreamWaitEvent on input writer event failed: {}",
                        cudaGetErrorString(err));
      }
    }

    // ---- Path 2: every output column is a pure passthrough — zero device copies. ----
    if (all_passthrough) {
      std::vector<cudf::column_view> cols;
      cols.reserve(column_plan.size());
      std::size_t referenced_bytes = 0;
      for (auto const& p : column_plan) {
        auto const& cv = input_view.column(p.index);
        cols.push_back(cv);
        referenced_bytes += estimate_column_view_bytes(cv);
      }
      cudf::table_view out_view(cols);
      // Owner = the input read-only lock: keeps the source columns alive AND pinned read-only for
      // the output batch's lifetime, so the view can never be freed/downgraded out from under us.
      output_batches.push_back(sirius::make_data_batch_from_view(
        out_view, std::move(input_ro), referenced_bytes, mem, stream));
      continue;
    }

    // ---- Path 3: mix of evaluated columns and passthroughs. ----
    auto evaluated_owned = gpu_expression_executor->execute(input_view);
    std::size_t referenced_bytes =
      evaluated_owned->alloc_size();  // (A) charge full referenced size
    // Move the evaluated table into a shared_ptr so it can live inside the (copy-constructible)
    // std::any owner alongside the input lock.
    std::shared_ptr<cudf::table> evaluated(std::move(evaluated_owned));
    auto eval_view = evaluated->view();

    std::vector<cudf::column_view> cols(column_plan.size());
    for (std::size_t i = 0; i < column_plan.size(); ++i) {
      auto const& p = column_plan[i];
      if (p.kind == projection_source::passthrough) {
        auto const& cv = input_view.column(p.index);
        cols[i]        = cv;
        referenced_bytes += estimate_column_view_bytes(cv);
      } else {
        cols[i] = eval_view.column(p.index);
      }
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
