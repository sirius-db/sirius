/*
 * Copyright 2026, Sirius Contributors.
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
#include "op/aggregate/group_by_bypass_analysis.hpp"

#include "cudf/cudf_utils.hpp"
#include "expression/ast/reference.hpp"
#include "memory/size_arithmetic.hpp"
#include "op/sirius_physical_filter.hpp"
#include "op/sirius_physical_grouped_aggregate_merge.hpp"
#include "op/sirius_physical_projection.hpp"

#include <cudf/utilities/traits.hpp>

#include <utility>
#include <variant>
#include <vector>

namespace sirius::op::group_by_bypass {

namespace {

/// Fixed-width integral types the v1 bypass model covers. Floats and decimals are excluded
/// deliberately: their merge-time partial states are not modelled here (see
/// docs/super-sirius/group-by-bypass.md).
[[nodiscard]] bool bypass_supported_column_type(int type_id) noexcept
{
  switch (static_cast<cudf::type_id>(type_id)) {
    case cudf::type_id::BOOL8:
    case cudf::type_id::INT8:
    case cudf::type_id::INT16:
    case cudf::type_id::INT32:
    case cudf::type_id::INT64:
    case cudf::type_id::UINT8:
    case cudf::type_id::UINT16:
    case cudf::type_id::UINT32:
    case cudf::type_id::UINT64: return true;
    default: return false;
  }
}

/// Whether the path from @p merge to the first downstream sink is a bounded result-collection
/// path: unary, non-expanding, and terminating at a RESULT_COLLECTOR.
///
/// Anything that buffers or reshapes — TOP_N, ORDER_BY, a second GROUP BY, a join, a further
/// PARTITION — makes the merge's output feed work this model does not account for, so v1 declines
/// rather than guessing. Along an accepted path, `row_bytes`/`columns` total the columns every
/// step materializes while the merge output is still live.
struct downstream_shape {
  bool supported          = false;
  std::uint64_t row_bytes = 0;
  std::size_t columns     = 0;
};

/// Walk from @p merge to its collector. @p row holds the fixed width of each merge output column;
/// it is empty when the merge's own schema is unknown, in which case the candidate is rejected on
/// its state before these sizes matter.
[[nodiscard]] downstream_shape classify_bypass_downstream(const sirius_physical_operator& merge,
                                                          std::vector<std::uint64_t> row)
{
  using T = SiriusPhysicalOperatorType;
  downstream_shape shape;
  downstream_shape const rejected;
  if (merge.owning_delim_join() != nullptr) { return rejected; }
  // Charge the columns a FILTER or LIMIT actually copies into its output row.
  auto charge_row = [&] {
    for (auto const width : row) {
      shape.row_bytes = memory::saturating_add(shape.row_bytes, width);
    }
    shape.columns += row.size();
  };
  for (auto* cur = merge.get_parent_op(); cur != nullptr; cur = cur->get_parent_op()) {
    if (cur->type == T::RESULT_COLLECTOR) {
      shape.supported = true;
      return shape;
    }
    if (cur->children.size() != 1) { return rejected; }
    switch (cur->type) {
      case T::FILTER: {
        auto const* filter = dynamic_cast<const sirius_physical_filter*>(cur);
        if (filter == nullptr) { return rejected; }
        if (auto const* indices =
              std::get_if<std::vector<cudf::size_type>>(&filter->output_columns)) {
          // Filters can gather a subset or reorder columns. Later references index this output,
          // not the original merge row; preserve physical widths rather than logical types.
          if (indices->empty()) { return rejected; }
          std::vector<std::uint64_t> next;
          next.reserve(indices->size());
          for (auto const index : *indices) {
            if (index < 0 || static_cast<std::size_t>(index) >= row.size()) { return rejected; }
            next.push_back(row[static_cast<std::size_t>(index)]);
          }
          row = std::move(next);
        }
        charge_row();
        break;
      }
      case T::LIMIT:
      case T::STREAMING_LIMIT: charge_row(); break;
      case T::PROJECTION: {
        // Pure references are zero-copy views of the input. Every evaluated expression allocates
        // a new column of its result type, so only fixed-width results can be sized; any other
        // result (a string from CASE or a function, say) has no bound here and is rejected.
        // Temporaries inside a nested expression are not sized and fall under the headroom.
        auto const* projection = dynamic_cast<const sirius_physical_projection*>(cur);
        if (projection == nullptr || projection->select_list.size() != cur->types.size()) {
          return rejected;
        }
        std::vector<std::uint64_t> next;
        next.reserve(projection->select_list.size());
        for (std::size_t i = 0; i < projection->select_list.size(); ++i) {
          auto const& expr = *projection->select_list[i];
          if (expr.holds<sirius::ast::reference>()) {
            auto const index = expr.get<sirius::ast::reference>().column_index;
            if (index >= row.size()) { return rejected; }
            next.push_back(row[index]);
            continue;
          }
          auto const dtype = try_get_cudf_type(cur->types[i]);
          if (!dtype.has_value() || !cudf::is_fixed_width(*dtype)) { return rejected; }
          auto const width = static_cast<std::uint64_t>(cudf::size_of(*dtype));
          shape.row_bytes  = memory::saturating_add(shape.row_bytes, width);
          shape.columns += 1;
          next.push_back(width);
        }
        row = std::move(next);
        break;
      }
      default: return rejected;
    }
  }
  return rejected;
}

// Check merge-time partial states, not SQL output types: COUNT is re-merged with SUM,
// while COUNT(DISTINCT) arrives as a LIST.
bool bypass_supported_aggregates(const sirius_physical_grouped_aggregate_merge& merge)
{
  // AVG and COUNT(DISTINCT) need post-merge projection (a cast/divide, or list element counting)
  // whose allocations are outside the model.
  if (merge.has_avg || merge.has_count_distinct) { return false; }
  if (merge.cudf_aggregates.empty()) { return false; }
  for (auto kind : merge.cudf_aggregates) {
    switch (kind) {
      case cudf::aggregation::Kind::SUM:
      case cudf::aggregation::Kind::MIN:
      case cudf::aggregation::Kind::MAX:
      case cudf::aggregation::Kind::COUNT_ALL:
      case cudf::aggregation::Kind::COUNT_VALID: break;
      // COLLECT_SET arrives as a LIST and re-merges with MERGE_SETS; not modelled.
      default: return false;
    }
  }
  return true;
}

}  // namespace

candidate_input make_candidate(const sirius_physical_grouped_aggregate_merge& merge,
                               const group_by_bypass_metadata& meta,
                               int natural,
                               int num_admitted_gpus,
                               double headroom_fraction)
{
  candidate_input candidate;
  candidate.auto_num_partitions = natural;
  candidate.num_admitted_gpus   = num_admitted_gpus;
  candidate.upstream_complete   = meta.upstream_complete;
  // A single admitted GPU must also mean the input really is on one device.
  candidate.single_gpu_resident          = meta.single_gpu_resident;
  candidate.headroom_fraction            = headroom_fraction;
  candidate.total_rows                   = meta.total_rows;
  candidate.admissible_additional_budget = meta.admissible_additional_budget;

  // Split the observed physical columns at the grouping-key boundary — the same boundary
  // merge_grouped_aggregate itself uses — and check each side against the whitelist.
  auto const num_group_cols = merge.group_idx.size();
  std::vector<std::uint64_t> output_widths;
  if (meta.columns.has_value() &&
      meta.columns->size() == num_group_cols + merge.cudf_aggregates.size()) {
    auto const& cols        = *meta.columns;
    bool types_ok           = bypass_supported_aggregates(merge);
    std::uint64_t key_width = 0;
    std::uint64_t agg_width = 0;
    std::size_t null_keys   = 0;
    std::size_t null_aggs   = 0;
    for (std::size_t c = 0; c < cols.size(); ++c) {
      auto const& col = cols[c];
      if (col.fixed_width_bytes == 0 || !bypass_supported_column_type(col.type_id)) {
        types_ok = false;
        break;
      }
      if (c < num_group_cols) {
        key_width += col.fixed_width_bytes;
        null_keys += col.nullable ? 1 : 0;
      } else {
        agg_width += col.fixed_width_bytes;
        null_aggs += col.nullable ? 1 : 0;
      }
    }
    candidate.supported_state = types_ok;
    if (types_ok) {
      candidate.key_width_bytes      = key_width;
      candidate.agg_width_bytes      = agg_width;
      candidate.key_columns          = num_group_cols;
      candidate.agg_columns          = cols.size() - num_group_cols;
      candidate.nullable_key_columns = null_keys;
      candidate.nullable_agg_columns = null_aggs;
      // The whitelisted states merge into columns of the same physical type, so the merge output
      // row has the partial input's widths.
      output_widths.reserve(cols.size());
      for (auto const& col : cols) {
        output_widths.push_back(col.fixed_width_bytes);
      }
    }
  } else {
    // Either the schema could not be read, or it does not match this merge's own column layout.
    // Both are "unknown", not "zero".
    candidate.supported_state = false;
  }

  auto const shape               = classify_bypass_downstream(merge, std::move(output_widths));
  candidate.supported_downstream = shape.supported;
  candidate.downstream_row_bytes = shape.row_bytes;
  candidate.downstream_columns   = shape.columns;

  return candidate;
}

}  // namespace sirius::op::group_by_bypass
