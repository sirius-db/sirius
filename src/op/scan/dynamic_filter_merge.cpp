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

#include <cudf/binaryop.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtx3/nvtx3.hpp>

#include <log/logging.hpp>
#include <op/scan/dynamic_filter_merge.hpp>

#include <algorithm>
#include <mutex>
#include <optional>
#include <vector>

namespace sirius::op::scan {

cudf::ast::expression const* merge_dynamic_filters_into_ast(
  cudf::ast::tree& tree,
  cudf::ast::expression const* existing_root,
  sirius::op::sirius_dynamic_filter_set const& filters,
  scan_plan const& plan)
{
  auto const* root = existing_root;
  for (auto const col_idx : filters.filtered_columns()) {
    if (col_idx >= plan.output_layout.size()) { continue; }
    auto const& entry = plan.output_layout[col_idx];
    if (entry.source != scan_plan::output_entry::DATA) { continue; }  // hive — skip
    auto const& parquet_col_name = plan.data_columns[entry.idx].name;

    for (auto const& f : filters.filters_for_column(col_idx)) {
      auto const* lowerable = dynamic_cast<sirius::op::sirius_ast_lowerable const*>(f.get());
      if (!lowerable) { continue; }
      auto const& col_ref  = tree.emplace<cudf::ast::column_name_reference>(parquet_col_name);
      auto const& fragment = lowerable->to_ast(tree, col_ref);
      root                 = root ? &tree.emplace<cudf::ast::operation>(
                      cudf::ast::ast_operator::LOGICAL_AND, *root, fragment)
                                  : &fragment;
    }
  }
  return root;
}

std::unique_ptr<cudf::table> apply_dynamic_filters_to_view(
  cudf::table_view const& input,
  sirius::op::sirius_dynamic_filter_set const& filters,
  rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode,
  dynamic_filter_gate* gate)
{
  nvtx3::scoped_range nvtx_range{"dynfilter::apply_output"};
  if (input.num_rows() == 0 || input.num_columns() == 0) { return nullptr; }

  auto const num_cols = static_cast<std::size_t>(input.num_columns());
  auto const mr       = cudf::get_current_device_resource_ref();

  // Filters apply as a CASCADE, most-selective-first (for membership filters) and AND-merge (for
  // AST-lowerable filters).
  std::unique_ptr<cudf::table> owned;  // most recent step's product backing `current`
  cudf::table_view current = input;
  auto const cascade_step  = [&](std::unique_ptr<cudf::column> mask) -> double {
    if (!mask || current.num_rows() == 0) { return 1.0; }
    auto const rows_before = current.num_rows();
    owned                  = cudf::apply_boolean_mask(current, mask->view(), stream, mr);
    current                = owned->view();
    return static_cast<double>(current.num_rows()) / static_cast<double>(rows_before);
  };

  auto const include_ast_masks = mode == dynamic_filter_apply_mode::include_ast_row_masks;

  // (1) AST-lowerable filters (zone-maps) -> one conjoined predicate via compute_column. Skipped
  // in membership_masks_only mode, where they are already used for row-group pruning at read.
  if (include_ast_masks) {
    cudf::ast::tree tree;
    cudf::ast::expression const* root = nullptr;
    for (auto const col_idx : filters.filtered_columns()) {
      if (col_idx >= num_cols) { continue; }
      cudf::ast::expression const* col_ref = nullptr;
      for (auto const& f : filters.filters_for_column(col_idx)) {
        auto const* lowerable = dynamic_cast<sirius::op::sirius_ast_lowerable const*>(f.get());
        if (!lowerable) { continue; }
        if (!col_ref) {
          col_ref =
            &tree.emplace<cudf::ast::column_reference>(static_cast<cudf::size_type>(col_idx));
        }
        auto const& fragment = lowerable->to_ast(tree, *col_ref);
        root                 = root ? &tree.emplace<cudf::ast::operation>(
                        cudf::ast::ast_operator::LOGICAL_AND, *root, fragment)
                                    : &fragment;
      }
    }
    if (root) {
      // Because AST-lowering is cross-column, we don't have a per-column keep ratio to gate on.
      // Ignore it.
      (void)cascade_step(cudf::compute_column(current, *root, stream, mr));
    }
  }

  // (2) Mask-applicable filters (IN-list / bloom membership). Order by recorded marginal keep
  // ratio (unmeasured filters last, where the cascade has already shrunk the row count, so their
  // first measurement is cheap); drop filters the gate has marked as not worth their kernel.
  struct membership_entry {
    std::size_t col_idx;
    sirius::op::sirius_mask_applicable const* filter;
    sirius::op::sirius_dynamic_filter const* identity;
    std::optional<double> recorded;
  };
  std::vector<membership_entry> entries;
  for (auto const col_idx : filters.filtered_columns()) {
    if (col_idx >= num_cols) { continue; }
    for (auto const& f : filters.filters_for_column(col_idx)) {
      auto const* applicable = dynamic_cast<sirius::op::sirius_mask_applicable const*>(f.get());
      if (!applicable) { continue; }
      auto recorded = gate ? gate->filter_keep_ratio(f.get()) : std::nullopt;
      if (recorded && dynamic_filter_gate::filter_skippable(*recorded)) { continue; }
      entries.push_back({col_idx, applicable, f.get(), recorded});
    }
  }
  std::stable_sort(entries.begin(), entries.end(), [](auto const& a, auto const& b) {
    return a.recorded.value_or(1.0) < b.recorded.value_or(1.0);
  });

  for (auto const& e : entries) {
    if (current.num_rows() == 0) { break; }
    auto const& probe = current.column(static_cast<cudf::size_type>(e.col_idx));
    auto const kept   = cascade_step(e.filter->compute_mask(probe, stream, mr));
    if (gate && !e.recorded) { gate->record_filter_keep_ratio(e.identity, kept); }
  }

  if (!owned) { return nullptr; }
  auto const* mode_name = mode == dynamic_filter_apply_mode::include_ast_row_masks
                            ? "include_ast_row_masks"
                            : "membership_masks_only";
  SIRIUS_LOG_DEBUG("[apply_dynamic_filters] mode={} apply: {} -> {} rows.",
                   mode_name,
                   input.num_rows(),
                   owned->num_rows());
  return owned;
}

std::optional<double> dynamic_filter_gate::filter_keep_ratio(
  sirius::op::sirius_dynamic_filter const* filter) const
{
  std::scoped_lock lock(_filter_ratios_mu);
  auto it = _filter_keep_ratios.find(filter);
  return it != _filter_keep_ratios.end() ? std::optional<double>{it->second} : std::nullopt;
}

void dynamic_filter_gate::record_filter_keep_ratio(sirius::op::sirius_dynamic_filter const* filter,
                                                   double kept)
{
  std::scoped_lock lock(_filter_ratios_mu);
  auto const [it, inserted] = _filter_keep_ratios.emplace(filter, kept);
  if (inserted && filter_skippable(kept)) {
    SIRIUS_LOG_DEBUG(
      "[apply_dynamic_filters] per-filter gate: marginal kept {:.3f} -> SKIP filter for the rest "
      "of this scan.",
      kept);
  }
}

bool dynamic_filter_gate::applicable(sirius::op::sirius_dynamic_filter_set const& filters) const
{
  if (!filters.has_filters()) { return false; }
  if (_state.load(std::memory_order_relaxed) != state::disabled) { return true; }
  // Re-arm: a filter that published after the disable decision may be selective (filters land
  // staggered, and the channel only ever grows), so the combined mask deserves one fresh
  // measurement. Purely a read — record_keep_ratio makes the new decision.
  return filters.filter_count() > _decided_filter_count.load(std::memory_order_relaxed);
}

void dynamic_filter_gate::record_keep_ratio(std::size_t rows_before,
                                            std::size_t rows_after,
                                            std::size_t observed_filter_count)
{
  constexpr double keep_threshold = 0.25;  // disable if a batch keeps > 25% of its rows
  if (rows_before == 0) { return; }
  auto const current = _state.load(std::memory_order_relaxed);
  if (current == state::active) { return; }  // proven useful — sticky
  if (current == state::disabled &&
      observed_filter_count <= _decided_filter_count.load(std::memory_order_relaxed)) {
    return;  // same filters the disabling batch saw — no new information
  }
  auto const kept = static_cast<double>(rows_after) / static_cast<double>(rows_before);
  _decided_filter_count.store(observed_filter_count, std::memory_order_relaxed);
  _state.store(kept > keep_threshold ? state::disabled : state::active, std::memory_order_relaxed);
  SIRIUS_LOG_DEBUG("[apply_dynamic_filters] selectivity gate: kept {:.3f} ({} filters) -> {}.",
                   kept,
                   observed_filter_count,
                   kept > keep_threshold ? "DISABLED" : "ACTIVE");
}

std::unique_ptr<cudf::table> apply_dynamic_filters_gated_view(
  cudf::table_view const& input,
  sirius::op::sirius_dynamic_filter_set const& filters,
  dynamic_filter_gate& gate,
  rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode)
{
  if (!gate.applicable(filters)) { return nullptr; }
  // Snapshot before computing the mask: a filter pushed mid-apply is not (reliably) in the mask,
  // so it must not be credited to this measurement — undercounting at worst re-arms once more.
  auto const observed_filters = filters.filter_count();
  auto const rows_before      = input.num_rows();
  auto filtered               = apply_dynamic_filters_to_view(input, filters, stream, mode, &gate);
  gate.record_keep_ratio(rows_before,
                         filtered ? static_cast<std::size_t>(filtered->num_rows()) : rows_before,
                         observed_filters);
  return filtered;
}

}  // namespace sirius::op::scan
