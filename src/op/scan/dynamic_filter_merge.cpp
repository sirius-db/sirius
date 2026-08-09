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
#include <op/dynamic_filter/dynamic_filter_device.hpp>
#include <op/scan/dynamic_filter_merge.hpp>

#include <algorithm>
#include <mutex>
#include <optional>
#include <vector>

namespace sirius::op::scan {

cudf::ast::expression const* merge_dynamic_filters_into_ast(
  cudf::ast::tree& tree,
  cudf::ast::expression const* existing_root,
  sirius::op::dynamic_filter_snapshot const& snapshot,
  scan_plan const& plan,
  int device_id)
{
  device_id        = sirius::op::detail::resolve_dynamic_filter_device_id(device_id);
  auto const* root = existing_root;
  for (auto const& column : snapshot.columns) {
    if (column.column >= plan.output_layout.size()) { continue; }
    auto const& entry = plan.output_layout[column.column];
    if (entry.source != scan_plan::output_entry::DATA) { continue; }  // hive — skip
    auto const& parquet_col_name = plan.data_columns[entry.idx].name;

    for (auto const& f : column.filters) {
      if (!f->is_available_on_device(device_id)) { continue; }
      cudf::ast::expression const* fragment = nullptr;
      if (auto const* lowerable = dynamic_cast<sirius::op::sirius_ast_lowerable const*>(f.get())) {
        auto const& col_ref = tree.emplace<cudf::ast::column_name_reference>(parquet_col_name);
        fragment            = &lowerable->to_ast(tree, col_ref, device_id);
      } else if (auto const* multi =
                   dynamic_cast<sirius::op::sirius_multi_column_ast_lowerable const*>(f.get())) {
        // A LEX boundary references several decoded columns; resolve each through the same
        // plan-driven name mapping, skipping the filter when any component is a hive partition
        // (its values never reach the reader).
        bool resolvable     = true;
        auto const resolver = [&](std::size_t ordinal) -> cudf::ast::expression const& {
          auto const& component = plan.output_layout[ordinal];
          return tree.emplace<cudf::ast::column_name_reference>(
            plan.data_columns[component.idx].name);
        };
        for (auto const ordinal : multi->referenced_ordinals()) {
          if (ordinal >= plan.output_layout.size() ||
              plan.output_layout[ordinal].source != scan_plan::output_entry::DATA) {
            resolvable = false;
            break;
          }
        }
        if (!resolvable) { continue; }
        fragment = &multi->to_ast(tree, resolver, device_id);
      }
      if (fragment == nullptr) { continue; }
      root = root ? &tree.emplace<cudf::ast::operation>(
                      cudf::ast::ast_operator::LOGICAL_AND, *root, *fragment)
                  : fragment;
    }
  }
  return root;
}

std::unique_ptr<cudf::table> apply_dynamic_filters_to_view(
  cudf::table_view const& input,
  sirius::op::dynamic_filter_snapshot const& snapshot,
  rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode,
  dynamic_filter_gate* gate,
  int device_id)
{
  nvtx3::scoped_range nvtx_range{"dynfilter::apply_output"};
  if (input.num_rows() == 0 || input.num_columns() == 0) { return nullptr; }

  device_id           = sirius::op::detail::resolve_dynamic_filter_device_id(device_id);
  auto const num_cols = static_cast<std::size_t>(input.num_columns());
  auto const mr       = cudf::get_current_device_resource_ref();

  // Apply one combined AST mask, then membership masks ordered by recorded selectivity.
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

  // Conjoin AST-lowerable filters into one row mask when the scan reader did not apply them.
  if (include_ast_masks) {
    cudf::ast::tree tree;
    cudf::ast::expression const* root = nullptr;
    for (auto const& column : snapshot.columns) {
      if (column.column >= num_cols) { continue; }
      cudf::ast::expression const* col_ref = nullptr;
      for (auto const& f : column.filters) {
        if (!f->is_available_on_device(device_id)) { continue; }
        auto const* lowerable = dynamic_cast<sirius::op::sirius_ast_lowerable const*>(f.get());
        if (!lowerable) { continue; }
        if (!col_ref) {
          col_ref =
            &tree.emplace<cudf::ast::column_reference>(static_cast<cudf::size_type>(column.column));
        }
        auto const& fragment = lowerable->to_ast(tree, *col_ref, device_id);
        root                 = root ? &tree.emplace<cudf::ast::operation>(
                        cudf::ast::ast_operator::LOGICAL_AND, *root, fragment)
                                    : &fragment;
      }
    }
    if (root) {
      // The AST mask can span columns, so only the scan-level gate records its combined effect.
      (void)cascade_step(cudf::compute_column(current, *root, stream, mr));
    }

    // Top-N boundary filters apply through the fused predicate+compaction kernel instead of a
    // BOOL8 mask, in the same situations the AST masks run: where no read-time dynamic phase
    // already applied them (native scans and sited endpoints).
    for (auto const& column : snapshot.columns) {
      if (column.column >= num_cols) { continue; }
      for (auto const& f : column.filters) {
        if (current.num_rows() == 0) { break; }
        // Deliberately no is_available_on_device guard: the boundary rides kernel launch
        // parameters, so compaction works on any device. That guard belongs to the AST path,
        // whose literals are per-device replicas.
        auto const* compactable =
          dynamic_cast<sirius::op::sirius_compaction_applicable const*>(f.get());
        if (compactable == nullptr) { continue; }

        std::vector<cudf::size_type> key_columns;
        if (auto const* multi =
              dynamic_cast<sirius::op::sirius_multi_column_ast_lowerable const*>(f.get())) {
          for (auto const ordinal : multi->referenced_ordinals()) {
            key_columns.push_back(static_cast<cudf::size_type>(ordinal));
          }
        } else {
          key_columns.push_back(static_cast<cudf::size_type>(column.column));
        }
        if (std::any_of(key_columns.begin(), key_columns.end(), [num_cols](auto ordinal) {
              return static_cast<std::size_t>(ordinal) >= num_cols;
            })) {
          continue;
        }

        auto result = compactable->apply_compact(current, key_columns, device_id, stream, mr);
        // A null table is the all-pass fast path: nothing was dropped, so the cascade is unchanged.
        if (!result.filtered) { continue; }
        owned   = std::move(result.filtered);
        current = owned->view();
      }
    }
  }

  // Apply membership filters in recorded selectivity order and omit filters the gate disabled.
  struct membership_entry {
    std::size_t col_idx;
    sirius::op::sirius_mask_applicable const* filter;
    sirius::op::sirius_dynamic_filter const* identity;
    std::optional<double> recorded;
  };
  // Scope every marginal ratio read and update to one channel-size snapshot.
  auto const observed_filter_count = snapshot.logical_filter_count;
  std::vector<membership_entry> entries;
  for (auto const& column : snapshot.columns) {
    if (column.column >= num_cols) { continue; }
    for (auto const& f : column.filters) {
      if (!f->is_available_on_device(device_id)) { continue; }
      auto const* applicable = dynamic_cast<sirius::op::sirius_mask_applicable const*>(f.get());
      if (!applicable) { continue; }
      auto recorded = gate ? gate->filter_keep_ratio(f.get(), observed_filter_count) : std::nullopt;
      if (recorded && dynamic_filter_gate::filter_skippable(*recorded)) { continue; }
      entries.push_back({column.column, applicable, f.get(), recorded});
    }
  }
  std::stable_sort(entries.begin(), entries.end(), [](auto const& a, auto const& b) {
    return a.recorded.value_or(1.0) < b.recorded.value_or(1.0);
  });

  for (auto const& e : entries) {
    if (current.num_rows() == 0) { break; }
    auto const& probe = current.column(static_cast<cudf::size_type>(e.col_idx));
    auto const kept   = cascade_step(e.filter->compute_mask(probe, device_id, stream, mr));
    if (gate && !e.recorded) {
      gate->record_filter_keep_ratio(e.identity, kept, observed_filter_count);
    }
  }

  if (!owned) { return nullptr; }
  auto const* mode_name = mode == dynamic_filter_apply_mode::include_ast_row_masks
                            ? "include_ast_row_masks"
                            : "membership_masks_only";
  SIRIUS_LOG_DEBUG("[apply_dynamic_filters] device={} mode={} apply: {} -> {} rows.",
                   device_id,
                   mode_name,
                   input.num_rows(),
                   owned->num_rows());
  return owned;
}

std::optional<double> dynamic_filter_gate::filter_keep_ratio(
  sirius::op::sirius_dynamic_filter const* filter, std::size_t observed_filter_count) const
{
  std::scoped_lock lock(_filter_ratios_mu);
  auto it = _filter_keep_ratios.find(filter);
  if (it == _filter_keep_ratios.end()) { return std::nullopt; }
  // A skippable verdict is permanent: rechecking it would cost the very kernel the verdict
  // avoids, and a wrong skip only forfeits pruning, never correctness.
  if (filter_skippable(it->second.kept)) { return it->second.kept; }
  // Channel growth changes the rows reaching a selective filter, so remeasure its stale ratio.
  if (it->second.observed_filter_count < observed_filter_count) { return std::nullopt; }
  return it->second.kept;
}

void dynamic_filter_gate::record_filter_keep_ratio(sirius::op::sirius_dynamic_filter const* filter,
                                                   double kept,
                                                   std::size_t observed_filter_count)
{
  std::scoped_lock lock(_filter_ratios_mu);
  auto const it = _filter_keep_ratios.find(filter);
  if (it != _filter_keep_ratios.end() &&
      it->second.observed_filter_count >= observed_filter_count) {
    return;  // already measured against at least this many filters -- no new information
  }
  _filter_keep_ratios.insert_or_assign(
    filter, filter_measurement{.kept = kept, .observed_filter_count = observed_filter_count});
  if (filter_skippable(kept)) {
    SIRIUS_LOG_DEBUG(
      "[apply_dynamic_filters] per-filter gate: marginal kept {:.3f} against {} filters -> SKIP "
      "filter permanently.",
      kept,
      observed_filter_count);
  }
}

bool dynamic_filter_gate::applicable(std::size_t observed_update_count) const
{
  if (observed_update_count == 0) { return false; }
  if (_state.load(std::memory_order_relaxed) != state::disabled) { return true; }
  // Count growth after a disable decision permits one measurement, like channel growth above.
  return observed_update_count > _decided_marker.load(std::memory_order_relaxed);
}

bool dynamic_filter_gate::applicable(sirius::op::dynamic_filter_snapshot const& snap) const
{
  if (snap.logical_filter_count == 0) { return false; }
  if (_state.load(std::memory_order_relaxed) != state::disabled) { return true; }
  // Generation growth after a disable decision permits one measurement: replacement never grows
  // filter_count, so the generation is the scan-side change signal.
  return snap.generation > _decided_marker.load(std::memory_order_relaxed);
}

void dynamic_filter_gate::record_keep_ratio(std::size_t rows_before,
                                            std::size_t rows_after,
                                            std::uint64_t observed_marker)
{
  if (rows_before == 0) { return; }

  // Serialize state and marker updates so an older batch cannot overwrite ACTIVE.
  std::scoped_lock decision_lock(_decision_mu);
  auto const current = _state.load(std::memory_order_relaxed);
  if (current == state::active) { return; }
  if (current == state::disabled &&
      observed_marker <= _decided_marker.load(std::memory_order_relaxed)) {
    return;  // same change-signal value the disabling batch saw — no new information
  }
  auto const kept = static_cast<double>(rows_after) / static_cast<double>(rows_before);
  _decided_marker.store(observed_marker, std::memory_order_relaxed);
  _state.store(kept > _keep_threshold ? state::disabled : state::active, std::memory_order_relaxed);
  SIRIUS_LOG_DEBUG("[apply_dynamic_filters] selectivity gate: kept {:.3f} (marker {}) -> {}.",
                   kept,
                   observed_marker,
                   kept > _keep_threshold ? "DISABLED" : "ACTIVE");
}

std::unique_ptr<cudf::table> apply_dynamic_filters_gated_view(
  cudf::table_view const& input,
  sirius::op::dynamic_filter_snapshot const& snapshot,
  dynamic_filter_gate& gate,
  rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode,
  int device_id)
{
  if (!gate.applicable(snapshot)) { return nullptr; }
  auto const rows_before = input.num_rows();
  auto filtered = apply_dynamic_filters_to_view(input, snapshot, stream, mode, &gate, device_id);
  // A device with no applicable local filter must not train the gate shared with other devices.
  if (!filtered) { return nullptr; }
  gate.record_keep_ratio(
    rows_before, static_cast<std::size_t>(filtered->num_rows()), snapshot.generation);
  return filtered;
}

}  // namespace sirius::op::scan
