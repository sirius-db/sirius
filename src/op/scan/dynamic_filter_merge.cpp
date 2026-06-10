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
#include <cudf/io/experimental/hybrid_scan.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/transform.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtx3/nvtx3.hpp>

#include <log/logging.hpp>
#include <op/scan/dynamic_filter_merge.hpp>

#include <utility>

namespace sirius::op::scan {

cudf::ast::expression const* merge_dynamic_filters_into_ast(
  cudf::ast::tree& tree,
  cudf::ast::expression const* existing_root,
  sirius::op::sirius_dynamic_filter_set const& filters,
  scan_plan const& plan)
{
  cudf::ast::expression const* root = existing_root;
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

std::unique_ptr<cudf::table> apply_dynamic_filters_to_output_table(
  std::unique_ptr<cudf::table> table,
  sirius::op::sirius_dynamic_filter_set const& filters,
  scan_plan const& plan,
  rmm::cuda_stream_view stream,
  bool include_ast_lowerable)
{
  nvtx3::scoped_range nvtx_range{"dynfilter::apply_output"};
  if (!table || table->num_rows() == 0 || table->num_columns() == 0) { return table; }

  auto const num_cols = static_cast<std::size_t>(table->num_columns());
  auto const mr       = cudf::get_current_device_resource_ref();

  // Accumulate one BOOL keep-mask by AND-ing every applicable filter's contribution; a row survives
  // only if it passes all of them. Output column `i` is plan.output_layout[i], so a filtered
  // consumer column index is that output position directly. LOGICAL_AND propagates nulls, and
  // apply_boolean_mask drops null-or-false rows — so a null key (which can never equi-join) is
  // dropped, as intended.
  std::unique_ptr<cudf::column> mask;
  auto const and_into = [&](std::unique_ptr<cudf::column> m) {
    if (!m) { return; }
    mask = mask ? cudf::binary_operation(mask->view(),
                                         m->view(),
                                         cudf::binary_operator::LOGICAL_AND,
                                         cudf::data_type{cudf::type_id::BOOL8},
                                         stream,
                                         mr)
                : std::move(m);
  };

  // (1) AST-lowerable filters (zone-maps) → one conjoined predicate via compute_column. Skipped on
  // the disk path, where they are already used for row-group pruning at read.
  if (include_ast_lowerable) {
    cudf::ast::tree tree;
    cudf::ast::expression const* root = nullptr;
    for (auto const col_idx : filters.filtered_columns()) {
      if (col_idx >= plan.output_layout.size() || col_idx >= num_cols) { continue; }
      if (plan.output_layout[col_idx].source != scan_plan::output_entry::DATA) {
        continue;
      }  // hive
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
    if (root) { and_into(cudf::compute_column(table->view(), *root, stream, mr)); }
  }

  // (2) Apply-lowerable filters (IN-list / bloom membership) → a BOOL mask per (column, filter).
  for (auto const col_idx : filters.filtered_columns()) {
    if (col_idx >= plan.output_layout.size() || col_idx >= num_cols) { continue; }
    if (plan.output_layout[col_idx].source != scan_plan::output_entry::DATA) { continue; }  // hive
    auto const probe = table->view().column(static_cast<cudf::size_type>(col_idx));
    for (auto const& f : filters.filters_for_column(col_idx)) {
      auto const* applyable = dynamic_cast<sirius::op::sirius_apply_lowerable const*>(f.get());
      if (!applyable) { continue; }
      and_into(applyable->compute_mask(probe, stream, mr));
    }
  }

  if (!mask) { return table; }
  // Keep `table` named so its stream-ordered destruction is sequenced after apply_boolean_mask.
  auto const rows_before = table->num_rows();
  auto filtered          = cudf::apply_boolean_mask(table->view(), mask->view(), stream, mr);
  SIRIUS_LOG_DEBUG("[apply_dynamic_filters] membership/ast={} apply: {} -> {} rows.",
                   include_ast_lowerable,
                   rows_before,
                   filtered->num_rows());
  return filtered;
}

std::vector<std::vector<cudf::size_type>> prune_row_groups_by_dynamic_filters(
  std::vector<cudf::io::parquet::FileMetaData const*> const& per_source_metadata,
  std::vector<std::vector<cudf::size_type>> const& rg_per_src,
  cudf::io::parquet_reader_options const& base_options,
  sirius::op::sirius_dynamic_filter_set const& filters,
  scan_plan const& plan,
  rmm::cuda_stream_view stream,
  std::size_t& pruned_out)
{
  nvtx3::scoped_range nvtx_range{"dynfilter::prune_row_groups"};
  pruned_out = 0;

  // Build the dynamic-only predicate once: column-name references resolve against the (shared)
  // schema, so one tree serves every source. Its lifetime must cover all stats-evaluation calls
  // below — `options` holds a reference into `tree`.
  cudf::ast::tree tree;
  auto const* root = merge_dynamic_filters_into_ast(tree, /*existing_root=*/nullptr, filters, plan);
  if (!root) {  // nothing AST-lowerable to prune by — pass every source through unchanged
    return rg_per_src;
  }
  auto options = base_options;
  options.set_filter(*root);

  std::vector<std::vector<cudf::size_type>> out;
  out.reserve(rg_per_src.size());
  for (std::size_t i = 0; i < rg_per_src.size(); ++i) {
    auto const& rgs = rg_per_src[i];
    auto const* md  = i < per_source_metadata.size() ? per_source_metadata[i] : nullptr;
    if (!md || rgs.empty()) {
      out.push_back(rgs);
      continue;
    }
    // Constructed from the already-parsed footer: no I/O. filter_row_groups_with_stats evaluates
    // the predicate against each given row group's column statistics and returns the conservative
    // survivors (a subset that may contain a match). Missing stats → the group is kept.
    std::unique_ptr<cudf::io::parquet::experimental::hybrid_scan_reader> reader;
    {
      nvtx3::scoped_range r{"dynfilter::reader_ctor"};
      reader = std::make_unique<cudf::io::parquet::experimental::hybrid_scan_reader>(*md, options);
    }
    std::vector<cudf::size_type> survivors;
    {
      nvtx3::scoped_range r{"dynfilter::frg_stats"};
      survivors = reader->filter_row_groups_with_stats(rgs, options, stream);
    }
    pruned_out += rgs.size() - survivors.size();
    out.push_back(std::move(survivors));
  }
  return out;
}

}  // namespace sirius::op::scan
