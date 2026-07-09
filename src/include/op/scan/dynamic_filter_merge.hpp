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

#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <op/scan/dynamic_filter_gate.hpp>
#include <op/scan/scan_plan.hpp>
#include <op/sirius_dynamic_filter.hpp>

#include <memory>

namespace sirius::op::scan {

/// Controls post-decode row-mask application.
/// - membership_masks_only applies only mask-applicable membership filters (IN-list / Bloom); use
///   it when AST filters already ran as scan-time row-group pruning.
/// - include_ast_row_masks also evaluates AST-capable filters row-wise with cudf::compute_column,
///   for materialized inputs that had no scan-time pruning phase.
enum class dynamic_filter_apply_mode { membership_masks_only, include_ast_row_masks };

/// @brief AND-merge AST-capable filters from @p filters into @p tree, AND-ing with @p
/// existing_root when non-null. Returns the new root, or @p existing_root unchanged when no
/// fragment contributed (empty set, filters lack the AST capability, or referenced columns are
/// all hive partitions).
///
/// Column index resolution uses @p plan: hive-partition columns are skipped (their values come
/// from the file path, not the parquet file itself), DATA columns resolve through
/// @c plan.output_layout to @c plan.data_columns[i].name. Emitted column references use
/// @c cudf::ast::column_name_reference so the result is compatible with the parquet reader's
/// @c set_filter API.
///
/// The returned pointer (when non-null) points into @p tree and remains valid for the lifetime
/// of @p tree. Device scalars referenced by filter literals are owned by the filter objects in
/// @p filters and must outlive any installed AST built from this call.
/// @p device_id selects device-local filter storage; -1 resolves to the current CUDA device.
[[nodiscard]] cudf::ast::expression const* merge_dynamic_filters_into_ast(
  cudf::ast::tree& tree,
  cudf::ast::expression const* existing_root,
  sirius::op::sirius_dynamic_filter_set const& filters,
  scan_plan const& plan,
  int device_id = -1);

/// @brief Drop rows of @p input that fail the dynamic filters in @p filters, returning only the
/// gathered survivors (the caller keeps @p input).
///
/// @p input must be in the scan's output layout: a filtered column index is its position in
/// @p input directly. Returns nullptr when no filter contributed a mask.
///
/// Two filter kinds combine into one keep-mask: AST-lowerable zone-maps (via @c
/// cudf::compute_column, only in @ref dynamic_filter_apply_mode::include_ast_row_masks) and
/// mask-applicable membership filters (IN-list / bloom). Membership filters apply as a
/// most-selective-first CASCADE: each filter's mask is computed only over the rows surviving the
/// filters before it. When @p gate is non-null it supplies per-filter marginal keep ratios:
/// measured-useless filters are dropped from the cascade, and each filter's first ratio is recorded
/// back. Pass null (tests, gate-less callers) to apply everything in channel insertion order.
/// @p device_id selects device-local filter storage; -1 resolves to the current CUDA device.
/// @note Should not be used by callers -- useful for testing. Callers should use
/// apply_dynamic_filters_gated_view instead, which wraps this with a gate early-out and keep-ratio
/// recording.
[[nodiscard]] std::unique_ptr<cudf::table> apply_dynamic_filters_to_view(
  cudf::table_view const& input,
  sirius::op::sirius_dynamic_filter_set const& filters,
  rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode = dynamic_filter_apply_mode::include_ast_row_masks,
  dynamic_filter_gate* gate      = nullptr,
  int device_id                  = -1);

/// @brief Apply dynamic filters with the scan-level gate.
///
/// Returns nullptr when the gate declines or no compatible device-local filter contributed a mask.
/// A mask-less apply does not train the gate: a GPU missing a best-effort replica must not disable
/// useful replicas on other GPUs.
/// @p device_id selects device-local filter storage; -1 resolves to the current CUDA device.
[[nodiscard]] std::unique_ptr<cudf::table> apply_dynamic_filters_gated_view(
  cudf::table_view const& input,
  sirius::op::sirius_dynamic_filter_set const& filters,
  dynamic_filter_gate& gate,
  rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode,
  int device_id = -1);

}  // namespace sirius::op::scan
