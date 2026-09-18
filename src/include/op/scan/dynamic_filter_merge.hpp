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

#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/dynamic_filter_gate.hpp>
#include <op/scan/scan_plan.hpp>

#include <memory>

namespace sirius::op::scan {

/// Post-decode mode: membership-only after scan-time AST filtering, or AST plus membership
/// otherwise.
enum class dynamic_filter_apply_mode { membership_masks_only, include_ast_row_masks };

/**
 * @brief ANDs compatible filters into @p tree
 *
 * Column references follow @p plan; hive partitions are skipped. The existing root is returned
 * when no filter applies. Returned expressions and filter-owned scalars must outlive installed
 * AST. A negative device ID selects the current device.
 */
[[nodiscard]] cudf::ast::expression const* merge_dynamic_filters_into_ast(
  cudf::ast::tree& tree,
  cudf::ast::expression const* existing_root,
  sirius::op::sirius_dynamic_filter_set const& filters,
  scan_plan const& plan,
  int device_id = -1);

/**
 * @brief Gathers rows that pass visible filters, or returns null when no mask applies
 *
 * Input uses scan output layout. A gate may suppress low-value masks; a negative device ID selects
 * the current device.
 */
[[nodiscard]] std::unique_ptr<cudf::table> apply_dynamic_filters_to_view(
  cudf::table_view const& input,
  sirius::op::sirius_dynamic_filter_set const& filters,
  rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode = dynamic_filter_apply_mode::include_ast_row_masks,
  dynamic_filter_gate* gate      = nullptr,
  int device_id                  = -1);

/**
 * @brief Applies filters through the scan-level gate
 *
 * A maskless attempt does not train the gate, preserving useful replicas on other GPUs.
 */
[[nodiscard]] std::unique_ptr<cudf::table> apply_dynamic_filters_gated_view(
  cudf::table_view const& input,
  sirius::op::sirius_dynamic_filter_set const& filters,
  dynamic_filter_gate& gate,
  rmm::cuda_stream_view stream,
  dynamic_filter_apply_mode mode,
  int device_id = -1);

}  // namespace sirius::op::scan
