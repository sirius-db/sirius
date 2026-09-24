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

#include <cuda/stream>

#include <op/dynamic_filter/sirius_dynamic_filter.hpp>
#include <op/scan/dynamic_filter_gate.hpp>
#include <op/scan/scan_plan.hpp>

#include <cstddef>
#include <memory>
#include <optional>
#include <span>

namespace sirius::op::scan {

/**
 * @brief Selects membership-only application after scan-time AST filtering, or AST plus membership
 * otherwise.
 */
enum class dynamic_filter_apply_mode { MEMBERSHIP_MASKS_ONLY, INCLUDE_AST_ROW_MASKS };

namespace detail {

/**
* @brief Compaction strategy for dynamic filter application
*
* @details The compaction strategy is used to determine how and when payload columns are compacted
*          during dynamic filter application.
* - CASCADE: Payload columns are compacted after each filter application.
* - DEFERRED_KEYS: Compact only the next filter's key column and row ids,
                   deferring compaction of payload until all filters have been applied.
* - GATHER_ONCE: AND all masks in original row space, then compact all columns once.
*/
enum class compaction_strategy { CASCADE, DEFERRED_KEYS, GATHER_ONCE };

/**
 * @brief Input parameters for compaction strategy selection.
 */
struct compaction_policy_input {
  std::size_t rows;
  std::optional<std::size_t> input_bytes;
  std::size_t candidate_step_count;
  std::span<std::optional<double> const> membership;
};

/**
 * @brief Chooses the compaction strategy based on the input policy.
 */
[[nodiscard]] compaction_strategy choose_compaction_strategy(
  compaction_policy_input const& input) noexcept;

/**
 * @brief Applies one explicitly selected compaction strategy with test-only invariant checks
 *
 * This seam bypasses strategy policy so tests can compare complete outputs from the same snapshot.
 */
[[nodiscard]] std::unique_ptr<cudf::table> apply_dynamic_filters_to_view_for_testing(
  cudf::table_view const& input,
  sirius::op::dynamic_filter_snapshot const& filters,
  ::cuda::stream_ref stream,
  compaction_strategy strategy,
  dynamic_filter_apply_mode mode = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS,
  dynamic_filter_gate* gate      = nullptr,
  int device_id                  = -1);

}  // namespace detail

/**
 * @brief ANDs compatible filters into @p tree
 *
 * Column references follow @p plan; hive partitions are skipped. The existing root is returned
 * when no filter applies. The caller retains the snapshot through the last GPU use of its
 * filter-owned scalars. A negative device ID selects the current device.
 */
[[nodiscard]] cudf::ast::expression const* merge_dynamic_filters_into_ast(
  cudf::ast::tree& tree,
  cudf::ast::expression const* existing_root,
  sirius::op::dynamic_filter_snapshot const& filters,
  scan_plan const& plan,
  int device_id = -1);

/**
 * @brief Gathers rows that pass visible filters, or returns null when no mask applies
 *
 * Input uses scan output layout. A gate may suppress low-value masks; a negative device ID selects
 * the current device. Submitted work completes before the snapshot can be released, including on
 * exceptional exits; no consumer waits for channel publication. Exact input bytes enable deferred
 * compaction policy decisions; callers without byte accounting retain the conservative cascade.
 */
[[nodiscard]] std::unique_ptr<cudf::table> apply_dynamic_filters_to_view(
  cudf::table_view const& input,
  sirius::op::dynamic_filter_snapshot const& filters,
  ::cuda::stream_ref stream,
  dynamic_filter_apply_mode mode         = dynamic_filter_apply_mode::INCLUDE_AST_ROW_MASKS,
  dynamic_filter_gate* gate              = nullptr,
  int device_id                          = -1,
  std::optional<std::size_t> input_bytes = std::nullopt);

/**
 * @brief Applies filters through the scan-level gate
 *
 * A maskless attempt does not train the gate, preserving useful replicas on other GPUs.
 */
[[nodiscard]] std::unique_ptr<cudf::table> apply_dynamic_filters_gated_view(
  cudf::table_view const& input,
  sirius::op::dynamic_filter_snapshot const& filters,
  dynamic_filter_gate& gate,
  ::cuda::stream_ref stream,
  dynamic_filter_apply_mode mode,
  int device_id                          = -1,
  std::optional<std::size_t> input_bytes = std::nullopt);

}  // namespace sirius::op::scan
