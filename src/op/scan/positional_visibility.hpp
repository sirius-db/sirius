/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "op/scan/gpu_row_positions.hpp"
#include "op/scan/materialization_contract.hpp"

#include <span>

namespace sirius::op::scan {
// The adapter validates applicability and retains sorted positions for each source.
// Positions outside the selected input ranges are ignored, including legacy negative entries.
// A missing required set must throw in the lookup, never mean empty.
using position_lookup = std::function<std::span<int64_t const>(std::string const&)>;
// Consumes the final internal position owner; user virtual columns remain in the table.
std::unique_ptr<cudf::table> apply_positional_visibility(std::unique_ptr<cudf::table>,
                                                         std::unique_ptr<gpu_row_positions>,
                                                         position_lookup const&,
                                                         ::cuda::stream_ref,
                                                         rmm::device_async_resource_ref);
std::unique_ptr<cudf::table> apply_positional_visibility(std::unique_ptr<cudf::table>,
                                                         std::span<batch_row_run const>,
                                                         position_lookup const&,
                                                         ::cuda::stream_ref,
                                                         rmm::device_async_resource_ref);
void apply_positional_visibility(materialized_batch&,
                                 position_lookup const&,
                                 ::cuda::stream_ref,
                                 rmm::device_async_resource_ref);
}  // namespace sirius::op::scan
