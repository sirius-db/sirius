/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "op/scan/materialization_contract.hpp"

#include <span>

namespace sirius::op::scan {
// The adapter validates applicability and retains the sorted, nonnegative positions
// for each source. A missing required set must throw in the lookup, never mean empty.
// An exclusive original-file row limit, independent of selected row groups.
using row_limit_lookup = std::function<std::optional<uint64_t>(std::string const&)>;
using position_lookup  = std::function<std::span<int64_t const>(std::string const&)>;
std::unique_ptr<cudf::table> apply_positional_visibility(std::unique_ptr<cudf::table>,
                                                         std::span<batch_row_run const>,
                                                         position_lookup const&,
                                                         ::cuda::stream_ref,
                                                         rmm::device_async_resource_ref,
                                                         row_limit_lookup const& = {});
void apply_positional_visibility(materialized_batch&,
                                 position_lookup const&,
                                 ::cuda::stream_ref,
                                 rmm::device_async_resource_ref);
}  // namespace sirius::op::scan
