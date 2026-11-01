/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "op/scan/arrow/arrow_input_profile.hpp"

#include <cudf/table/table.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <duckdb/common/arrow/arrow.hpp>

namespace sirius::op::scan::detail {
// Called only after metadata/extents/device/readiness validation on the task stream.
// Validates payloads before allocating destination columns. Never aliases source storage.
std::unique_ptr<cudf::table> import_arrow_device_columns(ArrowArray const&,
                                                         arrow_bound_schema const&,
                                                         arrow_source_profile const&,
                                                         ::cuda::stream_ref,
                                                         rmm::device_async_resource_ref);
}  // namespace sirius::op::scan::detail
