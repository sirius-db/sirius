/* Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0 */
#pragma once
#include "op/scan/arrow/arrow_input_profile.hpp"

#include <cudf/table/table.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <duckdb/common/arrow/arrow.hpp>

#include <memory>

namespace cucascade::exec {
class submission;
}

namespace sirius::op::scan::detail {
struct arrow_device_validation_status {
  unsigned invalid = 0, nulls = 0;
};
// Called only after metadata/extents/device/readiness validation on the task stream.
// Validates payloads before allocating destination columns. Never aliases source storage.
// The submission retains the readback owner through confirmed completion.
std::unique_ptr<cudf::table> import_arrow_device_columns(
  ArrowArray const&,
  arrow_bound_schema const&,
  arrow_source_profile const&,
  std::shared_ptr<arrow_device_validation_status>,
  cucascade::exec::submission&,
  rmm::device_async_resource_ref);
// The caller retains the readback storage through confirmed stream completion.
std::unique_ptr<cudf::table> import_arrow_device_columns(ArrowArray const&,
                                                         arrow_bound_schema const&,
                                                         arrow_source_profile const&,
                                                         arrow_device_validation_status&,
                                                         ::cuda::stream_ref,
                                                         rmm::device_async_resource_ref);
}  // namespace sirius::op::scan::detail
